use std::sync::Arc;

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, Path, State, multipart::MultipartRejection},
    http::{StatusCode, header},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use futures::{Stream, stream};
use koharu_translator::Language;
use serde::Serialize;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

use crate::jobs::{Input, Job, Jobs, Translation};

#[derive(OpenApi)]
#[openapi(paths(submit, status, events, cancel, result))]
struct Api;

pub(crate) fn router(jobs: Jobs) -> Router {
    Router::new()
        .route("/translations", post(submit))
        .route("/translations/{id}", get(status))
        .route("/translations/{id}/events", get(events))
        .route("/translations/{id}/cancel", post(cancel))
        .route("/translations/{id}/result", get(result))
        .layer(DefaultBodyLimit::disable())
        .with_state(jobs)
        .merge(SwaggerUi::new("/docs").url("/openapi.json", Api::openapi()))
}

#[derive(ToSchema)]
struct Upload {
    /// JPEG, PNG, or WebP page.
    #[schema(value_type = String, format = Binary)]
    image: Bytes,
    /// Language tag or name. Defaults to English.
    #[schema(value_type = Option<String>, example = "en")]
    target_language: Option<Language>,
    /// Additional translation guidance.
    instructions: Option<String>,
}

impl Upload {
    async fn parse(mut multipart: Multipart) -> Result<Self, ApiError> {
        let mut image = None;
        let mut target_language = None;
        let mut instructions = None;
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|_| ApiError::MULTIPART)?
        {
            match field.name() {
                Some("image") if image.is_none() => {
                    image = Some(field.bytes().await.map_err(|_| ApiError::MULTIPART)?);
                }
                Some("target_language") if target_language.is_none() => {
                    target_language = Some(
                        field
                            .text()
                            .await
                            .map_err(|_| ApiError::MULTIPART)?
                            .parse()
                            .map_err(|_| {
                                ApiError(StatusCode::BAD_REQUEST, "Unknown target language")
                            })?,
                    );
                }
                Some("instructions") if instructions.is_none() => {
                    instructions = Some(field.text().await.map_err(|_| ApiError::MULTIPART)?);
                }
                _ => {
                    return Err(ApiError(
                        StatusCode::BAD_REQUEST,
                        "Unexpected or repeated multipart field",
                    ));
                }
            }
        }
        Ok(Self {
            image: image.ok_or(ApiError(StatusCode::BAD_REQUEST, "Missing image field"))?,
            target_language,
            instructions,
        })
    }

    fn validate(self) -> Result<Input, ApiError> {
        let format = image::guess_format(&self.image).map_err(|_| ApiError::IMAGE)?;
        let media_type = match format {
            image::ImageFormat::Jpeg => "image/jpeg",
            image::ImageFormat::Png => "image/png",
            image::ImageFormat::WebP => "image/webp",
            _ => return Err(ApiError::IMAGE),
        };
        let decoded = image::load_from_memory_with_format(&self.image, format)
            .map_err(|_| ApiError::IMAGE)?;
        Ok(Input {
            source: Arc::from(self.image.as_ref()),
            media_type,
            width: decoded.width(),
            height: decoded.height(),
            language: self.target_language.unwrap_or(Language::English),
            instructions: self.instructions,
        })
    }
}

#[derive(Serialize, ToSchema)]
struct Problem {
    status: u16,
    title: &'static str,
    detail: &'static str,
}

#[derive(Debug)]
struct ApiError(StatusCode, &'static str);

impl ApiError {
    const MULTIPART: Self = Self(StatusCode::BAD_REQUEST, "Expected multipart/form-data");
    const IMAGE: Self = Self(
        StatusCode::BAD_REQUEST,
        "Expected a valid JPEG, PNG, or WebP image",
    );
    const NOT_FOUND: Self = Self(StatusCode::NOT_FOUND, "Translation job not found");
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let Self(status, detail) = self;
        (
            status,
            [(header::CONTENT_TYPE, "application/problem+json")],
            Json(Problem {
                status: status.as_u16(),
                title: status.canonical_reason().unwrap_or("Error"),
                detail,
            }),
        )
            .into_response()
    }
}

fn find(jobs: &Jobs, id: &str) -> Result<Arc<Job>, ApiError> {
    jobs.get(id).ok_or(ApiError::NOT_FOUND)
}

/// Submit a page for translation.
///
/// The accepted job runs independently of the connection.
#[utoipa::path(
    post, path = "/translations",
    request_body(content = Upload, content_type = "multipart/form-data"),
    responses(
        (status = 202, description = "Translation queued", body = Translation,
            headers(("Location" = String, description = "Translation status URL"))),
        (status = 400, description = "Invalid upload", body = Problem, content_type = "application/problem+json"),
        (status = 500, description = "Upload validation failed", body = Problem, content_type = "application/problem+json"),
        (status = 503, description = "Worker unavailable", body = Problem, content_type = "application/problem+json"),
    )
)]
async fn submit(
    State(jobs): State<Jobs>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<impl IntoResponse, ApiError> {
    let upload = Upload::parse(multipart.map_err(|_| ApiError::MULTIPART)?).await?;
    let input = tokio::task::spawn_blocking(move || upload.validate())
        .await
        .map_err(|error| {
            tracing::error!(%error, "image validation task failed");
            ApiError(StatusCode::INTERNAL_SERVER_ERROR, "Image validation failed")
        })??;
    let translation = jobs.submit(input).ok_or(ApiError(
        StatusCode::SERVICE_UNAVAILABLE,
        "Translation worker unavailable",
    ))?;
    Ok((
        StatusCode::ACCEPTED,
        [(
            header::LOCATION,
            format!("/translations/{}", translation.id),
        )],
        Json(translation),
    ))
}

/// Get the current status, stage timings, and result URL when ready.
///
/// Jobs and results are kept in memory for one hour after reaching a terminal state.
/// A server restart clears them immediately.
#[utoipa::path(
    get, path = "/translations/{id}",
    params(("id" = uuid::Uuid, Path, description = "Translation job ID")),
    responses(
        (status = 200, description = "Translation status", body = Translation),
        (status = 404, description = "Unknown or expired job", body = Problem, content_type = "application/problem+json"),
    )
)]
async fn status(
    State(jobs): State<Jobs>,
    Path(id): Path<String>,
) -> Result<Json<Translation>, ApiError> {
    Ok(Json(find(&jobs, &id)?.snapshot()))
}

/// Stream live translation status over Server-Sent Events.
///
/// Each connection starts with a `snapshot` event. It is followed by `progress`
/// events with the latest state, then `completed`, `failed`, or `cancelled`.
/// Every event carries the same JSON as the status endpoint. A terminal
/// snapshot or event ends the stream. EventSource clients should close their
/// connection when they see a terminal status.
#[utoipa::path(
    get, path = "/translations/{id}/events",
    params(("id" = uuid::Uuid, Path, description = "Translation job ID")),
    responses(
        (status = 200, description = "Live translation status", body = String, content_type = "text/event-stream"),
        (status = 404, description = "Unknown or expired job", body = Problem, content_type = "application/problem+json"),
    )
)]
async fn events(
    State(jobs): State<Jobs>,
    Path(id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, axum::Error>>>, ApiError> {
    let updates = find(&jobs, &id)?.subscribe();
    let stream = stream::unfold(Some((updates, true)), |next| async move {
        let (mut updates, first) = next?;
        if !first {
            updates.changed().await.ok()?;
        }
        let snapshot = updates.borrow_and_update().snapshot();
        let terminal = snapshot.status.terminal_event();
        let name = if first {
            "snapshot"
        } else {
            terminal.unwrap_or("progress")
        };
        let event = Event::default().event(name).json_data(snapshot);
        Some((event, terminal.is_none().then_some((updates, false))))
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// Cancel a queued job or request cooperative cancellation of a running job.
#[utoipa::path(
    post, path = "/translations/{id}/cancel",
    params(("id" = uuid::Uuid, Path, description = "Translation job ID")),
    responses(
        (status = 200, description = "Cancellation requested or complete", body = Translation),
        (status = 404, description = "Unknown or expired job", body = Problem, content_type = "application/problem+json"),
        (status = 409, description = "Job already completed or failed", body = Problem, content_type = "application/problem+json"),
    )
)]
async fn cancel(
    State(jobs): State<Jobs>,
    Path(id): Path<String>,
) -> Result<Json<Translation>, ApiError> {
    find(&jobs, &id)?.cancel().map(Json).ok_or(ApiError(
        StatusCode::CONFLICT,
        "Translation job already finished",
    ))
}

#[derive(ToSchema)]
#[schema(value_type = String, format = Binary)]
struct Png(Bytes);

impl IntoResponse for Png {
    fn into_response(self) -> Response {
        ([(header::CONTENT_TYPE, "image/png")], self.0).into_response()
    }
}

/// Download a completed translation as a PNG.
#[utoipa::path(
    get, path = "/translations/{id}/result",
    params(("id" = uuid::Uuid, Path, description = "Translation job ID")),
    responses(
        (status = 200, description = "Translated image", body = inline(Png), content_type = "image/png"),
        (status = 404, description = "Unknown or expired job", body = Problem, content_type = "application/problem+json"),
        (status = 409, description = "Job has no completed result", body = Problem, content_type = "application/problem+json"),
    )
)]
async fn result(State(jobs): State<Jobs>, Path(id): Path<String>) -> Result<Png, ApiError> {
    let png = find(&jobs, &id)?.result().ok_or(ApiError(
        StatusCode::CONFLICT,
        "Translation job has no completed result",
    ))?;
    Ok(Png(png))
}

#[cfg(test)]
mod tests;
