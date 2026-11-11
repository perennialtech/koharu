use std::io::Cursor;

use axum::{
    body::{Body, to_bytes},
    http::{Method, Request},
};
use futures::StreamExt;
use koharu_pipeline::{Progress, Stage};
use koharu_scene::EntityId;
use serde_json::Value;
use tower::ServiceExt;

use super::*;
use crate::jobs::Failure;

async fn request(app: &Router, method: Method, uri: &str) -> Response {
    app.clone()
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn json(response: Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
}

fn png() -> Vec<u8> {
    let mut png = Cursor::new(Vec::new());
    image::DynamicImage::new_rgba8(2, 2)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    png.into_inner()
}

async fn upload(app: &Router, image: &[u8]) -> Response {
    let mut body = b"--page\r\nContent-Disposition: form-data; name=\"image\"; filename=\"page.png\"\r\nContent-Type: image/png\r\n\r\n".to_vec();
    body.extend_from_slice(image);
    body.extend_from_slice(b"\r\n--page\r\nContent-Disposition: form-data; name=\"target_language\"\r\n\r\nja\r\n--page\r\nContent-Disposition: form-data; name=\"instructions\"\r\n\r\nKeep character names\r\n--page--\r\n");
    app.clone()
        .oneshot(
            Request::post("/translations")
                .header(header::CONTENT_TYPE, "multipart/form-data; boundary=page")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn submit_page(app: &Router, image: &[u8]) -> String {
    let response = upload(app, image).await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let location = response.headers()[header::LOCATION]
        .to_str()
        .unwrap()
        .to_owned();
    let status = json(response).await;
    assert_eq!(
        location,
        format!("/translations/{}", status["id"].as_str().unwrap())
    );
    assert_eq!(status["status"], "queued");
    location
}

#[tokio::test]
async fn progress_and_result_are_available_across_connections() {
    let (jobs, mut queue) = Jobs::new();
    let app = router(jobs);
    let png = png();
    let location = submit_page(&app, &png).await;
    let events = format!("{location}/events");
    let result = format!("{location}/result");
    assert_eq!(
        request(&app, Method::GET, &result).await.status(),
        StatusCode::CONFLICT
    );

    let mut first = request(&app, Method::GET, &events)
        .await
        .into_body()
        .into_data_stream();
    let frame = first.next().await.unwrap().unwrap();
    assert!(frame.starts_with(b"event: snapshot\n"));
    assert!(
        std::str::from_utf8(&frame)
            .unwrap()
            .contains("\"status\":\"queued\"")
    );

    let job = queue.recv().await.unwrap();
    let input = job.start().unwrap();
    assert_eq!(input.language, Language::Japanese);
    assert_eq!(input.instructions.as_deref(), Some("Keep character names"));
    assert_eq!(input.source.as_ref(), png);

    job.progress(Progress::Finished {
        page: EntityId::new(),
        stage: Stage::Detection,
        model: "detector".to_owned(),
        elapsed: std::time::Duration::from_millis(30),
    });
    let frame = first.next().await.unwrap().unwrap();
    assert!(frame.starts_with(b"event: progress\n"));
    assert!(
        std::str::from_utf8(&frame)
            .unwrap()
            .contains("\"elapsed_ms\":30")
    );
    drop(first);

    let status = json(request(&app, Method::GET, &location).await).await;
    assert_eq!(status["status"], "running");
    assert_eq!(status["completed"], 1);
    assert_eq!(status["stages"][0]["elapsed_ms"], 30);

    let mut reconnected = request(&app, Method::GET, &events)
        .await
        .into_body()
        .into_data_stream();
    let frame = reconnected.next().await.unwrap().unwrap();
    assert!(frame.starts_with(b"event: snapshot\n"));
    assert!(
        std::str::from_utf8(&frame)
            .unwrap()
            .contains("\"completed\":1")
    );
    job.finish(Ok(Some(Bytes::copy_from_slice(&png))));
    let frame = reconnected.next().await.unwrap().unwrap();
    assert!(frame.starts_with(b"event: completed\n"));
    assert!(reconnected.next().await.is_none());

    let status = json(request(&app, Method::GET, &location).await).await;
    assert_eq!(status["result_url"], result);
    let response = request(&app, Method::GET, &result).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "image/png");
    assert_eq!(
        to_bytes(response.into_body(), usize::MAX).await.unwrap(),
        png
    );

    let mut terminal = request(&app, Method::GET, &events)
        .await
        .into_body()
        .into_data_stream();
    let frame = terminal.next().await.unwrap().unwrap();
    assert!(frame.starts_with(b"event: snapshot\n"));
    assert!(
        std::str::from_utf8(&frame)
            .unwrap()
            .contains("\"status\":\"completed\"")
    );
    assert!(terminal.next().await.is_none());
}

#[tokio::test]
async fn shutdown_cancels_jobs_and_closes_active_events() {
    let (jobs, mut queue) = Jobs::new();
    let app = router(jobs.clone());
    let png = png();
    let running = submit_page(&app, &png).await;
    let queued = submit_page(&app, &png).await;

    let mut events = request(&app, Method::GET, &format!("{running}/events"))
        .await
        .into_body()
        .into_data_stream();
    assert!(
        events
            .next()
            .await
            .unwrap()
            .unwrap()
            .starts_with(b"event: snapshot\n")
    );

    let first = queue.recv().await.unwrap();
    assert!(first.start().is_some());
    let second = queue.recv().await.unwrap();
    jobs.shutdown();

    assert!(first.stop_token().stopped());
    assert_eq!(first.snapshot().status, crate::jobs::Status::Cancelling);
    assert_eq!(second.snapshot().status, crate::jobs::Status::Cancelled);
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(1), events.next())
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        upload(&app, &png).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        json(request(&app, Method::GET, &queued).await).await["status"],
        "cancelled"
    );
}

#[tokio::test]
async fn cancels_queued_and_running_jobs_independently() {
    let (jobs, mut queue) = Jobs::new();
    let app = router(jobs);
    let png = png();
    let running = submit_page(&app, &png).await;
    let queued = submit_page(&app, &png).await;
    assert_ne!(running, queued);

    let first = queue.recv().await.unwrap();
    assert!(first.start().is_some());
    let cancel = |location: &str| format!("{location}/cancel");
    let response = request(&app, Method::POST, &cancel(&queued)).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(json(response).await["status"], "cancelled");
    assert!(queue.recv().await.unwrap().start().is_none());

    let response = request(&app, Method::POST, &cancel(&running)).await;
    assert_eq!(json(response).await["status"], "cancelling");
    assert!(first.stop_token().stopped());
    first.finish(Ok(Some(Bytes::from(png))));
    assert_eq!(
        json(request(&app, Method::GET, &running).await).await["status"],
        "cancelled"
    );
    assert_eq!(
        request(&app, Method::GET, &format!("{running}/result"))
            .await
            .status(),
        StatusCode::CONFLICT
    );
}

#[tokio::test]
async fn serves_openapi_and_reports_failures() {
    let (jobs, mut queue) = Jobs::new();
    let app = router(jobs);
    let spec = json(request(&app, Method::GET, "/openapi.json").await).await;
    assert_eq!(spec["paths"].as_object().unwrap().len(), 5);
    assert_eq!(
        spec["components"]["schemas"]["Upload"]["properties"]["image"]["format"],
        "binary"
    );
    assert_eq!(
        spec["paths"]["/translations/{id}/result"]["get"]["responses"]["200"]["content"]["image/png"]
            ["schema"]["format"],
        "binary"
    );
    assert_eq!(
        request(&app, Method::GET, "/docs/").await.status(),
        StatusCode::OK
    );
    assert_eq!(
        request(&app, Method::POST, "/translate").await.status(),
        StatusCode::NOT_FOUND
    );

    let response = upload(&app, b"not an image").await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "application/problem+json"
    );
    assert_eq!(json(response).await["status"], 400);
    assert!(queue.try_recv().is_err());

    let location = submit_page(&app, &png()).await;
    let job = queue.recv().await.unwrap();
    assert!(job.start().is_some());
    job.progress(Progress::Loading {
        page: EntityId::new(),
        stage: Stage::Detection,
        model: "detector".to_owned(),
    });
    job.finish(Err(Failure::processing()));
    let status = json(request(&app, Method::GET, &location).await).await;
    assert_eq!(status["status"], "failed");
    assert_eq!(status["error"]["code"], "processing_failed");
    assert_eq!(status["stages"][0]["status"], "cancelled");
}
