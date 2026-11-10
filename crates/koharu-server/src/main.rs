//! A single-process HTTP interface to Koharu's headless translation pipeline.

use std::{collections::BTreeMap, io, net::SocketAddr, path::PathBuf, sync::Arc};

use anyhow::{Context, Result, bail, ensure};
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::post,
};
use clap::{Parser, ValueEnum};
use koharu_config::Config;
use koharu_pipeline::{
    Committer, Flux2KleinConfig, InpaintingModel, Operation, Pipeline, PipelineConfig, Progress,
    Request, RequestTranslation, RoremMixedConfig, Scope, StageOutput,
};
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::Renderer;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft, Session};
use koharu_translator::{Language, ModelSelection, Provider, ProvidersConfig};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tracing_subscriber::EnvFilter;
use url::Url;

const CACHE_VERSION: &[u8] = b"koharu-http-png-v1";

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProviderChoice {
    #[value(name = "local")]
    Local,
    #[value(name = "openrouter")]
    OpenRouter,
    #[value(name = "openai-compatible")]
    OpenAiCompatible,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InpaintingChoice {
    Lama,
    AotInpainting,
    Flux2Klein,
    RoremMixed,
}

#[derive(Debug, Parser)]
#[command(version, about = "Translate manga page images over HTTP")]
struct Arguments {
    /// Address for the HTTP listener.
    #[arg(long, env = "KOHARU_LISTEN", default_value = "0.0.0.0:8080")]
    listen: SocketAddr,

    /// Directory for cached translated PNG images.
    #[arg(long, env = "KOHARU_CACHE_DIR", default_value_os_t = default_cache_dir())]
    cache_dir: PathBuf,

    /// Local inference or an explicitly selected hosted translator.
    #[arg(long, env = "KOHARU_PROVIDER", value_enum, default_value = "local")]
    provider: ProviderChoice,

    /// Translation model. Local defaults to gemma4-e4b-it; hosted requires a model.
    #[arg(long, env = "KOHARU_MODEL")]
    model: Option<String>,

    /// OpenAI-compatible endpoint, including its /v1 path.
    #[arg(long, env = "KOHARU_OPENAI_COMPATIBLE_BASE_URL")]
    openai_compatible_base_url: Option<Url>,

    /// Inpainting model used to erase original text.
    #[arg(long, env = "KOHARU_INPAINTING", value_enum, default_value = "lama")]
    inpainting: InpaintingChoice,
}

fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("koharu/translations")
}

impl Arguments {
    fn configs(&self) -> Result<(PipelineConfig, ProvidersConfig)> {
        let mut pipeline = PipelineConfig::default();
        let mut providers = ProvidersConfig::default();
        pipeline.translation.model = match self.provider {
            ProviderChoice::Local => {
                if let Some(model) = self.model.as_deref() {
                    ensure!(
                        model == "gemma4-e4b-it",
                        "only gemma4-e4b-it is supported locally"
                    );
                }
                ModelSelection::default()
            }
            ProviderChoice::OpenRouter | ProviderChoice::OpenAiCompatible => {
                let model = self
                    .model
                    .as_deref()
                    .filter(|model| !model.is_empty())
                    .context("--model is required for hosted translation")?;
                if matches!(self.provider, ProviderChoice::OpenRouter) {
                    ensure!(
                        std::env::var("OPENROUTER_API_KEY").is_ok_and(|key| !key.is_empty()),
                        "OPENROUTER_API_KEY is required for OpenRouter"
                    );
                }
                ModelSelection {
                    provider: match self.provider {
                        ProviderChoice::OpenRouter => Provider::OpenRouter,
                        ProviderChoice::OpenAiCompatible => Provider::OpenAiCompatible,
                        ProviderChoice::Local => unreachable!(),
                    },
                    model: Some(model.to_owned()),
                    quantization: None,
                    vision: true,
                    reasoning: true,
                }
            }
        };
        if self.openai_compatible_base_url.is_some()
            && !matches!(self.provider, ProviderChoice::OpenAiCompatible)
        {
            bail!("--openai-compatible-base-url requires --provider openai-compatible");
        }
        if let Some(url) = &self.openai_compatible_base_url {
            providers.openai_compatible.base_url = Some(url.clone());
        }
        pipeline.inpainting = match self.inpainting {
            InpaintingChoice::Lama => InpaintingModel::LaMa {},
            InpaintingChoice::AotInpainting => InpaintingModel::AotInpainting {},
            InpaintingChoice::Flux2Klein => {
                InpaintingModel::Flux2Klein(Flux2KleinConfig::default())
            }
            InpaintingChoice::RoremMixed => {
                InpaintingModel::RoremMixed(RoremMixedConfig::default())
            }
        };
        Ok((pipeline, providers))
    }
}

struct Server {
    pipeline: Pipeline,
    renderer: Renderer,
    rasterizer: Rasterizer,
    cache_dir: PathBuf,
    model_key: Vec<u8>,
    admission: tokio::sync::Mutex<()>,
}

#[derive(Deserialize)]
struct TranslateQuery {
    target_language: Option<Language>,
    instructions: Option<String>,
}

enum ApiError {
    BadImage,
    Internal(anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            Self::BadImage => (
                StatusCode::BAD_REQUEST,
                "Expected a PNG, JPEG or WebP image",
            )
                .into_response(),
            Self::Internal(error) => {
                tracing::error!(error = ?error, "image translation failed");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Image translation failed",
                )
                    .into_response()
            }
        }
    }
}

fn png_response(bytes: Vec<u8>) -> Response {
    ([(header::CONTENT_TYPE, "image/png")], bytes).into_response()
}

async fn translate(
    State(server): State<Arc<Server>>,
    Query(query): Query<TranslateQuery>,
    body: Bytes,
) -> std::result::Result<Response, ApiError> {
    let language = query.target_language.unwrap_or(Language::English);
    let _lane = server.admission.lock().await;
    let key = cache_key(
        &server.model_key,
        &body,
        language,
        query.instructions.as_deref(),
    );
    let path = server.cache_dir.join(format!("{key}.png"));
    match tokio::fs::read(&path).await {
        Ok(png) => {
            tracing::info!(%key, "translation cache hit");
            return Ok(png_response(png));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(ApiError::Internal(error.into())),
    }
    tracing::info!(%key, "translation cache miss");
    let image_format = image::guess_format(&body).map_err(|_| ApiError::BadImage)?;
    let media_type = match image_format {
        image::ImageFormat::Jpeg => "image/jpeg",
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::WebP => "image/webp",
        _ => return Err(ApiError::BadImage),
    };
    let image =
        image::load_from_memory_with_format(&body, image_format).map_err(|_| ApiError::BadImage)?;
    let png = server
        .render(
            &body,
            media_type,
            image.width(),
            image.height(),
            language,
            query.instructions,
        )
        .await
        .map_err(ApiError::Internal)?;
    let pending = server
        .cache_dir
        .join(format!(".{key}.{}.tmp", std::process::id()));
    tokio::fs::write(&pending, &png)
        .await
        .map_err(|error| ApiError::Internal(error.into()))?;
    if let Err(error) = tokio::fs::rename(&pending, &path).await {
        let _ = tokio::fs::remove_file(&pending).await;
        return Err(ApiError::Internal(error.into()));
    }
    Ok(png_response(png))
}

impl Server {
    async fn render(
        &self,
        source: &[u8],
        media_type: &'static str,
        width: u32,
        height: u32,
        language: Language,
        instructions: Option<String>,
    ) -> Result<Vec<u8>> {
        let mut session = Session::memory().await?;
        let mut page = None;
        let patch = session.snapshot().patch(|edit| {
            let id = edit.add_page(
                PageDraft::new("page", f64::from(width), f64::from(height)),
                At::End,
            )?;
            edit.set_asset(
                id,
                &AssetRole::new("source")?,
                AssetInput::new(
                    Arc::<[u8]>::from(source),
                    media_type,
                    AssetMetadata {
                        width: Some(width),
                        height: Some(height),
                        attributes: BTreeMap::new(),
                    },
                ),
            )?;
            page = Some(id);
            Ok(())
        })?;
        session.commit(patch).await?;
        let page = page.context("page ID was not assigned")?;
        let snapshot = session.snapshot();
        let mut committer = SessionCommitter(&mut session);
        self.pipeline
            .execute(
                snapshot,
                Request {
                    operation: Operation::Full,
                    scope: Scope::Pages(vec![page]),
                    translation: Some(RequestTranslation {
                        target_language: language,
                        instructions,
                    }),
                    progress: Some(Arc::new(|event| {
                        if let Progress::Finished { stage, elapsed, .. } = event {
                            tracing::info!(%stage, ?elapsed, "pipeline stage finished");
                        }
                    })),
                    ..Request::default()
                },
                &mut committer,
            )
            .await
            .context("pipeline failed")?;
        let frame = self.renderer.render(&session.snapshot(), page).await?;
        let raster = self
            .rasterizer
            .rasterize(&frame.raster_frame()?, RasterOptions::default())?;
        let mut png = io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(raster.image)
            .write_to(&mut png, image::ImageFormat::Png)?;
        Ok(png.into_inner())
    }
}

struct SessionCommitter<'a>(&'a mut Session);

#[async_trait::async_trait]
impl Committer for SessionCommitter<'_> {
    async fn commit(&mut self, output: StageOutput) -> Result<koharu_scene::Snapshot> {
        Ok(self.0.commit(output.patch).await?.snapshot)
    }
}

fn cache_key(
    config: &[u8],
    image: &[u8],
    language: Language,
    instructions: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CACHE_VERSION);
    for data in [config, image] {
        hasher.update((data.len() as u64).to_be_bytes());
        hasher.update(data);
    }
    let request = serde_json::to_vec(&(language.tag(), instructions))
        .expect("language and instructions serialize to JSON");
    hasher.update(request);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let arguments = Arguments::parse();
    let (pipeline_config, provider_config) = arguments.configs()?;
    let model_key = serde_json::to_vec(&(&pipeline_config, &provider_config))?;
    tokio::fs::create_dir_all(&arguments.cache_dir)
        .await
        .with_context(|| {
            format!(
                "cannot create cache directory {}",
                arguments.cache_dir.display()
            )
        })?;
    koharu_ml::init()
        .await
        .context("cannot initialize model runtimes")?;
    let pipeline = Pipeline::from_config(
        Config::memory(pipeline_config),
        Config::memory(provider_config),
        koharu_ml::device(false),
    )?;
    let server = Arc::new(Server {
        pipeline,
        renderer: Renderer::new()?,
        rasterizer: Rasterizer::new()?,
        cache_dir: arguments.cache_dir,
        model_key,
        admission: tokio::sync::Mutex::new(()),
    });
    let app = Router::new()
        .route("/translate", post(translate))
        .layer(DefaultBodyLimit::disable())
        .with_state(server);
    let listener = tokio::net::TcpListener::bind(arguments.listen).await?;
    tracing::info!(address = %listener.local_addr()?, "Koharu HTTP server listening");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_includes_every_request_and_model_dimension() {
        let base = cache_key(b"local,lama", b"image", Language::English, None);
        assert_eq!(base.len(), 64);
        assert_eq!(
            base,
            cache_key(b"local,lama", b"image", Language::English, None)
        );
        assert_ne!(
            base,
            cache_key(b"local,lama", b"different", Language::English, None)
        );
        assert_ne!(
            base,
            cache_key(b"local,lama", b"image", Language::Japanese, None)
        );
        assert_ne!(
            base,
            cache_key(b"local,lama", b"image", Language::English, Some(""))
        );
        assert_ne!(
            base,
            cache_key(b"local,lama", b"image", Language::English, Some("lore"))
        );
        assert_ne!(
            base,
            cache_key(b"local,aot", b"image", Language::English, None)
        );
    }

    #[test]
    fn cli_keeps_local_only_and_selected_inpainters() {
        let arguments =
            Arguments::try_parse_from(["koharu-server", "--inpainting", "rorem-mixed"]).unwrap();
        let (config, _) = arguments.configs().unwrap();
        assert_eq!(
            config.translation.model.model.as_deref(),
            Some("gemma4-e4b-it")
        );
        assert!(matches!(config.inpainting, InpaintingModel::RoremMixed(_)));
    }

    #[test]
    fn hosted_translation_requires_a_model() {
        let arguments =
            Arguments::try_parse_from(["koharu-server", "--provider", "openai-compatible"])
                .unwrap();
        assert!(arguments.configs().is_err());
    }
}
