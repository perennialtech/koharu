//! A single-process HTTP interface to Koharu's headless translation pipeline.

mod api;
mod jobs;

use std::{collections::BTreeMap, io, net::SocketAddr, sync::Arc};

use anyhow::{Context, Result, bail, ensure};
use axum::body::Bytes;
use clap::{Parser, ValueEnum};
use koharu_config::Config;
use koharu_pipeline::{
    Committer, Flux2KleinConfig, InpaintingModel, Pipeline, PipelineConfig, PipelineError, Request,
    RequestTranslation, RunStatus, Scope, StageOutput,
};
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::Renderer;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft, Session};
use koharu_translator::{ModelSelection, Provider, ProvidersConfig};
use tracing_subscriber::EnvFilter;
use url::Url;

use jobs::{Failure, Input, Job, Jobs};

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
    Flux2Klein,
}

#[derive(Debug, Parser)]
#[command(version, about = "Translate manga page images over HTTP")]
struct Arguments {
    /// Address for the HTTP listener.
    #[arg(long, env = "KOHARU_LISTEN", default_value = "0.0.0.0:8080")]
    listen: SocketAddr,

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
                pipeline.translation.generation.reasoning = None;
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
            InpaintingChoice::Flux2Klein => {
                InpaintingModel::Flux2Klein(Flux2KleinConfig::default())
            }
        };
        Ok((pipeline, providers))
    }
}

struct Worker {
    pipeline: Pipeline,
    renderer: Renderer,
    rasterizer: Arc<Rasterizer>,
}

impl Worker {
    async fn run(self, mut queue: tokio::sync::mpsc::UnboundedReceiver<Arc<Job>>) {
        while let Some(job) = queue.recv().await {
            let Some(input) = job.start() else {
                continue;
            };
            let result = self.translate(input, &job).await.map_err(|error| {
                match error.downcast::<PipelineError>() {
                    Ok(error) => Failure::from(error),
                    Err(error) => {
                        tracing::error!(error = ?error, "image translation failed");
                        Failure::processing()
                    }
                }
            });
            job.finish(result);
        }
    }

    async fn translate(&self, input: Input, job: &Arc<Job>) -> Result<Option<Bytes>> {
        let Input {
            source,
            media_type,
            width,
            height,
            language,
            instructions,
        } = input;
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
                    source,
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
        let stop = job.stop_token();
        let progress = job.clone();
        let report = self
            .pipeline
            .execute(
                snapshot,
                Request {
                    scope: Scope::Pages(vec![page]),
                    translation: Some(RequestTranslation {
                        target_language: language,
                        instructions,
                    }),
                    stop: stop.clone(),
                    progress: Some(Arc::new(move |event| progress.progress(event))),
                    ..Request::default()
                },
                &mut committer,
            )
            .await?;
        if report.status == RunStatus::Stopped || stop.stopped() {
            return Ok(None);
        }
        let frame = self.renderer.render(&session.snapshot(), page).await?;
        if stop.stopped() {
            return Ok(None);
        }
        let frame = frame.raster_frame()?;
        let rasterizer = self.rasterizer.clone();
        tokio::task::spawn_blocking(move || {
            let raster = rasterizer.rasterize(&frame, RasterOptions::default())?;
            let mut png = io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(raster.image)
                .write_to(&mut png, image::ImageFormat::Png)?;
            Ok(Some(Bytes::from(png.into_inner())))
        })
        .await?
    }
}

struct SessionCommitter<'a>(&'a mut Session);

#[async_trait::async_trait]
impl Committer for SessionCommitter<'_> {
    async fn commit(&mut self, output: StageOutput) -> Result<koharu_scene::Snapshot> {
        Ok(self.0.commit(output.patch).await?.snapshot)
    }
}

fn main() -> Result<()> {
    // Torch never returns cached CUDA blocks to the driver while llama.cpp
    // models share the device, and page-sized detection buffers and
    // variable-size inpainting tiles fragment its cache. Expandable segments
    // keep that cache close to what is in use; torch reads the setting when its
    // allocator starts, so it is set before any thread can read the environment.
    if std::env::var_os("PYTORCH_CUDA_ALLOC_CONF").is_none() {
        // SAFETY: no other thread exists yet.
        unsafe { std::env::set_var("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") };
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(serve())
}

async fn serve() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let arguments = Arguments::parse();
    let (pipeline_config, provider_config) = arguments.configs()?;
    let mut features = vec![koharu_ml::Feature::Torch, koharu_ml::Feature::Llama];
    if matches!(&pipeline_config.inpainting, InpaintingModel::Flux2Klein(_)) {
        features.push(koharu_ml::Feature::Diffusion);
    }
    koharu_ml::init(&features)
        .await
        .context("cannot initialize model runtimes")?;
    let pipeline = Pipeline::from_config(
        Config::memory(pipeline_config),
        Config::memory(provider_config),
        koharu_ml::device(false),
    )?;
    let worker = Worker {
        pipeline,
        renderer: Renderer::new()?,
        rasterizer: Arc::new(Rasterizer::new()?),
    };
    let (jobs, queue) = Jobs::new();
    tokio::spawn(worker.run(queue));
    let app = api::router(jobs);
    let listener = tokio::net::TcpListener::bind(arguments.listen).await?;
    tracing::info!(address = %listener.local_addr()?, "Koharu HTTP server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

#[cfg(unix)]
async fn shutdown_signal() {
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");
    tokio::select! {
        result = tokio::signal::ctrl_c() => result.expect("failed to register SIGINT handler"),
        _ = sigterm.recv() => {}
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to register SIGINT handler");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_keeps_local_only_and_selected_inpainters() {
        let arguments =
            Arguments::try_parse_from(["koharu-server", "--inpainting", "flux2-klein"]).unwrap();
        let (config, _) = arguments.configs().unwrap();
        assert_eq!(
            config.translation.model.model.as_deref(),
            Some("gemma4-e4b-it")
        );
        assert!(matches!(config.inpainting, InpaintingModel::Flux2Klein(_)));
    }

    #[test]
    fn hosted_models_do_not_receive_unsolicited_reasoning_parameters() {
        let arguments = Arguments::try_parse_from([
            "koharu-server",
            "--provider",
            "openai-compatible",
            "--model",
            "vision-model",
        ])
        .unwrap();
        let (pipeline, _) = arguments.configs().unwrap();
        assert_eq!(pipeline.translation.generation.reasoning, None);
    }

    #[test]
    fn hosted_translation_requires_a_model() {
        let arguments =
            Arguments::try_parse_from(["koharu-server", "--provider", "openai-compatible"])
                .unwrap();
        assert!(arguments.configs().is_err());
    }
}
