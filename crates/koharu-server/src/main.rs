//! A single-process HTTP interface to Koharu's headless translation pipeline.

mod api;
mod jobs;

use std::{
    collections::BTreeMap, future::Future, io::Write, net::SocketAddr, sync::Arc, time::Duration,
};

use anyhow::{Context, Result};
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
use koharu_translator::{Provider, ProvidersConfig};
use tracing_subscriber::EnvFilter;
use url::Url;

use jobs::{Failure, Input, Job, Jobs};

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

    /// Translation engine to try when a request names none. Repeat it, or
    /// separate values with commas, to fail over in order. Providers:
    /// google-web, microsoft-edge, deeplx, deepl, google-cloud, azure,
    /// openrouter, openai-compatible. Defaults to google-web, then
    /// microsoft-edge.
    ///
    /// deepl reads DEEPL_API_KEY, google-cloud reads
    /// GOOGLE_CLOUD_TRANSLATION_API_KEY, azure reads AZURE_TRANSLATOR_KEY,
    /// openrouter reads OPENROUTER_API_KEY, and openai-compatible reads the
    /// optional OPENAI_COMPATIBLE_API_KEY.
    #[arg(long, env = "KOHARU_TRANSLATORS", value_delimiter = ',')]
    translator: Vec<Provider>,

    /// Model the openrouter provider uses, such as google/gemini-2.5-flash.
    #[arg(long, env = "KOHARU_OPENROUTER_MODEL")]
    openrouter_model: Option<String>,

    /// Model the openai-compatible provider uses.
    #[arg(long, env = "KOHARU_OPENAI_COMPATIBLE_MODEL")]
    openai_compatible_model: Option<String>,

    /// DeepLX server root, for the deeplx provider.
    #[arg(long, env = "KOHARU_DEEPLX_BASE_URL")]
    deeplx_base_url: Option<Url>,

    /// OpenAI-compatible endpoint, including its /v1 path, for the
    /// openai-compatible provider.
    #[arg(long, env = "KOHARU_OPENAI_COMPATIBLE_BASE_URL")]
    openai_compatible_base_url: Option<Url>,

    /// Azure AI Translator resource region, for regional resources.
    #[arg(long, env = "KOHARU_AZURE_REGION")]
    azure_region: Option<String>,

    /// Inpainting model used to erase original text.
    #[arg(long, env = "KOHARU_INPAINTING", value_enum, default_value = "lama")]
    inpainting: InpaintingChoice,
}

impl Arguments {
    fn configs(&self) -> Result<(PipelineConfig, ProvidersConfig)> {
        let mut pipeline = PipelineConfig::default();
        let providers = ProvidersConfig {
            deeplx_base_url: self.deeplx_base_url.clone(),
            openai_compatible_base_url: self.openai_compatible_base_url.clone(),
            openai_compatible_model: self.openai_compatible_model.clone(),
            openrouter_model: self.openrouter_model.clone(),
            azure_region: self.azure_region.clone(),
        };
        if !self.translator.is_empty() {
            pipeline.translation.engines = self.translator.clone();
        }
        for engine in &pipeline.translation.engines {
            providers
                .validate(*engine)
                .with_context(|| format!("cannot use translator {engine}"))?;
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
    /// Runs queued jobs until shutdown. Returning drops the models and GPU
    /// devices on this task, so shutdown awaits it before the process exits.
    async fn run(
        self,
        mut queue: tokio::sync::mpsc::UnboundedReceiver<Arc<Job>>,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
    ) {
        loop {
            // Connections that outlive the drain still hold queue senders, so
            // the queue alone does not end this loop.
            let job = tokio::select! {
                biased;
                _ = shutdown.wait_for(|stopping| *stopping) => break,
                job = queue.recv() => job,
            };
            let Some(job) = job else {
                break;
            };
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
            engines,
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
                        engines,
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
            Ok(Some(Bytes::from(raster.encode_png()?)))
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

fn main() {
    // Torch never returns cached CUDA blocks to the driver while llama.cpp
    // models share the device, and page-sized detection buffers and
    // variable-size inpainting tiles fragment its cache. Expandable segments
    // keep that cache close to what is in use; torch reads the setting when its
    // allocator starts, so it is set before any thread can read the environment.
    if std::env::var_os("PYTORCH_CUDA_ALLOC_CONF").is_none() {
        // SAFETY: no other thread exists yet.
        unsafe { std::env::set_var("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") };
    }
    let result = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(anyhow::Error::from)
        .and_then(|runtime| {
            let result = runtime.block_on(serve());
            // The pipeline has already been dropped. What remains are
            // connections that missed the drain deadline and blocking calls;
            // this bounds the wait inside Docker's 10-second stop window.
            runtime.shutdown_timeout(Duration::from_secs(1));
            result
        });
    let code = match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("Error: {error:?}");
            1
        }
    };
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // Returning from main would run libc atexit handlers and the C++ static
    // destructors of libtorch, ggml-cuda and the NVIDIA Vulkan driver while
    // other threads can still call into those libraries, which crashes the
    // process with SIGSEGV. The kernel and driver reclaim device memory when
    // the process exits, so skip that teardown.
    // SAFETY: nothing after this point needs to run.
    unsafe { libc::_exit(code) }
}

async fn serve() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let arguments = Arguments::parse();
    let (pipeline_config, provider_config) = arguments.configs()?;
    let engines = api::Engines {
        providers: provider_config.clone(),
        default: pipeline_config.translation.engines.clone(),
    };
    let mut shutdown = Box::pin(shutdown_signal()?);
    let handle = tokio::runtime::Handle::current();
    let mut startup = tokio::task::spawn_blocking(move || {
        handle.block_on(async move {
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
            let worker = tokio::spawn(worker.run(queue, jobs.subscribe_shutdown()));
            let shutdown_jobs = jobs.clone();
            let app = api::router(jobs, engines);
            let listener = tokio::net::TcpListener::bind(arguments.listen).await?;
            tracing::info!(address = %listener.local_addr()?, "Koharu HTTP server listening");
            Ok::<_, anyhow::Error>((listener, app, shutdown_jobs, worker))
        })
    });
    let (listener, app, shutdown_jobs, worker) = tokio::select! {
        biased;
        _ = &mut shutdown => {
            startup.abort();
            return Ok(());
        },
        started = &mut startup => started.context("server startup task failed")??,
    };
    // Docker waits 10 seconds before SIGKILL by default.
    serve_http(
        listener,
        app,
        shutdown_jobs,
        worker,
        shutdown,
        Duration::from_secs(8),
    )
    .await
}

/// Serves until `shutdown`, then gives HTTP connections and `worker` `grace`
/// to finish.
async fn serve_http(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    shutdown_jobs: Jobs,
    worker: tokio::task::JoinHandle<()>,
    shutdown: impl Future<Output = ()> + Send + 'static,
    grace: Duration,
) -> Result<()> {
    let (draining, drained) = tokio::sync::oneshot::channel();
    let server = async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                shutdown.await;
                shutdown_jobs.shutdown();
                let _ = draining.send(());
            })
            .await
    };
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result?,
        _ = drained => match tokio::time::timeout(grace, async {
            tokio::join!(&mut server, worker)
        })
        .await
        {
            Ok((result, worker)) => {
                if let Err(error) = worker {
                    tracing::error!(error = ?error, "translation worker failed");
                }
                result?
            }
            Err(_) => tracing::warn!("shutdown did not finish before its deadline"),
        },
    }
    Ok(())
}

#[cfg(unix)]
fn shutdown_signal() -> Result<impl Future<Output = ()>> {
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .context("failed to register SIGINT handler")?;
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("failed to register SIGTERM handler")?;
    Ok(async move {
        tokio::select! {
            _ = sigint.recv() => {}
            _ = sigterm.recv() => {}
        }
    })
}

#[cfg(not(unix))]
fn shutdown_signal() -> Result<impl Future<Output = ()>> {
    Ok(async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to register SIGINT handler");
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn shutdown_finishes_with_an_incomplete_multipart_upload() {
        use tokio::io::AsyncWriteExt;

        let (jobs, _queue) = Jobs::new();
        let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
        let app = api::router(
            jobs.clone(),
            api::Engines {
                providers: ProvidersConfig::default(),
                default: PipelineConfig::default().translation.engines,
            },
        )
        .layer(axum::middleware::from_fn(
            move |request: axum::extract::Request, next: axum::middleware::Next| {
                let started_tx = started_tx.clone();
                async move {
                    let _ = started_tx.send(());
                    next.run(request).await
                }
            },
        ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(serve_http(
            listener,
            app,
            jobs,
            tokio::spawn(async {}),
            async move {
                let _ = stop_rx.await;
            },
            Duration::from_millis(100),
        ));

        let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
        client
            .write_all(b"POST /translations HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=page\r\nContent-Length: 4096\r\n\r\n--page\r\nContent-Disposition: form-data; name=\"image\"; filename=\"page.png\"\r\n\r\n")
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), started_rx.recv())
            .await
            .unwrap()
            .unwrap();
        stop_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .expect("server did not finish draining the stalled upload")
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn shutdown_returns_after_the_worker_drops_its_state() {
        let (jobs, _queue) = Jobs::new();
        let mut stopping = jobs.subscribe_shutdown();
        let state = Arc::new(());
        let held = state.clone();
        let worker = tokio::spawn(async move {
            let _ = stopping.wait_for(|stopping| *stopping).await;
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(held);
        });
        let app = api::router(
            jobs.clone(),
            api::Engines {
                providers: ProvidersConfig::default(),
                default: PipelineConfig::default().translation.engines,
            },
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        serve_http(
            listener,
            app,
            jobs,
            worker,
            async {},
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        assert_eq!(
            Arc::strong_count(&state),
            1,
            "serve_http returned before the worker dropped its state"
        );
    }

    fn configs(arguments: &[&str]) -> Result<(PipelineConfig, ProvidersConfig)> {
        Arguments::try_parse_from(std::iter::once("koharu-server").chain(arguments.iter().copied()))
            .unwrap()
            .configs()
    }

    #[test]
    fn cli_defaults_to_keyless_engines_and_keeps_selected_inpainter() {
        let (config, _) = configs(&["--inpainting", "flux2-klein"]).unwrap();
        assert_eq!(
            config.translation.engines,
            [Provider::GoogleWeb, Provider::MicrosoftEdge]
        );
        assert!(matches!(config.inpainting, InpaintingModel::Flux2Klein(_)));
    }

    #[test]
    fn cli_translators_keep_their_order() {
        let (config, _) = configs(&[
            "--translator",
            "microsoft-edge,deeplx",
            "--translator",
            "openai-compatible",
            "--deeplx-base-url",
            "http://deeplx:1188",
            "--openai-compatible-base-url",
            "http://llm:8000/v1",
            "--openai-compatible-model",
            "qwen/qwen3:32b",
        ])
        .unwrap();
        assert_eq!(
            config.translation.engines,
            [
                Provider::MicrosoftEdge,
                Provider::DeepLx,
                Provider::OpenAiCompatible
            ]
        );
    }

    #[test]
    fn cli_rejects_translators_the_server_cannot_call() {
        assert!(configs(&["--translator", "deeplx"]).is_err());
        assert!(
            Arguments::try_parse_from(["koharu-server", "--translator", "openrouter:model"])
                .is_err()
        );
        assert!(
            configs(&[
                "--translator",
                "openai-compatible",
                "--openai-compatible-base-url",
                "http://llm:8000/v1",
            ])
            .is_err()
        );
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn signal_handlers_are_installed_before_polling() {
        const CHILD: &str = "KOHARU_EARLY_SIGNAL_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let _shutdown = shutdown_signal();
            assert!(
                std::process::Command::new("kill")
                    .arg("-TERM")
                    .arg(std::process::id().to_string())
                    .status()
                    .unwrap()
                    .success()
            );
            return;
        }

        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::signal_handlers_are_installed_before_polling",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "signal helper failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
