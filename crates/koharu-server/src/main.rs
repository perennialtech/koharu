//! An HTTP interface to Koharu's headless translation pipeline. The server
//! process handles HTTP and jobs; a worker process it starts runs the models.

mod api;
mod jobs;
mod supervisor;
mod wire;
mod worker;

use std::{future::Future, net::SocketAddr, time::Duration};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use koharu_pipeline::{Flux2KleinConfig, InpaintingModel, PipelineConfig};
use koharu_translator::{Provider, ProvidersConfig};
use tracing_subscriber::EnvFilter;
use url::Url;

use jobs::Jobs;

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

    /// Run as the model worker. The server starts it with this flag.
    #[arg(long, hide = true)]
    worker: bool,
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

fn main() {
    let arguments = Arguments::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    if arguments.worker {
        worker::main(&arguments);
    }
    let result = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(anyhow::Error::from)
        .and_then(|runtime| runtime.block_on(serve(arguments)));
    if let Err(error) = result {
        eprintln!("Error: {error:?}");
        std::process::exit(1);
    }
}

async fn serve(arguments: Arguments) -> Result<()> {
    let (pipeline_config, provider_config) = arguments.configs()?;
    let engines = api::Engines {
        providers: provider_config,
        default: pipeline_config.translation.engines,
    };
    let shutdown = shutdown_signal()?;
    let executable = std::env::current_exe().context("cannot find own executable")?;
    let (jobs, queue) = Jobs::new();
    let supervisor = supervisor::run(
        move || supervisor::worker_command(&executable),
        queue,
        jobs.subscribe_shutdown(),
    );
    let listener = tokio::net::TcpListener::bind(arguments.listen).await?;
    tracing::info!(address = %listener.local_addr()?, "Koharu HTTP server listening");
    // Docker waits 10 seconds before SIGKILL by default.
    serve_http(
        listener,
        api::router(jobs.clone(), engines),
        jobs,
        supervisor,
        shutdown,
        Duration::from_secs(8),
    )
    .await
}

/// Serves until `shutdown` or until `supervisor` or the server ends, then
/// gives HTTP connections `grace` to finish.
async fn serve_http(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    jobs: Jobs,
    supervisor: impl Future<Output = Result<()>>,
    shutdown: impl Future<Output = ()> + Send + 'static,
    grace: Duration,
) -> Result<()> {
    let mut stopping = jobs.subscribe_shutdown();
    let signal_jobs = jobs.clone();
    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
        tokio::select! {
            () = shutdown => {}
            _ = stopping.wait_for(|stopping| *stopping) => {}
        }
        signal_jobs.shutdown();
    });
    let mut stopped = jobs.subscribe_shutdown();
    let server = async {
        let result = tokio::select! {
            result = server => result.map_err(anyhow::Error::from),
            _ = async {
                let _ = stopped.wait_for(|stopping| *stopping).await;
                tokio::time::sleep(grace).await;
            } => {
                tracing::warn!("HTTP connections did not finish before the shutdown deadline");
                Ok(())
            }
        };
        jobs.shutdown();
        result
    };
    let supervisor = async {
        let result = supervisor.await;
        jobs.shutdown();
        result
    };
    let (server, supervisor) = tokio::join!(server, supervisor);
    supervisor?;
    server
}

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
        let mut stopping = jobs.subscribe_shutdown();
        let server = tokio::spawn(serve_http(
            listener,
            app,
            jobs,
            async move {
                let _ = stopping.wait_for(|stopping| *stopping).await;
                Ok(())
            },
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
    async fn a_failed_supervisor_stops_the_server_with_its_error() {
        let (jobs, _queue) = Jobs::new();
        let app = api::router(
            jobs.clone(),
            api::Engines {
                providers: ProvidersConfig::default(),
                default: PipelineConfig::default().translation.engines,
            },
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            serve_http(
                listener,
                app,
                jobs,
                async { anyhow::bail!("worker failed") },
                std::future::pending(),
                Duration::from_secs(1),
            ),
        )
        .await
        .expect("server kept running after the supervisor failed");
        assert_eq!(result.unwrap_err().to_string(), "worker failed");
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
