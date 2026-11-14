//! The worker process, which owns the models and GPU devices.
//!
//! The server starts it and kills it with SIGKILL, so nothing here shuts down
//! cleanly: the kernel and driver reclaim device memory when the process dies.

use std::{
    collections::BTreeMap,
    io::Write,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use axum::body::Bytes;
use koharu_config::Config;
use koharu_pipeline::{
    Committer, InpaintingModel, Pipeline, PipelineError, Request as PipelineRequest,
    RequestTranslation, RunStatus, Scope, StageOutput, StopToken,
};
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::Renderer;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft, Session};
use tokio::sync::mpsc;

use crate::{
    Arguments,
    jobs::{Failure, Input, StageProgress},
    wire::{self, Reply, Request},
};

pub(crate) fn main(arguments: &Arguments) -> ! {
    let result = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(anyhow::Error::from)
        .and_then(|runtime| runtime.block_on(run(arguments)));
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
    // other threads can still call into those libraries, which crashes with
    // SIGSEGV.
    // SAFETY: nothing after this point needs to run.
    unsafe { libc::_exit(code) }
}

/// Loads the models, then translates pages until the server closes the
/// socket.
async fn run(arguments: &Arguments) -> Result<()> {
    let (mut reader, mut writer) = wire::stdin_socket()
        .context("worker stdin is not a socket")?
        .into_split();
    let (pipeline_config, provider_config) = arguments.configs()?;
    let mut features = vec![
        koharu_ml::Feature::Torch,
        koharu_ml::Feature::Llama,
        koharu_ml::Feature::OnnxRuntime,
        koharu_ml::Feature::TensorRt,
    ];
    if matches!(&pipeline_config.inpainting, InpaintingModel::Flux2Klein(_)) {
        features.push(koharu_ml::Feature::Diffusion);
    }
    koharu_ml::init(&features)
        .await
        .context("cannot initialize model runtimes")?;
    let worker = Worker {
        pipeline: Pipeline::from_config(
            Config::memory(pipeline_config),
            Config::memory(provider_config),
            koharu_ml::device(false),
        )?,
        renderer: Renderer::new()?,
        rasterizer: Arc::new(Rasterizer::new()?),
    };

    // Replies come from the pipeline's progress callback as well as this
    // task, so one task owns the write half.
    let (replies, mut outgoing) = mpsc::unbounded_channel::<(Reply, Bytes)>();
    tokio::spawn(async move {
        while let Some((reply, body)) = outgoing.recv().await {
            if wire::send(&mut writer, &reply, &body).await.is_err() {
                break;
            }
        }
    });

    // `Stop` arrives while a translation runs, so another task reads.
    let current = Arc::new(Mutex::new(StopToken::default()));
    let (pages, mut queue) = mpsc::unbounded_channel();
    tokio::spawn({
        let current = current.clone();
        async move {
            while let Ok((request, body)) = wire::receive(&mut reader).await {
                match request {
                    Request::Translate(mut input) => {
                        input.source = body;
                        let stop = StopToken::default();
                        *current.lock().unwrap() = stop.clone();
                        if pages.send((input, stop)).is_err() {
                            break;
                        }
                    }
                    Request::Stop => current.lock().unwrap().stop(),
                }
            }
        }
    });

    let _ = replies.send((Reply::Ready, Bytes::new()));
    while let Some((input, stop)) = queue.recv().await {
        let reply = match worker.translate(input, &stop, &replies).await {
            Ok(Some(png)) => (Reply::Translated, png),
            Ok(None) => (Reply::Stopped, Bytes::new()),
            Err(error) => {
                let failure = match error.downcast::<PipelineError>() {
                    Ok(error) => Failure::from(error),
                    Err(error) => {
                        tracing::error!(error = ?error, "image translation failed");
                        Failure::processing()
                    }
                };
                (Reply::Failed(failure), Bytes::new())
            }
        };
        let _ = replies.send(reply);
    }
    Ok(())
}

struct Worker {
    pipeline: Pipeline,
    renderer: Renderer,
    rasterizer: Arc<Rasterizer>,
}

impl Worker {
    async fn translate(
        &self,
        input: Input,
        stop: &StopToken,
        replies: &mpsc::UnboundedSender<(Reply, Bytes)>,
    ) -> Result<Option<Bytes>> {
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
                    source.to_vec(),
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
        let progress = replies.clone();
        let report = self
            .pipeline
            .execute(
                snapshot,
                PipelineRequest {
                    scope: Scope::Pages(vec![page]),
                    translation: Some(RequestTranslation {
                        target_language: language,
                        instructions,
                        engines,
                    }),
                    stop: stop.clone(),
                    progress: Some(Arc::new(move |event| {
                        if let Some(update) = StageProgress::from_event(event) {
                            let _ = progress.send((Reply::Progress(update), Bytes::new()));
                        }
                    })),
                    ..PipelineRequest::default()
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
