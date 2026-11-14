//! Runs queued jobs on the worker process and restarts it when it dies.
//!
//! The server never loads the model runtimes or GPU drivers itself, so it can
//! exit normally. The worker is killed rather than shut down.

use std::{
    os::{fd::OwnedFd, unix::net::UnixStream},
    path::Path,
    process::Stdio,
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use axum::body::Bytes;
use tokio::{
    net::unix::OwnedWriteHalf,
    process::{Child, Command},
    sync::{mpsc, watch},
};

use crate::{
    jobs::{Failure, Job, Status},
    wire::{self, Reply, Request},
};

/// The command that starts a worker: this executable with the server's own
/// arguments plus `--worker`. The worker inherits the environment.
pub(crate) fn worker_command(executable: &Path) -> Command {
    let mut command = Command::new(executable);
    command.args(std::env::args_os().skip(1)).arg("--worker");
    // Torch never returns cached CUDA blocks to the driver while llama.cpp
    // models share the device, and page-sized detection buffers and
    // variable-size inpainting tiles fragment its cache. Expandable segments
    // keep that cache close to what is in use.
    if std::env::var_os("PYTORCH_CUDA_ALLOC_CONF").is_none() {
        command.env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True");
    }
    command
}

/// Runs jobs until shutdown. Fails if a worker exits before it is ready, which
/// means it cannot load its models and restarting would not help.
pub(crate) async fn run(
    command: impl Fn() -> Command,
    mut queue: mpsc::UnboundedReceiver<Arc<Job>>,
    mut shutdown: watch::Receiver<bool>,
) -> Result<()> {
    loop {
        let mut worker = tokio::select! {
            biased;
            () = stopped(&mut shutdown) => return Ok(()),
            worker = Worker::start(command()) => worker?,
        };
        loop {
            let job = tokio::select! {
                biased;
                () = stopped(&mut shutdown) => return Ok(()),
                job = queue.recv() => job,
            };
            let Some(job) = job else {
                return Ok(());
            };
            let Some(input) = job.start() else {
                continue;
            };
            let source = input.source.clone();
            if wire::send(&mut worker.writer, &Request::Translate(input), &source)
                .await
                .is_err()
                || !worker.finish(&job, &mut shutdown).await
            {
                if *shutdown.borrow() {
                    return Ok(());
                }
                job.finish(Err(Failure::processing()));
                let _ = worker.child.start_kill();
                let status = worker.child.wait().await;
                tracing::error!(?status, "translation worker exited, restarting it");
                break;
            }
        }
    }
}

async fn stopped(shutdown: &mut watch::Receiver<bool>) {
    let _ = shutdown.wait_for(|stopping| *stopping).await;
}

struct Worker {
    /// Dropping it kills the worker.
    child: Child,
    writer: OwnedWriteHalf,
    replies: mpsc::UnboundedReceiver<(Reply, Bytes)>,
}

impl Worker {
    /// Starts a worker and waits until it has loaded its models.
    async fn start(mut command: Command) -> Result<Self> {
        let (ours, theirs) = UnixStream::pair()?;
        let mut child = command
            .stdin(Stdio::from(OwnedFd::from(theirs)))
            .kill_on_drop(true)
            .spawn()
            .context("cannot start the translation worker")?;
        // Dropping `command` closed our copy of the worker's end, so the
        // socket reaches end of file when the worker dies.
        drop(command);
        ours.set_nonblocking(true)?;
        let (mut reader, writer) = tokio::net::UnixStream::from_std(ours)?.into_split();
        // Reading in its own task keeps a message whole when `finish` stops
        // waiting for it.
        let (sender, mut replies) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Ok(reply) = wire::receive(&mut reader).await {
                if sender.send(reply).is_err() {
                    break;
                }
            }
        });
        match replies.recv().await {
            Some((Reply::Ready, _)) => {
                tracing::info!(pid = child.id(), "translation worker ready");
                Ok(Self {
                    child,
                    writer,
                    replies,
                })
            }
            _ => bail!(
                "translation worker exited during startup: {}",
                child.wait().await?
            ),
        }
    }

    /// Relays the running job's progress, cancellation and result. Returns
    /// false if the worker died or shutdown began first.
    async fn finish(&mut self, job: &Job, shutdown: &mut watch::Receiver<bool>) -> bool {
        let mut state = job.subscribe();
        let mut stopping = false;
        loop {
            tokio::select! {
                biased;
                () = stopped(shutdown) => return false,
                reply = self.replies.recv() => match reply {
                    Some((Reply::Progress(progress), _)) => job.progress(progress),
                    Some((Reply::Translated, png)) => {
                        job.finish(Ok(Some(png)));
                        return true;
                    }
                    Some((Reply::Stopped, _)) => {
                        job.finish(Ok(None));
                        return true;
                    }
                    Some((Reply::Failed(failure), _)) => {
                        job.finish(Err(failure));
                        return true;
                    }
                    Some((Reply::Ready, _)) | None => return false,
                },
                _ = state.changed(), if !stopping => {
                    if job.status() == Status::Cancelling {
                        stopping = true;
                        if wire::send(&mut self.writer, &Request::Stop, &[]).await.is_err() {
                            return false;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use koharu_translator::Language;

    use super::*;
    use crate::jobs::{Input, Jobs};

    const FAKE_WORKER: &str = "KOHARU_FAKE_WORKER";

    fn page(instructions: Option<&str>) -> Input {
        Input {
            source: Bytes::from_static(b"page"),
            media_type: "image/png".to_owned(),
            width: 1,
            height: 1,
            language: Language::English,
            instructions: instructions.map(ToOwned::to_owned),
            engines: None,
        }
    }

    async fn wait_until_finished(jobs: &Jobs, id: uuid::Uuid) -> Arc<Job> {
        let job = jobs.get(&id.to_string()).unwrap();
        tokio::time::timeout(
            Duration::from_secs(10),
            job.subscribe()
                .wait_for(|state| state.snapshot().status.terminal_event().is_some()),
        )
        .await
        .unwrap()
        .unwrap();
        job
    }

    /// Acts as a worker that crashes on pages with the instructions "crash"
    /// and translates every other page into the bytes "png".
    async fn fake_worker() {
        let (mut reader, mut writer) = wire::stdin_socket().unwrap().into_split();
        wire::send(&mut writer, &Reply::Ready, &[]).await.unwrap();
        while let Ok((request, _)) = wire::receive::<Request>(&mut reader).await {
            if let Request::Translate(input) = request {
                if input.instructions.as_deref() == Some("crash") {
                    std::process::abort();
                }
                wire::send(&mut writer, &Reply::Translated, b"png")
                    .await
                    .unwrap();
            }
        }
    }

    #[tokio::test]
    async fn a_crashed_worker_fails_its_job_and_restarts() {
        if std::env::var_os(FAKE_WORKER).is_some() {
            return fake_worker().await;
        }
        let (jobs, queue) = Jobs::new();
        let supervisor = tokio::spawn(run(
            || {
                let mut command = Command::new(std::env::current_exe().unwrap());
                command
                    .args([
                        "--exact",
                        "supervisor::tests::a_crashed_worker_fails_its_job_and_restarts",
                    ])
                    .env(FAKE_WORKER, "1")
                    .stdout(Stdio::null());
                command
            },
            queue,
            jobs.subscribe_shutdown(),
        ));
        let crashed = jobs.submit(page(Some("crash"))).unwrap().id;
        let next = jobs.submit(page(None)).unwrap().id;

        assert_eq!(
            wait_until_finished(&jobs, crashed).await.status(),
            Status::Failed
        );
        let next = wait_until_finished(&jobs, next).await;
        assert_eq!(next.status(), Status::Completed);
        assert_eq!(next.result().unwrap(), "png");

        jobs.shutdown();
        supervisor.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn a_worker_that_exits_during_startup_stops_the_supervisor() {
        let (jobs, queue) = Jobs::new();
        let result = run(|| Command::new("false"), queue, jobs.subscribe_shutdown()).await;
        assert!(result.is_err());
    }
}
