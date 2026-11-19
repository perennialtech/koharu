use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::body::Bytes;
use koharu_pipeline::{ErrorKind, PipelineError, Progress, Stage};
use koharu_translator::{Language, Provider};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};
use utoipa::ToSchema;
use uuid::Uuid;

const RESULT_RETENTION: Duration = Duration::from_secs(60 * 60);

#[derive(Serialize, Deserialize)]
pub(crate) struct Input {
    /// Sent as the message body rather than in the JSON header.
    #[serde(skip)]
    pub(crate) source: Bytes,
    pub(crate) media_type: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) language: Language,
    pub(crate) instructions: Option<String>,
    pub(crate) engines: Option<Vec<Provider>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Status {
    Queued,
    Running,
    Cancelling,
    Completed,
    Failed,
    Cancelled,
}

impl Status {
    pub(crate) fn terminal_event(self) -> Option<&'static str> {
        match self {
            Self::Completed => Some("completed"),
            Self::Failed => Some("failed"),
            Self::Cancelled => Some("cancelled"),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
enum StageStatus {
    Pending,
    Loading,
    Running,
    Completed,
    Skipped,
    Failed,
    Cancelled,
}

#[derive(Clone, Serialize, Deserialize, ToSchema)]
pub(crate) struct StageProgress {
    /// Pipeline stage name.
    #[schema(value_type = String, example = "ocr")]
    stage: Stage,
    status: StageStatus,
    model: Option<String>,
    elapsed_ms: Option<u64>,
}

impl StageProgress {
    /// The stage update a pipeline event reports, if any.
    pub(crate) fn from_event(progress: Progress) -> Option<Self> {
        let (stage, status, model, elapsed_ms) = match progress {
            Progress::Started { .. } => return None,
            Progress::Loading { stage, model, .. } => {
                (stage, StageStatus::Loading, Some(model), None)
            }
            Progress::Running { stage, model, .. } => {
                (stage, StageStatus::Running, Some(model), None)
            }
            Progress::Finished {
                stage,
                model,
                elapsed,
                ..
            } => (
                stage,
                StageStatus::Completed,
                Some(model),
                Some(millis(elapsed)),
            ),
            Progress::Skipped { stage, .. } => (stage, StageStatus::Skipped, None, None),
        };
        Some(Self {
            stage,
            status,
            model,
            elapsed_ms,
        })
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FailureCode {
    ModelLoadFailed,
    ProcessingFailed,
}

#[derive(Clone, Copy, Serialize, Deserialize, ToSchema)]
pub(crate) struct Failure {
    code: FailureCode,
    #[schema(value_type = Option<String>, example = "ocr")]
    stage: Option<Stage>,
}

impl Failure {
    pub(crate) fn processing() -> Self {
        Self {
            code: FailureCode::ProcessingFailed,
            stage: None,
        }
    }
}

impl From<PipelineError> for Failure {
    fn from(error: PipelineError) -> Self {
        tracing::error!(error = ?error, "translation pipeline failed");
        Self {
            code: match error.kind {
                ErrorKind::ModelLoad => FailureCode::ModelLoadFailed,
                _ => FailureCode::ProcessingFailed,
            },
            stage: error.stage,
        }
    }
}

#[derive(Clone, Serialize, ToSchema)]
pub(crate) struct Translation {
    pub(crate) id: Uuid,
    pub(crate) status: Status,
    /// Stages that completed or skipped.
    completed: usize,
    total: usize,
    stages: Vec<StageProgress>,
    /// Time spent in the queue, in milliseconds.
    queue_ms: u64,
    /// Time since execution began, in milliseconds. Frozen on completion.
    elapsed_ms: Option<u64>,
    error: Option<Failure>,
    result_url: Option<String>,
}

pub(crate) struct JobState {
    view: Translation,
    input: Option<Input>,
    result: Option<Bytes>,
    submitted: Instant,
    started: Option<Instant>,
    finished: Option<Instant>,
}

impl JobState {
    pub(crate) fn snapshot(&self) -> Translation {
        let mut view = self.view.clone();
        let now = self.finished.unwrap_or_else(Instant::now);
        view.queue_ms = millis(self.started.unwrap_or(now).duration_since(self.submitted));
        view.elapsed_ms = self
            .started
            .map(|started| millis(now.duration_since(started)));
        view
    }

    fn expired(&self) -> bool {
        self.finished
            .is_some_and(|finished| finished.elapsed() >= RESULT_RETENTION)
    }

    fn mark_finished(&mut self) {
        self.finished = Some(Instant::now());
        for stage in &mut self.view.stages {
            if matches!(stage.status, StageStatus::Loading | StageStatus::Running) {
                stage.status = StageStatus::Cancelled;
            }
        }
    }
}

fn millis(duration: Duration) -> u64 {
    duration.as_millis() as u64
}

pub(crate) struct Job {
    state: watch::Sender<JobState>,
}

impl Job {
    fn new(id: Uuid, input: Input) -> Self {
        let stages = Stage::ALL;
        Self {
            state: watch::Sender::new(JobState {
                view: Translation {
                    id,
                    status: Status::Queued,
                    completed: 0,
                    total: stages.len(),
                    stages: stages
                        .into_iter()
                        .map(|stage| StageProgress {
                            stage,
                            status: StageStatus::Pending,
                            model: None,
                            elapsed_ms: None,
                        })
                        .collect(),
                    queue_ms: 0,
                    elapsed_ms: None,
                    error: None,
                    result_url: None,
                },
                input: Some(input),
                result: None,
                submitted: Instant::now(),
                started: None,
                finished: None,
            }),
        }
    }

    pub(crate) fn snapshot(&self) -> Translation {
        self.state.borrow().snapshot()
    }

    pub(crate) fn subscribe(&self) -> watch::Receiver<JobState> {
        self.state.subscribe()
    }

    pub(crate) fn result(&self) -> Option<Bytes> {
        self.state.borrow().result.clone()
    }

    pub(crate) fn start(&self) -> Option<Input> {
        let mut input = None;
        self.state.send_if_modified(|state| {
            input = state.input.take();
            if input.is_none() {
                return false;
            }
            state.started = Some(Instant::now());
            state.view.status = Status::Running;
            true
        });
        input
    }

    pub(crate) fn status(&self) -> Status {
        self.state.borrow().view.status
    }

    pub(crate) fn cancel(&self) -> Option<Translation> {
        let mut snapshot = None;
        self.state.send_if_modified(|state| {
            let changed = match state.view.status {
                Status::Queued => {
                    state.input = None;
                    state.view.status = Status::Cancelled;
                    state.mark_finished();
                    true
                }
                Status::Running => {
                    state.view.status = Status::Cancelling;
                    true
                }
                Status::Cancelling | Status::Cancelled => false,
                Status::Completed | Status::Failed => return false,
            };
            snapshot = Some(state.snapshot());
            changed
        });
        snapshot
    }

    pub(crate) fn progress(&self, progress: StageProgress) {
        self.state.send_if_modified(|state| {
            if state.view.status != Status::Running {
                return false;
            }
            if let Some(entry) = state
                .view
                .stages
                .iter_mut()
                .find(|entry| entry.stage == progress.stage)
            {
                if matches!(
                    progress.status,
                    StageStatus::Completed | StageStatus::Skipped
                ) {
                    state.view.completed += 1;
                }
                *entry = progress;
            }
            true
        });
    }

    pub(crate) fn finish(&self, result: Result<Option<Bytes>, Failure>) {
        self.state.send_modify(|state| {
            state.mark_finished();
            if state.view.status == Status::Cancelling {
                state.view.status = Status::Cancelled;
                return;
            }
            match result {
                Ok(Some(png)) => {
                    state.result = Some(png);
                    state.view.result_url = Some(format!("/translations/{}/result", state.view.id));
                    state.view.status = Status::Completed;
                }
                Ok(None) => state.view.status = Status::Cancelled,
                Err(error) => {
                    if let Some(stage) = state
                        .view
                        .stages
                        .iter_mut()
                        .find(|stage| Some(stage.stage) == error.stage)
                    {
                        stage.status = StageStatus::Failed;
                    }
                    state.view.error = Some(error);
                    state.view.status = Status::Failed;
                }
            }
        });
    }
}

#[derive(Clone)]
pub(crate) struct Jobs {
    entries: Arc<Mutex<HashMap<Uuid, Arc<Job>>>>,
    queue: mpsc::UnboundedSender<Arc<Job>>,
    shutdown: watch::Sender<bool>,
}

impl Jobs {
    pub(crate) fn new() -> (Self, mpsc::UnboundedReceiver<Arc<Job>>) {
        let (queue, receiver) = mpsc::unbounded_channel();
        let (shutdown, _) = watch::channel(false);
        let entries = Arc::new(Mutex::new(HashMap::<Uuid, Arc<Job>>::new()));
        let cleanup_entries = Arc::downgrade(&entries);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                let Some(entries) = cleanup_entries.upgrade() else {
                    break;
                };
                entries
                    .lock()
                    .unwrap()
                    .retain(|_, job| !job.state.borrow().expired());
            }
        });
        (
            Self {
                entries,
                queue,
                shutdown,
            },
            receiver,
        )
    }

    pub(crate) fn subscribe_shutdown(&self) -> watch::Receiver<bool> {
        self.shutdown.subscribe()
    }

    pub(crate) fn shutdown(&self) {
        self.shutdown.send_replace(true);
        for job in self.entries.lock().unwrap().values() {
            job.cancel();
        }
    }

    pub(crate) fn submit(&self, input: Input) -> Option<Translation> {
        let id = Uuid::new_v4();
        let job = Arc::new(Job::new(id, input));
        let snapshot = job.snapshot();
        let mut entries = self.entries.lock().unwrap();
        if *self.shutdown.borrow() {
            return None;
        }
        entries.insert(id, job.clone());
        if self.queue.send(job).is_err() {
            entries.remove(&id);
            return None;
        }
        Some(snapshot)
    }

    pub(crate) fn get(&self, id: &str) -> Option<Arc<Job>> {
        let id = Uuid::parse_str(id).ok()?;
        let job = self.entries.lock().unwrap().get(&id)?.clone();
        if job.state.borrow().expired() {
            return None;
        }
        Some(job)
    }
}
