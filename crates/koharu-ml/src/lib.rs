use anyhow::{Context, ensure};
use koharu_llama::llama_backend::LlamaBackend;
use koharu_runtime::{Hardware, Runtime};
use tokio::sync::OnceCell;

mod backend;

pub mod comic_text_bubble_detector;
pub mod flux2_klein;
pub mod koharu_layout_rfdetr_seg_2xl;
pub mod lama;
pub mod llm;
pub mod paddle_ocr_vl_quantized;

pub use koharu_diffusion as diffusion;
pub use koharu_llama as llama;
pub use koharu_runtime::{Backend, Device, DeviceType, Feature};
pub use koharu_torch as torch;

static LLAMA: OnceCell<LlamaBackend> = OnceCell::const_new();
static DIFFUSION: OnceCell<()> = OnceCell::const_new();
struct Initialized {
    device: Device,
    features: Vec<Feature>,
}

static READY: OnceCell<Initialized> = OnceCell::const_new();

/// Initializes the native runtimes required by this process.
///
/// Concurrent callers share the first initialization; later callers may
/// request only capabilities included in that initial selection.
#[tracing::instrument(skip_all)]
pub async fn init(features: &[Feature]) -> anyhow::Result<()> {
    let initialized = READY
        .get_or_try_init(|| async {
            let runtime = Runtime::discover(features.iter().copied())?;
            let device = runtime
                .initialize()
                .await
                .context("failed to initialize runtimes")?;

            if features.contains(&Feature::Llama) {
                LLAMA
                    .get_or_try_init(|| async {
                        koharu_llama::send_logs_to_tracing(koharu_llama::LogOptions::default());
                        LlamaBackend::init().context("failed to initialize llama.cpp backend")
                    })
                    .await?;
            }
            if features.contains(&Feature::Diffusion) {
                DIFFUSION
                    .get_or_try_init(|| async {
                        koharu_diffusion::send_logs_to_tracing()
                            .context("failed to redirect stable-diffusion.cpp logs")?;
                        Ok::<(), anyhow::Error>(())
                    })
                    .await?;
            }
            Ok::<Initialized, anyhow::Error>(Initialized {
                device,
                features: features.to_vec(),
            })
        })
        .await?;
    ensure!(
        features
            .iter()
            .all(|feature| initialized.features.contains(feature)),
        "the requested runtime was not included in process initialization"
    );
    Ok(())
}

/// Returns the runtime-owned device selected for every model backend.
#[must_use]
pub fn device(cpu: bool) -> Device {
    if cpu {
        Device::cpu()
    } else if let Some(ready) = READY.get() {
        ready.device.clone()
    } else if let Some(device) = Hardware::discover().device() {
        device.clone()
    } else {
        tracing::warn!("GPU is not available, falling back to CPU");
        Device::cpu()
    }
}

/// Returns the initialized process-wide llama.cpp backend.
#[must_use]
pub fn llama_backend() -> Option<&'static LlamaBackend> {
    LLAMA.get()
}

macro_rules! model_repository {
    ($repository:literal @ $revision:literal { $($name:ident = $filename:literal),+ $(,)? }) => {
        $(
            const $name: koharu_runtime::HuggingFaceFile<'static> =
                koharu_runtime::HuggingFaceFile::pinned(
                    $repository,
                    $revision,
                    $filename,
                );
        )+
    };
}

pub(crate) use model_repository;
