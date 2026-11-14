use std::path::Path;

use anyhow::Result;
use koharu_runtime::{Backend, Device};
use koharu_torch::{Cuda, Kind, nn};

pub(crate) trait TryIntoDevice<T> {
    fn try_into_device(self) -> Result<T>;
}

impl TryIntoDevice<koharu_torch::Device> for Device {
    fn try_into_device(self) -> Result<koharu_torch::Device> {
        match self.backend {
            Backend::Cpu => Ok(koharu_torch::Device::Cpu),
            Backend::Cuda => {
                Cuda::set_user_enabled_cudnn(true);
                Ok(koharu_torch::Device::Cuda(self.index))
            }
        }
    }
}

/// Whether a session's input shapes change from one run to the next.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Shapes {
    Fixed,
    Dynamic,
}

/// Opens an ONNX model on `device`. A CUDA device never falls back to the CPU
/// provider: a model that cannot run on the GPU fails to load instead.
/// TF32 is disabled to preserve detection accuracy.
///
/// On CUDA, intermediates are not planned into one preallocated block, and
/// with [`Shapes::Fixed`] the device arena grows by what each allocation
/// requests instead of by powers of two. Together they cut RF-DETR's peak
/// VRAM on an RTX 3080 from 6.3 to 3.1 GiB with no measurable change in
/// latency. [`Shapes::Dynamic`] keeps power-of-two growth: exact growth would
/// add an arena region for each new largest shape, and regions never merge.
pub(crate) fn onnx_session(
    device: &Device,
    model: &Path,
    shapes: Shapes,
) -> Result<ort::session::Session> {
    let mut builder = ort::session::Session::builder(crate::onnx_environment()?)?;
    if device.backend == Backend::Cuda {
        let arena = match shapes {
            Shapes::Fixed => ort::ep::ArenaExtendStrategy::SameAsRequested,
            Shapes::Dynamic => ort::ep::ArenaExtendStrategy::NextPowerOfTwo,
        };
        builder = builder
            .with_memory_pattern(false)
            .map_err(ort::Error::<()>::from)?
            .with_execution_providers([ort::ep::CUDA::default()
                .with_device_id(i32::try_from(device.index)?)
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Exhaustive)
                .with_arena_extend_strategy(arena)
                .with_tf32(false)
                .build()
                .error_on_failure()])
            .map_err(ort::Error::<()>::from)?;
    }
    Ok(builder.commit_from_file(model)?)
}

pub(crate) fn set_precision(var_store: &mut nn::VarStore) {
    let use_bfloat16 = matches!(var_store.device(), koharu_torch::Device::Cuda(_))
        && crate::device(false).compute_capability() >= 80;

    if var_store.is_empty() {
        var_store.set_kind(if use_bfloat16 {
            Kind::BFloat16
        } else {
            Kind::Float
        });
    } else if use_bfloat16 {
        var_store.bfloat16();
    } else {
        var_store.float();
    }
}
