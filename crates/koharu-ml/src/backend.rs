use std::path::Path;

use anyhow::Result;
use koharu_runtime::{Backend, Device};

/// Opens an ONNX model on `device`. A CUDA device never falls back to the CPU
/// provider: a model that cannot run on the GPU fails to load instead.
/// `allow_tf32` lets CUDA convolutions and matrix products round inputs to
/// TF32; models whose outputs move under that rounding pass `false`.
pub(crate) fn onnx_session(
    device: &Device,
    model: &Path,
    allow_tf32: bool,
) -> Result<ort::session::Session> {
    let mut builder = ort::session::Session::builder(crate::onnx_environment()?)?;
    if device.backend == Backend::Cuda {
        builder = builder
            .with_execution_providers([ort::ep::CUDA::default()
                .with_device_id(i32::try_from(device.index)?)
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Exhaustive)
                .with_tf32(allow_tf32)
                .build()
                .error_on_failure()])
            .map_err(ort::Error::<()>::from)?;
    }
    Ok(builder.commit_from_file(model)?)
}
