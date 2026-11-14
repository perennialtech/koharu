//! Safe TensorRT-RTX inference.
//!
//! A [`Session`] runs one ONNX model with fixed or dynamic shapes on one GPU.
//! The serialized engine is built once per model file and TensorRT-RTX
//! version; the kernels TensorRT-RTX compiles for the GPU are kept in a
//! runtime cache beside it, so later processes skip both steps.
//!
//! The TensorRT-RTX and CUDA runtime libraries must already be loaded by
//! koharu-runtime's package before the first call.

mod cuda;
mod engine;
mod session;

use std::ffi::{CStr, c_char};

use anyhow::{Result, anyhow};
use koharu_tensorrt_sys as sys;

pub use self::{
    cuda::{DeviceBuffer, Element, Stream},
    engine::{DataType, ShapeRange, TensorInfo},
    session::{Binding, Session},
};

/// Resolves the loaded libraries and initializes TensorRT-RTX. Later calls
/// are free and return the first call's result.
pub(crate) fn cudart() -> Result<&'static sys::Cudart> {
    // SAFETY: `log` only reads its arguments and forwards them to tracing,
    // which is safe from any thread.
    unsafe { sys::load(log) }.map_err(|error| anyhow!("failed to load TensorRT-RTX: {error}"))
}

/// The loaded TensorRT-RTX version, as major * 10000 + minor * 100 + patch.
pub fn version() -> Result<i32> {
    cudart()?;
    // SAFETY: the shim is initialized.
    Ok(unsafe { sys::ktrt_library_version() })
}

unsafe extern "C" fn log(severity: i32, message: *const c_char) {
    // SAFETY: TensorRT passes a NUL-terminated message valid for the call.
    let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    match severity {
        0 | 1 => tracing::error!(target: "tensorrt", "{message}"),
        2 => tracing::warn!(target: "tensorrt", "{message}"),
        3 => tracing::debug!(target: "tensorrt", "{message}"),
        _ => tracing::trace!(target: "tensorrt", "{message}"),
    }
}

fn shim_error(context: &str) -> anyhow::Error {
    let reason = sys::last_error();
    if reason.is_empty() {
        anyhow!("{context}")
    } else {
        anyhow!("{context}: {reason}")
    }
}

fn check(status: i32, context: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(shim_error(context))
    }
}
