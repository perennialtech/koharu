//! Owned TensorRT-RTX runtime, engine and execution context handles.

use std::{
    ffi::{CStr, CString},
    path::Path,
    ptr, slice,
};

use anyhow::{Context as _, Result, bail};
use koharu_tensorrt_sys as sys;

/// Element types of engine inputs and outputs (nvinfer1::DataType).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataType {
    Float,
    Half,
    Int8,
    Int32,
    Bool,
    Uint8,
    Fp8,
    BFloat16,
    Int64,
}

impl DataType {
    fn from_raw(raw: i32) -> Result<Self> {
        Ok(match raw {
            0 => Self::Float,
            1 => Self::Half,
            2 => Self::Int8,
            3 => Self::Int32,
            4 => Self::Bool,
            5 => Self::Uint8,
            6 => Self::Fp8,
            7 => Self::BFloat16,
            8 => Self::Int64,
            other => bail!("unsupported TensorRT data type {other}"),
        })
    }

    pub fn size(self) -> usize {
        match self {
            Self::Int8 | Self::Bool | Self::Uint8 | Self::Fp8 => 1,
            Self::Half | Self::BFloat16 => 2,
            Self::Float | Self::Int32 => 4,
            Self::Int64 => 8,
        }
    }
}

/// An engine input or output. A `-1` extent is set per run.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub input: bool,
    pub data_type: DataType,
    pub shape: Vec<i64>,
}

/// Serializes `onnx` into an engine. TensorRT-RTX builds without a GPU and
/// takes precision from the graph's types.
pub(crate) fn build(onnx: &Path) -> Result<Vec<u8>> {
    crate::cudart()?;
    let path = CString::new(onnx.to_str().context("ONNX path is not UTF-8")?)?;
    // SAFETY: the shim is initialized and `path` is NUL-terminated.
    let blob = Blob(unsafe { sys::ktrt_build_onnx(path.as_ptr()) });
    if blob.0.is_null() {
        return Err(crate::shim_error(&format!(
            "failed to build a TensorRT-RTX engine from {}",
            onnx.display()
        )));
    }
    Ok(blob.bytes().to_vec())
}

pub(crate) struct Blob(pub(crate) *mut sys::ktrt_blob);

impl Blob {
    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: a non-null blob owns `size` bytes until it is freed.
        unsafe {
            slice::from_raw_parts(
                sys::ktrt_blob_data(self.0).cast(),
                sys::ktrt_blob_size(self.0),
            )
        }
    }
}

impl Drop for Blob {
    fn drop(&mut self) {
        // SAFETY: the blob is owned and freed once; freeing null is a no-op.
        unsafe { sys::ktrt_blob_free(self.0) };
    }
}

/// A deserialized engine with the runtime it came from. Field order drops
/// the engine before the runtime, as TensorRT requires.
pub(crate) struct Engine {
    raw: *mut sys::ktrt_engine,
    runtime: *mut sys::ktrt_runtime,
    tensors: Vec<TensorInfo>,
}

// SAFETY: an ICudaEngine may move between threads; `Engine` exposes no
// shared mutation.
unsafe impl Send for Engine {}

impl Engine {
    pub(crate) fn deserialize(data: &[u8]) -> Result<Self> {
        crate::cudart()?;
        // SAFETY: the shim is initialized.
        let runtime = unsafe { sys::ktrt_runtime_create() };
        if runtime.is_null() {
            return Err(crate::shim_error("failed to create a TensorRT-RTX runtime"));
        }
        // SAFETY: `runtime` is live and `data` is a live slice.
        let raw =
            unsafe { sys::ktrt_engine_deserialize(runtime, data.as_ptr().cast(), data.len()) };
        if raw.is_null() {
            // SAFETY: `runtime` is owned and unused.
            unsafe { sys::ktrt_runtime_free(runtime) };
            return Err(crate::shim_error(
                "failed to deserialize a TensorRT-RTX engine",
            ));
        }
        let mut engine = Self {
            raw,
            runtime,
            tensors: Vec::new(),
        };
        engine.tensors = engine.read_tensors()?;
        Ok(engine)
    }

    fn read_tensors(&self) -> Result<Vec<TensorInfo>> {
        // SAFETY: the engine is live; names stay valid for its lifetime and
        // are copied before returning.
        unsafe {
            (0..sys::ktrt_engine_io_count(self.raw))
                .map(|index| {
                    let name = sys::ktrt_engine_io_name(self.raw, index);
                    let owned = CStr::from_ptr(name).to_string_lossy().into_owned();
                    let mut dims = [0i64; 8];
                    let rank = sys::ktrt_engine_io_shape(self.raw, name, dims.as_mut_ptr(), 8);
                    Ok(TensorInfo {
                        input: sys::ktrt_engine_io_is_input(self.raw, name) == 1,
                        data_type: DataType::from_raw(sys::ktrt_engine_io_dtype(self.raw, name))
                            .with_context(|| format!("tensor {owned}"))?,
                        shape: dims[..usize::try_from(rank)?].to_vec(),
                        name: owned,
                    })
                })
                .collect()
        }
    }

    pub(crate) fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub(crate) fn device_memory(&self) -> i64 {
        // SAFETY: the engine is live.
        unsafe { sys::ktrt_engine_device_memory(self.raw) }
    }

    pub(crate) fn raw(&self) -> *mut sys::ktrt_engine {
        self.raw
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // SAFETY: both are owned and freed once, the engine first.
        unsafe {
            sys::ktrt_engine_free(self.raw);
            sys::ktrt_runtime_free(self.runtime);
        }
    }
}

/// An execution context. It must be dropped before its engine.
pub(crate) struct ExecutionContext {
    raw: *mut sys::ktrt_context,
}

// SAFETY: an IExecutionContext may move between threads; all use goes
// through `&mut self`.
unsafe impl Send for ExecutionContext {}

impl ExecutionContext {
    pub(crate) fn new(engine: &Engine, cache: &[u8], cuda_graph: bool) -> Result<Self> {
        // SAFETY: the engine is live and `cache` is a live slice.
        let raw = unsafe {
            sys::ktrt_context_create(
                engine.raw(),
                if cache.is_empty() {
                    ptr::null()
                } else {
                    cache.as_ptr().cast()
                },
                cache.len(),
                i32::from(cuda_graph),
            )
        };
        if raw.is_null() {
            return Err(crate::shim_error(
                "failed to create a TensorRT-RTX execution context",
            ));
        }
        Ok(Self { raw })
    }

    pub(crate) fn runtime_cache(&self) -> Result<Vec<u8>> {
        // SAFETY: the context is live.
        let blob = Blob(unsafe { sys::ktrt_context_runtime_cache(self.raw) });
        if blob.0.is_null() {
            return Err(crate::shim_error("failed to serialize the runtime cache"));
        }
        Ok(blob.bytes().to_vec())
    }

    pub(crate) fn set_input_shape(&mut self, name: &CStr, shape: &[i64]) -> Result<()> {
        // SAFETY: the context is live; `shape` is a live slice of `rank` extents.
        crate::check(
            unsafe {
                sys::ktrt_context_set_input_shape(
                    self.raw,
                    name.as_ptr(),
                    shape.as_ptr(),
                    i32::try_from(shape.len())?,
                )
            },
            &format!("failed to set the shape of {}", name.to_string_lossy()),
        )
    }

    pub(crate) fn shape(&self, name: &CStr) -> Result<Vec<i64>> {
        let mut dims = [0i64; 8];
        // SAFETY: the context is live and `dims` holds 8 extents.
        let rank =
            unsafe { sys::ktrt_context_io_shape(self.raw, name.as_ptr(), dims.as_mut_ptr(), 8) };
        Ok(dims[..usize::try_from(rank)?].to_vec())
    }

    /// # Safety
    ///
    /// `address` must point to device memory large enough for the tensor,
    /// live until the enqueued work that uses it has finished.
    pub(crate) unsafe fn set_tensor_address(
        &mut self,
        name: &CStr,
        address: *mut std::ffi::c_void,
    ) -> Result<()> {
        crate::check(
            unsafe { sys::ktrt_context_set_tensor_address(self.raw, name.as_ptr(), address) },
            &format!("failed to bind {}", name.to_string_lossy()),
        )
    }

    /// # Safety
    ///
    /// Every tensor must be bound as `set_tensor_address` requires, and
    /// `stream` must be a live stream on the engine's device.
    pub(crate) unsafe fn enqueue(
        &mut self,
        stream: koharu_tensorrt_sys::cudaStream_t,
    ) -> Result<()> {
        crate::check(
            unsafe { sys::ktrt_context_enqueue(self.raw, stream) },
            "TensorRT-RTX inference failed",
        )
    }
}

impl Drop for ExecutionContext {
    fn drop(&mut self) {
        // SAFETY: the context is owned and freed once.
        unsafe { sys::ktrt_context_free(self.raw) };
    }
}
