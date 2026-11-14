//! Low-level FFI for TensorRT-RTX and the few CUDA runtime calls its callers
//! need.
//!
//! TensorRT-RTX has only a C++ API, so a small shim compiled by `build.rs`
//! exposes a C ABI over it. Neither the shim nor this crate links TensorRT-RTX
//! or the CUDA runtime: [`load`] resolves them from libraries the runtime
//! package has already brought into the process.

#![allow(non_camel_case_types, non_upper_case_globals)]

use std::{
    ffi::{CStr, c_char, c_int, c_void},
    sync::OnceLock,
};

use libloading::Library;

#[repr(C)]
pub struct ktrt_blob {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ktrt_runtime {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ktrt_engine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ktrt_context {
    _private: [u8; 0],
}

pub type ktrt_log_fn = unsafe extern "C" fn(severity: i32, message: *const c_char);

#[repr(C)]
pub struct ktrt_entry_points {
    pub create_infer_runtime: unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void,
    pub create_infer_builder: unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void,
    pub create_onnx_parser: unsafe extern "C" fn(*mut c_void, *mut c_void, i32) -> *mut c_void,
    pub get_infer_lib_version: unsafe extern "C" fn() -> i32,
    pub log: Option<ktrt_log_fn>,
}

unsafe extern "C" {
    pub fn ktrt_init(entry_points: *const ktrt_entry_points) -> i32;
    pub fn ktrt_last_error() -> *const c_char;
    pub fn ktrt_library_version() -> i32;

    pub fn ktrt_blob_data(blob: *const ktrt_blob) -> *const c_void;
    pub fn ktrt_blob_size(blob: *const ktrt_blob) -> usize;
    pub fn ktrt_blob_free(blob: *mut ktrt_blob);

    pub fn ktrt_build_onnx(path: *const c_char) -> *mut ktrt_blob;

    pub fn ktrt_runtime_create() -> *mut ktrt_runtime;
    pub fn ktrt_runtime_free(runtime: *mut ktrt_runtime);

    pub fn ktrt_engine_deserialize(
        runtime: *mut ktrt_runtime,
        data: *const c_void,
        size: usize,
    ) -> *mut ktrt_engine;
    pub fn ktrt_engine_free(engine: *mut ktrt_engine);
    pub fn ktrt_engine_io_count(engine: *const ktrt_engine) -> i32;
    pub fn ktrt_engine_io_name(engine: *const ktrt_engine, index: i32) -> *const c_char;
    pub fn ktrt_engine_io_is_input(engine: *const ktrt_engine, name: *const c_char) -> i32;
    pub fn ktrt_engine_io_dtype(engine: *const ktrt_engine, name: *const c_char) -> i32;
    pub fn ktrt_engine_io_shape(
        engine: *const ktrt_engine,
        name: *const c_char,
        dims: *mut i64,
        capacity: i32,
    ) -> i32;
    pub fn ktrt_engine_device_memory(engine: *const ktrt_engine) -> i64;

    pub fn ktrt_context_create(
        engine: *mut ktrt_engine,
        cache: *const c_void,
        cache_size: usize,
        cuda_graph: i32,
    ) -> *mut ktrt_context;
    pub fn ktrt_context_free(context: *mut ktrt_context);
    pub fn ktrt_context_runtime_cache(context: *const ktrt_context) -> *mut ktrt_blob;
    pub fn ktrt_context_set_input_shape(
        context: *mut ktrt_context,
        name: *const c_char,
        dims: *const i64,
        rank: i32,
    ) -> i32;
    pub fn ktrt_context_io_shape(
        context: *const ktrt_context,
        name: *const c_char,
        dims: *mut i64,
        capacity: i32,
    ) -> i32;
    pub fn ktrt_context_set_tensor_address(
        context: *mut ktrt_context,
        name: *const c_char,
        address: *mut c_void,
    ) -> i32;
    pub fn ktrt_context_enqueue(context: *mut ktrt_context, stream: *mut c_void) -> i32;
}

/// The message the shim left for the latest failed call on this thread.
pub fn last_error() -> String {
    // SAFETY: the shim returns a NUL-terminated thread-local string that
    // stays valid until the next failing call on this thread.
    unsafe { CStr::from_ptr(ktrt_last_error()) }
        .to_string_lossy()
        .into_owned()
}

pub type cudaError_t = c_int;
pub type cudaStream_t = *mut c_void;

pub const cudaSuccess: cudaError_t = 0;
pub const cudaStreamNonBlocking: u32 = 1;
pub const cudaMemcpyHostToDevice: c_int = 1;
pub const cudaMemcpyDeviceToHost: c_int = 2;
pub const cudaDevAttrComputeCapabilityMajor: c_int = 75;
pub const cudaDevAttrComputeCapabilityMinor: c_int = 76;

/// CUDA runtime functions, resolved from the process's CUDA runtime.
pub struct Cudart {
    pub set_device: unsafe extern "C" fn(c_int) -> cudaError_t,
    pub malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> cudaError_t,
    pub free: unsafe extern "C" fn(*mut c_void) -> cudaError_t,
    pub memcpy_async:
        unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, cudaStream_t) -> cudaError_t,
    pub stream_create_with_flags: unsafe extern "C" fn(*mut cudaStream_t, u32) -> cudaError_t,
    pub stream_destroy: unsafe extern "C" fn(cudaStream_t) -> cudaError_t,
    pub stream_synchronize: unsafe extern "C" fn(cudaStream_t) -> cudaError_t,
    pub device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> cudaError_t,
    pub driver_get_version: unsafe extern "C" fn(*mut c_int) -> cudaError_t,
    pub get_error_string: unsafe extern "C" fn(cudaError_t) -> *const c_char,
}

#[cfg(target_os = "linux")]
mod names {
    pub const TENSORRT: &str = "libtensorrt_rtx.so.1";
    pub const ONNX_PARSER: &str = "libtensorrt_onnxparser_rtx.so.1";
    pub const CUDART: &str = "libcudart.so.13";
}

#[cfg(target_os = "windows")]
mod names {
    pub const TENSORRT: &str = "tensorrt_rtx_1_7.dll";
    pub const ONNX_PARSER: &str = "tensorrt_onnxparser_rtx_1_7.dll";
    pub const CUDART: &str = "cudart64_13.dll";
}

struct Loaded {
    cudart: Cudart,
    // Keep the handles so the resolved functions stay valid.
    _libraries: [Library; 3],
}

static LOADED: OnceLock<Result<Loaded, String>> = OnceLock::new();

/// Resolves TensorRT-RTX and the CUDA runtime from libraries already loaded
/// into the process, and initializes the shim with `log` as TensorRT's
/// logger. Later calls return the first call's result.
///
/// # Safety
///
/// `log` must be safe to call from any thread for the life of the process.
pub unsafe fn load(log: ktrt_log_fn) -> Result<&'static Cudart, String> {
    LOADED
        .get_or_init(|| unsafe { load_once(log) })
        .as_ref()
        .map(|loaded| &loaded.cudart)
        .map_err(Clone::clone)
}

/// The CUDA runtime, if [`load`] succeeded.
pub fn cudart() -> Option<&'static Cudart> {
    LOADED.get()?.as_ref().ok().map(|loaded| &loaded.cudart)
}

unsafe fn load_once(log: ktrt_log_fn) -> Result<Loaded, String> {
    let tensorrt = unsafe { open_loaded(names::TENSORRT) }?;
    let parser = unsafe { open_loaded(names::ONNX_PARSER) }?;
    let cudart = unsafe { open_loaded(names::CUDART) }?;
    let entry_points = unsafe {
        ktrt_entry_points {
            create_infer_runtime: symbol(&tensorrt, b"createInferRuntime_INTERNAL\0")?,
            create_infer_builder: symbol(&tensorrt, b"createInferBuilder_INTERNAL\0")?,
            create_onnx_parser: symbol(&parser, b"createNvOnnxParser_INTERNAL\0")?,
            get_infer_lib_version: symbol(&tensorrt, b"getInferLibVersion\0")?,
            log: Some(log),
        }
    };
    if unsafe { ktrt_init(&entry_points) } != 0 {
        return Err(last_error());
    }
    let functions = unsafe {
        Cudart {
            set_device: symbol(&cudart, b"cudaSetDevice\0")?,
            malloc: symbol(&cudart, b"cudaMalloc\0")?,
            free: symbol(&cudart, b"cudaFree\0")?,
            memcpy_async: symbol(&cudart, b"cudaMemcpyAsync\0")?,
            stream_create_with_flags: symbol(&cudart, b"cudaStreamCreateWithFlags\0")?,
            stream_destroy: symbol(&cudart, b"cudaStreamDestroy\0")?,
            stream_synchronize: symbol(&cudart, b"cudaStreamSynchronize\0")?,
            device_get_attribute: symbol(&cudart, b"cudaDeviceGetAttribute\0")?,
            driver_get_version: symbol(&cudart, b"cudaDriverGetVersion\0")?,
            get_error_string: symbol(&cudart, b"cudaGetErrorString\0")?,
        }
    };
    Ok(Loaded {
        cudart: functions,
        _libraries: [tensorrt, parser, cudart],
    })
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| {
            format!(
                "missing symbol {}: {error}",
                String::from_utf8_lossy(&name[..name.len() - 1])
            )
        })
}

// Opening by name with RTLD_NOLOAD only finds a library already in the
// process, which keeps package activation the one place that decides which
// files load.
#[cfg(target_os = "linux")]
unsafe fn open_loaded(name: &str) -> Result<Library, String> {
    use libloading::os::unix::{Library, RTLD_LAZY};
    unsafe { Library::open(Some(name), RTLD_LAZY | libc::RTLD_NOLOAD) }
        .map(Into::into)
        .map_err(|error| format!("{name} is not loaded: {error}"))
}

#[cfg(target_os = "windows")]
unsafe fn open_loaded(name: &str) -> Result<Library, String> {
    libloading::os::windows::Library::open_already_loaded(name)
        .map(Into::into)
        .map_err(|error| format!("{name} is not loaded: {error}"))
}
