//! The CUDA memory and streams a [`crate::Session`] runs on.

use std::{
    ffi::{CStr, c_int, c_void},
    ptr,
};

use anyhow::{Result, bail, ensure};
use koharu_tensorrt_sys as sys;

fn check(cudart: &sys::Cudart, status: sys::cudaError_t, call: &str) -> Result<()> {
    if status == sys::cudaSuccess {
        return Ok(());
    }
    // SAFETY: cudaGetErrorString returns a static NUL-terminated string.
    let reason = unsafe { CStr::from_ptr((cudart.get_error_string)(status)) }.to_string_lossy();
    bail!("{call} failed: {reason}")
}

pub(crate) fn set_device(device: u32) -> Result<()> {
    let cudart = crate::cudart()?;
    // SAFETY: plain CUDA runtime call.
    check(
        cudart,
        unsafe { (cudart.set_device)(c_int::try_from(device)?) },
        "cudaSetDevice",
    )
}

pub(crate) fn compute_capability(device: u32) -> Result<(i32, i32)> {
    let cudart = crate::cudart()?;
    let device = c_int::try_from(device)?;
    let (mut major, mut minor) = (0, 0);
    // SAFETY: the out-pointers are valid for the calls.
    unsafe {
        check(
            cudart,
            (cudart.device_get_attribute)(
                &mut major,
                sys::cudaDevAttrComputeCapabilityMajor,
                device,
            ),
            "cudaDeviceGetAttribute",
        )?;
        check(
            cudart,
            (cudart.device_get_attribute)(
                &mut minor,
                sys::cudaDevAttrComputeCapabilityMinor,
                device,
            ),
            "cudaDeviceGetAttribute",
        )?;
    }
    Ok((major, minor))
}

pub(crate) fn driver_version() -> Result<i32> {
    let cudart = crate::cudart()?;
    let mut version = 0;
    // SAFETY: the out-pointer is valid for the call.
    check(
        cudart,
        unsafe { (cudart.driver_get_version)(&mut version) },
        "cudaDriverGetVersion",
    )?;
    Ok(version)
}

/// A non-blocking CUDA stream on the device current when it was created.
#[derive(Debug)]
pub struct Stream {
    raw: sys::cudaStream_t,
}

// SAFETY: a CUDA stream handle may be used from any thread.
unsafe impl Send for Stream {}

impl Stream {
    pub fn new(device: u32) -> Result<Self> {
        set_device(device)?;
        let cudart = crate::cudart()?;
        let mut raw = ptr::null_mut();
        // SAFETY: the out-pointer is valid for the call.
        check(
            cudart,
            unsafe { (cudart.stream_create_with_flags)(&mut raw, sys::cudaStreamNonBlocking) },
            "cudaStreamCreateWithFlags",
        )?;
        Ok(Self { raw })
    }

    /// Waits for all work enqueued on the stream.
    pub fn synchronize(&self) -> Result<()> {
        let cudart = crate::cudart()?;
        // SAFETY: the stream is live.
        check(
            cudart,
            unsafe { (cudart.stream_synchronize)(self.raw) },
            "cudaStreamSynchronize",
        )
    }

    pub(crate) fn raw(&self) -> sys::cudaStream_t {
        self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Some(cudart) = sys::cudart() {
            // SAFETY: the stream is live and dropped once. Pending work still
            // completes after destruction.
            unsafe { (cudart.stream_destroy)(self.raw) };
        }
    }
}

/// Device memory owned by Rust.
///
/// Host transfers are ordered on a [`Stream`]. Copies from pageable host
/// memory return once the source is staged and copies to it return once the
/// data has arrived, so a host slice is never read or written after its
/// borrow ends. Dropping the buffer frees it with `cudaFree`, which waits for
/// work still using it.
#[derive(Debug)]
pub struct DeviceBuffer {
    raw: *mut c_void,
    len: usize,
}

// SAFETY: device memory may be used from any thread.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    /// Allocates `len` bytes on `device`, uninitialized.
    pub fn new(device: u32, len: usize) -> Result<Self> {
        set_device(device)?;
        let cudart = crate::cudart()?;
        let mut raw = ptr::null_mut();
        // SAFETY: the out-pointer is valid for the call.
        check(
            cudart,
            unsafe { (cudart.malloc)(&mut raw, len.max(1)) },
            "cudaMalloc",
        )?;
        Ok(Self { raw, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies `data` to the start of the buffer.
    pub fn upload<T: Copy>(&mut self, data: &[T], stream: &Stream) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        ensure!(
            bytes <= self.len,
            "upload of {bytes} bytes into a {} byte buffer",
            self.len
        );
        let cudart = crate::cudart()?;
        // SAFETY: the destination holds `bytes` bytes and the source is a live
        // slice; see the type's note on host lifetimes.
        check(
            cudart,
            unsafe {
                (cudart.memcpy_async)(
                    self.raw,
                    data.as_ptr().cast(),
                    bytes,
                    sys::cudaMemcpyHostToDevice,
                    stream.raw(),
                )
            },
            "cudaMemcpyAsync",
        )
    }

    /// Copies the start of the buffer into `data`, waiting for the stream.
    pub fn download<T: Copy>(&self, data: &mut [T], stream: &Stream) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        ensure!(
            bytes <= self.len,
            "download of {bytes} bytes from a {} byte buffer",
            self.len
        );
        let cudart = crate::cudart()?;
        // SAFETY: as in `upload`, with the stream synchronized before the
        // borrow of `data` ends.
        check(
            cudart,
            unsafe {
                (cudart.memcpy_async)(
                    data.as_mut_ptr().cast(),
                    self.raw,
                    bytes,
                    sys::cudaMemcpyDeviceToHost,
                    stream.raw(),
                )
            },
            "cudaMemcpyAsync",
        )?;
        stream.synchronize()
    }

    pub(crate) fn raw(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Some(cudart) = sys::cudart() {
            // SAFETY: the allocation is live and freed once.
            unsafe { (cudart.free)(self.raw) };
        }
    }
}
