use std::{
    cell::Cell,
    fmt,
    marker::PhantomData,
    ptr::{self, NonNull},
    slice,
    sync::Arc,
};

use crate::{
    Audio, CancelMode, ContextParams, Error, ImageGenerationParams, Result, RgbImage, SampleMethod,
    Scheduler, Video, VideoGenerationParams, ffi::NativeCall, image::copy_rgb_from_raw, sys,
};

struct ContextInner {
    pointer: NonNull<sys::sd_ctx_t>,
}

// Context operations require `&mut Context`. The only operation exposed on a
// shared inner handle is native cancellation, which upstream implements using
// an atomic cancellation flag.
unsafe impl Send for ContextInner {}
unsafe impl Sync for ContextInner {}

impl Drop for ContextInner {
    fn drop(&mut self) {
        let _call = NativeCall::enter();
        unsafe { sys::free_sd_ctx(self.pointer.as_ptr()) };
    }
}

/// An owning stable-diffusion.cpp model context.
pub struct Context {
    inner: Arc<ContextInner>,
    // A context may move between threads, but normal operations are not
    // available concurrently through shared references.
    _not_sync: PhantomData<Cell<()>>,
}

impl fmt::Debug for Context {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Context")
            .field("pointer", &self.inner.pointer)
            .finish_non_exhaustive()
    }
}

impl Context {
    /// Loads a model context with the supplied components and backend settings.
    #[tracing::instrument(skip_all)]
    pub fn new(params: &ContextParams) -> Result<Self> {
        let native = params.to_native()?;
        let _call = NativeCall::enter();
        let pointer = unsafe { sys::new_sd_ctx(&raw const native.raw) };
        let pointer = NonNull::new(pointer).ok_or(Error::ContextCreationFailed)?;
        Ok(Self {
            inner: Arc::new(ContextInner { pointer }),
            _not_sync: PhantomData,
        })
    }

    #[must_use]
    pub fn supports_image_generation(&self) -> bool {
        let _call = NativeCall::enter();
        unsafe { sys::sd_ctx_supports_image_generation(self.inner.pointer.as_ptr()) }
    }

    #[must_use]
    pub fn supports_video_generation(&self) -> bool {
        let _call = NativeCall::enter();
        unsafe { sys::sd_ctx_supports_video_generation(self.inner.pointer.as_ptr()) }
    }

    /// Returns the native model-specific default sampling method.
    pub fn default_sample_method(&self) -> Result<SampleMethod> {
        let _call = NativeCall::enter();
        let raw = unsafe { sys::sd_get_default_sample_method(self.inner.pointer.as_ptr()) };
        SampleMethod::try_from(raw)
    }

    /// Returns the native model-specific scheduler default for a sampler.
    pub fn default_scheduler(&self, sample_method: SampleMethod) -> Result<Scheduler> {
        let _call = NativeCall::enter();
        let raw = unsafe {
            sys::sd_get_default_scheduler(self.inner.pointer.as_ptr(), sample_method.as_raw())
        };
        Scheduler::try_from(raw)
    }

    /// Generates one or more owned images.
    pub fn generate_image(&mut self, params: &ImageGenerationParams) -> Result<Vec<RgbImage>> {
        let native = params.to_native()?;
        let _call = NativeCall::enter();
        let mut output = RawImageResult::default();
        let succeeded = unsafe {
            sys::generate_image(
                self.inner.pointer.as_ptr(),
                &raw const native.raw,
                &raw mut output.pointer,
            )
        };
        if !succeeded {
            return Err(Error::ImageGenerationFailed);
        }
        output.copy("image")
    }

    /// Generates owned video frames and optional audio.
    pub fn generate_video(&mut self, params: &VideoGenerationParams) -> Result<Video> {
        let native = params.to_native()?;
        let _call = NativeCall::enter();
        let mut output = RawVideoResult::default();
        let succeeded = unsafe {
            sys::generate_video(
                self.inner.pointer.as_ptr(),
                &raw const native.raw,
                &raw mut output.pointer,
            )
        };
        if !succeeded {
            return Err(Error::VideoGenerationFailed);
        }
        output.copy()
    }

    /// Creates a thread-safe handle that can cancel generation from another thread.
    #[must_use]
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Changes the cancellation state directly.
    pub fn cancel(&self, mode: CancelMode) {
        self.cancel_handle().cancel(mode);
    }
}

/// A cloneable handle to the atomic native cancellation flag.
#[derive(Clone)]
pub struct CancelHandle {
    inner: Arc<ContextInner>,
}

impl fmt::Debug for CancelHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CancelHandle")
            .finish_non_exhaustive()
    }
}

impl CancelHandle {
    pub fn cancel(&self, mode: CancelMode) {
        let _call = NativeCall::enter();
        unsafe { sys::sd_cancel_generation(self.inner.pointer.as_ptr(), mode.as_raw()) };
    }
}

#[derive(Debug)]
pub(crate) struct RawImageResult {
    pub(crate) pointer: *mut sys::sd_image_generation_result_t,
}

impl Default for RawImageResult {
    fn default() -> Self {
        Self {
            pointer: ptr::null_mut(),
        }
    }
}

impl RawImageResult {
    pub(crate) fn copy(&self, kind: &'static str) -> Result<Vec<RgbImage>> {
        let result = NonNull::new(self.pointer).ok_or(Error::InvalidNativeOutput { kind })?;
        let result = unsafe { result.as_ref() };
        let count =
            usize::try_from(result.count).map_err(|_| Error::InvalidNativeOutput { kind })?;
        if count == 0 || result.items.is_null() {
            return Err(Error::InvalidNativeOutput { kind });
        }
        let items = unsafe { slice::from_raw_parts(result.items, count) };
        items
            .iter()
            .map(|item| unsafe { copy_rgb_from_raw(&item.image) })
            .collect()
    }
}

impl Drop for RawImageResult {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe { sys::free_sd_image_generation_result(self.pointer) };
        }
    }
}

#[derive(Debug)]
struct RawVideoResult {
    pointer: *mut sys::sd_video_generation_result_t,
}

impl Default for RawVideoResult {
    fn default() -> Self {
        Self {
            pointer: ptr::null_mut(),
        }
    }
}

impl RawVideoResult {
    fn copy(&self) -> Result<Video> {
        let result =
            NonNull::new(self.pointer).ok_or(Error::InvalidNativeOutput { kind: "video" })?;
        let result = unsafe { result.as_ref() };
        let kind = "video frame";
        let count =
            usize::try_from(result.frame_count).map_err(|_| Error::InvalidNativeOutput { kind })?;
        if count == 0 || result.frames.is_null() {
            return Err(Error::InvalidNativeOutput { kind });
        }
        let frames = unsafe { slice::from_raw_parts(result.frames, count) }
            .iter()
            .map(|raw| unsafe { copy_rgb_from_raw(raw) })
            .collect::<Result<Vec<_>>>()?;
        let audio = match NonNull::new(result.audio) {
            Some(audio) => Some(unsafe { Audio::copy_from_raw(audio.as_ref()) }?),
            None => None,
        };
        let fps = u32::try_from(result.fps)
            .ok()
            .filter(|fps| *fps > 0)
            .ok_or(Error::InvalidNativeOutput {
                kind: "video frame rate",
            })?;
        Ok(Video { frames, audio, fps })
    }
}

impl Drop for RawVideoResult {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe { sys::free_sd_video_generation_result(self.pointer) };
        }
    }
}
