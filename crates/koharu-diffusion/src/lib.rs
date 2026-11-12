//! Safe, owning wrappers for the stable-diffusion.cpp C API.
//!
//! The crate mirrors the upstream API closely while keeping C strings, input
//! buffers, contexts, and generated images under Rust
//! ownership. The native library is loaded dynamically by
//! [`koharu_diffusion_sys`].
//!
//! Raster inputs and outputs use the `image` crate directly: [`RgbImage`] for
//! generated/reference/control images, and [`GrayImage`] for
//! masks.
//!
//! ```no_run
//! use koharu_diffusion::{Context, ContextParams, ImageGenerationParams};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let context_params = ContextParams {
//!     model_path: Some("model.gguf".into()),
//!     ..ContextParams::default()
//! };
//! let mut context = Context::new(&context_params)?;
//! let image_params = ImageGenerationParams {
//!     prompt: "a watercolor fox reading by a window".into(),
//!     ..ImageGenerationParams::default()
//! };
//! let images = context.generate_image(&image_params)?;
//! println!("generated {} image(s)", images.len());
//! images[0].save("output.png")?;
//! # Ok(())
//! # }
//! ```

#![deny(unsafe_op_in_unsafe_fn)]

mod callbacks;
mod context;
mod enums;
mod error;
mod ffi;
mod image;
mod params;
mod system;

pub use ::image::{GrayImage, RgbImage};
pub use callbacks::{
    GraphEvaluation, LogMessage, Preview, PreviewOptions, Progress,
    clear_graph_evaluation_callback, clear_log_callback, clear_preview_callback,
    clear_progress_callback, send_logs_to_tracing, set_graph_evaluation_callback, set_log_callback,
    set_preview_callback, set_progress_callback,
};
pub use context::{CancelHandle, Context};
pub use enums::*;
pub use error::{Error, Result};
pub use image::RgbImageView;
pub use params::*;
pub use system::physical_core_count;

/// Low-level bindings for functionality not yet represented by the safe API.
pub use koharu_diffusion_sys as sys;
