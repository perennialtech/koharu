use std::{ffi::NulError, path::PathBuf};

/// A result returned by the safe stable-diffusion.cpp wrapper.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors reported while validating arguments or invoking stable-diffusion.cpp.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("{field} contains an interior NUL byte")]
    InteriorNul {
        field: &'static str,
        #[source]
        source: NulError,
    },
    #[error("{field} is not valid Unicode: {path}", path = path.display())]
    NonUnicodePath { field: &'static str, path: PathBuf },
    #[error("image dimensions overflow the addressable buffer size")]
    ImageDimensionsOverflow,
    #[error("image width, height, and channel count must all be non-zero")]
    ZeroImageDimension,
    #[error("native image has {actual} channels; expected {expected}")]
    UnexpectedNativeImageChannelCount { expected: u32, actual: u32 },
    #[error("{field} has {len} entries, which exceeds the native API limit")]
    CountOverflow { field: &'static str, len: usize },
    #[error("invalid parameter {name}: {reason}")]
    InvalidParameter {
        name: &'static str,
        reason: &'static str,
    },
    #[error("stable-diffusion.cpp failed to create a model context")]
    ContextCreationFailed,
    #[error("stable-diffusion.cpp failed to generate images")]
    ImageGenerationFailed,
    #[error("stable-diffusion.cpp returned an invalid {kind} output")]
    InvalidNativeOutput { kind: &'static str },
    #[error("a native callback cannot be reconfigured from inside a native call or callback")]
    ReentrantCallbackConfiguration,
    #[error("preview interval must be greater than zero")]
    InvalidPreviewInterval,
    #[error("invalid {kind} value returned by stable-diffusion.cpp: {value}")]
    InvalidEnum { kind: &'static str, value: i64 },
    #[error("unknown {kind} name: {value}")]
    UnknownEnumName { kind: &'static str, value: String },
}
