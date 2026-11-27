use thiserror::Error;

use crate::Language;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{provider} returned {actual} segments; expected {expected}")]
    SegmentCount {
        provider: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{provider} returned a blank translation for source segment at index {index}")]
    BlankTranslation {
        provider: &'static str,
        index: usize,
    },
    #[error("{provider} quota exhausted: {message}")]
    QuotaExceeded {
        provider: &'static str,
        message: String,
    },
    #[error("{provider} temporarily rate limited: {message}")]
    RateLimited {
        provider: &'static str,
        message: String,
    },
    #[error("{provider} API request failed with HTTP {status}: {message}")]
    Api {
        provider: &'static str,
        status: u16,
        message: String,
    },
    #[error("{provider} does not support {language}")]
    UnsupportedLanguage {
        provider: &'static str,
        language: Language,
    },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
