use serde::{Deserialize, Serialize};
use specta::Type;

/// Sampling options for LLM providers. Machine-translation providers ignore them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct GenerationConfig {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    /// Sent only when set, so models without reasoning controls accept the request.
    pub reasoning: Option<bool>,
    /// Attach the page image for LLM providers.
    pub vision: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            max_tokens: None,
            frequency_penalty: None,
            presence_penalty: None,
            reasoning: None,
            vision: true,
        }
    }
}
