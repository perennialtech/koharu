//! High-level local LLM inference backed by llama.cpp.

mod model;

use std::{num::NonZeroU32, path::PathBuf, time::Duration};

use anyhow::Result;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use self::model::Model;

pub use koharu_llama::{context::params::KvCacheType, model::params::LlamaLoadMode};

const DEFAULT_GPU_LAYERS: u32 = 1000;
const DEFAULT_MAX_TOKENS: usize = 512;
const DEFAULT_SEED: u32 = 299_792_458;

/// A loaded GGUF language model with optional image input.
#[derive(Debug)]
pub struct Llm {
    model: Model,
}

impl Llm {
    /// Loads a text-only GGUF model.
    pub async fn load(device: crate::Device, model_path: impl Into<PathBuf>) -> Result<Self> {
        Self::load_with_options(device, model_path, LoadOptions::default()).await
    }

    /// Loads a GGUF model with model and optional MTMD projector settings.
    pub async fn load_with_options(
        device: crate::Device,
        model_path: impl Into<PathBuf>,
        options: LoadOptions,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let model = tokio_rayon::spawn(move || Model::new(&device, model_path, options)).await?;
        Ok(Self { model })
    }

    /// Converts the model's beginning-of-sequence token to its textual form.
    pub fn bos_token(&self) -> Result<String> {
        self.model.bos_token()
    }

    /// Converts the model's end-of-sequence token to its textual form.
    pub fn eos_token(&self) -> Result<String> {
        self.model.eos_token()
    }

    /// Returns the number of transformer layers in the loaded model.
    #[must_use]
    pub fn n_layer(&self) -> u32 {
        self.model.n_layer()
    }

    /// Returns the total tensor size of the loaded model in bytes.
    #[must_use]
    pub fn model_size_bytes(&self) -> u64 {
        self.model.model_size_bytes()
    }

    /// Returns the model's sliding window attention size, or `0` if unused.
    #[must_use]
    pub fn n_swa(&self) -> i32 {
        self.model.n_swa()
    }

    /// Returns whether an MTP speculative decoding draft model is loaded.
    #[must_use]
    pub fn mtp_enabled(&self) -> bool {
        self.model.mtp_enabled()
    }

    /// Returns the auto-planned MoE device cache size in bytes, or `0` if none was planned.
    #[must_use]
    pub fn recommended_moe_cache_bytes(&self) -> usize {
        self.model.recommended_moe_cache_bytes()
    }

    /// Applies the GGUF chat template with a generation prompt.
    pub fn render_chat_prompt(&self, messages: &[ChatMessage]) -> Result<String> {
        self.model.render_chat_prompt(messages)
    }

    /// Generates text for several inputs together.
    ///
    /// Returns one generation per input, in input order.
    pub fn inference_batch(
        &self,
        inputs: &[Input<'_>],
        options: &GenerationOptions,
    ) -> Result<Vec<Generation>> {
        self.model.inference_batch(inputs, options)
    }
}

/// Controls CPU placement of Mixture-of-Experts tensors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoeOffload {
    /// Automatically fit MoE tensors and dynamic expert cache to available VRAM.
    #[default]
    Auto,
    /// Keep all MoE expert tensors on the selected accelerator.
    None,
    /// Place all MoE expert tensors on CPU.
    AllCpu,
    /// Place MoE expert tensors for the first `n` transformer layers on CPU.
    FirstLayers(u32),
}

/// Options used while loading a GGUF model.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Number of model layers to offload when an accelerator is selected.
    pub gpu_layers: u32,
    /// Controls how llama.cpp reads and retains model data.
    pub load_mode: LlamaLoadMode,
    /// Overrides an incorrect end-of-sequence token advertised by a GGUF file.
    pub eos_token_id: Option<i32>,
    /// Controls CPU placement of Mixture-of-Experts tensors.
    pub moe_offload: MoeOffload,
    /// Bypass host memory buffers so extra buffer types can be used.
    pub no_host: bool,
    /// Use extra buffer types (for weight repacking).
    pub use_extra_bufts: bool,
    /// Optional multimodal projector configuration.
    pub mtmd: Option<MtmdOptions>,
    /// Optional Gemma 4 Multi-Token Prediction (MTP) speculative decoding configuration.
    pub mtp: Option<MtpOptions>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            gpu_layers: DEFAULT_GPU_LAYERS,
            load_mode: LlamaLoadMode::Auto,
            eos_token_id: None,
            moe_offload: MoeOffload::Auto,
            no_host: false,
            use_extra_bufts: true,
            mtmd: None,
            mtp: None,
        }
    }
}

/// Configuration for Gemma 4 Multi-Token Prediction (MTP) speculative decoding.
#[derive(Debug, Clone)]
pub struct MtpOptions {
    /// Path to the MTP draft GGUF model.
    pub model_path: PathBuf,
    /// Number of draft layers to offload to the accelerator.
    pub gpu_layers: u32,
    /// Maximum number of draft tokens proposed per target forward pass.
    pub n_draft: usize,
    /// Minimum top-1 softmax probability required to keep a drafted token.
    pub p_min: f32,
}

impl MtpOptions {
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            gpu_layers: DEFAULT_GPU_LAYERS,
            n_draft: 4,
            p_min: 0.5,
        }
    }
}

/// Configuration for llama.cpp's multimodal (MTMD) image projector.
#[derive(Debug, Clone)]
pub struct MtmdOptions {
    pub projector_path: PathBuf,
    /// `-1` selects the projector default.
    pub image_min_tokens: i32,
    /// `-1` selects the projector default.
    pub image_max_tokens: i32,
}

impl MtmdOptions {
    #[must_use]
    pub fn new(projector_path: impl Into<PathBuf>) -> Self {
        Self {
            projector_path: projector_path.into(),
            image_min_tokens: -1,
            image_max_tokens: -1,
        }
    }

    #[must_use]
    pub fn with_image_token_range(mut self, min: i32, max: i32) -> Self {
        self.image_min_tokens = min;
        self.image_max_tokens = max;
        self
    }
}

/// A raw prompt and any images referenced by its MTMD markers.
#[derive(Debug)]
pub struct Input<'a> {
    prompt: &'a str,
    images: Vec<&'a DynamicImage>,
}

impl<'a> Input<'a> {
    #[must_use]
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            images: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_image(mut self, image: &'a DynamicImage) -> Self {
        self.images.push(image);
        self
    }

    #[must_use]
    pub fn prompt(&self) -> &str {
        self.prompt
    }

    #[must_use]
    pub fn images(&self) -> &[&'a DynamicImage] {
        &self.images
    }
}

/// GBNF grammar constraint for structured generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grammar {
    pub gbnf: String,
    pub root: String,
}

impl Grammar {
    #[must_use]
    pub fn new(gbnf: impl Into<String>, root: impl Into<String>) -> Self {
        Self {
            gbnf: gbnf.into(),
            root: root.into(),
        }
    }
}

/// Sampling and context settings for one inference call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub seed: u32,
    pub repeat_penalty: f32,
    /// Number of recent tokens to penalize; `0` disables repeat penalties.
    pub repeat_last_n: i32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// Whether llama.cpp or MTMD should add the model's special prompt tokens.
    pub add_special: bool,
    pub n_ctx: Option<NonZeroU32>,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub n_threads: Option<i32>,
    pub n_threads_batch: Option<i32>,
    /// Enables flash attention (`LLAMA_FLASH_ATTN_TYPE_ENABLED` when true).
    pub flash_attn: bool,
    /// Uses full-size SWA KV cache instead of sliding window allocation.
    pub swa_full: bool,
    #[serde(with = "serde_kv_cache_type")]
    pub kv_cache_type_k: Option<KvCacheType>,
    #[serde(with = "serde_kv_cache_type")]
    pub kv_cache_type_v: Option<KvCacheType>,
    /// Shares a unified KV buffer across sequences when enabled.
    pub kv_unified: bool,
    /// Offloads KQV operations to the accelerator.
    pub offload_kqv: bool,
    /// Offloads host tensor operations to the accelerator.
    pub op_offload: bool,
    /// Device cache for MoE experts in host memory. `None` uses the automatic planner's
    /// recommendation when available, `Some(0)` disables the cache, and `Some(n)` sets its size in bytes.
    pub moe_cache_size: Option<usize>,
    /// Share of MoE cache misses computed on CPU instead of uploaded (`0.0..=1.0`).
    pub moe_cache_cpu_share: f32,
    /// Optional GBNF grammar constraint enforced during sampling.
    pub grammar: Option<Grammar>,
    /// Whether to render special/control tokens into generated text.
    pub decode_special_tokens: bool,
    /// Whether to abort generation when a short token repetition loop is detected.
    pub abort_on_repetition_loop: bool,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: 0.1,
            top_k: None,
            top_p: None,
            min_p: None,
            seed: DEFAULT_SEED,
            repeat_penalty: 1.2,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            add_special: false,
            n_ctx: None,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            flash_attn: true,
            swa_full: false,
            kv_cache_type_k: None,
            kv_cache_type_v: None,
            kv_unified: false,
            offload_kqv: true,
            op_offload: true,
            moe_cache_size: None,
            moe_cache_cpu_share: 0.0,
            grammar: None,
            decode_special_tokens: false,
            abort_on_repetition_loop: true,
        }
    }
}

mod serde_kv_cache_type {
    use koharu_llama::context::params::KvCacheType;
    use serde::{Deserialize, Deserializer, Serializer, de, ser::Error as _};

    macro_rules! cache_types {
        ($($kind:ident),+ $(,)?) => {
            const NAMES: &[(KvCacheType, &str)] = &[$((KvCacheType::$kind, stringify!($kind))),+];

            fn name(kind: KvCacheType) -> Option<&'static str> {
                match kind {
                    $(KvCacheType::$kind => Some(stringify!($kind)),)+
                    KvCacheType::Unknown(_) => None,
                }
            }
        };
    }

    cache_types!(
        F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ2_XXS,
        IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, I8, I16, I32, I64, F64, IQ1_M, BF16,
        TQ1_0, TQ2_0, MXFP4,
    );

    pub fn serialize<S: Serializer>(
        value: &Option<KvCacheType>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            None => serializer.serialize_none(),
            Some(kind) => serializer.serialize_some(
                name(*kind)
                    .ok_or_else(|| S::Error::custom(format!("unknown KvCacheType: {kind:?}")))?,
            ),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<KvCacheType>, D::Error> {
        Option::<String>::deserialize(deserializer)?
            .map(|name| {
                NAMES
                    .iter()
                    .find(|(_, key)| key.eq_ignore_ascii_case(&name))
                    .map(|&(kind, _)| kind)
                    .ok_or_else(|| de::Error::custom(format!("unknown KvCacheType: {name}")))
            })
            .transpose()
    }

    #[cfg(test)]
    #[test]
    fn named_cache_types_round_trip_in_generation_options() {
        use super::GenerationOptions;
        for &(kind, name) in NAMES {
            let options = GenerationOptions {
                kv_cache_type_k: Some(kind),
                ..Default::default()
            };
            let json = serde_json::to_value(&options).unwrap();
            assert_eq!(json["kv_cache_type_k"], name);
            let decoded: GenerationOptions = serde_json::from_value(json).unwrap();
            assert_eq!(decoded.kv_cache_type_k, Some(kind));
            assert_eq!(decoded.kv_cache_type_v, None);
        }
        let unknown = GenerationOptions {
            kv_cache_type_v: Some(KvCacheType::Unknown(1000)),
            ..Default::default()
        };
        assert!(serde_json::to_value(unknown).is_err());
    }
}

/// Text and accounting information produced by one inference call.
#[derive(Debug, Clone)]
pub struct Generation {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_duration: Duration,
    pub generation_duration: Duration,
    pub finish_reason: FinishReason,
}

impl Generation {
    pub(super) fn empty(finish_reason: FinishReason) -> Self {
        Self {
            text: String::new(),
            prompt_tokens: 0,
            generated_tokens: 0,
            prompt_duration: Duration::ZERO,
            generation_duration: Duration::ZERO,
            finish_reason,
        }
    }

    #[must_use]
    pub fn prompt_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prompt_duration)
    }

    #[must_use]
    pub fn generated_tokens_per_second(&self) -> f64 {
        rate(self.generated_tokens, self.generation_duration)
    }
}

/// Why generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    StopToken,
    Length,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    #[must_use]
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }

    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

fn rate(tokens: usize, duration: Duration) -> f64 {
    if duration.as_secs_f64() > 0.0 {
        tokens as f64 / duration.as_secs_f64()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtmd_defaults_to_projector_token_budget() {
        let options = MtmdOptions::new("projector.gguf");
        assert_eq!(options.image_min_tokens, -1);
        assert_eq!(options.image_max_tokens, -1);
    }

    #[test]
    fn load_options_do_not_override_eos_by_default() {
        assert_eq!(LoadOptions::default().eos_token_id, None);
    }
}
