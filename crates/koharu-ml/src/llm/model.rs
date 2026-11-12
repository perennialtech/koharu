//! llama.cpp generation and MTMD evaluation.
//!
//! llama.cpp generation loop:
//! https://github.com/ggml-org/llama.cpp/blob/99f3dc32296f825fec94f202da1e9fede1e78cf9/examples/simple/simple.cpp
//! MTMD helper evaluation:
//! https://github.com/ggml-org/llama.cpp/blob/99f3dc32296f825fec94f202da1e9fede1e78cf9/tools/mtmd/mtmd-helper.cpp

use std::{
    fmt::{Debug, Formatter},
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use koharu_llama::{
    context::{
        LlamaContext,
        params::{LlamaAttentionType, LlamaContextParams},
    },
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use minijinja::{Environment, Error as TemplateError, ErrorKind as TemplateErrorKind, context};

use super::{ChatMessage, FinishReason, Generation, GenerationOptions, Input, LoadOptions};
use crate::Backend;

const DEFAULT_MAX_UBATCH: u32 = 512;
const CHAT_TEMPLATE_NAME: &str = "chat";

pub(super) struct Model {
    backend: &'static LlamaBackend,
    model: Arc<LlamaModel>,
    mtmd: Option<Mutex<MtmdContext>>,
    eos_token: LlamaToken,
    chat_template: ChatTemplate,
}

impl Debug for Model {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("model", &self.model)
            .field("vision", &self.mtmd.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct ChatTemplate {
    environment: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    fn from_model(model: &LlamaModel, eos_token: LlamaToken) -> Result<Self> {
        let template = model
            .chat_template(None)
            .context("GGUF model does not contain a chat template")?;
        let source = template
            .to_string()
            .context("GGUF chat template is not valid UTF-8")?;
        let mut environment = Environment::new();
        minijinja_contrib::add_to_environment(&mut environment);
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function("raise_exception", raise_template_exception);
        environment
            .add_template_owned(CHAT_TEMPLATE_NAME, source)
            .map_err(anyhow::Error::msg)
            .context("failed to compile GGUF chat template")?;
        Ok(Self {
            environment,
            bos_token: token_text(model, model.token_bos())
                .context("failed to decode model BOS token for chat template")?,
            eos_token: token_text(model, eos_token)
                .context("failed to decode model EOS token for chat template")?,
        })
    }

    fn render(&self, messages: &[ChatMessage]) -> Result<String> {
        let messages = messages
            .iter()
            .map(|message| {
                context! {
                    role => message.role.as_str(),
                    content => message.content.as_str(),
                }
            })
            .collect::<Vec<_>>();
        let template = self
            .environment
            .get_template(CHAT_TEMPLATE_NAME)
            .map_err(anyhow::Error::msg)
            .context("compiled GGUF chat template is unavailable")?;
        template
            .render(context! {
                messages => messages,
                bos_token => self.bos_token.as_str(),
                eos_token => self.eos_token.as_str(),
                add_generation_prompt => true,
                enable_thinking => false,
                tools => Vec::<()>::new(),
                documents => Vec::<()>::new(),
            })
            .map_err(anyhow::Error::msg)
            .context("failed to render GGUF chat template")
    }
}

fn raise_template_exception(message: String) -> std::result::Result<String, TemplateError> {
    Err(TemplateError::new(
        TemplateErrorKind::InvalidOperation,
        message,
    ))
}

impl Model {
    pub(super) fn new(
        device: &crate::Device,
        model_path: PathBuf,
        options: LoadOptions,
    ) -> Result<Self> {
        ensure_file(&model_path, "GGUF model")?;
        let backend = crate::llama_backend().context("llama.cpp backend is not initialized")?;
        let params = model_params(backend, device, &options)?;
        let model = Arc::new(
            LlamaModel::load_from_file(backend, &model_path, &params)
                .with_context(|| format!("failed to load GGUF model {}", model_path.display()))?,
        );
        ensure!(
            model.has_decoder(),
            "GGUF model does not advertise a decoder graph"
        );

        let eos_token = options
            .eos_token_id
            .map_or_else(|| model.token_eos(), LlamaToken::new);
        let chat_template = ChatTemplate::from_model(&model, eos_token)?;
        let mtmd = match options.mtmd {
            Some(options) => {
                ensure_file(&options.projector_path, "MTMD projector")?;
                validate_image_token_range(options.image_min_tokens, options.image_max_tokens)?;
                let projector_path = options.projector_path.to_str().with_context(|| {
                    format!(
                        "invalid MTMD projector path {}",
                        options.projector_path.display()
                    )
                })?;
                let mtmd = MtmdContext::init_from_file(
                    projector_path,
                    &model,
                    &MtmdContextParams {
                        use_gpu: device.backend != Backend::Cpu && backend.supports_gpu_offload(),
                        print_timings: false,
                        n_threads: available_threads(),
                        image_min_tokens: options.image_min_tokens,
                        image_max_tokens: options.image_max_tokens,
                        ..MtmdContextParams::default()
                    },
                )
                .context("failed to initialize MTMD projector")?;
                ensure!(
                    mtmd.support_vision(),
                    "MTMD projector does not support images"
                );
                Some(Mutex::new(mtmd))
            }
            None => None,
        };

        tracing::info!(
            path = %model_path.display(),
            has_encoder = model.has_encoder(),
            has_decoder = model.has_decoder(),
            vision = mtmd.is_some(),
            "loaded llama.cpp model"
        );
        Ok(Self {
            backend,
            model,
            mtmd,
            eos_token,
            chat_template,
        })
    }

    pub(super) fn bos_token(&self) -> Result<String> {
        token_text(&self.model, self.model.token_bos()).context("failed to decode model BOS token")
    }

    pub(super) fn eos_token(&self) -> Result<String> {
        token_text(&self.model, self.eos_token).context("failed to decode model EOS token")
    }

    pub(super) fn render_chat_prompt(&self, messages: &[ChatMessage]) -> Result<String> {
        self.chat_template.render(messages)
    }

    /// Generates unconstrained text for several inputs in one context.
    ///
    /// Each input is prefilled into its own sequence; the sequences then decode
    /// together, one token each per step, so a step costs about as much as a
    /// single-sequence step on a small model.
    pub(super) fn inference_batch(
        &self,
        inputs: &[Input<'_>],
        options: &GenerationOptions,
    ) -> Result<Vec<Generation>> {
        validate_generation_options(options)?;
        ensure!(
            !self.model.has_encoder(),
            "batched generation does not support encoder-decoder models"
        );
        if inputs.is_empty() || options.max_tokens == 0 {
            return Ok(inputs
                .iter()
                .map(|_| Generation::empty(FinishReason::Length))
                .collect());
        }

        let mtmd = if inputs.iter().all(|input| input.images().is_empty()) {
            None
        } else {
            Some(
                self.mtmd
                    .as_ref()
                    .context("image input requires an MTMD projector")?
                    .lock()
                    .map_err(|_| anyhow::anyhow!("MTMD projector lock is poisoned"))?,
            )
        };
        let prepared = inputs
            .iter()
            .map(|input| self.prepare_prompt(input, options.add_special, mtmd.as_deref()))
            .collect::<Result<Vec<_>>>()?;
        let context_config = batch_context_config(&prepared, options)?;
        let mut context = self
            .model
            .new_context(self.backend, context_config.params)
            .context("failed to create llama.cpp context")?;

        struct Sequence {
            sampler: LlamaSampler,
            decoder: encoding_rs::Decoder,
            next_token: LlamaToken,
            position: i32,
            generation: Generation,
            done: bool,
        }
        let mut sequences = Vec::with_capacity(prepared.len());
        for (index, prepared) in prepared.iter().enumerate() {
            let mut sampler = self.build_sampler(options, &prepared.history_tokens);
            let prompt_start = Instant::now();
            let (next_token, position) = self.prefill(
                prepared,
                mtmd.as_deref(),
                &mut context,
                &mut sampler,
                context_config.n_batch,
                i32::try_from(index).context("sequence index exceeds i32")?,
            )?;
            let mut generation = Generation::empty(FinishReason::Length);
            generation.prompt_tokens = prepared.prompt_tokens;
            generation.prompt_duration = prompt_start.elapsed();
            sequences.push(Sequence {
                sampler,
                decoder: encoding_rs::UTF_8.new_decoder(),
                next_token,
                position,
                generation,
                done: false,
            });
        }
        drop(mtmd);

        let generation_start = Instant::now();
        let mut batch = LlamaBatch::new(sequences.len(), 1);
        let mut decoding = Vec::with_capacity(sequences.len());
        loop {
            batch.clear();
            decoding.clear();
            for (index, sequence) in sequences.iter_mut().enumerate() {
                if sequence.done {
                    continue;
                }
                let generation = &mut sequence.generation;
                if self.should_stop(sequence.next_token) {
                    generation.finish_reason = FinishReason::StopToken;
                    sequence.done = true;
                    continue;
                }
                let piece = self
                    .model
                    .token_to_piece(sequence.next_token, &mut sequence.decoder, true, None)
                    .context("failed to decode generated token")?;
                generation.text.push_str(&piece);
                generation.generated_tokens += 1;
                if generation.generated_tokens >= options.max_tokens {
                    sequence.done = true;
                    continue;
                }
                batch
                    .add(
                        sequence.next_token,
                        sequence.position,
                        &[i32::try_from(index).context("sequence index exceeds i32")?],
                        true,
                    )
                    .context("failed to build generated-token batch")?;
                sequence.position = sequence
                    .position
                    .checked_add(1)
                    .context("generated position exceeds i32")?;
                decoding.push(index);
            }
            if decoding.is_empty() {
                break;
            }
            context
                .decode(&mut batch)
                .context("failed to decode generated tokens")?;
            for (output, &index) in decoding.iter().enumerate() {
                let sequence = &mut sequences[index];
                sequence.next_token = sequence.sampler.sample(
                    &context,
                    i32::try_from(output).context("batch output exceeds i32")?,
                );
            }
        }

        let generation_duration = generation_start.elapsed();
        Ok(sequences
            .into_iter()
            .map(|sequence| Generation {
                generation_duration,
                ..sequence.generation
            })
            .collect())
    }

    fn prepare_prompt(
        &self,
        input: &Input<'_>,
        add_special: bool,
        mtmd: Option<&MtmdContext>,
    ) -> Result<PreparedPrompt> {
        let Some(mtmd) = mtmd else {
            let tokens = self
                .model
                .str_to_token(
                    input.prompt(),
                    if add_special {
                        AddBos::Always
                    } else {
                        AddBos::Never
                    },
                )
                .context("failed to tokenize prompt")?;
            ensure!(!tokens.is_empty(), "prompt produced no tokens");
            let prompt_tokens = tokens.len();
            return Ok(PreparedPrompt {
                history_tokens: tokens.clone(),
                kind: PreparedPromptKind::Text(tokens),
                prompt_positions: prompt_tokens,
                prompt_tokens,
                batch_tokens: prompt_tokens,
                non_causal: false,
            });
        };

        let mut bitmaps = Vec::with_capacity(input.images().len());
        for image in input.images() {
            ensure!(
                image.width() > 0 && image.height() > 0,
                "image dimensions must be non-zero"
            );
            let rgb = image.to_rgb8();
            let (width, height) = rgb.dimensions();
            bitmaps.push(
                MtmdBitmap::from_image_data(width, height, &rgb.into_raw())
                    .context("failed to create MTMD image bitmap")?,
            );
        }
        let bitmap_refs = bitmaps.iter().collect::<Vec<_>>();
        let chunks = mtmd
            .tokenize(
                MtmdInputText {
                    text: input.prompt().to_owned(),
                    add_special,
                    parse_special: true,
                },
                &bitmap_refs,
            )
            .context("failed to tokenize multimodal input")?;
        ensure!(
            !chunks.is_empty(),
            "multimodal tokenization produced no chunks"
        );

        let prompt_positions = usize::try_from(chunks.total_positions())
            .context("multimodal prompt positions overflow usize")?;
        let prompt_tokens = chunks.total_tokens();
        let batch_tokens = max_chunk_tokens(&chunks).max(1);
        let history_tokens = text_chunk_tokens(&chunks);
        Ok(PreparedPrompt {
            kind: PreparedPromptKind::Multimodal {
                chunks,
                _bitmaps: bitmaps,
            },
            history_tokens,
            prompt_positions,
            prompt_tokens,
            batch_tokens,
            non_causal: mtmd.decode_use_non_causal(),
        })
    }

    fn prefill(
        &self,
        prepared: &PreparedPrompt,
        mtmd: Option<&MtmdContext>,
        context: &mut LlamaContext<'_>,
        sampler: &mut LlamaSampler,
        n_batch: u32,
        sequence: i32,
    ) -> Result<(LlamaToken, i32)> {
        match &prepared.kind {
            PreparedPromptKind::Text(tokens) => {
                let mut batch = LlamaBatch::new(tokens.len(), 1);
                batch
                    .add_sequence(tokens, sequence, false)
                    .context("failed to build prompt batch")?;
                if self.model.has_encoder() {
                    context
                        .encode(&mut batch)
                        .context("failed to encode prompt batch")?;
                    self.start_decoder(context, sampler)
                } else {
                    context
                        .decode(&mut batch)
                        .context("failed to decode prompt batch")?;
                    let position = i32::try_from(tokens.len()).context("prompt is too long")?;
                    Ok((sampler.sample(context, -1), position))
                }
            }
            PreparedPromptKind::Multimodal { chunks, .. } => {
                let mtmd = mtmd.context("multimodal prompt requires an MTMD projector")?;
                let past = chunks
                    .eval_chunks(
                        mtmd,
                        context,
                        0,
                        sequence,
                        i32::try_from(n_batch).context("batch size exceeds i32")?,
                        true,
                    )
                    .context("failed to evaluate multimodal prompt")?;
                if self.model.has_encoder() {
                    self.start_decoder(context, sampler)
                } else {
                    Ok((sampler.sample(context, -1), past))
                }
            }
        }
    }

    fn start_decoder(
        &self,
        context: &mut LlamaContext<'_>,
        sampler: &mut LlamaSampler,
    ) -> Result<(LlamaToken, i32)> {
        let decoder_start = self.model.decode_start_token();
        let decoder_start = if decoder_start.0 >= 0 {
            decoder_start
        } else {
            self.model.token_bos()
        };
        ensure!(
            decoder_start.0 >= 0,
            "encoder-decoder model has no decoder start or BOS token"
        );
        let mut batch = LlamaBatch::new(1, 1);
        batch
            .add(decoder_start, 0, &[0], true)
            .context("failed to build decoder start batch")?;
        context
            .decode(&mut batch)
            .context("failed to decode model start token")?;
        Ok((sampler.sample(context, -1), 1))
    }

    fn build_sampler(
        &self,
        options: &GenerationOptions,
        history_tokens: &[LlamaToken],
    ) -> LlamaSampler {
        let mut samplers = Vec::new();

        let has_repeat =
            (options.repeat_penalty - 1.0).abs() >= f32::EPSILON && options.repeat_last_n != 0;
        let has_frequency = options.frequency_penalty.abs() >= f32::EPSILON;
        let has_presence = options.presence_penalty.abs() >= f32::EPSILON;
        if has_repeat || has_frequency || has_presence {
            let mut penalties = LlamaSampler::penalties(
                options.repeat_last_n,
                if has_repeat {
                    options.repeat_penalty
                } else {
                    1.0
                },
                options.frequency_penalty,
                options.presence_penalty,
            );
            penalties.accept_many(history_tokens);
            samplers.push(penalties);
        }

        if options.temperature <= 0.0 {
            samplers.push(LlamaSampler::greedy());
            return LlamaSampler::chain_simple(samplers);
        }
        if let Some(top_k) = options.top_k.filter(|value| *value > 0) {
            samplers.push(LlamaSampler::top_k(
                i32::try_from(top_k).unwrap_or(i32::MAX),
            ));
        }
        if let Some(top_p) = options.top_p {
            samplers.push(LlamaSampler::top_p(top_p, 1));
        }
        if let Some(min_p) = options.min_p.filter(|value| *value > 0.0) {
            samplers.push(LlamaSampler::min_p(min_p, 1));
        }
        samplers.push(LlamaSampler::temp(options.temperature));
        samplers.push(LlamaSampler::dist(options.seed));
        LlamaSampler::chain_simple(samplers)
    }

    fn should_stop(&self, token: LlamaToken) -> bool {
        token == self.eos_token || self.model.is_eog_token(token)
    }
}

struct PreparedPrompt {
    kind: PreparedPromptKind,
    history_tokens: Vec<LlamaToken>,
    prompt_positions: usize,
    prompt_tokens: usize,
    batch_tokens: usize,
    non_causal: bool,
}

enum PreparedPromptKind {
    Text(Vec<LlamaToken>),
    Multimodal {
        chunks: MtmdInputChunks,
        _bitmaps: Vec<MtmdBitmap>,
    },
}

struct ContextConfig {
    params: LlamaContextParams,
    n_batch: u32,
}

fn model_params(
    backend: &LlamaBackend,
    device: &crate::Device,
    options: &LoadOptions,
) -> Result<LlamaModelParams> {
    let gpu_layers = if device.backend == Backend::Cpu || !backend.supports_gpu_offload() {
        0
    } else {
        options.gpu_layers
    };
    let mut params = LlamaModelParams::default()
        .with_n_gpu_layers(gpu_layers)
        .with_load_mode(options.load_mode);
    if gpu_layers > 0 {
        params = params
            .with_devices(&[device.index])
            .context("failed to select llama.cpp accelerator")?;
    }
    Ok(params)
}

/// Context for [`Model::inference_batch`]: one KV stream per sequence, each
/// sized like a single-sequence context, and a batch that holds a token from
/// every sequence.
fn batch_context_config(
    prepared: &[PreparedPrompt],
    options: &GenerationOptions,
) -> Result<ContextConfig> {
    let sequences = u32::try_from(prepared.len()).context("too many sequences")?;
    let mut n_ctx = 1;
    let mut n_batch = sequences;
    let mut non_causal = false;
    for prepared in prepared {
        let values = context_values(prepared, options)?;
        n_ctx = n_ctx.max(values.n_ctx.get());
        n_batch = n_batch.max(values.n_batch);
        non_causal |= prepared.non_causal;
    }
    let n_ubatch = if non_causal {
        n_batch
    } else {
        options
            .n_ubatch
            .unwrap_or_else(|| n_batch.min(DEFAULT_MAX_UBATCH))
            .clamp(1, n_batch)
    };
    // llama.cpp pads the context to a multiple of 256 cells and then splits it
    // evenly between sequences, so pad each sequence's share first.
    let n_ctx = n_ctx
        .checked_next_multiple_of(256)
        .and_then(|n_ctx| n_ctx.checked_mul(sequences))
        .context("context size exceeds u32")?;
    let mut params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_batch)
        .with_n_ubatch(n_ubatch)
        .with_n_seq_max(sequences)
        .with_n_outputs_max(sequences)
        .with_n_outputs_max_per_seq(1)
        .with_kv_unified(false);
    if non_causal {
        params = params.with_attention_type(LlamaAttentionType::NonCausal);
    }
    if let Some(n_threads) = options.n_threads {
        params = params.with_n_threads(n_threads);
    }
    if let Some(n_threads_batch) = options.n_threads_batch {
        params = params.with_n_threads_batch(n_threads_batch);
    }
    Ok(ContextConfig { params, n_batch })
}

struct ContextValues {
    n_ctx: NonZeroU32,
    n_batch: u32,
}

fn context_values(prepared: &PreparedPrompt, options: &GenerationOptions) -> Result<ContextValues> {
    let required = prepared
        .prompt_positions
        .saturating_add(options.max_tokens)
        .saturating_add(1)
        .max(
            prepared
                .prompt_tokens
                .saturating_add(options.max_tokens)
                .saturating_add(1),
        )
        .max(prepared.batch_tokens.saturating_add(1));
    let n_ctx = match options.n_ctx {
        Some(n_ctx) if usize::try_from(n_ctx.get()).unwrap_or(usize::MAX) >= required => n_ctx,
        Some(_) => bail!(
            "n_ctx is too small: need at least {required} positions for prompt and generation"
        ),
        None => NonZeroU32::new(u32::try_from(required).context("context size exceeds u32")?)
            .expect("required context size is non-zero"),
    };

    let required_batch =
        u32::try_from(prepared.batch_tokens.max(1)).context("prompt batch size exceeds u32")?;
    let n_batch = options.n_batch.unwrap_or(required_batch).max(1);
    ensure!(
        n_batch >= required_batch,
        "n_batch is too small: need at least {required_batch} tokens"
    );
    Ok(ContextValues { n_ctx, n_batch })
}

fn validate_generation_options(options: &GenerationOptions) -> Result<()> {
    ensure!(
        options.temperature.is_finite() && options.temperature >= 0.0,
        "temperature must be finite and non-negative"
    );
    if let Some(top_p) = options.top_p {
        ensure!(
            top_p.is_finite() && (0.0..=1.0).contains(&top_p),
            "top_p must be between 0 and 1"
        );
    }
    if let Some(min_p) = options.min_p {
        ensure!(
            min_p.is_finite() && (0.0..=1.0).contains(&min_p),
            "min_p must be between 0 and 1"
        );
    }
    ensure!(
        options.repeat_penalty.is_finite() && options.repeat_penalty > 0.0,
        "repeat_penalty must be finite and positive"
    );
    ensure!(
        options.frequency_penalty.is_finite(),
        "frequency_penalty must be finite"
    );
    ensure!(
        options.presence_penalty.is_finite(),
        "presence_penalty must be finite"
    );
    ensure!(
        options.repeat_last_n >= 0,
        "repeat_last_n must be non-negative"
    );
    if let Some(n_threads) = options.n_threads {
        ensure!(n_threads > 0, "n_threads must be positive");
    }
    if let Some(n_threads_batch) = options.n_threads_batch {
        ensure!(n_threads_batch > 0, "n_threads_batch must be positive");
    }
    Ok(())
}

fn max_chunk_tokens(chunks: &MtmdInputChunks) -> usize {
    (0..chunks.len())
        .filter_map(|index| chunks.get(index))
        .map(|chunk| chunk.n_tokens())
        .max()
        .unwrap_or(0)
}

fn text_chunk_tokens(chunks: &MtmdInputChunks) -> Vec<LlamaToken> {
    let mut tokens = Vec::new();
    for index in 0..chunks.len() {
        if let Some(chunk) = chunks.get(index)
            && let Some(text_tokens) = chunk.text_tokens()
        {
            tokens.extend_from_slice(text_tokens);
        }
    }
    tokens
}

fn token_text(model: &LlamaModel, token: LlamaToken) -> Result<String> {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    model
        .token_to_piece(token, &mut decoder, true, None)
        .map_err(Into::into)
}

fn ensure_file(path: &Path, description: &str) -> Result<()> {
    ensure!(
        path.exists(),
        "{description} does not exist: {}",
        path.display()
    );
    ensure!(
        path.is_file(),
        "{description} is not a file: {}",
        path.display()
    );
    Ok(())
}

fn validate_image_token_range(min: i32, max: i32) -> Result<()> {
    ensure!(min >= -1, "image_min_tokens must be -1 or non-negative");
    ensure!(max >= -1, "image_max_tokens must be -1 or non-negative");
    if min >= 0 && max >= 0 {
        ensure!(
            min <= max,
            "image_min_tokens must not exceed image_max_tokens"
        );
    }
    Ok(())
}

fn available_threads() -> i32 {
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .try_into()
        .unwrap_or(i32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_preserves_messages() {
        let mut environment = Environment::new();
        minijinja_contrib::add_to_environment(&mut environment);
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment
            .add_template(
                CHAT_TEMPLATE_NAME,
                "{{ bos_token }}{% for message in messages %}{{ message.role.upper() }}:{{ message.content.strip() }};{% endfor %}{{ eos_token }}|{{ add_generation_prompt }}|{{ enable_thinking }}|{{ tools|length }}|{{ documents|length }}",
            )
            .unwrap();
        let template = ChatTemplate {
            environment,
            bos_token: "<s>".to_owned(),
            eos_token: "</s>".to_owned(),
        };

        assert_eq!(
            template.render(&[ChatMessage::user(" OCR: ")]).unwrap(),
            "<s>USER:OCR:;</s>|True|False|0|0"
        );
    }

    fn prepared(tokens: usize, non_causal: bool) -> PreparedPrompt {
        PreparedPrompt {
            kind: PreparedPromptKind::Text(vec![LlamaToken(0); tokens]),
            history_tokens: Vec::new(),
            prompt_positions: tokens,
            prompt_tokens: tokens,
            batch_tokens: tokens,
            non_causal,
        }
    }

    #[test]
    fn context_includes_prompt_and_generation() {
        let options = GenerationOptions {
            max_tokens: 10,
            ..Default::default()
        };
        let config = context_values(&prepared(5, false), &options).unwrap();
        assert_eq!(config.n_ctx, NonZeroU32::new(16).unwrap());
        assert_eq!(config.n_batch, 5);
    }

    #[test]
    fn context_rejects_small_explicit_limits() {
        let options = GenerationOptions {
            max_tokens: 10,
            n_ctx: NonZeroU32::new(15),
            n_batch: Some(4),
            ..Default::default()
        };
        assert!(context_values(&prepared(5, false), &options).is_err());
    }
}
