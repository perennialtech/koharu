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
    GgufInfo,
    context::{
        LlamaContext, OwnedLlamaContext, OwnedLlamaContextPair,
        params::{KvCacheType, LlamaContextParams, LlamaContextType},
    },
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText},
    sampling::LlamaSampler,
    token::{LlamaToken, data::LlamaTokenData, data_array::LlamaTokenDataArray},
};
use minijinja::{Environment, Error as TemplateError, ErrorKind as TemplateErrorKind, context};

use super::{
    ChatMessage, FinishReason, Generation, GenerationOptions, Input, LoadOptions, MoeOffload,
    MtpOptions,
};
use crate::Backend;

const DEFAULT_MAX_BATCH: u32 = 2048;
const DEFAULT_MAX_UBATCH: u32 = 1024;
const DEFAULT_MIN_PERSISTENT_N_CTX: u32 = 2048;
const N_CTX_GROWTH_GRANULARITY: u32 = 1024;
// Matches llama_flash_attn_type in llama.h.
const LLAMA_FLASH_ATTN_TYPE_DISABLED: i32 = 0;
const LLAMA_FLASH_ATTN_TYPE_ENABLED: i32 = 1;
const CHAT_TEMPLATE_NAME: &str = "chat";

pub(super) struct Model {
    backend: &'static LlamaBackend,
    model: Arc<LlamaModel>,
    mtmd: Option<Mutex<MtmdContext>>,
    mtp: Option<MtpModel>,
    cached_context: Mutex<Option<CachedContext>>,
    recommended_moe_cache_bytes: usize,
    min_moe_cache_bytes_cpu_share: u64,
    eos_token: LlamaToken,
    chat_template: ChatTemplate,
}

struct MtpModel {
    model: Arc<LlamaModel>,
    n_draft: usize,
    p_min: f32,
}

enum InferenceContext {
    Single(OwnedLlamaContext),
    Paired {
        pair: OwnedLlamaContextPair,
        is_mem_shared: bool,
    },
}

impl InferenceContext {
    fn target(&self) -> &LlamaContext<'_> {
        match self {
            Self::Single(context) => context.as_context(),
            Self::Paired { pair, .. } => pair.target().as_context(),
        }
    }

    fn target_mut(&mut self) -> &mut LlamaContext<'_> {
        match self {
            Self::Single(context) => context.as_context_mut(),
            Self::Paired { pair, .. } => pair.target_mut(),
        }
    }
}

#[derive(Debug, Clone)]
struct ContextSpec {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_outputs_max_per_seq: u32,
    flash_attn: bool,
    swa_full: bool,
    kv_cache_type_k: Option<KvCacheType>,
    kv_cache_type_v: Option<KvCacheType>,
    kv_unified: bool,
    offload_kqv: bool,
    op_offload: bool,
    moe_cache_size: usize,
    moe_cache_cpu_share: f32,
    n_threads: i32,
    n_threads_batch: i32,
    non_causal: bool,
    mtp_enabled: bool,
}

impl ContextSpec {
    fn can_reuse(&self, req: &Self) -> bool {
        let per_seq_ok = self.kv_unified
            || (self.n_ctx / self.n_seq_max.max(1) >= req.n_ctx / req.n_seq_max.max(1));
        self.flash_attn == req.flash_attn
            && self.swa_full == req.swa_full
            && self.kv_cache_type_k == req.kv_cache_type_k
            && self.kv_cache_type_v == req.kv_cache_type_v
            && self.kv_unified == req.kv_unified
            && self.offload_kqv == req.offload_kqv
            && self.op_offload == req.op_offload
            && self.moe_cache_size == req.moe_cache_size
            && self.moe_cache_cpu_share.to_bits() == req.moe_cache_cpu_share.to_bits()
            && self.n_threads == req.n_threads
            && self.n_threads_batch == req.n_threads_batch
            && self.non_causal == req.non_causal
            && self.mtp_enabled == req.mtp_enabled
            && self.n_ctx >= req.n_ctx
            && per_seq_ok
            && self.n_seq_max >= req.n_seq_max
            && self.n_batch >= req.n_batch
            && self.n_ubatch == req.n_ubatch
            && self.n_outputs_max_per_seq >= req.n_outputs_max_per_seq
    }
}

struct CachedContext {
    context: InferenceContext,
    spec: ContextSpec,
    seq_tokens: Vec<Vec<LlamaToken>>,
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
        let target_info = (options.moe_offload != MoeOffload::None)
            .then(|| koharu_llama::inspect_gguf(&model_path))
            .flatten();
        let (effective_options, recommended_moe_cache_bytes) =
            resolve_auto_moe(&options, target_info.as_ref(), || {
                if device.backend == Backend::Cpu
                    || options.gpu_layers == 0
                    || !backend.supports_gpu_offload()
                {
                    return 0;
                }
                koharu_llama::gpu_backend_device_index(device.index)
                    .and_then(koharu_llama::backend_device_memory)
                    .map_or(0, |(free, _)| free as u64)
            });
        let params = model_params(backend, device, &effective_options)?;
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

        let mtp = match options.mtp {
            Some(mtp_opts) => {
                ensure_file(&mtp_opts.model_path, "MTP draft model")?;
                ensure!(mtp_opts.n_draft > 0, "MTP n_draft must be positive");
                ensure!(
                    mtp_opts.p_min.is_finite() && (0.0..=1.0).contains(&mtp_opts.p_min),
                    "MTP p_min must be between 0 and 1"
                );
                let draft_params =
                    mtp_model_params(backend, device, effective_options.load_mode, &mtp_opts)?;
                let draft_model = Arc::new(
                    LlamaModel::load_from_file(backend, &mtp_opts.model_path, &draft_params)
                        .with_context(|| {
                            format!(
                                "failed to load MTP draft model {}",
                                mtp_opts.model_path.display()
                            )
                        })?,
                );
                ensure!(
                    model.supports_mtp_draft(&draft_model),
                    "incompatible MTP target and draft model geometry"
                );
                Some(MtpModel {
                    model: draft_model,
                    n_draft: mtp_opts.n_draft,
                    p_min: mtp_opts.p_min,
                })
            }
            None => None,
        };

        tracing::info!(
            path = %model_path.display(),
            has_encoder = model.has_encoder(),
            has_decoder = model.has_decoder(),
            vision = mtmd.is_some(),
            mtp = mtp.is_some(),
            "loaded llama.cpp model"
        );
        Ok(Self {
            backend,
            model,
            mtmd,
            mtp,
            cached_context: Mutex::new(None),
            recommended_moe_cache_bytes,
            min_moe_cache_bytes_cpu_share: target_info.as_ref().map_or(0, |info| {
                moe_cache_floors(info, effective_options.moe_offload).1
            }),
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

    pub(super) fn n_layer(&self) -> u32 {
        self.model.n_layer()
    }

    pub(super) fn model_size_bytes(&self) -> u64 {
        self.model.size()
    }

    pub(super) fn n_swa(&self) -> i32 {
        self.model.n_swa()
    }

    pub(super) fn mtp_enabled(&self) -> bool {
        self.mtp.is_some()
    }

    pub(super) fn recommended_moe_cache_bytes(&self) -> usize {
        self.recommended_moe_cache_bytes
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
        let mtp_active = self.mtp.is_some()
            && prepared.len() == 1
            && matches!(prepared[0].kind, PreparedPromptKind::Text(_));
        let mtp_n_draft = mtp_active
            .then(|| self.mtp.as_ref().map(|mtp| mtp.n_draft))
            .flatten();
        let context_config = batch_context_config(
            &prepared,
            options,
            mtp_n_draft,
            self.recommended_moe_cache_bytes,
            self.min_moe_cache_bytes_cpu_share,
            self.model.n_ctx_train(),
        )?;
        let mut cached_guard = self
            .cached_context
            .lock()
            .map_err(|_| anyhow::anyhow!("cached llama.cpp context lock is poisoned"))?;
        let reuse = cached_guard
            .as_ref()
            .is_some_and(|cached| cached.spec.can_reuse(&context_config.spec));
        if !reuse {
            // Drop any existing context first so peak VRAM never doubles during reallocation.
            *cached_guard = None;
            let n_seq_max = usize::try_from(context_config.spec.n_seq_max)
                .context("n_seq_max exceeds usize")?;
            let context = self
                .model
                .new_context_owned(self.backend, context_config.params)
                .context("failed to create llama.cpp context")?;
            let context = if mtp_active && let Some(mtp) = &self.mtp {
                let flash_attn_policy = if context_config.spec.flash_attn {
                    LLAMA_FLASH_ATTN_TYPE_ENABLED
                } else {
                    LLAMA_FLASH_ATTN_TYPE_DISABLED
                };
                let draft_params = LlamaContextParams::default()
                    .with_context_type(LlamaContextType::Mtp)
                    .with_n_ctx(NonZeroU32::new(context.n_ctx()))
                    .with_n_batch(1)
                    .with_n_ubatch(1)
                    .with_n_seq_max(1)
                    .with_n_outputs_max(1)
                    .with_n_outputs_max_per_seq(1)
                    .with_n_rs_seq(0)
                    .with_flash_attention_policy(flash_attn_policy)
                    .with_swa_full(false)
                    .with_no_perf(true)
                    .with_n_threads(context_config.spec.n_threads)
                    .with_n_threads_batch(context_config.spec.n_threads_batch);
                let mut pair = context
                    .into_paired(self.backend, &mtp.model, draft_params)
                    .map_err(|(_, err)| err)
                    .context("failed to create MTP draft llama.cpp context")?;
                let is_mem_shared = pair.is_mem_shared();
                pair.target_mut().set_embeddings_nextn(true, is_mem_shared);
                pair.draft_mut().set_embeddings_nextn(true, true);
                tracing::debug!(
                    is_mem_shared,
                    n_draft = mtp.n_draft,
                    p_min = mtp.p_min,
                    n_ctx = pair.target().n_ctx(),
                    "initialized MTP draft context"
                );
                InferenceContext::Paired {
                    pair,
                    is_mem_shared,
                }
            } else {
                InferenceContext::Single(context)
            };
            *cached_guard = Some(CachedContext {
                context,
                spec: context_config.spec.clone(),
                seq_tokens: vec![Vec::new(); n_seq_max],
            });
        }
        let cached = cached_guard
            .as_mut()
            .expect("cached context is initialized");

        for seq_id in prepared.len()..cached.seq_tokens.len() {
            if let Ok(seq_u32) = u32::try_from(seq_id) {
                let _ = cached
                    .context
                    .target_mut()
                    .clear_kv_cache_seq(Some(seq_u32), None, None);
            }
            cached.seq_tokens[seq_id].clear();
        }

        struct Sequence<'a> {
            sampler: SequenceSampler<'a>,
            decoder: encoding_rs::Decoder,
            next_token: LlamaToken,
            position: i32,
            tokens: Vec<LlamaToken>,
            generation: Generation,
            done: bool,
        }
        let mut sequences = Vec::with_capacity(prepared.len());
        for (index, prepared) in prepared.iter().enumerate() {
            let mut sampler = self.build_sampler(options, &prepared.history_tokens)?;
            let prompt_start = Instant::now();
            let (next_token, position) = self.prefill(
                prepared,
                mtmd.as_deref(),
                cached.context.target_mut(),
                &mut cached.seq_tokens[index],
                &mut sampler,
                &context_config.spec,
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
                tokens: Vec::new(),
                generation,
                done: false,
            });
        }
        drop(mtmd);

        let generation_start = Instant::now();
        if let InferenceContext::Paired {
            pair,
            is_mem_shared: true,
        } = &mut cached.context
            && let Some(mtp) = &self.mtp
            && let Some(initial_h) = pair.target().get_embeddings_nextn_ith(-1)
        {
            let mut pending_h = initial_h.to_vec();
            let mut scratch_h = Vec::with_capacity(pending_h.len());
            let mut verify_batch = LlamaBatch::new(1 + mtp.n_draft, 1);
            let mut drafts = Vec::with_capacity(mtp.n_draft);
            let mut total_steps = 0usize;
            let mut total_drafted = 0usize;
            let mut total_accepted = 0usize;
            let sequence = &mut sequences[0];

            while !sequence.done {
                let generation = &mut sequence.generation;
                if self.should_stop(sequence.next_token) {
                    generation.finish_reason = FinishReason::StopToken;
                    generation.generation_duration = generation_start.elapsed();
                    sequence.done = true;
                    break;
                }
                let piece = self
                    .model
                    .token_to_piece(
                        sequence.next_token,
                        &mut sequence.decoder,
                        options.decode_special_tokens,
                        None,
                    )
                    .context("failed to decode generated token")?;
                sequence.tokens.push(sequence.next_token);
                generation.text.push_str(&piece);
                generation.generated_tokens += 1;
                if generation.generated_tokens >= options.max_tokens
                    || (options.abort_on_repetition_loop && has_repetition_loop(&sequence.tokens))
                {
                    generation.generation_duration = generation_start.elapsed();
                    sequence.done = true;
                    break;
                }

                // Draft up to `max_draft` tokens using the shared-memory MTP head.
                let max_draft = mtp.n_draft.min(
                    options
                        .max_tokens
                        .saturating_sub(generation.generated_tokens)
                        .saturating_sub(1),
                );
                let pos0 = sequence.position;
                let root_token = sequence.next_token;
                drafts.clear();
                let mut cur_tok = root_token;
                let mut use_scratch = false;
                for d in 0..max_draft {
                    let cur_h: &[f32] = if use_scratch { &scratch_h } else { &pending_h };
                    let mut batch_ext = pair
                        .draft_mut()
                        .batch_ext()
                        .context("failed to allocate MTP extended batch")?;
                    let idx = batch_ext
                        .add_token(0, cur_tok)
                        .context("failed to add token to MTP batch")?;
                    ensure!(
                        batch_ext.set_pos(idx, pos0),
                        "failed to set position in MTP batch"
                    );
                    ensure!(
                        batch_ext.set_embd_state(idx, cur_h),
                        "failed to set hidden state in MTP batch"
                    );
                    ensure!(
                        batch_ext.set_output_logits(idx, true),
                        "failed to enable output logits in MTP batch"
                    );
                    batch_ext
                        .decode()
                        .context("failed to decode MTP draft step")?;
                    let draft_logits = pair.draft().get_logits_ith(0);
                    let (draft_tok, top1_prob) = top1_token_and_prob(draft_logits);
                    if top1_prob < mtp.p_min {
                        break;
                    }
                    drafts.push(draft_tok);
                    if self.should_stop(draft_tok) {
                        break;
                    }
                    if d + 1 < max_draft {
                        let Some(next_h) = pair.draft().get_embeddings_nextn_ith(0) else {
                            break;
                        };
                        scratch_h.clear();
                        scratch_h.extend_from_slice(next_h);
                        cur_tok = draft_tok;
                        use_scratch = true;
                    }
                }

                // Verify `[root_token, drafts[0], ..., drafts[k-1]]` in one target forward pass.
                let k = drafts.len();
                total_steps += 1;
                total_drafted += k;
                verify_batch.clear();
                verify_batch
                    .add(root_token, pos0, &[0], true)
                    .context("failed to add root token to MTP verify batch")?;
                for (j, &dtok) in drafts.iter().enumerate() {
                    let pos = pos0
                        .checked_add(i32::try_from(j + 1).context("draft offset exceeds i32")?)
                        .context("draft position exceeds i32")?;
                    verify_batch
                        .add(dtok, pos, &[0], true)
                        .context("failed to add draft token to MTP verify batch")?;
                }
                if let Err(err) = pair.target_mut().decode(&mut verify_batch) {
                    for tokens in &mut cached.seq_tokens {
                        tokens.clear();
                    }
                    pair.target_mut().clear_kv_cache();
                    return Err(err).context("failed to decode MTP verify batch");
                }

                let mut n_accepted = 0usize;
                for i in 0..=k {
                    let sampled = sequence.sampler.sample(
                        pair.target().as_context(),
                        i32::try_from(i).context("verify index exceeds i32")?,
                    )?;
                    if drafts.get(i) == Some(&sampled) {
                        let generation = &mut sequence.generation;
                        if self.should_stop(sampled) {
                            generation.finish_reason = FinishReason::StopToken;
                            generation.generation_duration = generation_start.elapsed();
                            sequence.done = true;
                            n_accepted = i;
                            break;
                        }
                        let piece = self
                            .model
                            .token_to_piece(
                                sampled,
                                &mut sequence.decoder,
                                options.decode_special_tokens,
                                None,
                            )
                            .context("failed to decode accepted MTP token")?;
                        sequence.tokens.push(sampled);
                        generation.text.push_str(&piece);
                        generation.generated_tokens += 1;
                        if generation.generated_tokens >= options.max_tokens
                            || (options.abort_on_repetition_loop
                                && has_repetition_loop(&sequence.tokens))
                        {
                            generation.generation_duration = generation_start.elapsed();
                            sequence.done = true;
                            n_accepted = i;
                            break;
                        }
                        n_accepted = i + 1;
                    } else {
                        sequence.next_token = sampled;
                        n_accepted = i;
                        break;
                    }
                }
                total_accepted += n_accepted;

                // Roll back rejected speculative KV cache entries `[pos0 + 1 + n_accepted, ...)`.
                let kept_count = i32::try_from(1 + n_accepted).context("kept count exceeds i32")?;
                let next_pos = pos0
                    .checked_add(kept_count)
                    .context("generated position exceeds i32")?;
                if n_accepted < k {
                    let trim_pos = u32::try_from(next_pos).context("trim position exceeds u32")?;
                    let rollback =
                        pair.target_mut()
                            .clear_kv_cache_seq(Some(0), Some(trim_pos), None);
                    if !matches!(rollback, Ok(true)) {
                        cached.seq_tokens[0].clear();
                        // Only sequence zero is active in a paired context.
                        pair.target_mut().clear_kv_cache();
                        rollback.context("failed to roll back rejected MTP KV slots")?;
                        bail!("MTP KV cache does not support rollback at position {trim_pos}");
                    }
                }
                sequence.position = next_pos;
                cached.seq_tokens[0].push(root_token);
                cached.seq_tokens[0].extend_from_slice(&drafts[..n_accepted]);

                if !sequence.done {
                    let h_idx = i32::try_from(n_accepted).context("accepted index exceeds i32")?;
                    let Some(next_h) = pair.target().get_embeddings_nextn_ith(h_idx) else {
                        break;
                    };
                    pending_h.clear();
                    pending_h.extend_from_slice(next_h);
                }
            }

            let accept_rate = if total_drafted > 0 {
                total_accepted as f64 / total_drafted as f64
            } else {
                0.0
            };
            tracing::debug!(
                steps = total_steps,
                drafted = total_drafted,
                accepted = total_accepted,
                accept_rate,
                "MTP speculative decoding stats"
            );
        }

        let mut batch = LlamaBatch::new(sequences.len(), 1);
        let mut decoding: Vec<(usize, LlamaToken)> = Vec::with_capacity(sequences.len());
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
                    generation.generation_duration = generation_start.elapsed();
                    sequence.done = true;
                    continue;
                }
                let piece = self
                    .model
                    .token_to_piece(
                        sequence.next_token,
                        &mut sequence.decoder,
                        options.decode_special_tokens,
                        None,
                    )
                    .context("failed to decode generated token")?;
                sequence.tokens.push(sequence.next_token);
                generation.text.push_str(&piece);
                generation.generated_tokens += 1;
                if generation.generated_tokens >= options.max_tokens
                    || (options.abort_on_repetition_loop && has_repetition_loop(&sequence.tokens))
                {
                    generation.generation_duration = generation_start.elapsed();
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
                decoding.push((index, sequence.next_token));
            }
            if decoding.is_empty() {
                break;
            }
            if let Err(err) = cached.context.target_mut().decode(&mut batch) {
                for tokens in &mut cached.seq_tokens {
                    tokens.clear();
                }
                cached.context.target_mut().clear_kv_cache();
                return Err(err).context("failed to decode generated tokens");
            }
            for (output, &(index, token)) in decoding.iter().enumerate() {
                if matches!(prepared[index].kind, PreparedPromptKind::Text(_)) {
                    cached.seq_tokens[index].push(token);
                }
                let sequence = &mut sequences[index];
                sequence.next_token = sequence.sampler.sample(
                    cached.context.target(),
                    i32::try_from(output).context("batch output exceeds i32")?,
                )?;
            }
        }

        Ok(sequences
            .into_iter()
            .map(|sequence| sequence.generation)
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

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        prepared: &PreparedPrompt,
        mtmd: Option<&MtmdContext>,
        context: &mut LlamaContext<'_>,
        cached_seq_tokens: &mut Vec<LlamaToken>,
        sampler: &mut SequenceSampler<'_>,
        spec: &ContextSpec,
        sequence: i32,
    ) -> Result<(LlamaToken, i32)> {
        let seq_u32 = u32::try_from(sequence).context("sequence index exceeds u32")?;
        match &prepared.kind {
            PreparedPromptKind::Text(tokens) => {
                if self.model.has_encoder() {
                    let _ = context.clear_kv_cache_seq(Some(seq_u32), None, None);
                    cached_seq_tokens.clear();
                    let mut batch = LlamaBatch::new(tokens.len(), 1);
                    batch
                        .add_sequence(tokens, sequence, false)
                        .context("failed to build prompt batch")?;
                    context
                        .encode(&mut batch)
                        .context("failed to encode prompt batch")?;
                    self.start_decoder(context, sampler)
                } else {
                    let common = common_prefix_len(cached_seq_tokens, tokens);
                    // Always decode at least 1 token so the final prompt position produces
                    // fresh logits and next-token hidden states.
                    let keep = common.min(tokens.len().saturating_sub(1));
                    let kept = if keep > 0
                        && can_reuse_kv_prefix(
                            keep,
                            context.kv_cache_seq_pos_min(sequence),
                            context.kv_cache_seq_pos_max(sequence),
                            self.model.n_swa(),
                            spec.swa_full,
                        ) {
                        let keep_u32 =
                            u32::try_from(keep).context("cached prefix length exceeds u32")?;
                        match context.clear_kv_cache_seq(Some(seq_u32), Some(keep_u32), None) {
                            Ok(true) => {
                                tracing::debug!(
                                    seq_id = sequence,
                                    kept_tokens = keep,
                                    total_tokens = tokens.len(),
                                    "reused KV prefix"
                                );
                                keep
                            }
                            Ok(false) => {
                                // Partial removal unsupported (e.g. SWA range): fall back to full clear.
                                let _ = context.clear_kv_cache_seq(Some(seq_u32), None, None);
                                0
                            }
                            Err(err) => {
                                cached_seq_tokens.clear();
                                return Err(err).context("failed to trim cached KV sequence");
                            }
                        }
                    } else {
                        let _ = context.clear_kv_cache_seq(Some(seq_u32), None, None);
                        0
                    };

                    cached_seq_tokens.clear();
                    let suffix = &tokens[kept..];
                    let chunk_size = usize::try_from(spec.n_batch).unwrap_or(1).max(1);
                    let mut batch = LlamaBatch::new(chunk_size, 1);
                    for (chunk_index, chunk) in suffix.chunks(chunk_size).enumerate() {
                        batch.clear();
                        let offset = kept + chunk_index * chunk_size;
                        for (i, &token) in chunk.iter().enumerate() {
                            let pos = i32::try_from(offset + i).context("prompt is too long")?;
                            let logits = offset + i + 1 == tokens.len();
                            batch
                                .add(token, pos, &[sequence], logits)
                                .context("failed to build prompt batch")?;
                        }
                        if let Err(err) = context.decode(&mut batch) {
                            let _ = context.clear_kv_cache_seq(Some(seq_u32), None, None);
                            return Err(err).context("failed to decode prompt batch");
                        }
                    }
                    cached_seq_tokens.extend_from_slice(tokens);
                    let position = i32::try_from(tokens.len()).context("prompt is too long")?;
                    Ok((sampler.sample(context, -1)?, position))
                }
            }
            PreparedPromptKind::Multimodal { chunks, .. } => {
                let _ = context.clear_kv_cache_seq(Some(seq_u32), None, None);
                cached_seq_tokens.clear();
                let mtmd = mtmd.context("multimodal prompt requires an MTMD projector")?;
                let past = chunks
                    .eval_chunks(
                        mtmd,
                        context,
                        0,
                        sequence,
                        i32::try_from(spec.n_batch).context("batch size exceeds i32")?,
                        true,
                    )
                    .context("failed to evaluate multimodal prompt")?;
                if self.model.has_encoder() {
                    self.start_decoder(context, sampler)
                } else {
                    Ok((sampler.sample(context, -1)?, past))
                }
            }
        }
    }

    fn start_decoder(
        &self,
        context: &mut LlamaContext<'_>,
        sampler: &mut SequenceSampler<'_>,
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
        Ok((sampler.sample(context, -1)?, 1))
    }

    fn build_sampler<'a>(
        &self,
        options: &'a GenerationOptions,
        history_tokens: &[LlamaToken],
    ) -> Result<SequenceSampler<'a>> {
        let has_repeat =
            (options.repeat_penalty - 1.0).abs() >= f32::EPSILON && options.repeat_last_n != 0;
        let has_frequency = options.frequency_penalty.abs() >= f32::EPSILON;
        let has_presence = options.presence_penalty.abs() >= f32::EPSILON;
        let mut penalties = if has_repeat || has_frequency || has_presence {
            let mut p = LlamaSampler::penalties(
                options.repeat_last_n,
                if has_repeat {
                    options.repeat_penalty
                } else {
                    1.0
                },
                options.frequency_penalty,
                options.presence_penalty,
            );
            p.accept_many(history_tokens);
            Some(p)
        } else {
            None
        };

        let grammar = match &options.grammar {
            Some(grammar) => Some(
                LlamaSampler::grammar(&self.model, &grammar.gbnf, &grammar.root)
                    .context("failed to initialize GBNF grammar sampler")?,
            ),
            None => None,
        };

        let mut samplers = Vec::new();
        if grammar.is_none() {
            samplers.extend(penalties.take());
        }
        push_selector_samplers(&mut samplers, options);
        Ok(SequenceSampler {
            penalties,
            grammar,
            selector: LlamaSampler::chain_simple(samplers),
            options,
        })
    }

    fn should_stop(&self, token: LlamaToken) -> bool {
        token == self.eos_token || self.model.is_eog_token(token)
    }
}

const FAST_GRAMMAR_TOP_K: usize = 64;

fn push_selector_samplers(samplers: &mut Vec<LlamaSampler>, options: &GenerationOptions) {
    if options.temperature <= 0.0 {
        samplers.push(LlamaSampler::greedy());
        return;
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
}

/// Sequence sampler that fast-paths GBNF grammar evaluation over top-1 / top-K candidates
/// before falling back to a full-vocabulary grammar scan.
struct SequenceSampler<'a> {
    penalties: Option<LlamaSampler>,
    grammar: Option<LlamaSampler>,
    selector: LlamaSampler,
    options: &'a GenerationOptions,
}

impl SequenceSampler<'_> {
    fn sample(&mut self, ctx: &LlamaContext<'_>, idx: i32) -> Result<LlamaToken> {
        let Some(grammar) = self.grammar.as_mut() else {
            return Ok(self.selector.sample(ctx, idx));
        };
        let logits = ctx
            .try_get_logits_ith(idx)
            .context("logits unavailable for sampled token index")?;
        let penalized = self.penalties.as_ref().map(|penalties| {
            let mut full = full_token_data_array(logits);
            penalties.apply(&mut full);
            full
        });
        let mut candidates = grammar_candidates(logits, penalized, self.options, |array| {
            grammar.apply(array)
        })?;
        self.selector.apply(&mut candidates);
        let token = candidates
            .selected_token()
            .context("selector did not select a grammar token")?;
        if let Some(penalties) = &mut self.penalties {
            penalties.accept(token);
        }
        grammar.accept(token);
        Ok(token)
    }
}

fn grammar_candidates(
    logits: &[f32],
    penalized: Option<LlamaTokenDataArray>,
    options: &GenerationOptions,
    mut apply_grammar: impl FnMut(&mut LlamaTokenDataArray),
) -> Result<LlamaTokenDataArray> {
    let mut legal = |mut array: LlamaTokenDataArray| {
        apply_grammar(&mut array);
        array.data.retain(|item| item.logit().is_finite());
        array
    };
    let greedy = options.temperature <= 0.0;
    if greedy {
        let best = if let Some(full) = &penalized {
            full.data
                .iter()
                .min_by(|a, b| compare_desc_candidate(a, b))
                .copied()
        } else {
            token_data(logits).min_by(compare_desc_candidate)
        };
        if let Some(best) = best {
            let single = legal(LlamaTokenDataArray::new(vec![best], true));
            if !single.data.is_empty() {
                return Ok(single);
            }
        }
    }

    let k = FAST_GRAMMAR_TOP_K.max(options.top_k.unwrap_or(0));
    let shortlist = match &penalized {
        Some(full) => top_k_token_data(full.data.iter().copied(), k),
        None => top_k_token_data(token_data(logits), k),
    };
    let shortlist_len = shortlist.len();
    if let Some(boundary) = shortlist.last().map(LlamaTokenData::logit) {
        let valid = legal(LlamaTokenDataArray::new(shortlist, true));
        if shortlist_is_exact(options, &valid.data, boundary, shortlist_len, logits.len()) {
            return Ok(valid);
        }
    }

    let mut full = legal(penalized.unwrap_or_else(|| full_token_data_array(logits)));
    ensure!(
        !full.data.is_empty(),
        "GBNF grammar rejected all vocabulary tokens"
    );
    full.data.sort_unstable_by(compare_desc_candidate);
    full.sorted = true;
    Ok(full)
}

fn shortlist_is_exact(
    options: &GenerationOptions,
    valid: &[LlamaTokenData],
    boundary_logit: f32,
    shortlist_len: usize,
    vocab_len: usize,
) -> bool {
    let Some(best) = valid.first() else {
        return false;
    };
    if options.temperature <= 0.0 || shortlist_len >= vocab_len {
        return true;
    }
    // top_k runs first. top_p needs the mass of the entire remaining distribution.
    if let Some(k) = options.top_k.filter(|&k| k > 0) {
        return valid.len() >= k;
    }
    if options.top_p.is_some() {
        return false;
    }
    if let Some(mp) = options.min_p.filter(|&mp| mp > 0.0) {
        return boundary_logit < best.logit() + mp.ln() - 1e-4;
    }
    // At this distance the omitted softmax weights round to zero in f32.
    (best.logit() - boundary_logit) / options.temperature >= 104.0
}

fn token_data(logits: &[f32]) -> impl ExactSizeIterator<Item = LlamaTokenData> {
    logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| LlamaTokenData::new(LlamaToken(i as i32), logit, 0.0))
}

fn full_token_data_array(logits: &[f32]) -> LlamaTokenDataArray {
    LlamaTokenDataArray::from_iter(token_data(logits), false)
}

fn compare_desc_candidate(a: &LlamaTokenData, b: &LlamaTokenData) -> std::cmp::Ordering {
    if a.logit() == b.logit() {
        a.id().0.cmp(&b.id().0)
    } else {
        b.logit().total_cmp(&a.logit())
    }
}

/// Retains only the top k entries during the vocabulary scan, breaking ties by token ID.
fn top_k_token_data(
    mut candidates: impl Iterator<Item = LlamaTokenData>,
    k: usize,
) -> Vec<LlamaTokenData> {
    let mut buf: Vec<_> = candidates.by_ref().take(k).collect();
    let worst = |buf: &[LlamaTokenData]| {
        buf.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| compare_desc_candidate(a, b))
            .map(|(i, _)| i)
    };
    if let Some(mut index) = worst(&buf) {
        for candidate in candidates {
            if compare_desc_candidate(&candidate, &buf[index]).is_lt() {
                buf[index] = candidate;
                index = worst(&buf).expect("top-k buffer is nonempty");
            }
        }
    }
    buf.sort_unstable_by(compare_desc_candidate);
    buf
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
    spec: ContextSpec,
}

fn common_prefix_len(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn can_reuse_kv_prefix(
    keep: usize,
    pos_min: i32,
    pos_max: i32,
    n_swa: i32,
    swa_full: bool,
) -> bool {
    if keep == 0 || pos_min < 0 || pos_max < 0 || (pos_max as usize) + 1 < keep {
        return false;
    }
    pos_min == 0
        || (n_swa > 0 && !swa_full && pos_min as usize <= keep.saturating_sub(n_swa as usize))
}

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const AUTO_FULL_GPU_HEADROOM_BYTES: u64 = 1536 * MIB;
const AUTO_FIXED_VRAM_RESERVE_BYTES: u64 = 1856 * MIB;
const AUTO_VRAM_UTILIZATION_PCT: u64 = 92;
const AUTO_MOE_CACHE_VRAM_PCT: u64 = 22;
const AUTO_MOE_CACHE_MAX_BYTES: u64 = 4 * GIB;
const AUTO_MOE_CACHE_ALIGN_BYTES: u64 = 64 * MIB;
const AUTO_MOE_WEIGHT_FRACTION_PCT: u64 = 90;
const AUTO_PINNED_RAM_HEADROOM_BYTES: u64 = 4 * GIB;
const DEFAULT_AUTO_MOE_LAYERS_EST: u32 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AutoMoePlan {
    offload: MoeOffload,
    moe_cache_bytes: usize,
    prefer_pinned_host: bool,
}

fn moe_cache_floors(info: &GgufInfo, offload: MoeOffload) -> (u64, u64) {
    let host_layers = match offload {
        MoeOffload::AllCpu => None,
        MoeOffload::FirstLayers(n) => Some(n),
        MoeOffload::Auto | MoeOffload::None => return (0, 0),
    };
    info.min_moe_cache_bytes_for_host_layers(host_layers)
}

/// Computes an automatic MoE CPU-offload and dynamic VRAM cache plan from model size,
/// free accelerator VRAM, and estimated transformer layer count.
fn plan_auto_moe_budget(
    model_bytes: u64,
    free_vram_bytes: u64,
    n_layers_est: u32,
    upload_floor: impl FnOnce(MoeOffload) -> u64,
) -> AutoMoePlan {
    if free_vram_bytes == 0
        || n_layers_est == 0
        || model_bytes.saturating_add(AUTO_FULL_GPU_HEADROOM_BYTES) <= free_vram_bytes
    {
        return AutoMoePlan {
            offload: MoeOffload::None,
            moe_cache_bytes: 0,
            prefer_pinned_host: false,
        };
    }

    let target_budget = free_vram_bytes.saturating_mul(AUTO_VRAM_UTILIZATION_PCT) / 100;
    let available_for_moe = target_budget.saturating_sub(AUTO_FIXED_VRAM_RESERVE_BYTES);
    let raw_cache = (free_vram_bytes.saturating_mul(AUTO_MOE_CACHE_VRAM_PCT) / 100)
        .min(AUTO_MOE_CACHE_MAX_BYTES)
        .min(available_for_moe / 2);
    let mut moe_cache_bytes_u64 =
        (raw_cache / AUTO_MOE_CACHE_ALIGN_BYTES) * AUTO_MOE_CACHE_ALIGN_BYTES;
    let moe_bytes_est = model_bytes.saturating_mul(AUTO_MOE_WEIGHT_FRACTION_PCT) / 100;
    let per_layer_moe = (moe_bytes_est / u64::from(n_layers_est)).max(1);
    let offload_for_cache = |cache_bytes: u64| {
        let static_moe_vram_budget = available_for_moe - cache_bytes;
        let gpu_layers = (static_moe_vram_budget / per_layer_moe).min(u64::from(n_layers_est));
        match n_layers_est - gpu_layers as u32 {
            0 => MoeOffload::None,
            n if n == n_layers_est => MoeOffload::AllCpu,
            n => MoeOffload::FirstLayers(n),
        }
    };
    let mut offload = offload_for_cache(moe_cache_bytes_u64);
    if offload == MoeOffload::None || moe_cache_bytes_u64 < upload_floor(offload) {
        moe_cache_bytes_u64 = 0;
        offload = offload_for_cache(0);
    }
    AutoMoePlan {
        offload,
        moe_cache_bytes: usize::try_from(moe_cache_bytes_u64).unwrap_or(usize::MAX),
        prefer_pinned_host: offload != MoeOffload::None,
    }
}

fn resolve_auto_moe(
    options: &LoadOptions,
    info: Option<&GgufInfo>,
    free_vram: impl FnOnce() -> u64,
) -> (LoadOptions, usize) {
    let mut resolved = options.clone();
    if options.moe_offload != MoeOffload::Auto {
        return (resolved, 0);
    }
    resolved.moe_offload = MoeOffload::None;
    let Some(info) = info.filter(|info| info.expert_count > 0) else {
        return (resolved, 0);
    };
    let model_bytes = info.total_model_bytes;
    let free_vram_bytes = free_vram();
    let n_layers_est = info.block_count.unwrap_or(DEFAULT_AUTO_MOE_LAYERS_EST);

    let plan = plan_auto_moe_budget(model_bytes, free_vram_bytes, n_layers_est, |offload| {
        moe_cache_floors(info, offload).0
    });
    resolved.moe_offload = plan.offload;

    if plan.prefer_pinned_host
        && resolved.load_mode == koharu_llama::model::params::LlamaLoadMode::Auto
        && available_system_ram_bytes()
            .is_some_and(|ram| ram >= model_bytes.saturating_add(AUTO_PINNED_RAM_HEADROOM_BYTES))
    {
        resolved.load_mode = koharu_llama::model::params::LlamaLoadMode::None;
    }

    tracing::info!(
        model_gib = format_args!("{:.2}", model_bytes as f64 / GIB as f64),
        free_vram_gib = format_args!("{:.2}", free_vram_bytes as f64 / GIB as f64),
        n_layers_est,
        resolved_offload = ?resolved.moe_offload,
        resolved_load_mode = ?resolved.load_mode,
        moe_cache_mib = plan.moe_cache_bytes / (MIB as usize),
        "planned automatic MoE VRAM budget"
    );

    (resolved, plan.moe_cache_bytes)
}

#[cfg(target_os = "linux")]
fn available_system_ram_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return kb.checked_mul(1024);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn available_system_ram_bytes() -> Option<u64> {
    None
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
        .with_load_mode(options.load_mode)
        .with_no_host(options.no_host)
        .with_use_extra_bufts(options.use_extra_bufts);
    if gpu_layers > 0 {
        let backend_device = koharu_llama::gpu_backend_device_index(device.index)
            .with_context(|| format!("llama.cpp GPU device ordinal {} not found", device.index))?;
        params = params
            .with_devices(&[backend_device])
            .context("failed to select llama.cpp accelerator")?;
    }
    match options.moe_offload {
        MoeOffload::Auto | MoeOffload::None => {}
        MoeOffload::AllCpu => {
            params = params.with_cpu_moe();
        }
        MoeOffload::FirstLayers(n_layers) => {
            params = params.with_n_cpu_moe(n_layers);
        }
    }
    Ok(params)
}

/// Context for [`Model::inference_batch`]: one KV stream per sequence, each
/// sized like a single-sequence context, and a batch that holds a token from
/// every sequence.
fn mtp_model_params(
    backend: &LlamaBackend,
    device: &crate::Device,
    load_mode: koharu_llama::model::params::LlamaLoadMode,
    options: &MtpOptions,
) -> Result<LlamaModelParams> {
    let gpu_layers = if device.backend == Backend::Cpu || !backend.supports_gpu_offload() {
        0
    } else {
        options.gpu_layers
    };
    let mut params = LlamaModelParams::default()
        .with_n_gpu_layers(gpu_layers)
        .with_load_mode(load_mode);
    if gpu_layers > 0 {
        let backend_device = koharu_llama::gpu_backend_device_index(device.index)
            .with_context(|| format!("llama.cpp GPU device ordinal {} not found", device.index))?;
        params = params
            .with_devices(&[backend_device])
            .context("failed to select llama.cpp accelerator for MTP draft model")?;
    }
    Ok(params)
}

fn top1_token_and_prob(logits: &[f32]) -> (LlamaToken, f32) {
    if logits.is_empty() {
        return (LlamaToken(0), 0.0);
    }
    let mut best_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            best_idx = idx;
        }
    }
    if !max_val.is_finite() {
        return (LlamaToken(i32::try_from(best_idx).unwrap_or(0)), 0.0);
    }
    let mut sum_exp = 0.0_f64;
    for &val in logits {
        let diff = f64::from(val) - f64::from(max_val);
        if diff > -80.0 {
            sum_exp += diff.exp();
        }
    }
    let prob = if sum_exp > 0.0 {
        (1.0_f64 / sum_exp) as f32
    } else {
        0.0
    };
    (LlamaToken(i32::try_from(best_idx).unwrap_or(0)), prob)
}

fn plan_context_spec(
    prepared: &[PreparedPrompt],
    options: &GenerationOptions,
    mtp_n_draft: Option<usize>,
    recommended_moe_cache_bytes: usize,
    min_moe_cache_bytes_cpu_share: u64,
    model_n_ctx_train: u32,
) -> Result<ContextSpec> {
    let sequences = u32::try_from(prepared.len()).context("too many sequences")?;
    let mtp_enabled = mtp_n_draft.is_some();
    let n_outputs_max_per_seq = mtp_n_draft
        .map(|d| u32::try_from(d.saturating_add(1)).unwrap_or(u32::MAX))
        .unwrap_or(1);
    let _ = sequences
        .checked_mul(n_outputs_max_per_seq)
        .context("n_outputs_max exceeds u32")?;
    let mut n_ctx = 1;
    let mut n_batch = sequences.max(n_outputs_max_per_seq);
    let mut non_causal = false;
    for prepared in prepared {
        let values = context_values(prepared, options)?;
        n_ctx = n_ctx.max(values.n_ctx.get());
        n_batch = n_batch.max(values.n_batch);
        non_causal |= prepared.non_causal;
    }
    if options.n_ctx.is_none() && sequences == 1 && !non_causal {
        let rounded = if n_ctx <= DEFAULT_MIN_PERSISTENT_N_CTX {
            DEFAULT_MIN_PERSISTENT_N_CTX
        } else {
            n_ctx
                .checked_next_multiple_of(N_CTX_GROWTH_GRANULARITY)
                .unwrap_or(n_ctx)
        };
        n_ctx = if model_n_ctx_train > 0 {
            rounded.min(model_n_ctx_train.max(n_ctx))
        } else {
            rounded
        };
    }
    if options.n_batch.is_none() && !non_causal {
        // Keep default n_batch (and therefore default n_ubatch) stable across growing
        // prompts so persistent CachedContext reuse does not thrash on prompt length.
        n_batch = n_batch.max(n_ctx.min(DEFAULT_MAX_BATCH));
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
    let n_threads = options.n_threads.unwrap_or_else(available_threads);
    let n_threads_batch = options.n_threads_batch.unwrap_or_else(available_threads);
    let moe_cache_size = match options.moe_cache_size {
        Some(size) => size,
        None if options.moe_cache_cpu_share > 0.0
            && (recommended_moe_cache_bytes as u64) < min_moe_cache_bytes_cpu_share =>
        {
            0
        }
        None => recommended_moe_cache_bytes,
    };
    Ok(ContextSpec {
        n_ctx,
        n_batch,
        n_ubatch,
        n_seq_max: sequences,
        n_outputs_max_per_seq,
        flash_attn: options.flash_attn,
        swa_full: options.swa_full,
        kv_cache_type_k: options.kv_cache_type_k,
        kv_cache_type_v: options.kv_cache_type_v,
        kv_unified: options.kv_unified,
        offload_kqv: options.offload_kqv,
        op_offload: options.op_offload,
        moe_cache_size,
        moe_cache_cpu_share: options.moe_cache_cpu_share,
        n_threads,
        n_threads_batch,
        non_causal,
        mtp_enabled,
    })
}

fn batch_context_config(
    prepared: &[PreparedPrompt],
    options: &GenerationOptions,
    mtp_n_draft: Option<usize>,
    recommended_moe_cache_bytes: usize,
    min_moe_cache_bytes_cpu_share: u64,
    model_n_ctx_train: u32,
) -> Result<ContextConfig> {
    let spec = plan_context_spec(
        prepared,
        options,
        mtp_n_draft,
        recommended_moe_cache_bytes,
        min_moe_cache_bytes_cpu_share,
        model_n_ctx_train,
    )?;
    let n_outputs_max = spec
        .n_seq_max
        .checked_mul(spec.n_outputs_max_per_seq)
        .context("n_outputs_max exceeds u32")?;
    let flash_attn_policy = if spec.flash_attn {
        LLAMA_FLASH_ATTN_TYPE_ENABLED
    } else {
        LLAMA_FLASH_ATTN_TYPE_DISABLED
    };
    let mut params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(spec.n_ctx))
        .with_n_batch(spec.n_batch)
        .with_n_ubatch(spec.n_ubatch)
        .with_n_seq_max(spec.n_seq_max)
        .with_n_outputs_max(n_outputs_max)
        .with_n_outputs_max_per_seq(spec.n_outputs_max_per_seq)
        .with_flash_attention_policy(flash_attn_policy)
        .with_swa_full(spec.swa_full)
        .with_kv_unified(spec.kv_unified)
        .with_offload_kqv(spec.offload_kqv)
        .with_op_offload(spec.op_offload)
        .with_moe_cache_size(spec.moe_cache_size)
        .with_moe_cache_cpu_share(spec.moe_cache_cpu_share)
        .with_no_perf(true)
        .with_n_threads(spec.n_threads)
        .with_n_threads_batch(spec.n_threads_batch);
    if let Some(type_k) = spec.kv_cache_type_k {
        params = params.with_type_k(type_k);
    }
    if let Some(type_v) = spec.kv_cache_type_v {
        params = params.with_type_v(type_v);
    }
    Ok(ContextConfig { params, spec })
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

    let n_batch = match &prepared.kind {
        PreparedPromptKind::Text(_) => {
            let default_batch = u32::try_from(
                prepared
                    .prompt_tokens
                    .min(DEFAULT_MAX_BATCH as usize)
                    .max(1),
            )
            .unwrap_or(DEFAULT_MAX_BATCH);
            match options.n_batch {
                Some(0) => bail!("n_batch must be positive"),
                Some(n) => n,
                None => default_batch,
            }
        }
        PreparedPromptKind::Multimodal { .. } => {
            let required_batch = u32::try_from(prepared.batch_tokens.max(1))
                .context("prompt batch size exceeds u32")?;
            let n_batch = match options.n_batch {
                Some(0) => bail!("n_batch must be positive"),
                Some(n) => n,
                None => required_batch,
            };
            ensure!(
                n_batch >= required_batch,
                "n_batch is too small: need at least {required_batch} tokens"
            );
            n_batch
        }
    };
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
    if let Some(n_batch) = options.n_batch {
        ensure!(n_batch > 0, "n_batch must be positive");
    }
    if let Some(n_ubatch) = options.n_ubatch {
        ensure!(n_ubatch > 0, "n_ubatch must be positive");
    }
    ensure!(
        options.moe_cache_cpu_share.is_finite()
            && (0.0..=1.0).contains(&options.moe_cache_cpu_share),
        "moe_cache_cpu_share must be between 0 and 1"
    );
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

    #[test]
    fn text_prompt_allows_chunked_batch_smaller_than_prompt() {
        let default_opts = GenerationOptions {
            max_tokens: 128,
            ..Default::default()
        };
        let default_cfg = context_values(&prepared(4096, false), &default_opts).unwrap();
        assert_eq!(default_cfg.n_batch, DEFAULT_MAX_BATCH);

        let chunked_opts = GenerationOptions {
            max_tokens: 128,
            n_batch: Some(512),
            ..Default::default()
        };
        let chunked_cfg = context_values(&prepared(4096, false), &chunked_opts).unwrap();
        assert_eq!(chunked_cfg.n_batch, 512);
    }

    fn base_spec() -> ContextSpec {
        ContextSpec {
            n_ctx: 4096,
            n_batch: 1024,
            n_ubatch: 512,
            n_seq_max: 2,
            n_outputs_max_per_seq: 1,
            flash_attn: true,
            swa_full: false,
            kv_cache_type_k: Some(KvCacheType::Q8_0),
            kv_cache_type_v: Some(KvCacheType::Q8_0),
            kv_unified: false,
            offload_kqv: true,
            op_offload: true,
            moe_cache_size: 2048 * 1024 * 1024,
            moe_cache_cpu_share: 0.0,
            n_threads: 8,
            n_threads_batch: 16,
            non_causal: false,
            mtp_enabled: false,
        }
    }

    #[test]
    fn common_prefix_len_handles_edge_cases() {
        let a = [LlamaToken(1), LlamaToken(2), LlamaToken(3), LlamaToken(4)];
        let b = [LlamaToken(1), LlamaToken(2), LlamaToken(9), LlamaToken(4)];
        assert_eq!(common_prefix_len(&[], &[]), 0);
        assert_eq!(common_prefix_len(&[], &a), 0);
        assert_eq!(common_prefix_len(&a, &[]), 0);
        assert_eq!(common_prefix_len(&a, &a), 4);
        assert_eq!(common_prefix_len(&a, &b), 2);
        assert_eq!(common_prefix_len(&a[..2], &a), 2);
        assert_eq!(common_prefix_len(&[LlamaToken(9)], &a), 0);
    }

    #[test]
    fn context_spec_can_reuse_capacity_and_config_rules() {
        let cached = base_spec();
        assert!(cached.can_reuse(&cached));

        // Smaller capacity requests reuse cleanly when per-seq context fits.
        let mut smaller = base_spec();
        smaller.n_ctx = 2048;
        smaller.n_seq_max = 1;
        smaller.n_batch = 512;
        assert!(cached.can_reuse(&smaller));

        // Larger capacity requires reallocation.
        let mut larger_ctx = base_spec();
        larger_ctx.n_ctx = 8192;
        assert!(!cached.can_reuse(&larger_ctx));

        // Non-unified KV cache splits n_ctx across n_seq_max, so 1 seq needing 3072
        // cannot reuse a 4096/2 = 2048 per-seq context.
        let mut larger_per_seq = base_spec();
        larger_per_seq.n_ctx = 3072;
        larger_per_seq.n_seq_max = 1;
        assert!(!cached.can_reuse(&larger_per_seq));

        let mut larger_batch = base_spec();
        larger_batch.n_batch = 2048;
        assert!(!cached.can_reuse(&larger_batch));

        let mut larger_seq = base_spec();
        larger_seq.n_seq_max = 4;
        assert!(!cached.can_reuse(&larger_seq));

        let mut larger_outputs = base_spec();
        larger_outputs.n_outputs_max_per_seq = 5;
        assert!(!cached.can_reuse(&larger_outputs));

        // ubatch must match exactly.
        let mut diff_ubatch = base_spec();
        diff_ubatch.n_ubatch = 256;
        assert!(!cached.can_reuse(&diff_ubatch));

        // Non-capacity config mismatches must not reuse.
        let mut diff_fa = base_spec();
        diff_fa.flash_attn = false;
        assert!(!cached.can_reuse(&diff_fa));

        let mut diff_kv = base_spec();
        diff_kv.kv_cache_type_k = Some(KvCacheType::F16);
        assert!(!cached.can_reuse(&diff_kv));

        let mut diff_moe = base_spec();
        diff_moe.moe_cache_size = 0;
        assert!(!cached.can_reuse(&diff_moe));

        let mut diff_mtp = base_spec();
        diff_mtp.mtp_enabled = true;
        assert!(!cached.can_reuse(&diff_mtp));
    }

    #[test]
    fn top1_token_and_prob_computes_stable_softmax() {
        let (tok, prob) = top1_token_and_prob(&[10.0, 12.0, 10.0]);
        assert_eq!(tok, LlamaToken(1));
        let expected = 1.0_f32 / (1.0 + 2.0 * (-2.0_f32).exp());
        assert!((prob - expected).abs() < 1e-5);

        let (tok_large, prob_large) = top1_token_and_prob(&[1000.0, 1000.0]);
        assert_eq!(tok_large, LlamaToken(0));
        assert!((prob_large - 0.5).abs() < 1e-5);

        let (tok_empty, prob_empty) = top1_token_and_prob(&[]);
        assert_eq!(tok_empty, LlamaToken(0));
        assert_eq!(prob_empty, 0.0);
    }

    #[test]
    fn grammar_top_k_retains_lower_ids_across_the_boundary() {
        let logits = [2.0, 2.0, 3.0, 2.0];
        let ids = |data: Vec<LlamaTokenData>| data.iter().map(|d| d.id().0).collect::<Vec<_>>();
        assert_eq!(ids(top_k_token_data(token_data(&logits), 2)), [2, 0]);
        let mut data: Vec<_> = token_data(&logits).collect();
        data.reverse();
        assert_eq!(ids(top_k_token_data(data.into_iter(), 2)), [2, 0]);
    }

    #[test]
    fn stochastic_shortlist_requires_exact_selector_support() {
        let valid = token_data(&[0.0, -0.1]).collect::<Vec<_>>();
        let opts = GenerationOptions {
            temperature: 1.0,
            ..Default::default()
        };
        let exact =
            |opts: &GenerationOptions, boundary| shortlist_is_exact(opts, &valid, boundary, 64, 65);
        assert!(!exact(&opts, -103.0));
        assert!(exact(&opts, -104.0));
        assert!(exact(
            &GenerationOptions {
                temperature: 0.5,
                ..opts.clone()
            },
            -52.0
        ));
        assert!(shortlist_is_exact(&opts, &valid, -0.1, 64, 64));
        assert!(!shortlist_is_exact(&opts, &[], -200.0, 64, 64));
        let min_p = GenerationOptions {
            min_p: Some(0.5),
            top_k: Some(0),
            ..opts.clone()
        };
        assert!(exact(&min_p, -0.7));
        assert!(!exact(&min_p, 0.5_f32.ln()));
        let top_p = GenerationOptions {
            top_p: Some(0.9),
            ..min_p
        };
        assert!(!exact(&top_p, -110.0));
        assert!(exact(
            &GenerationOptions {
                top_k: Some(2),
                ..top_p.clone()
            },
            -1.0
        ));
        assert!(!exact(
            &GenerationOptions {
                top_k: Some(3),
                ..top_p
            },
            -110.0
        ));
    }

    #[test]
    fn grammar_mask_drops_forbidden_tokens_and_rejects_an_empty_vocabulary() {
        let logits = (0..65).map(|i| -(i as f32) / 100.0).collect::<Vec<_>>();
        for penalize in [true, false] {
            let candidates = |options: &GenerationOptions, reject_all: bool| {
                grammar_candidates(
                    &logits,
                    penalize.then(|| full_token_data_array(&logits)),
                    options,
                    |array| {
                        for token in &mut array.data {
                            if reject_all || token.id().0 == 0 {
                                token.set_logit(f32::NEG_INFINITY);
                            }
                        }
                    },
                )
            };
            let opts = GenerationOptions {
                top_k: Some(2),
                ..Default::default()
            };
            let shortlist = candidates(&opts, false).unwrap();
            assert!(shortlist.sorted);
            assert_eq!(shortlist.data[0].id(), LlamaToken(1));
            assert!(shortlist.data.iter().all(|d| d.logit().is_finite()));
            let full = candidates(
                &GenerationOptions {
                    top_k: Some(64),
                    ..opts.clone()
                },
                false,
            )
            .unwrap();
            assert!(full.sorted);
            assert_eq!(full.data.len(), 64);
            assert_eq!(full.data.last().unwrap().id(), LlamaToken(64));
            assert!(full.data.iter().all(|d| d.logit().is_finite()));
            for temperature in [0.0, 1.0] {
                let error = candidates(
                    &GenerationOptions {
                        temperature,
                        ..opts.clone()
                    },
                    true,
                )
                .unwrap_err();
                assert!(error.to_string().contains("rejected all vocabulary tokens"));
            }
        }
    }

    #[test]
    fn grammar_shortlist_and_full_vocabulary_break_top_k_ties_by_token_id() {
        let mut legal_logits = vec![8.0, 8.0, 9.0, 9.0, 10.0];
        legal_logits.resize(65, 1.0);
        let options = GenerationOptions {
            top_k: Some(2),
            ..Default::default()
        };
        for penalize in [false, true] {
            let candidates = |masked_logit| {
                let mut logits = legal_logits.clone();
                logits.resize(logits.len() + FAST_GRAMMAR_TOP_K - 1, masked_logit);
                grammar_candidates(
                    &logits,
                    penalize.then(|| full_token_data_array(&logits)),
                    &options,
                    |array| {
                        for item in &mut array.data {
                            if item.id().0 as usize >= legal_logits.len() {
                                item.set_logit(f32::NEG_INFINITY);
                            }
                        }
                    },
                )
                .unwrap()
            };
            let shortlist = candidates(0.0);
            // The masked tokens now displace all but one legal shortlist entry.
            let full = candidates(11.0);
            assert!(shortlist.sorted && full.sorted);
            assert_eq!(shortlist.data.len(), FAST_GRAMMAR_TOP_K);
            assert_eq!(full.data.len(), legal_logits.len());
            assert_eq!(shortlist.data, full.data[..FAST_GRAMMAR_TOP_K]);
            assert_eq!(shortlist.data[0].id(), LlamaToken(4));
            assert_eq!(shortlist.data[1].id(), LlamaToken(2));
        }
    }

    #[test]
    fn kv_prefix_requires_the_entire_attention_window() {
        assert!(!can_reuse_kv_prefix(300, 3600, 4627, 1024, false));
        assert!(can_reuse_kv_prefix(300, 0, 4627, 1024, false));
        assert!(can_reuse_kv_prefix(4096, 3072, 4627, 1024, false));
        assert!(!can_reuse_kv_prefix(4096, 3073, 4627, 1024, false));
        assert!(!can_reuse_kv_prefix(4096, 1, 4627, 0, false));
        assert!(!can_reuse_kv_prefix(4096, 1, 4627, 1024, true));
        for (keep, min, max) in [(0, 0, 0), (3, -1, 2), (3, 0, -1), (3, 0, 1)] {
            assert!(!can_reuse_kv_prefix(keep, min, max, 1024, false));
        }
    }

    #[test]
    fn auto_moe_skips_dense_and_uninspectable_models() {
        let options = LoadOptions::default();
        let dense = GgufInfo::default();
        for info in [Some(&dense), None] {
            let (resolved, cache) =
                resolve_auto_moe(&options, info, || panic!("dense models need no VRAM plan"));
            assert_eq!(resolved.moe_offload, MoeOffload::None);
            assert_eq!(resolved.load_mode, options.load_mode);
            assert_eq!(cache, 0);
        }
    }

    #[test]
    fn automatic_moe_cache_respects_the_cpu_share_floor_and_explicit_size() {
        for (setting, cpu_share, recommended, expected) in [
            (None, 0.0, 1234, 1234),
            (None, 0.5, 1234, 0),
            (None, 0.5, 2048, 2048),
            (Some(0), 0.0, 2048, 0),
            (Some(567), 0.5, 1234, 567),
        ] {
            let opts = GenerationOptions {
                moe_cache_size: setting,
                moe_cache_cpu_share: cpu_share,
                ..Default::default()
            };
            let spec =
                plan_context_spec(&[prepared(5, false)], &opts, None, recommended, 2048, 2048)
                    .unwrap();
            assert_eq!(spec.moe_cache_size, expected);
        }
    }

    #[test]
    fn plan_auto_moe_budget_handles_small_and_large_models() {
        // Small model (~0.6 GiB on 10 GiB GPU) stays 100% on GPU with zero offload/cache.
        let small = plan_auto_moe_budget(600 * MIB, 10 * GIB, 28, |_| 0);
        assert_eq!(small.offload, MoeOffload::None);
        assert_eq!(small.moe_cache_bytes, 0);
        assert!(!small.prefer_pinned_host);

        // Gemma 4 26B A4B (~14.2 GiB, 30 layers) across RTX 3080 free VRAM range (9.5-9.9 GiB):
        // tightens to FirstLayers(18..=19) + ~2 GiB dynamic MoE cache while staying under 92% budget.
        for (free_mib, expected_ncmoe) in [
            (9_728u64, 19u32),  // 9.5 GiB
            (9_933u64, 19u32),  // 9.7 GiB
            (10_138u64, 18u32), // 9.9 GiB
        ] {
            let free_bytes = free_mib * MIB;
            let plan = plan_auto_moe_budget(14_540 * MIB, free_bytes, 30, |offload| {
                assert_eq!(offload, MoeOffload::FirstLayers(expected_ncmoe));
                128 * MIB
            });
            assert_eq!(
                plan.offload,
                MoeOffload::FirstLayers(expected_ncmoe),
                "unexpected offload for free_mib={free_mib}"
            );
            assert!(
                plan.moe_cache_bytes >= 2048 * (MIB as usize)
                    && plan.moe_cache_bytes <= 2304 * (MIB as usize),
                "unexpected moe_cache_bytes={} for free_mib={free_mib}",
                plan.moe_cache_bytes
            );
            let gpu_layers = 30 - u64::from(expected_ncmoe);
            let est_static_moe =
                (14_540 * MIB * AUTO_MOE_WEIGHT_FRACTION_PCT / 100) / 30 * gpu_layers;
            let est_total =
                AUTO_FIXED_VRAM_RESERVE_BYTES + plan.moe_cache_bytes as u64 + est_static_moe;
            assert!(
                est_total <= free_bytes * AUTO_VRAM_UTILIZATION_PCT / 100,
                "estimated VRAM {est_total} exceeds 92% of {free_bytes}"
            );
            assert!(plan.prefer_pinned_host);
        }

        // Very tight VRAM (2.5 GiB free) offloads all MoE layers to CPU cleanly.
        let tight = plan_auto_moe_budget(14_540 * MIB, 2_560 * MIB, 30, |_| 0);
        assert_eq!(tight.offload, MoeOffload::AllCpu);
        assert!(tight.prefer_pinned_host);
        let below_floor = plan_auto_moe_budget(14_540 * MIB, 2_560 * MIB, 30, |offload| {
            assert_eq!(offload, MoeOffload::AllCpu);
            567 * MIB / 2
        });
        assert_eq!(below_floor.moe_cache_bytes, 0);
        assert_eq!(below_floor.offload, MoeOffload::FirstLayers(29));

        // Large 24 GiB GPU easily fits the 14.2 GiB model with full GPU residency.
        let large_gpu = plan_auto_moe_budget(14_540 * MIB, 24 * GIB, 30, |_| 0);
        assert_eq!(large_gpu.offload, MoeOffload::None);
        assert_eq!(large_gpu.moe_cache_bytes, 0);
        assert!(!large_gpu.prefer_pinned_host);
    }

    #[test]
    fn persistent_context_headroom_and_default_ubatch_reuse_across_pages() {
        assert_eq!(DEFAULT_MAX_UBATCH, 1024);
        let options = GenerationOptions {
            max_tokens: 512,
            ..Default::default()
        };
        let page1 =
            plan_context_spec(&[prepared(451, false)], &options, None, 0, 0, 131_072).unwrap();
        let page2 =
            plan_context_spec(&[prepared(663, false)], &options, None, 0, 0, 131_072).unwrap();
        let page3 =
            plan_context_spec(&[prepared(859, false)], &options, None, 0, 0, 131_072).unwrap();

        assert_eq!(page1.n_ctx, 2048);
        assert_eq!(page1.n_batch, 2048);
        assert_eq!(page1.n_ubatch, 1024);
        assert!(page1.can_reuse(&page2));
        assert!(page1.can_reuse(&page3));

        // Prompts exceeding 2048 round up to the next 1024 multiple.
        let long =
            plan_context_spec(&[prepared(1800, false)], &options, None, 0, 0, 131_072).unwrap();
        assert_eq!(long.n_ctx, 3072);

        // Rounding respects model_n_ctx_train cap when positive.
        let capped =
            plan_context_spec(&[prepared(1800, false)], &options, None, 0, 0, 2560).unwrap();
        assert_eq!(capped.n_ctx, 2560);
    }
}

const MAX_LOOP_PERIOD: usize = 8;
const MIN_LOOP_REPEATS: usize = 8;

fn has_repetition_loop(tokens: &[LlamaToken]) -> bool {
    for period in 1..=MAX_LOOP_PERIOD {
        let window = period * MIN_LOOP_REPEATS;
        if tokens.len() < window {
            continue;
        }
        let tail = &tokens[tokens.len() - window..];
        let pattern = &tail[..period];
        if tail.chunks_exact(period).all(|chunk| chunk == pattern) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod loop_tests {
    use super::*;

    #[test]
    fn detects_single_and_multi_token_repetition_loops() {
        let single: Vec<_> = (0..8).map(|_| LlamaToken(42)).collect();
        assert!(has_repetition_loop(&single));

        let short: Vec<_> = (0..7).map(|_| LlamaToken(42)).collect();
        assert!(!has_repetition_loop(&short));

        let cycle: Vec<_> = [1, 2, 3]
            .into_iter()
            .cycle()
            .take(24)
            .map(LlamaToken)
            .collect();
        assert!(has_repetition_loop(&cycle));
    }
}
