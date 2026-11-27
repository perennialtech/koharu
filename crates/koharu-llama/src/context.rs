//! Safe wrapper around `llama_context`.

use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;

use crate::context::params::{LlamaContextParams, LlamaContextType};
use crate::llama_backend::LlamaBackend;
use crate::llama_batch::{BatchExtError, LlamaBatch, LlamaBatchExtScope};
use crate::model::{LlamaLoraAdapter, LlamaModel};
use crate::sampling::LlamaSampler;
use crate::timing::LlamaTimings;
use crate::token::LlamaToken;
use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::{
    DecodeError, EmbeddingsError, EncodeError, LlamaContextLoadError, LlamaLoraAdapterRemoveError,
    LlamaLoraAdapterSetError,
};

pub mod kv_cache;
pub mod params;
pub mod session;

/// Storage for a native context and the model that backs it.
pub struct ModelContext<M: ?Sized> {
    pub(crate) context: NonNull<koharu_llama_sys::llama_context>,
    batch_ext: Option<NonNull<koharu_llama_sys::llama_batch_ext>>,
    pub(crate) initialized_logits: Vec<i32>,
    embeddings_enabled: bool,
    pub(crate) requires_embd_state: bool,
    kv_streams: u32,
    _backend_samplers: Vec<(i32, LlamaSampler)>,
    // Erasing the model holder makes the borrowed context unsized. A mutable
    // borrow can operate on the native context but cannot replace or extract it.
    model: M,
}

/// A native context whose model stays alive for its entire lifetime.
pub type LlamaContext<'model> = ModelContext<dyn Borrow<LlamaModel> + 'model>;

impl<M: ?Sized> Debug for ModelContext<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("context", &self.context)
            .finish()
    }
}

fn kv_stream_count(params: &LlamaContextParams) -> u32 {
    if params.kv_unified() {
        1
    } else {
        params.n_seq_max().max(1)
    }
}

impl<M> ModelContext<M> {
    pub(crate) fn new(
        model: M,
        context: NonNull<koharu_llama_sys::llama_context>,
        params: &LlamaContextParams,
        backend_samplers: Vec<(i32, LlamaSampler)>,
    ) -> Self {
        Self {
            context,
            batch_ext: None,
            initialized_logits: Vec::new(),
            embeddings_enabled: params.embeddings(),
            requires_embd_state: params.context_type() == LlamaContextType::Mtp,
            kv_streams: kv_stream_count(params),
            _backend_samplers: backend_samplers,
            model,
        }
    }
}

impl<'model> LlamaContext<'model> {
    /// The model that determines this context's tensor and vocabulary sizes.
    pub fn model(&self) -> &LlamaModel {
        self.model.borrow()
    }

    /// Gets the max number of logical tokens that can be submitted to decode. Must be greater than or equal to [`Self::n_ubatch`].
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        unsafe { koharu_llama_sys::llama_n_batch(self.context.as_ptr()) }
    }

    /// Gets the max number of physical tokens (hardware level) to decode in batch. Must be less than or equal to [`Self::n_batch`].
    #[must_use]
    pub fn n_ubatch(&self) -> u32 {
        unsafe { koharu_llama_sys::llama_n_ubatch(self.context.as_ptr()) }
    }

    /// Gets the size of the context.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { koharu_llama_sys::llama_n_ctx(self.context.as_ptr()) }
    }

    /// Decodes the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        if self.requires_embd_state {
            // The legacy batch API cannot carry MTP hidden states.
            return Err(DecodeError::MissingEmbdState);
        }
        crate::clear_allocation_failure();
        let result =
            unsafe { koharu_llama_sys::llama_decode(self.context.as_ptr(), batch.llama_batch) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Encodes the batch.
    ///
    /// # Errors
    ///
    /// - `EncodeError` for MTP contexts or if encoding fails.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn encode(&mut self, batch: &mut LlamaBatch) -> Result<(), EncodeError> {
        if self.requires_embd_state {
            return Err(EncodeError::Unknown(-1));
        }
        crate::clear_allocation_failure();
        let result =
            unsafe { koharu_llama_sys::llama_encode(self.context.as_ptr(), batch.llama_batch) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(EncodeError::from(error)),
        }
    }

    /// Get the embeddings for the `i`th sequence in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch. Its size is the
    /// model's output embedding width, or classifier output width for rank pooling.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - If the current model had a pooling type of [`koharu_llama_sys::LLAMA_POOLING_TYPE_NONE`]
    /// - If the given sequence index exceeds the max sequence id.
    ///
    /// # Panics
    ///
    /// * The output width does not fit into a `usize`.
    pub fn embeddings_seq_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        unsafe {
            let embedding = koharu_llama_sys::llama_get_embeddings_seq(self.context.as_ptr(), i);

            // Technically also possible whenever `i >= max(batch.n_seq)`, but can't check that here.
            if embedding.is_null() {
                Err(EmbeddingsError::NonePoolType)
            } else {
                Ok(slice::from_raw_parts(embedding, self.embeddings_out_len()))
            }
        }
    }

    /// Get the embeddings for the `i`th token in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch of the given token.
    /// Its size is the model's output embedding width, or classifier output width for rank pooling.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - When the given token didn't have logits enabled when it was passed.
    /// - If the given token index exceeds the max token id.
    ///
    /// # Panics
    ///
    /// * The output width does not fit into a `usize`.
    pub fn embeddings_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        unsafe {
            let embedding = koharu_llama_sys::llama_get_embeddings_ith(self.context.as_ptr(), i);
            // Technically also possible whenever `i >= batch.n_tokens`, but no good way of checking `n_tokens` here.
            if embedding.is_null() {
                Err(EmbeddingsError::LogitsNotEnabled)
            } else {
                Ok(slice::from_raw_parts(embedding, self.embeddings_out_len()))
            }
        }
    }

    fn embeddings_out_len(&self) -> usize {
        let pooling = unsafe { koharu_llama_sys::llama_pooling_type(self.context.as_ptr()) };
        if pooling == koharu_llama_sys::LLAMA_POOLING_TYPE_RANK {
            usize::try_from(self.model().n_cls_out()).expect("n_cls_out does not fit into a usize")
        } else {
            usize::try_from(self.model().n_embd_out())
                .expect("n_embd_out does not fit into a usize")
        }
    }

    /// Get the logits for the last token in the context.
    ///
    /// # Returns
    /// An iterator over unsorted `LlamaTokenData` containing the
    /// logits for the last token in the context.
    ///
    /// # Panics
    ///
    /// - underlying logits data is null
    pub fn candidates(&self) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits()).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the token data array for the last token in the context.
    ///
    /// This is a convience method that implements:
    /// ```ignore
    /// LlamaTokenDataArray::from_iter(ctx.candidates(), false)
    /// ```
    ///
    /// # Panics
    ///
    /// - underlying logits data is null
    #[must_use]
    pub fn token_data_array(&self) -> LlamaTokenDataArray {
        LlamaTokenDataArray::from_iter(self.candidates(), false)
    }

    /// Token logits obtained from the last call to `decode()`.
    /// The logits for which `batch.logits[i] != 0` are stored contiguously
    /// in the order they have appeared in the batch.
    /// Rows: number of tokens for which `batch.logits[i] != 0`
    /// Cols: `n_vocab`
    ///
    /// # Returns
    ///
    /// A slice containing the logits for the last decoded token.
    /// The size corresponds to the `n_vocab` parameter of the context's model.
    ///
    /// # Panics
    ///
    /// - `n_vocab` does not fit into a usize
    /// - token data returned is null
    #[must_use]
    pub fn get_logits(&self) -> &[f32] {
        let data = unsafe { koharu_llama_sys::llama_get_logits(self.context.as_ptr()) };
        assert!(!data.is_null(), "logits data for last token is null");
        let len =
            usize::try_from(self.model().n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    pub fn candidates_ith(&self, i: i32) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits_ith(i)).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the token data array for the ith token in the context.
    ///
    /// This is a convience method that implements:
    /// ```ignore
    /// LlamaTokenDataArray::from_iter(ctx.candidates_ith(i), false)
    /// ```
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    #[must_use]
    pub fn token_data_array_ith(&self, i: i32) -> LlamaTokenDataArray {
        LlamaTokenDataArray::from_iter(self.candidates_ith(i), false)
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - `i` is greater than `n_ctx`
    /// - `n_vocab` does not fit into a usize
    /// - logit `i` is not initialized.
    #[must_use]
    pub fn get_logits_ith(&self, i: i32) -> &[f32] {
        assert!(
            self.initialized_logits.contains(&i),
            "logit {i} is not initialized. only {:?} is",
            self.initialized_logits
        );
        assert!(
            self.n_ctx() > u32::try_from(i).expect("i does not fit into a u32"),
            "n_ctx ({}) must be greater than i ({})",
            self.n_ctx(),
            i
        );

        self.try_get_logits_ith(i).expect("logits unavailable")
    }

    /// Returns logits for the `i`-th token in the last decoded batch, supporting negative
    /// indices such as `-1` for the last output token. Returns `None` if unavailable.
    #[must_use]
    pub fn try_get_logits_ith(&self, i: i32) -> Option<&[f32]> {
        let data = unsafe { koharu_llama_sys::llama_get_logits_ith(self.context.as_ptr(), i) };
        if data.is_null() {
            return None;
        }
        let len = usize::try_from(self.model().n_vocab()).ok()?;
        Some(unsafe { slice::from_raw_parts(data, len) })
    }

    /// Reset the timings for the context.
    pub fn reset_timings(&mut self) {
        unsafe { koharu_llama_sys::llama_perf_context_reset(self.context.as_ptr()) }
    }

    /// Returns the timings for the context.
    pub fn timings(&mut self) -> LlamaTimings {
        let timings = unsafe { koharu_llama_sys::llama_perf_context(self.context.as_ptr()) };
        LlamaTimings { timings }
    }

    /// Sets a lora adapter.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterSetError`] for more information.
    pub fn lora_adapter_set(
        &self,
        adapter: &mut LlamaLoraAdapter,
        scale: f32,
    ) -> Result<(), LlamaLoraAdapterSetError> {
        let mut adapters = [adapter.lora_adapter.as_ptr()];
        let mut scales = [scale];
        let err_code = unsafe {
            koharu_llama_sys::llama_set_adapters_lora(
                self.context.as_ptr(),
                adapters.as_mut_ptr(),
                1,
                scales.as_mut_ptr(),
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterSetError::ErrorResult(err_code));
        }

        tracing::debug!("Set lora adapter");
        Ok(())
    }

    /// Remove all lora adapters.
    ///
    /// Note: The upstream API now replaces all adapters at once via
    /// `llama_set_adapters_lora`. This clears all adapters from the context.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterRemoveError`] for more information.
    pub fn lora_adapter_remove(
        &self,
        _adapter: &mut LlamaLoraAdapter,
    ) -> Result<(), LlamaLoraAdapterRemoveError> {
        let err_code = unsafe {
            koharu_llama_sys::llama_set_adapters_lora(
                self.context.as_ptr(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterRemoveError::ErrorResult(err_code));
        }

        tracing::debug!("Remove lora adapter");
        Ok(())
    }

    /// Returns the backend-sampled token at `i`, or `None` if no token was sampled.
    #[must_use]
    pub fn sampled_token_ith(&self, i: i32) -> Option<LlamaToken> {
        let token =
            unsafe { koharu_llama_sys::llama_get_sampled_token_ith(self.context.as_ptr(), i) };
        (token != -1).then_some(LlamaToken(token))
    }

    /// Starts an empty extended batch backed by this context's memory.
    pub fn batch_ext(&mut self) -> Result<LlamaBatchExtScope<'_, 'model>, BatchExtError> {
        let batch = match self.batch_ext {
            Some(batch) => batch,
            None => {
                let ptr = unsafe { koharu_llama_sys::llama_batch_ext_init(self.context.as_ptr()) };
                let batch = NonNull::new(ptr).ok_or(BatchExtError::NullInit)?;
                self.batch_ext = Some(batch);
                batch
            }
        };
        unsafe { koharu_llama_sys::llama_batch_ext_clear(batch.as_ptr()) };
        Ok(LlamaBatchExtScope {
            ctx: self,
            batch,
            n_entries: 0,
            next_embd_state: 0,
            initialized_logits: Vec::new(),
        })
    }

    /// Configures whether this context outputs NextN hidden states for MTP speculative decoding.
    pub fn set_embeddings_nextn(&mut self, enabled: bool, masked: bool) {
        unsafe {
            koharu_llama_sys::llama_set_embeddings_nextn(self.context.as_ptr(), enabled, masked);
        }
    }

    /// Returns the NextN hidden state slice for the `i`-th output token (supports negative indices like `-1`).
    #[must_use]
    pub fn get_embeddings_nextn_ith(&self, i: i32) -> Option<&[f32]> {
        let ptr =
            unsafe { koharu_llama_sys::llama_get_embeddings_nextn_ith(self.context.as_ptr(), i) };
        if ptr.is_null() {
            return None;
        }
        let len = usize::try_from(self.model().n_embd_out()).ok()?;
        Some(unsafe { slice::from_raw_parts(ptr, len) })
    }
}

impl<M: ?Sized> Drop for ModelContext<M> {
    fn drop(&mut self) {
        unsafe {
            if let Some(batch) = self.batch_ext {
                koharu_llama_sys::llama_batch_ext_free(batch.as_ptr());
            }
            koharu_llama_sys::llama_free(self.context.as_ptr());
        }
    }
}

/// A native context that owns a reference to its model.
#[derive(Debug)]
pub struct OwnedLlamaContext {
    pub(crate) context: ModelContext<Arc<LlamaModel>>,
}

// SAFETY: owned contexts have an exclusive native handle and a thread-safe model.
// A context that references another context is only accessible through its pair.
unsafe impl Send for OwnedLlamaContext {}

impl OwnedLlamaContext {
    /// Borrows this context for reading native outputs.
    pub fn as_context(&self) -> &LlamaContext<'_> {
        &self.context
    }

    /// Borrows this context exclusively for operations that change native state.
    pub fn as_context_mut(&mut self) -> &mut LlamaContext<'_> {
        &mut self.context
    }

    pub(crate) fn as_ptr(&self) -> *mut koharu_llama_sys::llama_context {
        self.context.context.as_ptr()
    }

    /// Returns the model owner.
    pub fn model(&self) -> &Arc<LlamaModel> {
        &self.context.model
    }

    /// Gets the max number of logical tokens that can be submitted to decode.
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        self.as_context().n_batch()
    }

    /// Gets the max number of physical tokens (hardware level) to decode in batch.
    #[must_use]
    pub fn n_ubatch(&self) -> u32 {
        self.as_context().n_ubatch()
    }

    /// Gets the size of the context.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        self.as_context().n_ctx()
    }

    /// Decodes the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if decoding failed.
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        self.as_context_mut().decode(batch)
    }

    /// Encodes the batch.
    ///
    /// # Errors
    ///
    /// - `EncodeError` if encoding failed.
    pub fn encode(&mut self, batch: &mut LlamaBatch) -> Result<(), EncodeError> {
        self.as_context_mut().encode(batch)
    }

    /// Clears the entire KV cache.
    pub fn clear_kv_cache(&mut self) {
        self.as_context_mut().clear_kv_cache();
    }

    /// Clears the KV cache for the given sequence within `[p0, p1)`.
    ///
    /// Returns `Ok(false)` when partial removal is unsupported (e.g. certain SWA ranges).
    ///
    /// # Errors
    ///
    /// - `KvCacheConversionError` if any coordinate exceeds `i32::MAX`.
    pub fn clear_kv_cache_seq(
        &mut self,
        src: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<bool, kv_cache::KvCacheConversionError> {
        self.as_context_mut().clear_kv_cache_seq(src, p0, p1)
    }

    /// Returns logits for the last decoded token.
    #[must_use]
    pub fn get_logits(&self) -> &[f32] {
        self.as_context().get_logits()
    }

    /// Returns logits for the `i`-th token in the last decoded batch.
    #[must_use]
    pub fn get_logits_ith(&self, i: i32) -> &[f32] {
        self.as_context().get_logits_ith(i)
    }

    /// Returns logits for the `i`-th token in the last decoded batch, supporting negative
    /// indices such as `-1`.
    #[must_use]
    pub fn try_get_logits_ith(&self, i: i32) -> Option<&[f32]> {
        self.as_context().try_get_logits_ith(i)
    }

    /// Starts an empty extended batch backed by this context's memory.
    pub fn batch_ext(&mut self) -> Result<LlamaBatchExtScope<'_, '_>, BatchExtError> {
        self.as_context_mut().batch_ext()
    }

    /// Returns the smallest cached position for the sequence.
    pub fn kv_cache_seq_pos_min(&self, seq_id: i32) -> i32 {
        self.as_context().kv_cache_seq_pos_min(seq_id)
    }

    /// Returns the largest cached position for the sequence.
    pub fn kv_cache_seq_pos_max(&self, seq_id: i32) -> i32 {
        self.as_context().kv_cache_seq_pos_max(seq_id)
    }

    /// Configures whether this context outputs NextN hidden states for MTP speculative decoding.
    pub fn set_embeddings_nextn(&mut self, enabled: bool, masked: bool) {
        self.as_context_mut().set_embeddings_nextn(enabled, masked);
    }

    /// Returns the NextN hidden state slice for the `i`-th output token (supports negative indices like `-1`).
    #[must_use]
    pub fn get_embeddings_nextn_ith(&self, i: i32) -> Option<&[f32]> {
        self.as_context().get_embeddings_nextn_ith(i)
    }

    /// Creates a Gemma 4 assistant MTP context sharing the target's KV cache.
    /// A populated target, incompatible models or mismatched KV stream counts
    /// return [`LlamaContextLoadError::IncompatiblePair`].
    /// On failure the target is returned to the caller.
    pub fn into_paired(
        self,
        backend: &LlamaBackend,
        draft_model: &Arc<LlamaModel>,
        mut draft_params: LlamaContextParams,
    ) -> Result<OwnedLlamaContextPair, (Self, LlamaContextLoadError)> {
        // The draft resizes the shared KV cell table to its own stream count.
        if draft_params.context_type() != LlamaContextType::Mtp
            || self.context.kv_streams != kv_stream_count(&draft_params)
            || self.context.requires_embd_state
            || !self.model().supports_mtp_draft(draft_model)
        {
            return Err((self, LlamaContextLoadError::IncompatiblePair));
        }
        // A single KV stream can hold any native sequence ID. Pairing resizes
        // its cell table, so every sequence in the target must be empty.
        let seq_count = if self.context.kv_streams == 1 {
            unsafe { koharu_llama_sys::llama_max_parallel_sequences() }
        } else {
            self.context.kv_streams as usize
        };
        if (0..seq_count).any(|seq| self.kv_cache_seq_pos_max(seq as i32) >= 0) {
            return Err((self, LlamaContextLoadError::IncompatiblePair));
        }
        draft_params.context_params.ctx_other = self.as_ptr();
        match draft_model.new_context_owned(backend, draft_params) {
            Ok(draft) => Ok(OwnedLlamaContextPair {
                draft,
                target: self,
            }),
            Err(error) => Err((self, error)),
        }
    }
}

/// A draft and its target with exclusive access to their shared native memory.
///
/// The mutable view cannot be swapped out, even with a context from another pair.
///
/// ```compile_fail,E0277
/// use koharu_llama::context::OwnedLlamaContextPair;
/// fn swap_targets(a: &mut OwnedLlamaContextPair, b: &mut OwnedLlamaContextPair) {
///     std::mem::swap(a.target_mut(), b.target_mut());
/// }
/// ```
///
/// A live batch on either context keeps exclusive access to the pair.
///
/// ```compile_fail,E0502
/// use koharu_llama::context::OwnedLlamaContextPair;
/// fn process(pair: &mut OwnedLlamaContextPair) {
///     let batch = pair.draft_mut().batch_ext().unwrap();
///     let _ = pair.target().n_ctx();
///     batch.decode().unwrap();
/// }
/// ```
#[derive(Debug)]
pub struct OwnedLlamaContextPair {
    // Rust drops fields in declaration order. The draft borrows native target memory.
    draft: OwnedLlamaContext,
    target: OwnedLlamaContext,
}

// SAFETY: moving the pair transfers both native contexts together. Mutable views
// cannot extract either handle or alias operations on the other context.
unsafe impl Send for OwnedLlamaContextPair {}

impl OwnedLlamaContextPair {
    pub fn target(&self) -> &OwnedLlamaContext {
        &self.target
    }

    pub fn target_mut(&mut self) -> &mut LlamaContext<'_> {
        self.target.as_context_mut()
    }

    pub fn draft(&self) -> &OwnedLlamaContext {
        &self.draft
    }

    pub fn draft_mut(&mut self) -> &mut LlamaContext<'_> {
        self.draft.as_context_mut()
    }

    pub fn is_mem_shared(&self) -> bool {
        unsafe {
            koharu_llama_sys::llama_get_ctx_other(self.draft.as_ptr()) == self.target.as_ptr()
        }
    }
}
