//! Safe wrapper around `llama_batch`.

use crate::DecodeError;
use crate::context::LlamaContext;
use crate::token::LlamaToken;
use koharu_llama_sys::{llama_batch, llama_batch_free, llama_batch_init, llama_pos, llama_seq_id};
use std::marker::PhantomData;
use std::num::NonZeroI32;

/// A safe wrapper around `llama_batch`.
#[derive(Debug)]
pub struct LlamaBatch<'a> {
    /// The number of tokens the batch was allocated with. they are safe to write to - but not necessarily read from as they are not necessarily initialized
    allocated: usize,
    /// The logits that are initialized. Used by [`LlamaContext`] to ensure that only initialized logits are accessed.
    pub(crate) initialized_logits: Vec<i32>,
    #[allow(clippy::doc_markdown)]
    /// The llama_cpp batch. always initialize by `koharu_llama_sys::llama_batch_init(allocated, <unknown>, <unknown>)`
    pub(crate) llama_batch: llama_batch,
    phantom: PhantomData<&'a [LlamaToken]>,
}

/// Errors that can occur when adding a token to a batch.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchAddError {
    /// There was not enough space in the batch to add the token.
    #[error("Insufficient Space of {0}")]
    InsufficientSpace(usize),
    /// Empty buffer is provided for [`LlamaBatch::get_one`]
    #[error("Empty buffer")]
    EmptyBuffer,
}

impl<'a> LlamaBatch<'a> {
    /// Clear the batch. This does not free the memory associated with the batch, but it does reset
    /// the number of tokens to 0.
    pub fn clear(&mut self) {
        self.llama_batch.n_tokens = 0;
        self.initialized_logits.clear();
    }

    /// add a token to the batch for sequences `seq_ids` at position `pos`. If `logits` is true, the
    /// token will be initialized and can be read from after the next decode.
    ///
    /// # Panics
    ///
    /// - [`self.llama_batch.n_tokens`] does not fit into a usize
    /// - [`seq_ids.len()`] does not fit into a [`llama_seq_id`]
    ///
    /// # Errors
    ///
    /// returns a error if there is insufficient space in the buffer
    pub fn add(
        &mut self,
        LlamaToken(id): LlamaToken,
        pos: llama_pos,
        seq_ids: &[i32],
        logits: bool,
    ) -> Result<(), BatchAddError> {
        if self.allocated
            < usize::try_from(self.n_tokens() + 1).expect("cannot fit n_tokens into a usize")
        {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }
        let offset = self.llama_batch.n_tokens;
        let offset_usize = usize::try_from(offset).expect("cannot fit n_tokens into a usize");
        unsafe {
            // batch.token   [batch.n_tokens] = id;
            self.llama_batch.token.add(offset_usize).write(id);
            // batch.pos     [batch.n_tokens] = pos,
            self.llama_batch.pos.add(offset_usize).write(pos);
            // batch.n_seq_id[batch.n_tokens] = seq_ids.size();
            self.llama_batch.n_seq_id.add(offset_usize).write(
                llama_seq_id::try_from(seq_ids.len())
                    .expect("cannot fit seq_ids.len() into a llama_seq_id"),
            );
            // for (size_t i = 0; i < seq_ids.size(); ++i) {
            //     batch.seq_id[batch.n_tokens][i] = seq_ids[i];
            // }
            for (i, seq_id) in seq_ids.iter().enumerate() {
                let tmp = *self.llama_batch.seq_id.add(offset_usize);
                tmp.add(i).write(*seq_id);
            }
            // batch.logits  [batch.n_tokens] = logits;
            self.llama_batch
                .logits
                .add(offset_usize)
                .write(i8::from(logits));
        }

        if logits {
            self.initialized_logits.push(offset);
        } else {
            self.initialized_logits.retain(|l| l != &offset);
        }

        // batch.n_tokens++;
        self.llama_batch.n_tokens += 1;

        Ok(())
    }

    /// Add a sequence of tokens to the batch for the given sequence id. If `logits_all` is true, the
    /// tokens will be initialized and can be read from after the next decode.
    ///
    /// Either way the last token in the sequence will have its logits set to `true`.
    ///
    /// # Errors
    ///
    /// Returns an error if there is insufficient space in the buffer
    ///
    /// # Panics
    ///
    /// - [`self.llama_batch.n_tokens`] does not fit into a [`usize`]
    /// - [`n_tokens - 1`] does not fit into a [`llama_pos`]
    pub fn add_sequence(
        &mut self,
        tokens: &[LlamaToken],
        seq_id: i32,
        logits_all: bool,
    ) -> Result<(), BatchAddError> {
        let n_tokens_0 =
            usize::try_from(self.llama_batch.n_tokens).expect("cannot fit n_tokens into a usize");
        let n_tokens = tokens.len();

        if self.allocated < n_tokens_0 + n_tokens {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }

        let last_index = llama_pos::try_from(n_tokens.saturating_sub(1))
            .expect("cannot fit n_tokens into a llama_pos");
        for (i, token) in (0..).zip(tokens.iter()) {
            self.add(*token, i, &[seq_id], logits_all || i == last_index)?;
        }

        Ok(())
    }

    /// Create a new `LlamaBatch` that can contain up to `n_tokens` tokens.
    ///
    /// # Arguments
    ///
    /// - `n_tokens`: the maximum number of tokens that can be added to the batch
    /// - `n_seq_max`: the maximum number of sequences that can be added to the batch (generally 1 unless you know what you are doing)
    ///
    /// # Panics
    ///
    /// Panics if `n_tokens` is greater than `i32::MAX`.
    #[must_use]
    pub fn new(n_tokens: usize, n_seq_max: i32) -> Self {
        let n_tokens_i32 = i32::try_from(n_tokens).expect("cannot fit n_tokens into a i32");
        let batch = unsafe { llama_batch_init(n_tokens_i32, 0, n_seq_max) };

        LlamaBatch {
            allocated: n_tokens,
            initialized_logits: vec![],
            llama_batch: batch,
            phantom: PhantomData,
        }
    }

    /// ``llama_batch_get_one``
    /// Return batch for single sequence of tokens
    ///
    /// NOTE: this is a helper function to facilitate transition to the new batch API
    ///
    /// # Errors
    /// If the provided token buffer is empty.
    ///
    /// # Panics
    /// If the number of tokens in ``tokens`` exceeds [`i32::MAX`].
    pub fn get_one(tokens: &'a [LlamaToken]) -> Result<Self, BatchAddError> {
        if tokens.is_empty() {
            return Err(BatchAddError::EmptyBuffer);
        }
        let batch = unsafe {
            let ptr = tokens.as_ptr() as *mut i32;
            koharu_llama_sys::llama_batch_get_one(
                ptr,
                tokens
                    .len()
                    .try_into()
                    .expect("number of tokens exceeds i32::MAX"),
            )
        };
        let batch = Self {
            allocated: 0,
            initialized_logits: vec![
                (tokens.len() - 1)
                    .try_into()
                    .expect("number of tokens exceeds i32::MAX + 1"),
            ],
            llama_batch: batch,
            phantom: PhantomData,
        };
        Ok(batch)
    }

    /// Returns the number of tokens in the batch.
    #[must_use]
    pub fn n_tokens(&self) -> i32 {
        self.llama_batch.n_tokens
    }
}

impl<'a> Drop for LlamaBatch<'a> {
    /// Drops the `LlamaBatch`.
    ///
    /// ```no_run
    /// # use koharu_llama::llama_batch::LlamaBatch;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let batch = LlamaBatch::new(512, 1);
    /// // frees the memory associated with the batch. (allocated by llama.cpp)
    /// drop(batch);
    /// # Ok(())
    /// # }
    fn drop(&mut self) {
        unsafe {
            if self.allocated > 0 {
                llama_batch_free(self.llama_batch);
            }
        }
    }
}

/// Errors returned when adding entries to a [`LlamaBatchExtScope`].
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchExtError {
    /// Failed to allocate `llama_batch_ext`.
    #[error("llama_batch_ext_init returned null")]
    NullInit,
    /// The batch is full.
    #[error("extended batch is full")]
    Full,
    /// Invalid token id or embedding.
    #[error("invalid token or embedding for extended batch")]
    InvalidToken,
    /// Invalid sequence id.
    #[error("invalid sequence id for extended batch")]
    InvalidSeqId,
    /// Unknown negative error code from `llama_batch_ext_add*`.
    #[error("extended batch error {0}")]
    Unknown(i32),
}

/// An extended batch borrowing its originating context until decoded or discarded.
#[derive(Debug)]
pub struct LlamaBatchExtScope<'ctx, 'model> {
    pub(crate) ctx: &'ctx mut LlamaContext<'model>,
    pub(crate) batch: std::ptr::NonNull<koharu_llama_sys::llama_batch_ext>,
    pub(crate) n_entries: usize,
    pub(crate) next_embd_state: usize,
    pub(crate) initialized_logits: Vec<i32>,
}

impl LlamaBatchExtScope<'_, '_> {
    /// Adds a token entry for `seq_id` and returns its batch index.
    ///
    /// # Errors
    ///
    /// Returns [`BatchExtError`] if llama.cpp rejects the entry.
    pub fn add_token(&mut self, seq_id: i32, token: LlamaToken) -> Result<i32, BatchExtError> {
        if !(0..self.ctx.model().n_vocab()).contains(&token.0) {
            return Err(BatchExtError::InvalidToken);
        }
        if seq_id < 0 {
            return Err(BatchExtError::InvalidSeqId);
        }
        let idx = unsafe {
            koharu_llama_sys::llama_batch_ext_add_token(self.batch.as_ptr(), seq_id, token.0)
        };
        if idx >= 0 {
            if self.ctx.requires_embd_state {
                self.n_entries += 1;
            }
            Ok(idx)
        } else {
            Err(match idx {
                -1 => BatchExtError::Full,
                -2 => BatchExtError::InvalidToken,
                -3 => BatchExtError::InvalidSeqId,
                other => BatchExtError::Unknown(other),
            })
        }
    }

    /// Associates an additional sequence id with batch entry `idx`.
    pub fn add_seq(&mut self, idx: i32, seq_id: i32) -> bool {
        unsafe { koharu_llama_sys::llama_batch_ext_add_seq(self.batch.as_ptr(), idx, seq_id) }
    }

    /// Sets the position for text token entry `idx`.
    pub fn set_pos(&mut self, idx: i32, pos: llama_pos) -> bool {
        unsafe {
            koharu_llama_sys::llama_batch_ext_set_pos(self.batch.as_ptr(), idx, [pos; 4].as_ptr())
        }
    }

    /// Sets the single-row hidden state embedding for entry `idx` (e.g. MTP target state).
    /// MTP states must be set once each in token order because the native batch
    /// appends state rows in the order they are supplied.
    pub fn set_embd_state(&mut self, idx: i32, state: &[f32]) -> bool {
        if self.ctx.requires_embd_state && usize::try_from(idx) != Ok(self.next_embd_state) {
            return false;
        }
        let embd = koharu_llama_sys::llama_embd {
            data: state.as_ptr(),
            n_rows: 1,
            n_embd: state.len(),
        };
        let ok = unsafe {
            koharu_llama_sys::llama_batch_ext_set_embd_state(self.batch.as_ptr(), idx, embd)
        };
        if ok && self.ctx.requires_embd_state {
            self.next_embd_state += 1;
        }
        ok
    }

    /// Enables or disables output logits for entry `idx`.
    pub fn set_output_logits(&mut self, idx: i32, value: bool) -> bool {
        let ok = unsafe {
            koharu_llama_sys::llama_batch_ext_set_output_logits(self.batch.as_ptr(), idx, value)
        };
        if ok {
            if value {
                if !self.initialized_logits.contains(&idx) {
                    self.initialized_logits.push(idx);
                }
            } else {
                self.initialized_logits.retain(|l| l != &idx);
            }
        }
        ok
    }

    /// Decodes this batch and makes its requested logits available on success.
    pub fn decode(self) -> Result<(), DecodeError> {
        if self.ctx.requires_embd_state {
            if self.n_entries == 0 {
                return Err(DecodeError::NTokensZero);
            }
            if self.next_embd_state != self.n_entries {
                return Err(DecodeError::MissingEmbdState);
            }
        }
        self.ctx.initialized_logits.clear();
        crate::clear_allocation_failure();
        let result = unsafe {
            koharu_llama_sys::llama_process(
                self.ctx.context.as_ptr(),
                koharu_llama_sys::LLAMA_PROCESS_TYPE_DECODE,
                self.batch.as_ptr(),
            )
        };
        if let Some(error) = NonZeroI32::new(result) {
            return Err(DecodeError::from(error));
        }
        self.ctx.initialized_logits = self.initialized_logits;
        Ok(())
    }
}
