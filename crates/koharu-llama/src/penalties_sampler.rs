//! Repetition penalties applied to the tokens in the penalty window.
//!
//! Mirrors llama.cpp's `penalties` sampler:
//! https://github.com/ggml-org/llama.cpp/blob/b11642/src/llama-sampler.cpp#L2925-L2986
//! The upstream sampler looks up every candidate in its token counts on every
//! step, which costs a hash lookup per vocabulary entry (262k for Gemma 4).
//! This one visits only the windowed tokens. A candidate array built from a
//! context's logits holds token `i` at index `i`, so each token is found
//! directly; any other arrangement is searched. The penalty arithmetic is
//! unchanged, so sampled tokens are identical.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;

use crate::sampling::LlamaSampler;

#[derive(Clone)]
struct Penalties {
    last_n: usize,
    repeat: f32,
    frequency: f32,
    presence: f32,
    window: VecDeque<i32>,
    counts: HashMap<i32, i32>,
}

impl Penalties {
    fn disabled(&self) -> bool {
        self.last_n == 0 || (self.repeat == 1.0 && self.frequency == 0.0 && self.presence == 0.0)
    }

    fn accept(&mut self, token: i32) {
        if self.last_n == 0 {
            return;
        }
        *self.counts.entry(token).or_default() += 1;
        if self.window.len() >= self.last_n
            && let Some(old) = self.window.pop_front()
            && let Some(count) = self.counts.get_mut(&old)
        {
            *count -= 1;
            if *count == 0 {
                self.counts.remove(&old);
            }
        }
        self.window.push_back(token);
    }

    fn apply(&self, candidates: &mut [koharu_llama_sys::llama_token_data]) {
        if self.disabled() {
            return;
        }
        for (&token, &count) in &self.counts {
            let index = usize::try_from(token)
                .ok()
                .filter(|&index| candidates.get(index).is_some_and(|data| data.id == token))
                .or_else(|| candidates.iter().position(|data| data.id == token));
            let Some(candidate) = index.map(|index| &mut candidates[index]) else {
                continue;
            };
            if candidate.logit <= 0.0 {
                candidate.logit *= self.repeat;
            } else {
                candidate.logit /= self.repeat;
            }
            candidate.logit -=
                count as f32 * self.frequency + f32::from(u8::from(count > 0)) * self.presence;
        }
    }
}

unsafe extern "C" fn name(
    _smpl: *const koharu_llama_sys::llama_sampler,
) -> *const std::os::raw::c_char {
    c"penalties".as_ptr()
}

unsafe extern "C" fn accept(
    smpl: *mut koharu_llama_sys::llama_sampler,
    token: koharu_llama_sys::llama_token,
) {
    let penalties = unsafe { &mut *(*smpl).ctx.cast::<Penalties>() };
    penalties.accept(token);
}

unsafe extern "C" fn apply(
    smpl: *mut koharu_llama_sys::llama_sampler,
    cur_p: *mut koharu_llama_sys::llama_token_data_array,
) {
    let penalties = unsafe { &*(*smpl).ctx.cast::<Penalties>() };
    let cur_p = unsafe { &mut *cur_p };
    if penalties.disabled() {
        return;
    }
    let candidates = unsafe { std::slice::from_raw_parts_mut(cur_p.data, cur_p.size) };
    penalties.apply(candidates);
    cur_p.sorted = false;
}

unsafe extern "C" fn reset(smpl: *mut koharu_llama_sys::llama_sampler) {
    let penalties = unsafe { &mut *(*smpl).ctx.cast::<Penalties>() };
    penalties.window.clear();
    penalties.counts.clear();
}

unsafe extern "C" fn clone(
    smpl: *const koharu_llama_sys::llama_sampler,
) -> *mut koharu_llama_sys::llama_sampler {
    let penalties = unsafe { &*(*smpl).ctx.cast::<Penalties>() };
    init(penalties.clone())
}

unsafe extern "C" fn free(smpl: *mut koharu_llama_sys::llama_sampler) {
    let penalties = unsafe { (*smpl).ctx.cast::<Penalties>() };
    if !penalties.is_null() {
        drop(unsafe { Box::from_raw(penalties) });
    }
}

static mut PENALTIES_I: koharu_llama_sys::llama_sampler_i = koharu_llama_sys::llama_sampler_i {
    name: Some(name),
    accept: Some(accept),
    apply: Some(apply),
    reset: Some(reset),
    clone: Some(clone),
    free: Some(free),
    backend_init: None,
    backend_accept: None,
    backend_apply: None,
    backend_set_input: None,
    backend_reset: None,
    copy_state: None,
};

fn init(penalties: Penalties) -> *mut koharu_llama_sys::llama_sampler {
    unsafe {
        koharu_llama_sys::llama_sampler_init(
            &raw mut PENALTIES_I,
            Box::into_raw(Box::new(penalties)).cast::<c_void>(),
        )
    }
}

pub(crate) fn penalties(last_n: i32, repeat: f32, frequency: f32, presence: f32) -> LlamaSampler {
    let last_n = usize::try_from(last_n).unwrap_or(0);
    LlamaSampler {
        sampler: init(Penalties {
            last_n,
            repeat,
            frequency,
            presence,
            window: VecDeque::with_capacity(last_n),
            counts: HashMap::new(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates(logits: &[f32]) -> Vec<koharu_llama_sys::llama_token_data> {
        logits
            .iter()
            .enumerate()
            .map(|(id, &logit)| koharu_llama_sys::llama_token_data {
                id: id as i32,
                logit,
                p: 0.0,
            })
            .collect()
    }

    #[test]
    fn penalizes_only_tokens_in_the_window() {
        let mut penalties = Penalties {
            last_n: 2,
            repeat: 2.0,
            frequency: 0.5,
            presence: 0.25,
            window: VecDeque::new(),
            counts: HashMap::new(),
        };
        for token in [0, 1, 1] {
            penalties.accept(token);
        }
        let mut data = candidates(&[4.0, 4.0, -4.0]);
        penalties.apply(&mut data);
        assert_eq!(data[0].logit, 4.0);
        assert_eq!(data[1].logit, 4.0 / 2.0 - (2.0 * 0.5 + 0.25));
        assert_eq!(data[2].logit, -4.0);

        penalties.accept(2);
        let mut data = candidates(&[4.0, 4.0, -4.0]);
        data.reverse();
        penalties.apply(&mut data);
        assert_eq!(data[0].logit, -4.0 * 2.0 - (0.5 + 0.25));
        assert_eq!(data[1].logit, 4.0 / 2.0 - (0.5 + 0.25));
        assert_eq!(data[2].logit, 4.0);
    }
}
