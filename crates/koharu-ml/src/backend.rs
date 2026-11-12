use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use anyhow::Result;
use koharu_runtime::{Backend, Device};
use koharu_torch::{Cuda, Kind, TchError, nn};

pub(crate) trait TryIntoDevice<T> {
    fn try_into_device(self) -> Result<T>;
}

impl TryIntoDevice<koharu_torch::Device> for Device {
    fn try_into_device(self) -> Result<koharu_torch::Device> {
        match self.backend {
            Backend::Cpu => Ok(koharu_torch::Device::Cpu),
            Backend::Cuda => {
                Cuda::set_user_enabled_cudnn(true);
                Ok(koharu_torch::Device::Cuda(self.index))
            }
        }
    }
}

pub(crate) fn recover_torch_oom<T>(run: impl FnOnce() -> Result<T>) -> Result<T> {
    let panic = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(result) => return result,
        Err(panic) => panic,
    };
    // Infallible tensor wrappers unwrap TchError. Keep allocator failures on
    // the stage's error path without treating unrelated panics as recoverable.
    if let Some(error) = panic.downcast_ref::<String>().and_then(|message| {
        message
            .strip_prefix("called `Result::unwrap()` on an `Err` value: Torch(\"")?
            .strip_suffix("\")")
    }) {
        let lower = error.to_ascii_lowercase();
        if lower.contains("cuda out of memory")
            || lower.contains("cuda error: out of memory")
            || lower.contains("cuda_error_out_of_memory")
        {
            return Err(TchError::Torch(error.to_owned()).into());
        }
    }
    resume_unwind(panic)
}

pub(crate) fn set_precision(var_store: &mut nn::VarStore) {
    let use_bfloat16 = matches!(var_store.device(), koharu_torch::Device::Cuda(_))
        && crate::device(false).compute_capability() >= 80;

    if var_store.is_empty() {
        var_store.set_kind(if use_bfloat16 {
            Kind::BFloat16
        } else {
            Kind::Float
        });
    } else if use_bfloat16 {
        var_store.bfloat16();
    } else {
        var_store.float();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_oom_from_an_infallible_torch_operation_returns_an_error() {
        fn allocate() -> std::result::Result<(), TchError> {
            Err(TchError::Torch(
                "CUDA out of memory. Tried to allocate 38 MiB".into(),
            ))
        }

        let error = recover_torch_oom(|| {
            allocate().unwrap();
            Ok(())
        })
        .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<TchError>(),
            Some(TchError::Torch(message)) if message.starts_with("CUDA out of memory")
        ));
    }

    #[test]
    #[should_panic(expected = "unrelated panic")]
    fn unrelated_panic_keeps_unwinding() {
        let _: Result<()> = recover_torch_oom(|| panic!("unrelated panic"));
    }
}
