use anyhow::Result;
use koharu_runtime::{Backend, Device};
use koharu_torch::{Cuda, Kind, nn};

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
