use serde::{Deserialize, Serialize};

/// Compute backend used by a machine-learning device.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Hash, Serialize, strum::Display)]
pub enum Backend {
    #[strum(serialize = "CPU")]
    Cpu,
    #[strum(serialize = "CUDA")]
    Cuda,
}

/// Broad device category shared by native model backends.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Hash, Serialize)]
pub enum DeviceType {
    Cpu,
    Gpu,
    IntegratedGpu,
}

/// The process-wide device selected for ONNX Runtime, llama.cpp, and stable-diffusion.cpp.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Hash, Serialize)]
pub struct Device {
    pub index: usize,
    pub name: String,
    pub description: String,
    pub backend: Backend,
    pub device_type: DeviceType,
    pub memory_total: usize,
    pub memory_free: usize,
    #[serde(skip)]
    pub(crate) compute_capability: u32,
}

impl Device {
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            index: 0,
            name: "CPU".to_owned(),
            description: "CPU".to_owned(),
            backend: Backend::Cpu,
            device_type: DeviceType::Cpu,
            memory_total: 0,
            memory_free: 0,
            compute_capability: 0,
        }
    }

    #[must_use]
    pub fn cuda(index: usize) -> Self {
        let name = format!("CUDA{index}");
        Self {
            index,
            description: name.clone(),
            name,
            backend: Backend::Cuda,
            device_type: DeviceType::Gpu,
            memory_total: 0,
            memory_free: 0,
            compute_capability: 0,
        }
    }

    #[must_use]
    pub fn is_integrated(&self) -> bool {
        self.device_type == DeviceType::IntegratedGpu
    }

    #[must_use]
    pub fn compute_capability(&self) -> u32 {
        self.compute_capability
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}
