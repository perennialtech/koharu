mod cuda;

use std::{cmp::Reverse, sync::OnceLock};

use crate::{Backend, Device};

/// CUDA devices and their process-wide selection priority.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Hardware {
    pub(crate) devices: Vec<Device>,
    pub(crate) candidates: Vec<usize>,
    pub(crate) selected: Option<usize>,
}

impl Hardware {
    #[must_use]
    pub fn discover() -> Self {
        static HARDWARE: OnceLock<Hardware> = OnceLock::new();
        HARDWARE.get_or_init(Self::probe).clone()
    }

    fn probe() -> Self {
        let devices = cuda::probe();
        let mut candidates = (0..devices.len()).collect::<Vec<_>>();
        candidates.sort_by_key(|&position| {
            let device = &devices[position];
            (
                device.is_integrated(),
                Reverse(device.memory_total),
                Reverse(device.compute_capability),
                device.index,
            )
        });
        let selected = candidates.first().copied();
        Self {
            devices,
            candidates,
            selected,
        }
    }

    #[must_use]
    pub fn device(&self) -> Option<&Device> {
        self.selected.and_then(|index| self.devices.get(index))
    }

    pub(crate) fn candidates(&self) -> impl Iterator<Item = Self> + '_ {
        self.candidates.iter().copied().map(|selected| Self {
            selected: Some(selected),
            ..self.clone()
        })
    }

    #[must_use]
    pub fn supports_cuda(&self) -> bool {
        matches!(self.device(), Some(device) if device.backend == Backend::Cuda)
    }
}
