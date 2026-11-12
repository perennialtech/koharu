use std::path::PathBuf;

use anyhow::Result;

use crate::{
    Hardware, Store, download,
    runtime::{
        DiscoverablePackage, Package, RuntimePackage, graph::Component, loader, packages::Cuda,
        sealed,
    },
    source::extract,
};

const RELEASE: &str = "koharu-c72cfa8";

// perennialtech/diffusion only builds Linux x86_64 CUDA for sm_86, the
// RTX 3080 Koharu runs on. Other platforms have no diffusion runtime.
const ASSET: &str = "x86_64-unknown-linux-gnu-cuda.tar.gz";
const LIBRARY: &str = "libstable-diffusion.so";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, strum::Display)]
pub(crate) enum Diffusion {
    #[strum(serialize = "linux-cuda")]
    LinuxCuda,
}

impl sealed::Sealed for Diffusion {}

impl Package for Diffusion {
    async fn install(self) -> Result<PathBuf> {
        let target = Store::root()
            .join("diffusion")
            .join(RELEASE)
            .join(self.to_string());
        Store::directory(
            target,
            move |path| path.join(LIBRARY).is_file(),
            move |stage| async move {
                let url = format!(
                    "https://github.com/perennialtech/diffusion/releases/download/{RELEASE}/{ASSET}"
                );
                let archive = tempfile::Builder::new().suffix(".tar.gz").tempfile()?;
                download::fetch(&url, archive.path()).await?;
                extract(archive.path(), &stage, &["**/*.so", "**/*.so.*"])
            },
        )
        .await
    }
}

impl DiscoverablePackage for Diffusion {
    fn discover(hardware: &Hardware) -> Option<Self> {
        // The packaged binary contains SM86 native kernels and no PTX fallback.
        // CUDA 8.x devices with capability 8.6 or newer can run those kernels.
        hardware.device().and_then(|device| {
            (hardware.supports_cuda() && (86..90).contains(&device.compute_capability()))
                .then_some(Self::LinuxCuda)
        })
    }
}

impl RuntimePackage for Diffusion {
    const NAME: &'static str = "diffusion";

    fn dependencies(self, _hardware: &Hardware) -> Result<Vec<Component>> {
        Ok(vec![
            Component::Cuda(Cuda::Runtime13),
            Component::Cuda(Cuda::Blas13),
        ])
    }

    async fn activate(self) -> Result<()> {
        let root = self.install().await?;
        loader::load(root.join(LIBRARY), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_only_cuda_devices_compatible_with_the_sm86_package() {
        for (capability, supported) in [
            (75, false),
            (80, false),
            (86, true),
            (89, true),
            (90, false),
        ] {
            let mut device = crate::Device::cuda(0);
            device.compute_capability = capability;
            let hardware = Hardware {
                devices: vec![device],
                candidates: vec![0],
                selected: Some(0),
            };
            assert_eq!(
                Diffusion::discover(&hardware).is_some(),
                supported,
                "SM{capability}"
            );
        }
    }
}
