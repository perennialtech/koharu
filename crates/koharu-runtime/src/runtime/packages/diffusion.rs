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
        (cfg!(all(target_os = "linux", target_arch = "x86_64")) && hardware.supports_cuda())
            .then_some(Self::LinuxCuda)
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
