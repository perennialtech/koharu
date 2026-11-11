use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::{
    Hardware, Store, download,
    runtime::{
        DiscoverablePackage, Package, RuntimePackage, graph::Component, loader, packages::Cuda,
        sealed,
    },
    source::extract,
};

const RELEASE: &str = "b11642";
const ASSET: &str = "x86_64-unknown-linux-gnu-cuda.tar.gz";
const LIBRARIES: [&str; 2] = ["libllama.so", "libmtmd.so"];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, strum::Display)]
pub(crate) enum Llama {
    #[strum(serialize = "linux-cuda")]
    LinuxCuda,
}

impl Llama {
    fn complete(self, path: &Path) -> bool {
        LIBRARIES.iter().all(|name| path.join(name).is_file())
    }
}

impl sealed::Sealed for Llama {}

impl Package for Llama {
    async fn install(self) -> Result<PathBuf> {
        let target = Store::root()
            .join("llama")
            .join(RELEASE)
            .join(self.to_string());
        Store::directory(
            target,
            move |path| self.complete(path),
            move |stage| async move {
                let url = format!(
                    "https://github.com/koharu-rs/llama/releases/download/{RELEASE}/{ASSET}"
                );
                let archive = tempfile::Builder::new().suffix(".tar.gz").tempfile()?;
                download::fetch(&url, archive.path()).await?;
                extract(archive.path(), &stage, &["**/*.so", "**/*.so.*"])
            },
        )
        .await
    }
}

impl DiscoverablePackage for Llama {
    fn discover(hardware: &Hardware) -> Option<Self> {
        hardware.supports_cuda().then_some(Self::LinuxCuda)
    }
}

impl RuntimePackage for Llama {
    const NAME: &'static str = "llama";

    fn dependencies(self, _hardware: &Hardware) -> Result<Vec<Component>> {
        Ok(vec![
            Component::Cuda(Cuda::Runtime13),
            Component::Cuda(Cuda::Blas13),
        ])
    }

    async fn activate(self) -> Result<()> {
        let root = self.install().await?;
        for library in LIBRARIES {
            loader::load(root.join(library), false)
                .with_context(|| format!("failed to activate llama library {library}"))?;
        }
        Ok(())
    }
}
