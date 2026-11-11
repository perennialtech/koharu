use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::{
    Hardware, Store, download,
    runtime::{
        DiscoverablePackage, Package, RuntimePackage, graph::Component, loader, packages::Cuda,
        sealed,
    },
    source::extract,
};

const RELEASE: &str = "v2.13.0.7";
const ASSET: &str = "x86_64-unknown-linux-gnu-cuda.tar.gz";
const LIBRARIES: &str = "libc10.so,libc10_cuda.so,libcaffe2_nvrtc.so,libtorch_global_deps.so,libtorch_cpu.so,libtorch_cuda.so,libtorch.so,libkoharu-torch.so";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, strum::Display)]
pub enum Torch {
    #[strum(serialize = "cuda")]
    Cuda,
}

impl Torch {
    pub fn library_names(self) -> impl Iterator<Item = &'static str> {
        LIBRARIES.split(',')
    }

    fn complete(self, root: &Path) -> bool {
        self.library_names().all(|name| root.join(name).is_file())
    }
}

impl sealed::Sealed for Torch {}

impl Package for Torch {
    async fn install(self) -> Result<PathBuf> {
        let path = Store::root()
            .join("torch")
            .join(RELEASE)
            .join(self.to_string());

        Store::directory(
            path,
            move |path| self.complete(path),
            move |stage| async move {
                let url = format!(
                    "https://github.com/koharu-rs/torch/releases/download/{RELEASE}/{ASSET}"
                );
                let archive = tempfile::Builder::new().suffix(".tar.gz").tempfile()?;
                download::fetch(&url, archive.path()).await?;
                extract(archive.path(), &stage, &["**/*.so", "**/*.so.*"])
            },
        )
        .await
    }
}

impl DiscoverablePackage for Torch {
    fn discover(hardware: &Hardware) -> Option<Self> {
        hardware.supports_cuda().then_some(Self::Cuda)
    }
}

impl RuntimePackage for Torch {
    const NAME: &'static str = "Torch";

    fn dependencies(self, _hardware: &Hardware) -> Result<Vec<Component>> {
        let packages = [
            Cuda::Runtime13,
            Cuda::JitLink13,
            Cuda::Rtc13,
            Cuda::Blas13,
            Cuda::Fft12,
            Cuda::Rand10,
            Cuda::Sparse12,
            Cuda::Solver12,
            Cuda::Dnn925,
        ];
        Ok(packages.into_iter().map(Component::Cuda).collect())
    }

    async fn activate(self) -> Result<()> {
        let directory = self.install().await?;
        for library in self.library_names() {
            loader::load(directory.join(library), false)?;
        }
        Ok(())
    }
}
