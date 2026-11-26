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

const RELEASE: &str = "1.31.1";
const PACKAGE: &str = "microsoft.ml.onnxruntime.gpu.linux";
const ROOT: &str = "runtimes/linux-x64/native";
const LIBRARY: &str = "libonnxruntime.so";
// ONNX Runtime loads its execution providers from the directory that holds
// the core library, so they are extracted beside it but never loaded here.
const LIBRARIES: [&str; 3] = [
    LIBRARY,
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_cuda.so",
];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, strum::Display)]
pub enum OnnxRuntime {
    #[strum(serialize = "cuda13")]
    Cuda13,
}

impl OnnxRuntime {
    /// Installs the package if needed and returns the core library for
    /// `ort::init_from`.
    pub async fn library(self) -> Result<PathBuf> {
        Ok(self.install().await?.join(ROOT).join(LIBRARY))
    }

    fn complete(self, path: &Path) -> bool {
        LIBRARIES
            .iter()
            .all(|name| path.join(ROOT).join(name).is_file())
    }
}

impl sealed::Sealed for OnnxRuntime {}

impl Package for OnnxRuntime {
    async fn install(self) -> Result<PathBuf> {
        let target = Store::root()
            .join("onnxruntime")
            .join(RELEASE)
            .join(self.to_string());
        Store::directory(
            target,
            move |path| self.complete(path),
            move |stage| async move {
                let url = format!(
                    "https://api.nuget.org/v3-flatcontainer/{PACKAGE}/{RELEASE}/{PACKAGE}.{RELEASE}.nupkg"
                );
                let archive = tempfile::Builder::new().suffix(".zip").tempfile()?;
                download::fetch(&url, archive.path()).await?;
                let patterns = LIBRARIES.map(|name| format!("{ROOT}/{name}"));
                let patterns = patterns.iter().map(String::as_str).collect::<Vec<_>>();
                extract(archive.path(), &stage, &patterns)
            },
        )
        .await
    }
}

impl DiscoverablePackage for OnnxRuntime {
    fn discover(hardware: &Hardware) -> Option<Self> {
        hardware.supports_cuda().then_some(Self::Cuda13)
    }
}

impl RuntimePackage for OnnxRuntime {
    const NAME: &'static str = "ONNX Runtime";

    fn dependencies(self, _hardware: &Hardware) -> Result<Vec<Component>> {
        // The CUDA provider links cuBLAS, cuRAND and the CUDA runtime, and
        // opens cuDNN by soname when a session first uses it.
        let packages = [Cuda::Runtime13, Cuda::Blas13, Cuda::Rand10, Cuda::Dnn925];
        Ok(packages.into_iter().map(Component::Cuda).collect())
    }

    async fn activate(self) -> Result<()> {
        let library = self.library().await?;
        loader::load(&library, false)
            .with_context(|| format!("failed to activate {}", library.display()))
    }
}
