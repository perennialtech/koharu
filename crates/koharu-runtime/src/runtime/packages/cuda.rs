use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use strum::EnumProperty;
use walkdir::WalkDir;

use crate::{
    Hardware, Store, download,
    runtime::{Package, RuntimePackage, graph::Component, loader, sealed},
    source::{extract, wheel},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, strum::Display, strum::EnumProperty)]
pub(crate) enum Cuda {
    #[strum(
        serialize = "runtime-13",
        props(project = "nvidia-cuda-runtime/13.3.29", libraries = "libcudart.so.13")
    )]
    Runtime13,
    #[strum(
        serialize = "cublas-13",
        props(
            project = "nvidia-cublas/13.6.0.2",
            libraries = "libcublasLt.so.13,libcublas.so.13"
        )
    )]
    Blas13,
    #[strum(
        serialize = "cufft-12",
        props(project = "nvidia-cufft/12.3.0.29", libraries = "libcufft.so.12")
    )]
    Fft12,
    #[strum(
        serialize = "curand-10",
        props(project = "nvidia-curand/10.4.3.29", libraries = "libcurand.so.10")
    )]
    Rand10,
    #[strum(
        serialize = "cudnn-9.25",
        props(
            project = "nvidia-cudnn-cu13/9.25.1.1",
            libraries = "libcudnn.so.9,libcudnn_adv.so.9,libcudnn_cnn.so.9,libcudnn_engines_precompiled.so.9,libcudnn_engines_runtime_compiled.so.9,libcudnn_engines_tensor_ir.so.9,libcudnn_ext.so.9,libcudnn_graph.so.9,libcudnn_heuristic.so.9,libcudnn_ops.so.9"
        )
    )]
    Dnn925,
    #[strum(
        serialize = "nvrtc-13",
        props(
            project = "nvidia-cuda-nvrtc/13.3.33",
            libraries = "libnvrtc-builtins.so.13.3,libnvrtc.so.13"
        )
    )]
    Rtc13,
    #[strum(
        serialize = "nvjitlink-13",
        props(project = "nvidia-nvjitlink/13.3.33", libraries = "libnvJitLink.so.13")
    )]
    JitLink13,
}

impl Cuda {
    fn libraries(self) -> impl Iterator<Item = &'static str> {
        self.get_str("libraries")
            .expect("CUDA package has libraries")
            .split(',')
    }

    fn library_paths(self, root: &Path) -> Result<Vec<PathBuf>> {
        let names = self.libraries().collect::<Vec<_>>();
        let mut paths = vec![None; names.len()];
        for entry in WalkDir::new(root) {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            if let Some(index) = names
                .iter()
                .position(|name| entry.file_name() == std::ffi::OsStr::new(name))
            {
                paths[index] = Some(entry.into_path());
            }
        }
        names
            .into_iter()
            .zip(paths)
            .map(|(name, path)| path.with_context(|| format!("missing CUDA library {name}")))
            .collect()
    }
}

impl sealed::Sealed for Cuda {}

impl Package for Cuda {
    async fn install(self) -> Result<PathBuf> {
        let project = self.get_str("project").expect("CUDA package has a project");
        let target = Store::root().join("cuda").join(project.replace('/', "--"));
        Store::directory(
            target,
            move |path| self.library_paths(path).is_ok(),
            move |stage| async move {
                let url = wheel(project).await?;
                let archive = tempfile::Builder::new().suffix(".whl").tempfile()?;
                download::fetch(&url, archive.path()).await?;
                extract(archive.path(), &stage, &["**/*.so", "**/*.so.*"])
            },
        )
        .await
    }
}

impl RuntimePackage for Cuda {
    const NAME: &'static str = "CUDA";

    fn dependencies(self, _hardware: &Hardware) -> Result<Vec<Component>> {
        let packages = match self {
            Self::Blas13 => vec![Self::Rtc13],
            Self::Fft12 => vec![Self::JitLink13],
            Self::Dnn925 => vec![Self::Blas13],
            Self::Runtime13 | Self::Rand10 | Self::Rtc13 | Self::JitLink13 => Vec::new(),
        };
        Ok(packages.into_iter().map(Component::Cuda).collect())
    }

    async fn activate(self) -> Result<()> {
        let directory = self.install().await?;
        for library in self.library_paths(&directory)? {
            loader::load(library, false)?;
        }
        Ok(())
    }
}
