use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use strum::EnumProperty;
use walkdir::WalkDir;

use crate::{
    Hardware, Store,
    runtime::{Package, RuntimePackage, graph::Component, loader, sealed},
    source::{SHARED_LIBRARIES, Wheel, install_archive},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, strum::Display, strum::EnumProperty)]
pub(crate) enum Cuda {
    #[strum(
        serialize = "runtime-13",
        props(
            project = "nvidia-cuda-runtime",
            version = "13.3.29",
            platform = "manylinux2014_x86_64.manylinux_2_17_x86_64",
            libraries = "libcudart.so.13"
        )
    )]
    Runtime13,
    #[strum(
        serialize = "cublas-13",
        props(
            project = "nvidia-cublas",
            version = "13.6.0.2",
            platform = "manylinux_2_27_x86_64",
            libraries = "libcublasLt.so.13,libcublas.so.13"
        )
    )]
    Blas13,
    #[strum(
        serialize = "cufft-12",
        props(
            project = "nvidia-cufft",
            version = "12.3.0.29",
            platform = "manylinux2014_x86_64.manylinux_2_17_x86_64",
            libraries = "libcufft.so.12"
        )
    )]
    Fft12,
    #[strum(
        serialize = "curand-10",
        props(
            project = "nvidia-curand",
            version = "10.4.3.29",
            platform = "manylinux_2_27_x86_64",
            libraries = "libcurand.so.10"
        )
    )]
    Rand10,
    #[strum(
        serialize = "cudnn-9.25",
        props(
            project = "nvidia-cudnn-cu13",
            version = "9.25.1.1",
            platform = "manylinux_2_27_x86_64",
            libraries = "libcudnn.so.9,libcudnn_adv.so.9,libcudnn_cnn.so.9,libcudnn_engines_precompiled.so.9,libcudnn_engines_runtime_compiled.so.9,libcudnn_engines_tensor_ir.so.9,libcudnn_ext.so.9,libcudnn_graph.so.9,libcudnn_heuristic.so.9,libcudnn_ops.so.9"
        )
    )]
    Dnn925,
    #[strum(
        serialize = "nvrtc-13",
        props(
            project = "nvidia-cuda-nvrtc",
            version = "13.3.33",
            platform = "manylinux2010_x86_64.manylinux_2_12_x86_64",
            libraries = "libnvrtc-builtins.so.13.3,libnvrtc.so.13"
        )
    )]
    Rtc13,
    #[strum(
        serialize = "nvjitlink-13",
        props(
            project = "nvidia-nvjitlink",
            version = "13.3.33",
            platform = "manylinux2010_x86_64.manylinux_2_12_x86_64",
            libraries = "libnvJitLink.so.13"
        )
    )]
    JitLink13,
    #[strum(
        serialize = "cusparse-12",
        props(
            project = "nvidia-cusparse",
            version = "12.8.2.51",
            platform = "manylinux2014_x86_64.manylinux_2_17_x86_64",
            libraries = "libcusparse.so.12"
        )
    )]
    Sparse12,
    #[strum(
        serialize = "cusolver-12",
        props(
            project = "nvidia-cusolver",
            version = "12.2.6.9",
            platform = "manylinux_2_27_x86_64",
            libraries = "libcusolver.so.12,libcusolverMg.so.12"
        )
    )]
    Solver12,
}

impl Cuda {
    fn wheel(self) -> Wheel {
        Wheel {
            project: self.get_str("project").expect("CUDA wheel has a project"),
            version: self.get_str("version").expect("CUDA wheel has a version"),
            platform: self.get_str("platform").expect("CUDA wheel has a platform"),
        }
    }

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
        let wheel = self.wheel();
        let target = Store::root().join("cuda").join(wheel.directory());
        Store::directory(
            target,
            move |path| self.library_paths(path).is_ok(),
            move |stage| async move {
                let url = wheel.url();
                install_archive(&url, stage, SHARED_LIBRARIES).await
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
            Self::Fft12 | Self::Sparse12 => vec![Self::JitLink13],
            Self::Solver12 => vec![Self::JitLink13, Self::Blas13, Self::Sparse12],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_the_linux_wheel_url() {
        assert_eq!(
            Cuda::Rtc13.wheel().url(),
            "https://files.pythonhosted.org/packages/py3/n/nvidia-cuda-nvrtc/nvidia_cuda_nvrtc-13.3.33-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl"
        );
    }
}
