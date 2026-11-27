use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::{
    Hardware, Store,
    runtime::{
        DiscoverablePackage, Package, RuntimePackage, graph::Component, loader, packages::Cuda,
        sealed,
    },
    source::{Wheel, install_archive},
};

const WHEEL: Wheel = Wheel {
    project: "tensorrt-rtx-cu13-libs",
    version: "1.7.1.116",
    platform: "manylinux_2_28_x86_64",
};
const ROOT: &str = "tensorrt_rtx_libs";
// The ONNX parser opens the core library by soname, so the core loads first.
const LIBRARIES: [&str; 2] = ["libtensorrt_rtx.so.1", "libtensorrt_onnxparser_rtx.so.1"];

/// TensorRT-RTX. It links the CUDA runtime statically and needs only the
/// driver; its callers allocate buffers and streams through the CUDA runtime
/// package.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, strum::Display)]
pub enum TensorRt {
    #[strum(serialize = "rtx-cu13")]
    RtxCuda13,
}

impl TensorRt {
    fn complete(self, path: &Path) -> bool {
        LIBRARIES
            .iter()
            .all(|name| path.join(ROOT).join(name).is_file())
    }
}

impl sealed::Sealed for TensorRt {}

impl Package for TensorRt {
    async fn install(self) -> Result<PathBuf> {
        let target = Store::root().join("tensorrt").join(WHEEL.directory());
        Store::directory(
            target,
            move |path| self.complete(path),
            move |stage| async move {
                let url = WHEEL.url();
                let patterns = LIBRARIES.map(|name| format!("{ROOT}/{name}"));
                install_archive(&url, stage, patterns).await
            },
        )
        .await
    }
}

impl DiscoverablePackage for TensorRt {
    fn discover(hardware: &Hardware) -> Option<Self> {
        hardware.supports_cuda().then_some(Self::RtxCuda13)
    }
}

impl RuntimePackage for TensorRt {
    const NAME: &'static str = "TensorRT";

    fn dependencies(self, _hardware: &Hardware) -> Result<Vec<Component>> {
        Ok(vec![Component::Cuda(Cuda::Runtime13)])
    }

    async fn activate(self) -> Result<()> {
        let directory = self.install().await?.join(ROOT);
        for library in LIBRARIES {
            loader::load(directory.join(library), false)?;
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
            WHEEL.url(),
            "https://files.pythonhosted.org/packages/py3/t/tensorrt-rtx-cu13-libs/tensorrt_rtx_cu13_libs-1.7.1.116-py3-none-manylinux_2_28_x86_64.whl"
        );
    }
}
