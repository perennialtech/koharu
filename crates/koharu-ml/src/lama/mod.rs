//! LaMa inference with IOPaint-compatible orchestration.

mod config;
mod model;
mod processor;
mod tensorrt;

use std::path::PathBuf;

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_runtime::Backend;
use koharu_torch::Device;

use crate::backend::TryIntoDevice;

pub use self::config::{HDStrategy, InpaintRequest};
use self::{config::FFCResNetGeneratorConfig, model::Model, processor::InpaintModel};

crate::model_repository!("mayocream/lama-manga" @ "f91c85b26913b3e83f9877867b4c336da3675238" {
    WEIGHTS = "lama-manga.safetensors"
});

/// Prototype only: a directory with the fixed-shape TensorRT-RTX exports
/// (`lama-manga-<size>.onnx`). Shipping this needs the exports published as
/// pinned files of the model repository.
const TENSORRT_ONNX: &str = "KOHARU_LAMA_TENSORRT_ONNX";

#[derive(Debug)]
pub struct LaMa {
    network: Network,
}

#[derive(Debug)]
enum Network {
    Torch {
        model: Model,
        processor: InpaintModel,
    },
    TensorRt(tensorrt::Network),
}

impl LaMa {
    pub async fn load(device: crate::Device) -> Result<Self> {
        // The TensorRT exports are BF16, which needs Ampere or newer.
        if let Some(directory) = std::env::var_os(TENSORRT_ONNX).map(PathBuf::from)
            && device.backend == Backend::Cuda
            && device.compute_capability() >= 80
        {
            let network = tensorrt::Network::load(&directory, u32::try_from(device.index)?)
                .context("failed to load LaMa on TensorRT-RTX")?;
            return Ok(Self {
                network: Network::TensorRt(network),
            });
        }
        let device: Device = device.try_into_device()?;
        let weights_path = WEIGHTS
            .resolve()
            .await
            .context("failed to resolve LaMa weights")?;
        let mut model = Model::new(&FFCResNetGeneratorConfig::default(), device);
        model
            .load(&weights_path)
            .context("failed to load LaMa safetensors")?;
        Ok(Self {
            network: Network::Torch {
                model,
                processor: InpaintModel::new(device),
            },
        })
    }

    /// The crop size the network runs at for a `width` x `height` crop, when
    /// it runs at fixed sizes. A caller with more image around the crop can
    /// grow the crop to it; otherwise the crop is mirrored out to it.
    pub fn crop_size(&self, width: u32, height: u32) -> Option<u32> {
        match &self.network {
            Network::Torch { .. } => None,
            Network::TensorRt(_) => tensorrt::crop_size(width, height),
        }
    }

    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        match &self.network {
            Network::Torch { model, processor } => koharu_torch::no_grad(|| {
                processor::inpaint(image, mask, config, |image, mask| {
                    processor.pad_forward(model, image, mask, config)
                })
            }),
            Network::TensorRt(network) => processor::inpaint(image, mask, config, |image, mask| {
                network.forward(image, mask, config.sd_keep_unmasked_area)
            }),
        }
    }
}
