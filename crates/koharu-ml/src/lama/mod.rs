//! LaMa inference with IOPaint-compatible orchestration.

mod config;
mod model;
mod processor;

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, RgbImage};

pub use self::config::{HDStrategy, InpaintRequest};
use self::model::Model;

// ONNX export of mayocream/lama-manga@f91c85b26913b3e83f9877867b4c336da3675238
// (lama-manga.safetensors); the repository holds the export script.
crate::model_repository!("perennialtech/koharu-onnx" @ "0000000000000000000000000000000000000000" {
    MODEL = "lama-manga/lama-manga.onnx"
});

#[derive(Debug)]
pub struct LaMa {
    model: Model,
}

impl LaMa {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let path = MODEL
            .resolve()
            .await
            .context("failed to resolve LaMa model")?;
        let model = tokio_rayon::spawn(move || Model::load(&device, &path))
            .await
            .context("failed to load LaMa")?;
        Ok(Self { model })
    }

    pub fn inference(
        &mut self,
        image: &DynamicImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        processor::inpaint(&mut self.model, image, mask, config)
    }
}
