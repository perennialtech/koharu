//! Comic speech-bubble and text detection with ogkalu's RT-DETRv2 checkpoint.
//!
//! Checkpoint (Apache-2.0):
//! https://huggingface.co/ogkalu/comic-text-and-bubble-detector/tree/16e8a622f91fabc6b5b65c96d32d1183f8843546
//! The model is box-only; callers that need masks derive them.

mod model;
mod processor;

use anyhow::{Context, Result};
use image::DynamicImage;
use koharu_torch::Device;

use crate::backend::TryIntoDevice;

pub use self::processor::{ComicTextBubbleDetection, ComicTextBubbleLabel};

use self::model::Model;

crate::model_repository!("ogkalu/comic-text-and-bubble-detector" @ "16e8a622f91fabc6b5b65c96d32d1183f8843546" {
    WEIGHTS = "model.safetensors",
});

#[derive(Debug)]
pub struct ComicTextBubbleDetector {
    device: Device,
    model: Model,
}

impl ComicTextBubbleDetector {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let weights_path = WEIGHTS
            .resolve()
            .await
            .context("failed to resolve comic text and bubble detector weights")?;
        let mut model = Model::new(device);
        model
            .load(&weights_path)
            .with_context(|| format!("failed to load {}", weights_path.display()))?;
        Ok(Self { device, model })
    }

    /// Returns every detection scoring above `threshold`, best first.
    pub fn inference(
        &self,
        image: &DynamicImage,
        threshold: f32,
    ) -> Result<Vec<ComicTextBubbleDetection>> {
        koharu_torch::no_grad(|| {
            let pixel_values = processor::preprocess(image, self.device)?;
            let output = self.model.forward(&pixel_values);
            processor::postprocess(&output, image.width(), image.height(), threshold)
        })
    }
}
