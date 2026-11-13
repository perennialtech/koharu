//! Comic speech-bubble and text detection with ogkalu's RT-DETRv2 checkpoint.
//!
//! Checkpoint (Apache-2.0):
//! https://huggingface.co/ogkalu/comic-text-and-bubble-detector/tree/16e8a622f91fabc6b5b65c96d32d1183f8843546
//! The model is box-only; callers that need masks derive them.

mod model;
mod processor;

use anyhow::{Context, Result};
use image::DynamicImage;

pub use self::processor::{ComicTextBubbleDetection, ComicTextBubbleLabel};

use self::model::Model;

// ONNX export of the checkpoint's model.safetensors; the repository holds the
// export script.
crate::model_repository!("8q0sb/koharu-onnx" @ "fda0b1ad6a31da8887f86bbeaa77ab47da2aac70" {
    MODEL = "comic-text-and-bubble-detector/comic-text-and-bubble-detector.onnx",
});

#[derive(Debug)]
pub struct ComicTextBubbleDetector {
    model: Model,
}

impl ComicTextBubbleDetector {
    /// Bubble recall on the hard-page set is flat between 0.2 and 0.6; 0.5
    /// keeps the fewest false positives at that recall.
    pub const RECOMMENDED_BUBBLE_THRESHOLD: f32 = 0.5;

    pub async fn load(device: crate::Device) -> Result<Self> {
        let path = MODEL
            .resolve()
            .await
            .context("failed to resolve comic text and bubble detector model")?;
        let model = tokio_rayon::spawn(move || Model::load(&device, &path))
            .await
            .context("failed to load comic text and bubble detector")?;
        Ok(Self { model })
    }

    /// Returns every detection scoring above `threshold`, best first.
    pub fn inference(
        &mut self,
        image: &DynamicImage,
        threshold: f32,
    ) -> Result<Vec<ComicTextBubbleDetection>> {
        let pixel_values = processor::preprocess(image)?;
        let output = self.model.forward(&pixel_values)?;
        processor::postprocess(&output, image.width(), image.height(), threshold)
    }
}
