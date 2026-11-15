//! Segmentation of censorship already in a page (bars and mosaics) with
//! kitsumed's YOLO11m-seg.
//!
//! Weights (AGPL-3.0), used as published:
//! https://huggingface.co/kitsumed/yolov11m_seg-2d-censorship-detection/tree/3acab0bd97658a9c63fcb4a577eec815e4b38641
//! Pre- and postprocessing follow the Ultralytics segmentation predictor
//! with `retina_masks=True`, at the release the evaluation ran:
//! https://github.com/ultralytics/ultralytics/tree/v8.4.211/ultralytics/models/yolo/segment/predict.py
//!
//! The model was trained at 640 px. It runs at 960 px with a confidence
//! threshold of 0.4, the operating point chosen on real censored art in
//! https://github.com/perennialtech/censor-detect-eval: at 640 thin bars and
//! fine mosaics on large pages shrink to a few pixels, while 1280 and up
//! raise false alarms on uncensored art.

mod network;
mod processor;

use std::path::Path;

use anyhow::{Context, Result, ensure};
use image::{GrayImage, Luma, RgbImage};

use self::{network::Network, processor::Letterbox};

crate::model_repository!("kitsumed/yolov11m_seg-2d-censorship-detection" @ "3acab0bd97658a9c63fcb4a577eec815e4b38641" {
    NETWORK = "model_fp16_dynamic.onnx",
});

/// The checkpoint's classes, in its class-index order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CensorKind {
    /// "Censor Bar"
    Bar,
    /// "Pixelation"
    Mosaic,
}

/// One censored region, in source-image pixels.
#[derive(Clone, Debug)]
pub struct Censor {
    pub kind: CensorKind,
    pub score: f32,
    /// `[left, top, right, bottom]`, clipped to the image.
    pub bbox: [f32; 4],
    /// The region's pixels (255) within the box's pixel span, whose top-left
    /// corner in the image is `mask_origin`.
    pub mask: GrayImage,
    pub mask_origin: (u32, u32),
}

impl Censor {
    /// Sets the region's pixels in `target`, an image the size of the source.
    pub fn paint(&self, target: &mut GrayImage) {
        let (left, top) = self.mask_origin;
        for (x, y, pixel) in self.mask.enumerate_pixels() {
            if pixel[0] != 0 {
                target.put_pixel(left + x, top + y, Luma([u8::MAX]));
            }
        }
    }
}

pub struct KitsumedYolo11mSeg {
    network: Network,
}

impl std::fmt::Debug for KitsumedYolo11mSeg {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KitsumedYolo11mSeg")
            .finish_non_exhaustive()
    }
}

impl KitsumedYolo11mSeg {
    /// Loads the network on a CUDA device; TensorRT-RTX must be initialized.
    pub async fn load(device: crate::Device) -> Result<Self> {
        ensure!(
            device.backend == crate::Backend::Cuda && crate::initialized(crate::Feature::TensorRt),
            "kitsumed-yolo11m-seg needs a CUDA device with TensorRT-RTX"
        );
        let network = NETWORK
            .resolve()
            .await
            .context("failed to resolve kitsumed-yolo11m-seg")?;
        let device = u32::try_from(device.index)?;
        let network = tokio_rayon::spawn(move || Network::load(device, Path::new(&network)))
            .await
            .context("failed to load kitsumed-yolo11m-seg on TensorRT-RTX")?;
        Ok(Self { network })
    }

    /// Censored regions above the confidence threshold, highest score first.
    pub fn inference(&mut self, image: &RgbImage) -> Result<Vec<Censor>> {
        ensure!(
            image.width() > 0 && image.height() > 0,
            "cannot segment an empty image"
        );
        let letterbox = Letterbox::new(image.width(), image.height());
        let input = letterbox.apply(image);
        let output = self.network.forward(&input, letterbox.input)?;
        Ok(processor::postprocess(&output, &letterbox))
    }
}
