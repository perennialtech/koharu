//! High-resolution manga layout instance segmentation with RF-DETR Seg 2XL.
//!
//! Checkpoint and strict Python loader:
//! https://huggingface.co/mayocream/koharu-layout-highres/tree/68077eb09401a46e548190ff2786bd405113ea8b
//! RF-DETR upstream implementation:
//! https://github.com/roboflow/rf-detr/tree/4ab7c18729de9d02ffd0495795d0831b5630f01b

mod config;
mod model;
mod processor;
mod tensorrt;

use std::path::PathBuf;

use anyhow::{Context, Result};
use image::DynamicImage;
use koharu_runtime::Backend;

pub use self::{
    config::{KoharuLayoutRFDetrSeg2XLConfig, KoharuLayoutThresholds},
    processor::{
        KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask,
        KoharuLayoutRFDetrImageProcessor,
    },
};

use self::model::Model;

crate::model_repository!("mayocream/koharu-layout-highres" @ "68077eb09401a46e548190ff2786bd405113ea8b" {
    CONFIG = "config.json",
});

// ONNX export of the checkpoint above (model.safetensors); the repository holds
// the export script.
crate::model_repository!("8q0sb/koharu-onnx" @ "26af0c10ce9ced0774a2afbc8750493bf0cd42cd" {
    NETWORK = "koharu-layout-highres/koharu-layout-highres.onnx",
    MASK_HEAD = "koharu-layout-highres/koharu-layout-highres-mask-head.onnx",
});

/// Prototype only: the network export with standard ONNX Attention nodes
/// (export.py without its MultiHeadAttention rewrite) for TensorRT-RTX.
/// Shipping this needs the export pinned in the model repository above.
const TENSORRT_ONNX: &str = "KOHARU_RFDETR_TENSORRT_ONNX";

#[derive(Debug)]
pub struct KoharuLayoutRFDetrSeg2XL {
    model: Model,
    processor: KoharuLayoutRFDetrImageProcessor,
}

impl KoharuLayoutRFDetrSeg2XL {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let config_path = CONFIG
            .resolve()
            .await
            .context("failed to resolve KoharuLayout RF-DETR inference config")?;
        let network = NETWORK
            .resolve()
            .await
            .context("failed to resolve KoharuLayout RF-DETR network")?;
        let mask_head = MASK_HEAD
            .resolve()
            .await
            .context("failed to resolve KoharuLayout RF-DETR mask head")?;
        let config = KoharuLayoutRFDetrSeg2XLConfig::from_file(&config_path)?;
        let processor = KoharuLayoutRFDetrImageProcessor::new(&config)?;
        // TensorRT-RTX runs on Turing and newer.
        let tensorrt = std::env::var_os(TENSORRT_ONNX)
            .map(PathBuf::from)
            .filter(|_| device.backend == Backend::Cuda && device.compute_capability() >= 75);
        let model = match tensorrt {
            Some(network) => {
                let index = u32::try_from(device.index)?;
                tokio_rayon::spawn(move || Model::load_tensorrt(index, &network, &mask_head))
                    .await
                    .context("failed to load KoharuLayout RF-DETR on TensorRT-RTX")?
            }
            None => tokio_rayon::spawn(move || Model::load(&device, &network, &mask_head))
                .await
                .context("failed to load KoharuLayout RF-DETR")?,
        };
        Ok(Self { model, processor })
    }

    pub fn inference(&mut self, image: &DynamicImage) -> Result<KoharuLayoutDetections> {
        self.inference_with_thresholds(image, self.processor.recommended_thresholds())
    }

    pub fn inference_with_thresholds(
        &mut self,
        image: &DynamicImage,
        thresholds: KoharuLayoutThresholds,
    ) -> Result<KoharuLayoutDetections> {
        let start = std::time::Instant::now();
        let pixel_values = self.processor.preprocess(image)?;
        let preprocessed = start.elapsed();
        let output = self.model.forward(&pixel_values)?;
        let forwarded = start.elapsed();
        let detections = self.processor.postprocess(
            &mut self.model,
            &output,
            image.width(),
            image.height(),
            thresholds,
        );
        tracing::debug!(
            pre_ms = preprocessed.as_secs_f64() * 1e3,
            net_ms = (forwarded - preprocessed).as_secs_f64() * 1e3,
            post_ms = (start.elapsed() - forwarded).as_secs_f64() * 1e3,
            "rfdetr_split"
        );
        detections
    }

    pub fn recommended_thresholds(&self) -> KoharuLayoutThresholds {
        self.processor.recommended_thresholds()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyhow::Result;

    use super::KoharuLayoutRFDetrSeg2XL;

    #[tokio::test]
    #[ignore = "downloads the checkpoint and takes minutes on CPU"]
    async fn checkpoint_matches_rfdetr_upstream_structured_output() -> Result<()> {
        crate::init(&[crate::Feature::OnnxRuntime]).await?;
        let image = image::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("benches/fixtures/object_detection/1.jpg"),
        )?;
        let mut model = KoharuLayoutRFDetrSeg2XL::load(crate::Device::cpu()).await?;
        let result = model.inference(&image)?;

        // rfdetr 1.7.0, CPU FP32, shape=(1920, 1920), the checkpoint loaded strictly.
        // Both sides run FP32 on CPU. Kernels vary across runtimes and releases, so compare
        // structured geometry and mask area with bounded confidence differences.
        let best = &result.detections[0];
        assert_eq!(best.label, "bubble");
        assert!((best.score - 0.981_6).abs() < 0.01);
        for (actual, expected) in best.bbox.into_iter().zip([567.63, 546.78, 691.9, 720.57]) {
            assert!((actual - expected).abs() < 6.0);
        }
        assert!(best.area.abs_diff(18_900) < 100);

        let lower_panel = result
            .detections
            .iter()
            .find(|detection| detection.label == "panel" && detection.bbox[1] > 700.0)
            .expect("lower page panel");
        let middle_panel = result
            .detections
            .iter()
            .find(|detection| {
                detection.label == "panel" && detection.bbox[1] > 500.0 && detection.bbox[1] < 700.0
            })
            .expect("middle page panel");
        for (actual, expected) in lower_panel
            .bbox
            .into_iter()
            .zip([68.45, 799.89, 701.19, 1008.47])
        {
            assert!((actual - expected).abs() < 7.0);
        }
        for (actual, expected) in middle_panel
            .bbox
            .into_iter()
            .zip([69.46, 555.17, 698.4, 787.68])
        {
            assert!((actual - expected).abs() < 3.0);
        }
        assert!(lower_panel.area.abs_diff(137_424) < 1_000);
        assert!(middle_panel.area.abs_diff(143_583) < 2_000);
        Ok(())
    }
}
