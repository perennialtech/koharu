//! Hayai OCR v2.7 Supernova: crop-level OCR for Japanese, Chinese, Korean,
//! and English text. A SigLIP2 NaFlex encoder's 2x2 patch blocks are merged
//! into one token each by a projector that also reads them in both writing
//! directions, then a small GQA transformer decoder reads them over a shared
//! visual and textual prefix, with every encoder depth fused into its visual
//! tokens, block-causal attention, and 2D multimodal RoPE.
//!
//! Upstream:
//! https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.7-supernova/tree/b00317ed69fccee5f787187d913fcf8a96f37f65

mod config;
mod model;
mod processor;

use crate::backend::TryIntoDevice;
use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use koharu_torch::{Device, Kind};

use self::{
    config::HayaiConfig,
    model::Model,
    processor::{ImageProcessor, Tokenizer},
};

crate::model_repository!("JustANormalTinkerer/hayai-ocr-v2.7-supernova" @ "b00317ed69fccee5f787187d913fcf8a96f37f65" {
    CONFIG = "config.json",
    WEIGHTS = "model.safetensors",
    TOKENIZER = "tokenizer.json",
});

// Each crop is padded to the largest patch grid in the batch. Cap the
// number of simultaneous vision activations and decoder KV caches per page.
const MAX_BATCH: usize = 24;

#[derive(Debug)]
pub struct HayaiOcr {
    device: Device,
    model: Model,
    processor: ImageProcessor,
    tokenizer: Tokenizer,
}

impl HayaiOcr {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let kind = if device.backend == crate::Backend::Cuda && device.compute_capability() >= 80 {
            Kind::BFloat16
        } else {
            Kind::Float
        };
        let device: Device = device.try_into_device()?;
        let (config, weights, tokenizer) =
            tokio::try_join!(CONFIG.resolve(), WEIGHTS.resolve(), TOKENIZER.resolve())
                .context("failed to resolve Hayai OCR's files")?;
        let config: HayaiConfig = serde_json::from_slice(&std::fs::read(config)?)?;
        config.validate()?;
        let tokenizer = Tokenizer::from_file(&tokenizer)?;
        ensure!(
            tokenizer.len() == config.vocab_size as usize,
            "Hayai OCR vocabulary has {} entries but the decoder has {} outputs",
            tokenizer.len(),
            config.vocab_size
        );
        let _guard = koharu_torch::no_grad_guard();
        let processor = ImageProcessor::new(device);

        let mut model = Model::new(&config, &tokenizer, device, kind)?;
        model.load(&weights)?;
        Ok(Self {
            device,
            model,
            processor,
            tokenizer,
        })
    }

    /// The text of each crop, the crops decoded together.
    pub fn inference(&self, images: &[DynamicImage]) -> Result<Vec<String>> {
        recognize_batches(images, |images| {
            let _guard = koharu_torch::no_grad_guard();
            let inputs = images
                .iter()
                .map(|image| self.processor.preprocess(image))
                .collect::<Result<Vec<_>>>()?;
            let token_ids = self.model.generate(&inputs, &self.tokenizer, self.device)?;
            token_ids
                .iter()
                .map(|token_ids| self.tokenizer.decode(token_ids))
                .collect()
        })
    }
}

fn recognize_batches(
    images: &[DynamicImage],
    mut recognize: impl FnMut(&[DynamicImage]) -> Result<Vec<String>>,
) -> Result<Vec<String>> {
    let mut texts = Vec::with_capacity(images.len());
    for batch in images.chunks(MAX_BATCH) {
        texts.extend(recognize(batch)?);
    }
    Ok(texts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_crops_skip_inference() -> Result<()> {
        assert!(recognize_batches(&[], |_| unreachable!())?.is_empty());
        Ok(())
    }

    #[test]
    fn bounded_batches_keep_crop_order() -> Result<()> {
        let images: Vec<_> = (1..=50).map(|w| DynamicImage::new_rgb8(w, 1)).collect();
        let mut sizes = Vec::new();
        let texts = recognize_batches(&images, |batch| {
            sizes.push(batch.len());
            Ok(batch.iter().map(|crop| crop.width().to_string()).collect())
        })?;
        assert_eq!(sizes, [24, 24, 2]);
        assert_eq!(texts, (1..=50).map(|w| w.to_string()).collect::<Vec<_>>());
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires LibTorch, the checkpoint, and upstream benches/fixtures/ocr/title.png"]
    async fn checkpoint_matches_upstream_character_sequence() -> anyhow::Result<()> {
        let input =
            std::env::var_os("KOHARU_HAYAI_TITLE_FIXTURE").context(
                "set KOHARU_HAYAI_TITLE_FIXTURE to upstream crates/koharu-ml/benches/fixtures/ocr/title.png",
            )?;
        let crop = image::open(input)?;
        crate::init(&[crate::Feature::Torch]).await?;
        let device = crate::device(false);
        let model = HayaiOcr::load(device).await?;
        let text = model.inference(&[crop])?;
        // Verified against the pinned upstream `HayaiModel.generate` in greedy
        // mode on the identical fixture crop.
        assert_eq!(text, ["対策委員会です!"]);
        Ok(())
    }
}
