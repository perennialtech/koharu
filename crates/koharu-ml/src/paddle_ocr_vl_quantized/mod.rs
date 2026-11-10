//! Quantized PaddleOCR-VL-1.6 inference through llama.cpp.

use anyhow::{Context, Result};
use image::DynamicImage;
use koharu_llama::mtmd::mtmd_default_marker;
use serde::{Deserialize, Serialize};

use crate::{
    Device,
    llm::{ChatMessage, GenerationOptions, Input, Llm, LoadOptions, MtmdOptions},
};

const MAX_NEW_TOKENS: usize = 512;
const REPETITION_PENALTY: f32 = 1.2;
// Each sequence keeps its own KV stream sized for the prompt plus
// MAX_NEW_TOKENS, about 14 MiB here, so pages with many text regions are
// recognized in groups to bound VRAM.
const BATCH_SIZE: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaddleOCRVLTask {
    Ocr,
    Table,
    Formula,
    Chart,
    Spotting,
    Seal,
}

impl PaddleOCRVLTask {
    const fn prompt(self) -> &'static str {
        match self {
            Self::Ocr => "OCR:",
            Self::Table => "Table Recognition:",
            Self::Formula => "Formula Recognition:",
            Self::Chart => "Chart Recognition:",
            Self::Spotting => "Spotting:",
            Self::Seal => "Seal Recognition:",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaddleOCRVLResult {
    pub text: String,
}

crate::model_repository!("PaddlePaddle/PaddleOCR-VL-1.6-GGUF" @ "511b09642bb324401f15f97cc23bc67e8f0a291d" {
    MODEL = "PaddleOCR-VL-1.6-GGUF.gguf",
    PROJECTOR = "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
});

#[derive(Debug)]
pub struct PaddleOCRVLQuantized {
    model: Llm,
}

impl PaddleOCRVLQuantized {
    pub async fn load(device: Device) -> Result<Self> {
        let (model_path, projector_path) =
            tokio::try_join!(MODEL.resolve(), PROJECTOR.resolve())
                .context("failed to resolve quantized PaddleOCR-VL-1.6 files")?;
        let model = Llm::load_with_options(
            device,
            model_path,
            LoadOptions {
                mtmd: Some(MtmdOptions::new(projector_path)),
                ..LoadOptions::default()
            },
        )
        .await
        .context("failed to load quantized PaddleOCR-VL-1.6")?;
        Ok(Self { model })
    }

    /// Recognizes each image with the same task, returning results in order.
    pub fn inference(
        &self,
        images: &[&DynamicImage],
        task: PaddleOCRVLTask,
    ) -> Result<Vec<PaddleOCRVLResult>> {
        let prompt = self.model.render_chat_prompt(&[ChatMessage::user(format!(
            "{}{}",
            mtmd_default_marker(),
            task.prompt()
        ))])?;
        let options = GenerationOptions {
            max_tokens: MAX_NEW_TOKENS,
            temperature: 0.0,
            repeat_penalty: REPETITION_PENALTY,
            repeat_last_n: MAX_NEW_TOKENS as i32,
            add_special: true,
            ..GenerationOptions::default()
        };
        let mut results = Vec::with_capacity(images.len());
        for group in images.chunks(BATCH_SIZE) {
            let inputs = group
                .iter()
                .map(|image| Input::new(&prompt).with_image(image))
                .collect::<Vec<_>>();
            results.extend(
                self.model
                    .inference_batch(&inputs, &options)?
                    .into_iter()
                    .map(|generation| PaddleOCRVLResult {
                        text: generation.text,
                    }),
            );
        }
        Ok(results)
    }
}
