use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use image::{DynamicImage, ImageFormat};
use koharu_ml::waifu2x::Waifu2x;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole};

use super::{StageInput, StageProcessor, finish, generation};
use crate::ModelCell;

const PRODUCER: &str = "dev.koharu.pipeline.upscaling";
const MODEL: &str = "waifu2x-cunet";
/// The renderer draws this page asset in place of `source`, so a scaled
/// render samples the upscaler's pixels instead of resampling the page.
const UPSCALED: &str = "upscaled";

pub(super) struct Processor {
    device: koharu_ml::Device,
    model: ModelCell<Arc<Mutex<Waifu2x>>>,
}

impl Processor {
    pub(super) fn new(device: koharu_ml::Device) -> Self {
        Self {
            device,
            model: ModelCell::new(),
        }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        MODEL
    }

    fn skip(&self, input: &StageInput) -> Result<bool> {
        Ok(input
            .scene
            .asset(input.page, &AssetRole::new(UPSCALED)?)?
            .is_some())
    }

    fn unload(&self) -> bool {
        self.model.unload()
    }

    async fn load(&self) -> Result<()> {
        self.model
            .ensure(|| async {
                Ok(Arc::new(Mutex::new(
                    Waifu2x::load(self.device.clone()).await?,
                )))
            })
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let model = self
            .model
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| anyhow!("upscaling model is not loaded"))?
            .clone();
        let page = input.page;
        let source = input
            .images
            .get(&input.scene, page, "source")
            .await?
            .ok_or_else(|| anyhow!("page {page} has no source image"))?;
        let (png, width, height) = tokio_rayon::spawn(move || -> Result<_> {
            let upscaled = model
                .lock()
                .map_err(|_| anyhow!("waifu2x model lock is poisoned"))?
                .inference(&source.to_rgb8())?;
            let (width, height) = upscaled.dimensions();
            let mut png = Cursor::new(Vec::new());
            DynamicImage::ImageRgb8(upscaled).write_to(&mut png, ImageFormat::Png)?;
            Ok((png.into_inner(), width, height))
        })
        .await?;

        let mut edit = input.scene.edit_as(generation(PRODUCER, MODEL)?);
        edit.observe_assets(page)?;
        edit.set_asset(
            page,
            &AssetRole::new(UPSCALED)?,
            AssetInput::new(
                Arc::<[u8]>::from(png),
                "image/png",
                AssetMetadata {
                    width: Some(width),
                    height: Some(height),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        finish(edit)
    }
}
