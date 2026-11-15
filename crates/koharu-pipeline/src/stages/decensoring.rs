use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use image::{DynamicImage, GrayImage, ImageFormat, Luma, RgbImage, Rgba, RgbaImage};
use koharu_ml::{
    kitsumed_yolo11m_seg::{CensorKind, KitsumedYolo11mSeg},
    waifu_inpaint_xl::WaifuInpaintXl,
};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, Origin, RasterLayer, RasterLayerKind,
};

use super::{StageInput, StageProcessor, finish, generation, inpainting::inpaint_tiled};
use crate::ModelCell;

const PRODUCER: &str = "dev.koharu.pipeline.decensoring";
const MODEL: &str = "kitsumed-yolo11m-seg+waifu-inpaint-xl";
/// Pixels added around each detected censor. The detector's outlines stop
/// short of a censor's edge (it covers 87% of censored pixels on real art),
/// and a sliver left behind keeps the censor visible.
const MASK_GROWTH: u8 = 8;
/// How far down the schedule mosaics are refined from.
const MOSAIC_STRENGTH: f32 = 0.8;
/// Known pixels kept around each generated crop.
const TILE_CONTEXT: u32 = 128;

/// Repaints censorship already in the page source: bars and mosaics that
/// kitsumed's YOLO11m-seg finds, filled by Waifu-Inpaint-XL into a
/// decensor layer above the source.
pub(super) struct Processor {
    device: koharu_ml::Device,
    models: ModelCell<Models>,
}

struct Models {
    detector: Arc<Mutex<KitsumedYolo11mSeg>>,
    inpainter: Arc<WaifuInpaintXl>,
}

impl Processor {
    pub(super) fn new(device: koharu_ml::Device) -> Self {
        Self {
            device,
            models: ModelCell::new(),
        }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        MODEL
    }

    fn skip(&self, input: &StageInput) -> Result<bool> {
        let source = AssetRole::new("source")?;
        for entity in input.scene.children(input.page)? {
            if input
                .scene
                .component::<RasterLayer>(entity)?
                .is_some_and(|layer| layer.kind == RasterLayerKind::Decensor)
                && input.scene.asset(entity, &source)?.is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn unload(&self) -> bool {
        self.models.unload()
    }

    async fn load(&self) -> Result<()> {
        self.models
            .ensure(|| async {
                let detector = KitsumedYolo11mSeg::load(self.device.clone()).await?;
                let inpainter = WaifuInpaintXl::load(self.device.clone()).await?;
                Ok(Models {
                    detector: Arc::new(Mutex::new(detector)),
                    inpainter: Arc::new(inpainter),
                })
            })
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let (detector, inpainter) = {
            let models = self.models.lock().await;
            let models = models
                .as_ref()
                .ok_or_else(|| anyhow!("decensoring models are not loaded"))?;
            (models.detector.clone(), models.inpainter.clone())
        };
        let page = input.page;
        let source = input
            .images
            .get(&input.scene, page, "source")
            .await?
            .ok_or_else(|| anyhow!("page {page} has no source image"))?;
        let region = input.region;
        let generated = tokio_rayon::spawn(move || -> Result<Option<(GrayImage, RgbImage)>> {
            let image = source.to_rgb8();
            let censors = detector
                .lock()
                .map_err(|_| anyhow!("kitsumed-yolo11m-seg lock is poisoned"))?
                .inference(&image)?;
            let (width, height) = image.dimensions();
            let mut image = DynamicImage::ImageRgb8(image);
            // Bars hide everything under them and are generated from noise.
            // Mosaics keep the coarse colours and shapes they cover, so
            // they are refined from the page instead, after the bars.
            let mut repainted = GrayImage::new(width, height);
            for (kind, strength) in [
                (CensorKind::Bar, 1.0),
                (CensorKind::Mosaic, MOSAIC_STRENGTH),
            ] {
                let mut mask = GrayImage::new(width, height);
                for censor in censors.iter().filter(|censor| censor.kind == kind) {
                    censor.paint(&mut mask);
                }
                let mut mask = imageproc::morphology::dilate(
                    &mask,
                    imageproc::distance_transform::Norm::LInf,
                    MASK_GROWTH,
                );
                for (x, y, pixel) in mask.enumerate_pixels_mut() {
                    let outside = region.is_some_and(|bounds| {
                        f64::from(x + 1) <= bounds.x
                            || f64::from(y + 1) <= bounds.y
                            || f64::from(x) >= bounds.x + bounds.width
                            || f64::from(y) >= bounds.y + bounds.height
                    });
                    if outside || repainted.get_pixel(x, y)[0] != 0 {
                        *pixel = Luma([0]);
                    }
                }
                if mask.as_raw().iter().all(|value| *value == 0) {
                    continue;
                }
                image = inpaint_tiled(
                    &image,
                    &mask,
                    &GrayImage::new(width, height),
                    &[],
                    TILE_CONTEXT,
                    // Grown to its generation bucket's aspect ratio with
                    // page context instead of being stretched.
                    |width, height| Some(WaifuInpaintXl::crop_size(width, height)),
                    |image, mask| {
                        Ok(DynamicImage::ImageRgb8(inpainter.inference(
                            &image.to_rgb8(),
                            mask,
                            strength,
                        )?))
                    },
                )?;
                for (target, value) in repainted.iter_mut().zip(mask.as_raw()) {
                    *target = (*target).max(*value);
                }
            }
            Ok(repainted
                .as_raw()
                .iter()
                .any(|value| *value != 0)
                .then(|| (repainted, image.to_rgb8())))
        })
        .await?;
        let Some((mask, generated)) = generated else {
            return finish(input.scene.edit());
        };
        if generated.dimensions() != mask.dimensions() {
            bail!("decensored image dimensions do not match page {page}");
        }
        let mut overlay = RgbaImage::new(mask.width(), mask.height());
        for (x, y, target) in overlay.enumerate_pixels_mut() {
            if mask.get_pixel(x, y)[0] >= 127 {
                let [red, green, blue] = generated.get_pixel(x, y).0;
                *target = Rgba([red, green, blue, u8::MAX]);
            }
        }
        let (width, height) = overlay.dimensions();
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(overlay).write_to(&mut bytes, ImageFormat::Png)?;

        let mut edit = input.scene.edit_as(generation(PRODUCER, MODEL)?);
        edit.observe_assets(page)?;
        let entity = edit.add_entity(page, At::Start)?;
        edit.set(
            entity,
            &RasterLayer {
                origin: Origin::User,
                name: "Decensor".to_owned(),
                kind: RasterLayerKind::Decensor,
            },
        )?;
        edit.set_asset(
            entity,
            &AssetRole::new("source")?,
            AssetInput::new(
                Arc::<[u8]>::from(bytes.into_inner()),
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
