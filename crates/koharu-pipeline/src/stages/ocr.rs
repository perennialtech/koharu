use std::sync::{Arc, Mutex};

use super::{StageInput, StageProcessor, finish, generation};
use crate::{ModelCell, OcrModel, scope::geometry_extents};
use anyhow::{Context as _, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use image::DynamicImage;
use koharu_ml::paddle_ocr_vl_quantized::{PaddleOCRVLQuantized, PaddleOCRVLTask};
use koharu_scene::{
    Authored, EntityId, Geometry, LanguageTag, OcrAnalysis, Origin, RecognizedFrom, Region,
    RegionSpec, SourceText, TextDirection, TextRegion,
};

const PRODUCER: &str = "dev.koharu.pipeline.ocr";

pub(super) struct Processor {
    device: koharu_ml::Device,
    model: ModelCell<Model>,
}

impl Processor {
    pub(super) fn new(_config: OcrModel, device: koharu_ml::Device) -> Self {
        Self {
            device,
            model: ModelCell::new(),
        }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        "paddleocr-vl-1.6"
    }

    fn unload(&self) -> bool {
        self.model.unload()
    }

    async fn load(&self) -> Result<()> {
        self.model.ensure(|| Model::load(self.device.clone())).await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        self.model
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| anyhow!("OCR model is not loaded"))?
            .run(input)
            .await
    }
}

struct Model(Arc<Mutex<PaddleOCRVLQuantized>>);

impl Model {
    async fn load(device: koharu_ml::Device) -> Result<Self> {
        Ok(Self(Arc::new(Mutex::new(
            PaddleOCRVLQuantized::load(device).await?,
        ))))
    }

    async fn run(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let page = input.page;
        let mut targets = Vec::new();
        let source = input
            .images
            .get(&input.scene, page, "source")
            .await?
            .ok_or_else(|| anyhow!("page {page} has no source image"))?;
        for entity in input.scene.descendants(page)? {
            let region = entity.id();
            if !input.contains_entity(region)? {
                continue;
            }
            let is_text_region = input
                .scene
                .component::<Region>(region)?
                .is_some_and(|value| value.kind == TextRegion::kind());
            if !is_text_region {
                continue;
            }
            let geometry = input
                .scene
                .component::<Geometry>(region)?
                .ok_or_else(|| anyhow!("text region {region} has no geometry"))?;
            let crop = crop(&source, &geometry)
                .with_context(|| format!("text region {region} is outside its source image"))?;
            for relation in input.scene.relations_to_as::<RecognizedFrom>(region) {
                let content = relation.value().source;
                let previous = input.scene.component::<SourceText>(content)?;
                if previous
                    .as_ref()
                    .is_some_and(|value| matches!(value.text.origin, Origin::User))
                {
                    continue;
                }
                targets.push(OcrTarget {
                    content,
                    region,
                    geometry: geometry.clone(),
                    previous,
                    image: crop.clone(),
                });
            }
        }

        let results = infer_text(self.0.clone(), targets, |model, images| {
            Ok(model
                .inference(images, PaddleOCRVLTask::Ocr)?
                .into_iter()
                .map(|result| result.text)
                .collect())
        })
        .await?;

        let generation = generation(PRODUCER, "paddleocr-vl-1.6")?;
        let mut edit = input.scene.edit_as(generation.clone());
        edit.observe_assets(page)?;
        for result in &results {
            edit.observe::<Region>(result.region)?;
            edit.observe::<Geometry>(result.region)?;
            edit.observe::<SourceText>(result.content)?;
        }
        for result in results {
            let language = result
                .previous
                .and_then(|value| value.language)
                .or_else(|| LanguageTag::new("ja-JP").ok());
            edit.set(
                result.content,
                &SourceText {
                    text: Authored::generated(result.text, generation.clone()),
                    language,
                },
            )?;
            let (min_x, min_y, max_x, max_y) = geometry_extents(&result.geometry)
                .ok_or_else(|| anyhow!("text region {} has empty geometry", result.region))?;
            edit.set(
                result.region,
                &OcrAnalysis {
                    origin: Origin::Generated(generation.clone()),
                    direction: if max_y - min_y >= (max_x - min_x) * 1.15 {
                        TextDirection::Vertical
                    } else {
                        TextDirection::Horizontal
                    },
                    confidence: None,
                    line_boundaries: Vec::new(),
                },
            )?;
        }
        finish(edit)
    }
}

struct OcrTarget {
    content: EntityId,
    region: EntityId,
    geometry: Geometry,
    previous: Option<SourceText>,
    image: DynamicImage,
}

struct OcrResult {
    content: EntityId,
    region: EntityId,
    geometry: Geometry,
    previous: Option<SourceText>,
    text: String,
}

async fn infer_text<M: Send + 'static>(
    model: Arc<Mutex<M>>,
    targets: Vec<OcrTarget>,
    inference: impl Fn(&M, &[&DynamicImage]) -> Result<Vec<String>> + Send + Sync + 'static,
) -> Result<Vec<OcrResult>> {
    tokio_rayon::spawn(move || {
        let model = model
            .lock()
            .map_err(|_| anyhow!("OCR model lock is poisoned"))?;
        let images = targets
            .iter()
            .map(|target| &target.image)
            .collect::<Vec<_>>();
        let texts = inference(&model, &images)?;
        ensure!(
            texts.len() == targets.len(),
            "OCR returned {} results for {} regions",
            texts.len(),
            targets.len()
        );
        Ok(targets
            .into_iter()
            .zip(texts)
            .map(|(target, text)| OcrResult {
                content: target.content,
                region: target.region,
                geometry: target.geometry,
                previous: target.previous,
                text: normalize_ocr_text(text),
            })
            .collect())
    })
    .await
}

// Manga OCR can emit replacement-box glyphs for an isolated Japanese ellipsis.
// Normalize only an all-placeholder sequence so ordinary OCR output is preserved.
fn normalize_ocr_text(text: String) -> String {
    let visible = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<Vec<_>>();
    if visible.len() >= 2
        && visible
            .iter()
            .all(|character| matches!(character, '☐' | '□' | '▢' | '▣' | '�'))
    {
        "…".to_owned()
    } else {
        text
    }
}

// Detected text boxes hug the ink, and PaddleOCR-VL drops or misreads glyphs
// that touch the crop edge. A margin of a fraction of the line thickness, the
// box's shorter side, restores them without reaching neighbouring lines.
const CROP_MARGIN: f64 = 0.15;

fn crop(source: &DynamicImage, geometry: &Geometry) -> Result<DynamicImage> {
    let (min_x, min_y, max_x, max_y) =
        geometry_extents(geometry).ok_or_else(|| anyhow!("geometry is empty"))?;
    let margin = (max_x - min_x).min(max_y - min_y).max(0.0) * CROP_MARGIN;
    let x = (min_x - margin).floor().max(0.0) as u32;
    let y = (min_y - margin).floor().max(0.0) as u32;
    let right = (max_x + margin)
        .ceil()
        .max(0.0)
        .min(f64::from(source.width())) as u32;
    let bottom = (max_y + margin)
        .ceil()
        .max(0.0)
        .min(f64::from(source.height())) as u32;
    if right <= x || bottom <= y {
        bail!("geometry does not overlap the image");
    }
    Ok(source.crop_imm(x, y, right - x, bottom - y))
}

#[cfg(test)]
mod tests {
    use image::DynamicImage;
    use koharu_scene::Geometry;

    use super::{crop, normalize_ocr_text};

    #[test]
    fn crops_keep_a_margin_around_the_text_within_the_page() {
        let source = DynamicImage::new_luma8(200, 100);

        let column = crop(&source, &Geometry::rectangle(50.0, 20.0, 20.0, 60.0)).unwrap();
        let edge = crop(&source, &Geometry::rectangle(0.0, 0.0, 40.0, 100.0)).unwrap();

        assert_eq!((column.width(), column.height()), (26, 66));
        assert_eq!((edge.width(), edge.height()), (46, 100));
    }

    #[test]
    fn repeated_placeholder_glyphs_are_an_ellipsis() {
        assert_eq!(normalize_ocr_text("☐ ☐ ☐".to_owned()), "…");
        assert_eq!(normalize_ocr_text("□\n□".to_owned()), "…");
    }

    #[test]
    fn ordinary_text_and_single_boxes_are_unchanged() {
        assert_eq!(normalize_ocr_text("待って…".to_owned()), "待って…");
        assert_eq!(normalize_ocr_text("☐".to_owned()), "☐");
    }
}
