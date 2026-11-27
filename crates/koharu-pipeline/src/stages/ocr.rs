use std::sync::{Arc, Mutex};

use super::{
    StageInput, StageProcessor,
    detection::{GLYPH_MASK, clip_polygon},
    finish, generation,
};
use crate::{ModelCell, OcrModel, scope::geometry_extents};
use anyhow::{Context as _, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::geometric_transformations::{Border, Interpolation, warp_into_with};
use koharu_ml::hayai_ocr::HayaiOcr;
use koharu_scene::{
    AssetRole, Authored, Edit, EntityId, Geometry, LanguageTag, OcrAnalysis, Origin, Presents,
    RecognizedFrom, Region, RegionSpec, SourceText, TextDirection, TextRegion, Typography,
    WritingMode,
};

const MODEL: &str = "hayai-ocr";
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
        MODEL
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

struct Model(Arc<Mutex<HayaiOcr>>);

impl Model {
    async fn load(device: koharu_ml::Device) -> Result<Self> {
        Ok(Self(Arc::new(Mutex::new(HayaiOcr::load(device).await?))))
    }

    async fn run(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let generation = generation(PRODUCER, MODEL)?;
        let mut edit = input.scene.edit_as(generation.clone());
        let source = input.require_page_image().await?;
        let targets = prepare_targets(&input, &mut edit, &source).await?;
        if targets.is_empty() {
            return finish(edit);
        }
        let (targets, images): (Vec<_>, Vec<_>) = targets.into_iter().unzip();
        let model = Arc::clone(&self.0);
        let texts = tokio_rayon::spawn(move || {
            model
                .lock()
                .map_err(|_| anyhow!("OCR model lock is poisoned"))?
                .inference(&images)
        })
        .await?;
        ensure!(
            texts.len() == targets.len(),
            "OCR returned {} results for {} regions",
            texts.len(),
            targets.len()
        );

        for (target, text) in targets.into_iter().zip(texts) {
            let language = target
                .previous
                .and_then(|value| value.language)
                .or_else(|| LanguageTag::new("ja-JP").ok());
            edit.set(
                target.content,
                &SourceText {
                    text: Authored::generated(normalize_ocr_text(text), generation.clone()),
                    language,
                },
            )?;
            if let Some(direction) = target.direction {
                let mut analysis = input
                    .scene
                    .component::<OcrAnalysis>(target.region)?
                    .unwrap_or_else(|| OcrAnalysis {
                        origin: Origin::Generated(generation.clone()),
                        direction,
                        confidence: None,
                        line_boundaries: Vec::new(),
                    });
                analysis.direction = direction;
                edit.set(target.region, &analysis)?;
            }
        }
        finish(edit)
    }
}

async fn prepare_targets(
    input: &StageInput,
    edit: &mut Edit,
    source: &DynamicImage,
) -> Result<Vec<(OcrTarget, DynamicImage)>> {
    edit.observe_assets(input.page)?;
    let role = AssetRole::new(GLYPH_MASK)?;
    let mut targets = Vec::new();
    let mut masks = Vec::new();
    // A new region can claim ink in an existing crop. A cleanup raster on
    // the same page has no bearing on the source image or its glyph masks.
    for region in edit.observe_entities_with::<Region>(input.page)? {
        edit.observe::<Region>(region)?;
        let entity = input.scene.entity(region)?;
        if entity
            .component::<Region>()?
            .is_none_or(|r| r.kind != TextRegion::kind())
        {
            continue;
        }
        edit.observe_assets(region)?;
        edit.observe::<Geometry>(region)?;
        edit.observe::<OcrAnalysis>(region)?;
        let asset = input.scene.asset(region, &role)?;
        let has_glyph_mask = asset.is_some();
        if let Some(asset) = asset {
            let coordinate = |name: &str| -> Result<u32> {
                asset
                    .metadata
                    .attributes
                    .get(name)
                    .with_context(|| format!("glyph mask on {region} has no {name}"))?
                    .parse()
                    .with_context(|| format!("invalid glyph mask {name} on {region}"))
            };
            let x = coordinate("x")?;
            let y = coordinate("y")?;
            let (Some(w), Some(h)) = (asset.metadata.width, asset.metadata.height) else {
                bail!("glyph mask on {region} has no dimensions");
            };
            masks.push((region, [x, y, x.saturating_add(w), y.saturating_add(h)]));
        }
        if !input.contains_entity(region)? {
            continue;
        }
        for content in edit.observe_relation_sources::<RecognizedFrom>(region)? {
            edit.observe::<SourceText>(content)?;
            let previous = input.scene.component::<SourceText>(content)?;
            if previous
                .as_ref()
                .is_some_and(|v| v.text.origin == Origin::User)
            {
                continue;
            }
            let geometry = entity
                .component::<Geometry>()?
                .ok_or_else(|| anyhow!("text region {region} has no geometry"))?;
            let Some(window) = CropWindow::new(source, &geometry)? else {
                continue;
            };
            // Detection owns measured direction. Existing source analysis
            // (including user annotations) keeps its provenance and lines.
            let current = entity.component::<OcrAnalysis>()?;
            let direction = if current.as_ref().is_some_and(|a| match &a.origin {
                Origin::User => true,
                Origin::Generated(g) => {
                    g.producer.as_str() != PRODUCER && a.direction != TextDirection::Auto
                }
            }) {
                None
            } else {
                let mut writing_mode = None;
                for layer in edit.observe_relation_sources::<Presents>(content)? {
                    edit.observe::<Typography>(layer)?;
                    writing_mode = input
                        .scene
                        .component::<Typography>(layer)?
                        .and_then(|t| t.writing_mode);
                    if writing_mode.is_some() {
                        break;
                    }
                }
                let direction = match writing_mode {
                    Some(WritingMode::Vertical) => TextDirection::Vertical,
                    Some(WritingMode::Horizontal) => TextDirection::Horizontal,
                    None => current
                        .as_ref()
                        .map(|a| a.direction)
                        .filter(|d| *d != TextDirection::Auto)
                        .unwrap_or(if window.height >= window.width * 1.15 {
                            TextDirection::Vertical
                        } else {
                            TextDirection::Horizontal
                        }),
                };
                current
                    .is_none_or(|a| a.direction != direction)
                    .then_some(direction)
            };
            targets.push((
                OcrTarget {
                    region,
                    content,
                    previous,
                    direction,
                },
                window,
                has_glyph_mask,
            ));
        }
    }

    let mut glyphs = Vec::new();
    for (region, bounds) in masks {
        if targets
            .iter()
            .any(|(_, window, has_mask)| *has_mask && overlaps(window.bounds, bounds))
        {
            let image = input
                .images
                .get(&input.scene, region, GLYPH_MASK)
                .await?
                .context("glyph mask disappeared from the snapshot")?;
            let mask = image
                .as_luma8()
                .context("glyph mask must be an 8-bit grayscale image")?;
            ensure!(
                mask.dimensions() == (bounds[2] - bounds[0], bounds[3] - bounds[1]),
                "glyph mask dimensions differ from metadata on {region}"
            );
            glyphs.push((region, bounds, image));
        }
    }
    Ok(targets
        .into_iter()
        .map(|(target, window, has_glyph_mask)| {
            // Without an ownership mask the user's entire crop is the selection.
            // An explicit empty mask still lets neighbours claim their own ink.
            let masks = if has_glyph_mask {
                glyphs.as_slice()
            } else {
                &[]
            };
            let image = window.crop(source, target.region, masks);
            (target, image)
        })
        .collect())
}

struct OcrTarget {
    content: EntityId,
    region: EntityId,
    direction: Option<TextDirection>,
    previous: Option<SourceText>,
}

// Columns of dots in silent reaction balloons can read as boxes or circles.
// Normalize only text made entirely of such symbols and middle dots, with at
// least one symbol, so ordinary OCR output, dots read as dots, masked words
// such as ○○…, and hearts and notes used as lettering are preserved.
fn normalize_ocr_text(text: String) -> String {
    let visible = text
        .chars()
        .filter(|character| !character.is_whitespace() && *character != '\u{FE0F}')
        .collect::<Vec<_>>();
    if visible.len() >= 2
        && visible.iter().copied().any(misread_dot)
        && visible
            .iter()
            .copied()
            .all(|character| misread_dot(character) || matches!(character, '・' | '･' | '·'))
    {
        "…".to_owned()
    } else {
        text
    }
}

fn misread_dot(character: char) -> bool {
    matches!(
        character,
        '\u{25A0}'..='\u{27BF}' | '\u{1F300}'..='\u{1FAFF}' | '∗' | '•' | '�'
    ) && !matches!(character, '♡' | '♥' | '❤' | '♩' | '♪' | '♫' | '♬')
}

type GlyphMask = (EntityId, [u32; 4], Arc<DynamicImage>);

fn overlaps(a: [u32; 4], b: [u32; 4]) -> bool {
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

struct CropWindow {
    bounds: [u32; 4],
    width: f64,
    height: f64,
    angle: f64,
    center: [f64; 2],
}

impl CropWindow {
    fn new(source: &DynamicImage, geometry: &Geometry) -> Result<Option<Self>> {
        let (left, top, right, bottom) = geometry_extents(geometry).context("geometry is empty")?;
        let (mut width, mut height, mut angle) = (right - left, bottom - top, 0.0);
        if let [a, b, c, d] = geometry.points.as_slice() {
            let (ux, uy, vx, vy) = (b.x - a.x, b.y - a.y, d.x - a.x, d.y - a.y);
            let (w, h) = (ux.hypot(uy), vx.hypot(vy));
            // Free-form scene polygons keep their axis-aligned covering crop.
            if w > 0.0
                && h > 0.0
                && (ux * vx + uy * vy).abs() < 1e-6 * w * h
                && (c.x - b.x - vx).hypot(c.y - b.y - vy) < 1e-6 * w.max(h)
            {
                (width, height, angle) = (w, h, uy.atan2(ux));
            }
        }
        // Hayai reads the detected source frame without additional crop padding.
        let bounds = [
            left.floor().max(0.0) as u32,
            top.floor().max(0.0) as u32,
            right.ceil().clamp(0.0, f64::from(source.width())) as u32,
            bottom.ceil().clamp(0.0, f64::from(source.height())) as u32,
        ];
        if bounds[0] >= bounds[2] || bounds[1] >= bounds[3] {
            return Ok(None);
        }
        let page = crate::Bounds {
            x: 0.0,
            y: 0.0,
            width: f64::from(source.width()),
            height: f64::from(source.height()),
        };
        if clip_polygon(&geometry.points, page).is_none() {
            return Ok(None);
        }
        Ok(Some(Self {
            bounds,
            width,
            height,
            angle,
            center: [(left + right) / 2.0, (top + bottom) / 2.0],
        }))
    }

    fn crop(&self, source: &DynamicImage, own: EntityId, masks: &[GlyphMask]) -> DynamicImage {
        let [left, top, right, bottom] = self.bounds;
        let mut crop = source
            .crop_imm(left, top, right - left, bottom - top)
            .to_rgb8();
        let mine = masks.iter().find(|(id, ..)| *id == own);
        let others: Vec<_> = masks
            .iter()
            .filter(|(id, bounds, _)| *id != own && overlaps(self.bounds, *bounds))
            .collect();
        if others.is_empty() && self.angle == 0.0 {
            return crop.into();
        }
        // Per-channel upper median of the original crop, before blanking or interpolation.
        let mut histogram = [[0usize; 256]; 3];
        for pixel in crop.pixels() {
            for (channel, value) in histogram.iter_mut().zip(pixel.0) {
                channel[usize::from(value)] += 1;
            }
        }
        let middle = (crop.as_raw().len() / 3) / 2;
        let paper = Rgb(histogram.map(|bins| {
            let mut seen = 0;
            bins.into_iter()
                .position(|n| {
                    seen += n;
                    seen > middle
                })
                .unwrap_or(255) as u8
        }));
        let contains = |mask: &GlyphMask, x: u32, y: u32| {
            let [mx, my, mr, mb] = mask.1;
            x >= mx
                && y >= my
                && x < mr
                && y < mb
                && mask.2.as_luma8().expect("validated glyph mask")[(x - mx, y - my)][0] != 0
        };
        for other in others {
            let [x0, y0, x1, y1] = other.1;
            for y in top.max(y0)..bottom.min(y1) {
                for x in left.max(x0)..right.min(x1) {
                    if contains(other, x, y) && !mine.is_some_and(|m| contains(m, x, y)) {
                        crop.put_pixel(x - left, y - top, paper);
                    }
                }
            }
        }
        if self.angle == 0.0 {
            return crop.into();
        }

        // Map output pixel centres back into the page-aligned crop, so the
        // source's clockwise tilt is undone in one bilinear resampling.
        let (sin, cos) = self.angle.sin_cos();
        let (cx, cy) = (
            self.center[0] - f64::from(left),
            self.center[1] - f64::from(top),
        );
        let (w, h) = (self.width / 2.0, self.height / 2.0);
        // Limit an off-page rectangle to the part of its frame the scan covers.
        let mut ext = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for x in [-cx, f64::from(crop.width()) - cx] {
            for y in [-cy, f64::from(crop.height()) - cy] {
                let (u, v) = (x * cos + y * sin, y * cos - x * sin);
                ext = [ext[0].min(u), ext[1].min(v), ext[2].max(u), ext[3].max(v)];
            }
        }
        let (x0, y0) = (ext[0].max(-w), ext[1].max(-h));
        let mut upright = RgbImage::new(
            (ext[2].min(w) - x0).ceil().max(1.0) as u32,
            (ext[3].min(h) - y0).ceil().max(1.0) as u32,
        );
        warp_into_with(
            &crop,
            |x, y| {
                let (u, v) = (f64::from(x) + 0.5 + x0, f64::from(y) + 0.5 + y0);
                (
                    (cx + u * cos - v * sin - 0.5) as f32,
                    (cy + u * sin + v * cos - 0.5) as f32,
                )
            },
            Interpolation::Bilinear,
            Border::Constant(paper),
            &mut upright,
        );
        upright.into()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CropWindow, GLYPH_MASK, OcrTarget, StageInput, generation, normalize_ocr_text,
        prepare_targets,
    };
    use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};
    use koharu_scene::{
        AssetInput, AssetMetadata, AssetRole, At, Authored, EntityId, Geometry, OcrAnalysis,
        Origin, PageDraft, RecognizedFrom, Session, SourceText, TextDirection, TextRegion,
    };

    fn crop(source: &DynamicImage, geometry: &Geometry) -> anyhow::Result<DynamicImage> {
        Ok(CropWindow::new(source, geometry)?
            .expect("crop overlaps page")
            .crop(source, koharu_scene::EntityId::new(), &[]))
    }

    #[test]
    fn deskews_a_tilted_crop_about_its_own_center() {
        use koharu_scene::Point;
        for degrees in [-25.0_f64, 22.0] {
            let (sin, cos) = degrees.to_radians().sin_cos();
            let center = [73.0, 49.0];
            let turn = |u: f64, v: f64| Point {
                x: center[0] + u * cos - v * sin,
                y: center[1] + u * sin + v * cos,
            };
            let paper = Rgb([211, 182, 157]);
            let source = RgbImage::from_fn(120, 100, |x, y| {
                let (dx, dy) = (
                    f64::from(x) + 0.5 - center[0],
                    f64::from(y) + 0.5 - center[1],
                );
                let (u, v) = (dx * cos + dy * sin, dy * cos - dx * sin);
                if u.abs() < 18.0 && v.abs() < 2.0 {
                    Rgb([8; 3])
                } else {
                    paper
                }
            });
            let geometry = Geometry {
                origin: koharu_scene::Origin::User,
                points: vec![
                    turn(-20., -4.),
                    turn(20., -4.),
                    turn(20., 4.),
                    turn(-20., 4.),
                ],
            };
            let image = crop(&source.into(), &geometry).unwrap().to_rgb8();
            for x in 5..35 {
                assert!(image[(x, 4)][0] < 40, "{degrees} at {x}");
                assert_eq!(image[(x, 0)], paper);
            }
        }
    }

    #[test]
    fn deskew_keeps_the_tall_center_glyph_of_a_tilted_line() {
        use crate::stages::detection::rotated_rectangle_geometry;
        use koharu_ml::koharu_layout_rfdetr_seg_2xl::{KoharuLayoutDetection, KoharuLayoutMask};
        for degrees in [-20.0_f32, 20.0] {
            let (sin, cos) = f64::from(degrees).to_radians().sin_cos();
            let mut source = RgbImage::from_pixel(120, 80, Rgb([255; 3]));
            let mut mask = KoharuLayoutMask {
                x: 10,
                y: 20,
                width: 100,
                height: 40,
                pixels: vec![0; 100 * 40],
            };
            for y in 0..40 {
                for x in 0..100 {
                    let (dx, dy) = (f64::from(x) - 49.5, f64::from(y) - 19.5);
                    let (u, v) = (dx * cos + dy * sin, dy * cos - dx * sin);
                    if v.abs() < 1.5 || (u.abs() < 4.0 && v.abs() < 14.0) {
                        source.put_pixel(
                            mask.x + x,
                            mask.y + y,
                            Rgb(if v < -10.0 {
                                [220, 0, 0]
                            } else if v > 10.0 {
                                [0, 0, 220]
                            } else {
                                [0; 3]
                            }),
                        );
                        mask.pixels[(y * 100 + x) as usize] = 255;
                    }
                }
            }
            let text = KoharuLayoutDetection {
                label_id: 0,
                label: "text".into(),
                score: 0.9,
                bbox: [10., 20., 110., 60.],
                area: mask.pixels.iter().filter(|p| **p != 0).count() as u32,
                mask: mask.into(),
            };
            let geometry = rotated_rectangle_geometry(&text, degrees);
            let (top, bottom) = (geometry.points[0], geometry.points[3]);
            let height = (bottom.x - top.x).hypot(bottom.y - top.y);
            assert!(
                (28.0..30.0).contains(&height),
                "{degrees}° frame height {height}"
            );
            let upright = crop(&source.into(), &geometry).unwrap().to_rgb8();
            assert!(
                upright
                    .rows()
                    .take(4)
                    .flatten()
                    .any(|p| p[0] > 180 && p[1] < 50)
            );
            assert!(
                upright
                    .rows()
                    .skip(upright.height() as usize - 4)
                    .flatten()
                    .any(|p| p[2] > 180 && p[1] < 50)
            );
        }
    }

    struct OcrPage {
        session: Session,
        input: StageInput,
        target: EntityId,
        neighbor: EntityId,
        content: EntityId,
        source: DynamicImage,
    }

    fn mask(x: u32, w: u32, value: u8) -> AssetInput {
        let mut bytes = std::io::Cursor::new(Vec::new());
        GrayImage::from_pixel(w, 1, Luma([value]))
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        AssetInput::new(
            bytes.into_inner(),
            "image/png",
            AssetMetadata {
                width: Some(w),
                height: Some(1),
                attributes: [("x".into(), x.to_string()), ("y".into(), "15".into())].into(),
            },
        )
    }

    impl OcrPage {
        async fn new() -> anyhow::Result<Self> {
            let mut session = Session::memory().await?;
            let mut edit = session.snapshot().edit();
            let page = edit.add_page(PageDraft::new("page", 60., 40.), At::End)?;
            let target = edit.add_analysis_region::<TextRegion>(
                page,
                At::End,
                &Geometry::rectangle(12., 10., 22., 12.),
                None,
            )?;
            let neighbor = edit.add_analysis_region::<TextRegion>(
                page,
                At::End,
                &Geometry::rectangle(28., 10., 8., 14.),
                None,
            )?;
            let content = edit.add_text_content(page, At::End)?;
            let other = edit.add_text_content(page, At::End)?;
            edit.relate::<RecognizedFrom>(content, target)?;
            edit.relate::<RecognizedFrom>(other, neighbor)?;
            edit.set(
                other,
                &SourceText {
                    text: Authored::user("edited".to_owned()),
                    language: None,
                },
            )?;
            edit.set(
                target,
                &OcrAnalysis {
                    origin: Origin::User,
                    direction: TextDirection::Vertical,
                    confidence: None,
                    line_boundaries: vec![],
                },
            )?;
            let role = AssetRole::new(GLYPH_MASK)?;
            edit.set_asset(target, &role, mask(22, 7, 255))?;
            edit.set_asset(neighbor, &role, mask(27, 4, 255))?;
            let snapshot = session.commit(edit.finish()?).await?.snapshot;
            let source = RgbImage::from_fn(60, 40, |x, y| {
                if y == 15 && (22..33).contains(&x) {
                    Rgb([13, 14, 15])
                } else if (x + y) % 4 == 0 {
                    Rgb([248, 247, 246])
                } else {
                    Rgb([201, 176, 143])
                }
            });
            let input = StageInput::new(
                snapshot,
                page,
                Some(std::sync::Arc::new([target].into())),
                None,
                Default::default(),
                None,
            );
            Ok(Self {
                session,
                input,
                target,
                neighbor,
                content,
                source: source.into(),
            })
        }

        async fn prepare(
            &self,
        ) -> anyhow::Result<(Vec<(OcrTarget, DynamicImage)>, koharu_scene::Edit)> {
            let mut edit = self
                .input
                .scene
                .edit_as(generation(super::PRODUCER, "test")?);
            let targets = prepare_targets(&self.input, &mut edit, &self.source).await?;
            Ok((targets, edit))
        }

        async fn recognition_patch(&self) -> anyhow::Result<koharu_scene::Patch> {
            let (targets, mut edit) = self.prepare().await?;
            let [(target, _)] = targets.as_slice() else {
                panic!("expected one OCR target")
            };
            edit.set(
                target.content,
                &SourceText {
                    text: Authored::generated(
                        "recognized".into(),
                        generation(super::PRODUCER, "test")?,
                    ),
                    language: None,
                },
            )?;
            Ok(edit.finish()?)
        }
    }

    #[tokio::test]
    async fn scoped_ocr_blanks_exclusive_neighbor_ink_and_observes_mask_assets()
    -> anyhow::Result<()> {
        let mut fixture = OcrPage::new().await?;
        let (targets, edit) = fixture.prepare().await?;
        assert_eq!(targets.len(), 1);
        let (found, image) = &targets[0];
        assert_eq!(found.content, fixture.content);
        assert!(
            found.direction.is_none(),
            "keep the existing vertical analysis"
        );
        let crop = image.as_rgb8().unwrap();
        // Crop origin (12,10). Target-only and shared ink survive. The two
        // exclusively neighbouring pixels take the original channel medians.
        assert_eq!(crop[(22 - 12, 15 - 10)], Rgb([13, 14, 15]));
        assert_eq!(crop[(28 - 12, 15 - 10)], Rgb([13, 14, 15]));
        assert_eq!(crop[(30 - 12, 15 - 10)], Rgb([201, 176, 143]));
        let derived = edit.finish()?;
        let base = &fixture.input.scene;
        for (entity, x) in [(fixture.neighbor, 31), (fixture.target, 19)] {
            let changed = base.patch(|edit| {
                edit.set_asset(entity, &AssetRole::new(GLYPH_MASK)?, mask(x, 3, 255))
            })?;
            assert!(derived.rebase_on(&base.preview([&changed])?).is_err());
        }
        fixture.input.entities = None;
        let (targets, _) = fixture.prepare().await?;
        assert_eq!(targets.len(), 1, "user-authored source text is skipped");
        Ok(())
    }

    #[tokio::test]
    async fn ocr_rebases_after_inpainting_commits_on_the_same_page() -> anyhow::Result<()> {
        use koharu_scene::{RasterLayer, RasterLayerKind};
        let mut fixture = OcrPage::new().await?;
        let ocr = fixture.recognition_patch().await?;

        let base = &fixture.input.scene;
        let mut inpainting = base.edit_as(generation("dev.koharu.pipeline.inpainting", "lama")?);
        inpainting.observe_assets(fixture.input.page)?;
        let cleanup = inpainting.add_entity(fixture.input.page, At::Start)?;
        inpainting.set(
            cleanup,
            &RasterLayer {
                origin: Origin::User,
                name: "Cleanup".into(),
                kind: RasterLayerKind::Cleanup,
            },
        )?;
        let role = AssetRole::new("source")?;
        let mut bytes = std::io::Cursor::new(Vec::new());
        fixture
            .source
            .to_rgba8()
            .write_to(&mut bytes, image::ImageFormat::Png)?;
        let overlay = AssetInput::new(
            bytes.into_inner(),
            "image/png",
            AssetMetadata {
                width: Some(fixture.source.width()),
                height: Some(fixture.source.height()),
                attributes: Default::default(),
            },
        );
        inpainting.set_asset(cleanup, &role, overlay)?;
        let inpainting = inpainting.finish()?;
        assert_eq!(ocr.base_revision(), inpainting.base_revision());
        let advanced = fixture.session.commit(inpainting).await?.snapshot;
        let snapshot = fixture
            .session
            .commit(ocr.rebase_on(&advanced)?)
            .await?
            .snapshot;
        assert_eq!(
            snapshot
                .component::<SourceText>(fixture.content)?
                .unwrap()
                .text
                .value,
            "recognized"
        );
        assert_eq!(
            snapshot.asset(cleanup, &role)?,
            advanced.asset(cleanup, &role)?
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_off_page_region_does_not_abort_other_ocr_targets() -> anyhow::Result<()> {
        let mut fixture = OcrPage::new().await?;
        let mut edit = fixture.input.scene.edit();
        let diamond = |center: f64| Geometry {
            origin: Origin::User,
            points: [(0., -6.), (6., 0.), (0., 6.), (-6., 0.)]
                .map(|(dx, dy)| koharu_scene::Point {
                    x: center + dx,
                    y: center + dy,
                })
                .into(),
        };
        // The first diamond's bounding box reaches (2, 2), the next only
        // touches the page at (0, 0), and the last covers a triangle of it.
        let mut expected = vec![fixture.content];
        for (geometry, on_page) in [
            (Geometry::rectangle(70., 4., 10., 10.), false),
            (diamond(-4.), false),
            (diamond(-3.), false),
            (diamond(-2.), true),
        ] {
            let region = edit.add_analysis_region::<TextRegion>(
                fixture.input.page,
                At::Start,
                &geometry,
                None,
            )?;
            let content = edit.add_text_content(fixture.input.page, At::End)?;
            edit.relate::<RecognizedFrom>(content, region)?;
            if on_page {
                expected.push(content);
            }
        }
        fixture.input.scene = fixture.session.commit(edit.finish()?).await?.snapshot;
        fixture.input.entities = None;
        let (targets, _) = fixture.prepare().await?;
        let mut actual: Vec<_> = targets.iter().map(|(target, _)| target.content).collect();
        actual.sort();
        expected.sort();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn a_new_masked_neighbor_invalidates_prepared_ocr() -> anyhow::Result<()> {
        let mut fixture = OcrPage::new().await?;
        let derived = fixture.recognition_patch().await?;
        let (before, _) = fixture.prepare().await?;
        assert_eq!(
            before[0].1.as_rgb8().unwrap()[(31 - 12, 5)],
            Rgb([13, 14, 15])
        );
        let mut edit = fixture.input.scene.edit();
        let neighbor = edit.add_analysis_region::<TextRegion>(
            fixture.input.page,
            At::End,
            &Geometry::rectangle(30., 12., 5., 6.),
            None,
        )?;
        edit.set_asset(neighbor, &AssetRole::new(GLYPH_MASK)?, mask(31, 2, 255))?;
        fixture.input.scene = fixture.session.commit(edit.finish()?).await?.snapshot;
        assert!(matches!(
            derived.rebase_on(&fixture.input.scene),
            Err(koharu_scene::Error::PatchConflict(_))
        ));
        let (after, _) = fixture.prepare().await?;
        assert_eq!(
            after[0].1.as_rgb8().unwrap()[(31 - 12, 5)],
            Rgb([201, 176, 143])
        );
        Ok(())
    }

    #[tokio::test]
    async fn repointing_recognized_from_invalidates_prepared_ocr() -> anyhow::Result<()> {
        let mut fixture = OcrPage::new().await?;
        let derived = fixture.recognition_patch().await?;
        let base = &fixture.input.scene;
        let relation = base
            .relations_from_as::<RecognizedFrom>(fixture.content)
            .next()
            .unwrap();
        let mut edit = base.edit();
        edit.remove_relation(relation.id())?;
        edit.relate::<RecognizedFrom>(fixture.content, fixture.neighbor)?;
        let advanced = fixture.session.commit(edit.finish()?).await?.snapshot;
        assert!(matches!(
            derived.rebase_on(&advanced),
            Err(koharu_scene::Error::PatchConflict(_))
        ));
        Ok(())
    }

    #[tokio::test]
    async fn a_user_crop_without_a_mask_keeps_pixels_claimed_by_a_neighbor() -> anyhow::Result<()> {
        let mut fixture = OcrPage::new().await?;
        let role = AssetRole::new(GLYPH_MASK)?;
        let mut edit = fixture.input.scene.edit();
        edit.remove_asset(fixture.target, &role)?;
        let snapshot = fixture.session.commit(edit.finish()?).await?.snapshot;
        let mut edit = snapshot.edit_as(generation("dev.koharu.pipeline.detection", "test")?);
        let generated = edit.add_analysis_region::<TextRegion>(
            fixture.input.page,
            At::End,
            &Geometry::rectangle(20., 12., 15., 6.),
            None,
        )?;
        edit.set_asset(generated, &role, mask(22, 11, 255))?;
        fixture.input.scene = fixture.session.commit(edit.finish()?).await?.snapshot;
        let (targets, _) = fixture.prepare().await?;
        let crop = targets[0].1.as_rgb8().unwrap();
        for x in 22..33 {
            assert_eq!(crop[(x - 12, 5)], Rgb([13, 14, 15]));
        }

        // The empty asset can be stored outside the crop: its presence, even
        // when its PNG is never decoded, enables exclusive neighbour masking.
        let mut edit = fixture.input.scene.edit();
        edit.set_asset(fixture.target, &role, mask(50, 1, 0))?;
        fixture.input.scene = fixture.session.commit(edit.finish()?).await?.snapshot;
        let (targets, _) = fixture.prepare().await?;
        let crop = targets[0].1.as_rgb8().unwrap();
        for x in 22..33 {
            assert_eq!(crop[(x - 12, 5)], Rgb([201, 176, 143]));
        }
        Ok(())
    }

    #[test]
    fn crops_cover_the_source_frame_clipped_to_the_page() {
        let source = DynamicImage::new_luma8(200, 100);

        let column = crop(&source, &Geometry::rectangle(50.0, 20.0, 20.0, 60.0)).unwrap();
        let edge = crop(&source, &Geometry::rectangle(-4.0, -7.0, 40.0, 100.0)).unwrap();

        assert_eq!((column.width(), column.height()), (20, 60));
        assert_eq!((edge.width(), edge.height()), (36, 93));
    }

    #[test]
    fn symbols_read_from_a_column_of_dots_are_an_ellipsis() {
        for text in [
            "☐ ☐ ☐",
            "□\n□",
            "○●●",
            "☑ ☐ ☒ ☐ ☐",
            "● ● ● ● ●",
            "☀️ 🌞 ☁️ ☉ ⚪️ ☛️",
            "・ ●  ・ ●  ・",
            "●・",
        ] {
            assert_eq!(normalize_ocr_text(text.to_owned()), "…", "{text}");
        }
    }

    #[test]
    fn ordinary_text_lettering_symbols_and_single_boxes_are_unchanged() {
        for text in [
            "待って…",
            "・・・",
            "……",
            "!!",
            "♡♡",
            "♪♪",
            "♡・",
            "○○…",
            "☐",
            "●待って",
        ] {
            assert_eq!(normalize_ocr_text(text.to_owned()), text);
        }
    }
}
