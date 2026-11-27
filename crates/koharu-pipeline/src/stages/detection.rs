use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::{Arc, Mutex},
};

use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use image::{
    DynamicImage, ExtendedColorType, GrayImage, ImageEncoder as _, Luma, RgbImage,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use imageproc::{
    contours::{BorderType, find_contours_with_threshold},
    distance_transform::{Norm, distance_transform},
    geometry::convex_hull,
};
use koharu_ml::{
    comic_text_bubble_detector::{
        ComicTextBubbleDetection, ComicTextBubbleDetector, ComicTextBubbleLabel,
    },
    koharu_layout_rfdetr_seg_2xl::{
        GlyphProposal, KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask,
        KoharuLayoutRFDetrSeg2XL, KoharuLayoutThresholds, fit_glyph_masks,
    },
};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, BubbleRegion, DetectionAnalysis, DetectionLabel,
    EntityId, EntityOrigin, FitsTo, FlowsIn, Generation, Geometry, Inside, OcrAnalysis, Origin,
    PanelRegion, Point, RecognizedFrom, Region, RegionKind, RegionSpec, RemovePolicy,
    TextDirection, TextLayout, TextLayoutKind, TextRegion, TextRole, Typography, WritingMode,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use specta::Type;

use super::{StageInput, StageProcessor, finish, generation};
use crate::{DetectionModel, ModelCell};

const MODEL_ID: &str = "mayocream/koharu-layout-highres+ogkalu/comic-text-and-bubble-detector";
const MODEL_NAME: &str = "koharu-layout-rfdetr-seg-2xl";
const PRODUCER: &str = "dev.koharu.pipeline.detection";
/// A region-local, Luma8 PNG. Its x and y attributes locate its top-left page pixel.
pub(super) const GLYPH_MASK: &str = "glyph-mask";
const TYPOGRAPHY_SAMPLE_MARGIN: f32 = 2.0;
const COLOR_SNAP_DARK: u8 = 88;
const COLOR_SNAP_LIGHT: u8 = 191;
const COLOR_CLUSTER_MIN_DISTANCE_SQUARED: u32 = 32 * 32;
const COLOR_CLUSTER_COUNT: usize = 4;
const MIN_EXTREME_COLOR_PIXELS: u32 = 4;
const MIN_MEASURED_STROKE_WIDTH: u8 = 2;
const DUPLICATE_IOU: f32 = 0.5;
const DIALOGUE_MASK_CONTAINMENT_THRESHOLD: f32 = 0.9;
/// Largest gap, in line thicknesses, between lines of one text block.
const TEXT_BLOCK_GAP: f32 = 0.6;
/// Share of a text block a bubble must hold to be joined with the rest of a
/// split balloon.
const SPLIT_BUBBLE_MIN_SHARE: f32 = 0.2;
// RF-DETR proposals far below its own bubble threshold still carry usable
// masks for bubbles that the box-only bubble detector finds.
const BUBBLE_MASK_CANDIDATE_THRESHOLD: f32 = 0.05;
const BUBBLE_MASK_MIN_IOU: f32 = 0.5;
const RFDETR_TEXT_LABEL_ID: usize = 0;
const RFDETR_BUBBLE_LABEL_ID: usize = 2;
// RF-DETR labels handwritten asides `onomatopoeia`, like sound effects. The
// comic detector boxes the asides as text even at low confidence and never
// boxes sound effects, so its agreement separates the two.
const COMIC_TEXT_AGREEMENT_THRESHOLD: f32 = 0.3;
const COMIC_TEXT_AGREEMENT_CONTAINMENT: f32 = 0.5;
// Text the comic detector finds alone, such as short exclamations in narrow
// balloons, needs a confident box.
const COMIC_TEXT_THRESHOLD: f32 = 0.5;
const COMIC_TEXT_MAX_COVERAGE: f32 = 0.3;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct KoharuLayoutRFDetrSeg2XLConfig {
    pub text_threshold: Option<f32>,
    /// Applies to the comic bubble detector, which supplies every bubble box.
    pub bubble_threshold: Option<f32>,
    pub panel_threshold: Option<f32>,
}

pub(super) struct Processor {
    config: DetectionModel,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
}

impl Processor {
    pub(super) fn new(mut config: DetectionModel, device: koharu_ml::Device) -> Self {
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(settings) = &mut config;
        for (name, value) in [
            ("text", &mut settings.text_threshold),
            ("bubble", &mut settings.bubble_threshold),
            ("panel", &mut settings.panel_threshold),
        ] {
            // A stored threshold is only a preference; refusing to start over one
            // leaves the application unusable until the file is edited by hand.
            if let Some(threshold) = *value
                && !(threshold.is_finite() && (0.0..=1.0).contains(&threshold))
            {
                tracing::warn!(
                    label = name,
                    threshold = %threshold,
                    "confidence threshold is not between zero and one; using the model default"
                );
                *value = None;
            }
        }

        Self {
            config,
            device,
            model: ModelCell::new(),
        }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        MODEL_NAME
    }

    fn skip(&self, input: &StageInput) -> Result<bool> {
        for entity in input.scene.descendants(input.page)? {
            let id = entity.id();
            if input.contains_entity(id)?
                && entity
                    .component::<Region>()?
                    .is_some_and(|region| region.kind == TextRegion::kind())
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn unload(&self) -> bool {
        self.model.unload()
    }

    async fn load(&self) -> Result<()> {
        self.model
            .ensure(|| Model::load(self.device.clone(), &self.config))
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        self.model
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| anyhow!("detection model is not loaded"))?
            .run(input)
            .await
    }
}

/// RF-DETR supplies text, panels and masks. Bubble boxes come from the comic
/// bubble detector, which finds translucent balloons RF-DETR misses.
struct Model {
    network: Arc<Mutex<KoharuLayoutRFDetrSeg2XL>>,
    bubbles: Arc<Mutex<ComicTextBubbleDetector>>,
    thresholds: KoharuLayoutThresholds,
    bubble_threshold: f32,
}

impl Model {
    async fn load(device: koharu_ml::Device, config: &DetectionModel) -> Result<Self> {
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(config) = config;
        let network = KoharuLayoutRFDetrSeg2XL::load(device.clone())
            .await?
            .with_bubble_proposals();
        let bubbles = ComicTextBubbleDetector::load(device).await?;
        let mut thresholds = network.recommended_thresholds();
        thresholds.text = config.text_threshold.unwrap_or(thresholds.text);
        thresholds.bubble = BUBBLE_MASK_CANDIDATE_THRESHOLD;
        thresholds.panel = config.panel_threshold.unwrap_or(thresholds.panel);
        Ok(Self {
            network: Arc::new(Mutex::new(network)),
            bubbles: Arc::new(Mutex::new(bubbles)),
            thresholds,
            bubble_threshold: config
                .bubble_threshold
                .unwrap_or(ComicTextBubbleDetector::RECOMMENDED_BUBBLE_THRESHOLD),
        })
    }

    async fn run(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let image = input.require_page_image().await?;
        let output = self.detect(image.clone()).await?;
        build_patch(&input, &image, output, &generation(PRODUCER, MODEL_ID)?).await
    }

    async fn detect(&self, image: Arc<DynamicImage>) -> Result<Vec<Candidate>> {
        let network = self.network.clone();
        let bubbles = self.bubbles.clone();
        let thresholds = self.thresholds;
        let bubble_threshold = self.bubble_threshold;
        tokio_rayon::spawn(move || {
            let layout = network
                .lock()
                .map_err(|_| anyhow!("layout model lock is poisoned"))?
                .inference_with_thresholds(&image, thresholds)?;
            let comic = bubbles
                .lock()
                .map_err(|_| anyhow!("bubble model lock is poisoned"))?
                .inference(&image, bubble_threshold.min(COMIC_TEXT_AGREEMENT_THRESHOLD))?;
            Ok(with_comic_detections(layout, &comic, bubble_threshold))
        })
        .await
    }
}

struct Candidate {
    detection: KoharuLayoutDetection,
    proposal: GlyphProposal,
}

/// Combines RF-DETR with the comic detector, which finds text and balloons
/// RF-DETR misses.
///
/// Text and onomatopoeia boxes first grow to their masks within the comic
/// detector's text boxes. Onomatopoeia the comic detector also boxes as text
/// becomes text, and confident comic text boxes that no RF-DETR box covers
/// use the comic box as a source-ink search window.
///
/// RF-DETR bubbles are replaced with the comic detector's boxes. Each box takes
/// the mask of the best-overlapping RF-DETR bubble proposal, clipped to the
/// box so a proposal spanning both lobes of a joined balloon cannot hand one
/// lobe the other's area; otherwise its
/// mask is the inscribed ellipse plus the text it contains, so containment
/// still links that text to the bubble.
fn with_comic_detections(
    layout: KoharuLayoutDetections,
    comic: &[ComicTextBubbleDetection],
    bubble_threshold: f32,
) -> Vec<Candidate> {
    let KoharuLayoutDetections {
        image_width,
        image_height,
        detections,
    } = layout;
    let page = [0.0, 0.0, image_width as f32, image_height as f32];
    let (proposals, mut detections): (Vec<_>, Vec<_>) = detections
        .into_iter()
        .filter(|d| intersection_area(d.bbox, page) > 0.0)
        .partition(|detection| detection.label == "bubble");
    let original_boxes: Vec<_> = detections.iter().map(|d| d.bbox).collect();
    let comic_text = comic
        .iter()
        .filter(|detection| {
            detection.label != ComicTextBubbleLabel::Bubble
                && detection.score > COMIC_TEXT_AGREEMENT_THRESHOLD
        })
        .collect::<Vec<_>>();
    for detection in &mut detections {
        if detection.label != "text" && detection.label != "onomatopoeia" {
            continue;
        }
        fit_to_mask(detection, &comic_text);
        if detection.label == "onomatopoeia"
            && comic_text.iter().any(|text| {
                containment(text.bbox, detection.bbox) >= COMIC_TEXT_AGREEMENT_CONTAINMENT
            })
        {
            detection.label = "text".to_owned();
            detection.label_id = RFDETR_TEXT_LABEL_ID;
        }
    }
    let mut added_text = Vec::new();
    for text in comic_text
        .iter()
        .filter(|text| text.score > COMIC_TEXT_THRESHOLD)
    {
        let bbox = clip_bounds(text.bbox, page);
        let covered = detections
            .iter()
            .filter(|detection| detection.label == "text" || detection.label == "onomatopoeia")
            .map(|detection| intersection_area(detection.bbox, bbox))
            .sum::<f32>();
        if area(bbox) < 1.0 || covered >= area(bbox) * COMIC_TEXT_MAX_COVERAGE {
            continue;
        }
        let mask = box_mask(bbox);
        added_text.push(KoharuLayoutDetection {
            label_id: RFDETR_TEXT_LABEL_ID,
            label: "text".to_owned(),
            score: text.score,
            bbox,
            area: mask.width * mask.height,
            mask: mask.into(),
        });
    }
    detections.extend(added_text);
    let mut bubble_detections = Vec::new();
    for bubble in comic.iter().filter(|bubble| {
        bubble.label == ComicTextBubbleLabel::Bubble && bubble.score > bubble_threshold
    }) {
        let bbox = clip_bounds(bubble.bbox, page);
        if area(bbox) < 1.0 {
            continue;
        }
        let mask = proposals
            .iter()
            .filter(|proposal| proposal.area > 0 && valid_mask(&proposal.mask))
            .map(|proposal| (intersection_over_union(proposal.bbox, bbox), proposal))
            .filter(|(iou, _)| *iou >= BUBBLE_MASK_MIN_IOU)
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .map_or_else(
                || ellipse_mask(bbox, &detections),
                |(_, proposal)| {
                    // Over translucent balloons the proposal mask hugs the
                    // text or loses bites to the artwork behind it, while the
                    // detector's box follows the balloon. The inscribed
                    // ellipse restores the body; the proposal keeps corners
                    // of balloons that are not elliptical.
                    let mut mask = ellipse_mask(bbox, &detections);
                    let mut image = GrayImage::from_raw(mask.width, mask.height, mask.pixels)
                        .expect("ellipse mask matches its size");
                    stamp_mask(
                        &mut image,
                        &clip_mask(&proposal.mask, covering_pixels(bbox.map(f64::from))),
                        mask.x,
                        mask.y,
                    );
                    mask.pixels = image.into_raw();
                    mask
                },
            );
        bubble_detections.push(KoharuLayoutDetection {
            label_id: RFDETR_BUBBLE_LABEL_ID,
            label: "bubble".to_owned(),
            score: bubble.score,
            bbox,
            area: mask.pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: mask.into(),
        });
    }
    detections.extend(bubble_detections);
    detections
        .into_iter()
        .enumerate()
        .map(|(i, detection)| Candidate {
            proposal: original_boxes
                .get(i)
                .copied()
                .map_or(GlyphProposal::Box, GlyphProposal::Segmented),
            detection,
        })
        .collect()
}

/// RF-DETR predicts a box and a mask per query, and the box can end a glyph or
/// more short of the column its mask covers, which cuts that glyph from OCR.
/// Masks also run on into neighbouring columns and asides, which the box
/// must not take in. The comic detector boxes whole text blocks, so the box
/// grows to its mask only within the comic text boxes it overlaps, and mask
/// pixels outside them are dropped. Without such a box the box is kept.
fn fit_to_mask(detection: &mut KoharuLayoutDetection, comic_text: &[&ComicTextBubbleDetection]) {
    let Some(window) = comic_text
        .iter()
        .filter(|text| intersection_area(text.bbox, detection.bbox) > 0.0)
        .map(|text| text.bbox)
        .reduce(|left, right| {
            [
                left[0].min(right[0]),
                left[1].min(right[1]),
                left[2].max(right[2]),
                left[3].max(right[3]),
            ]
        })
    else {
        return;
    };
    if !valid_mask(&detection.mask) {
        return;
    }
    let [left, top, right, bottom] = detection.bbox;
    let mask = clip_mask(
        &detection.mask,
        covering_pixels(
            [
                window[0].min(left),
                window[1].min(top),
                window[2].max(right),
                window[3].max(bottom),
            ]
            .map(f64::from),
        ),
    );
    let Some([x, y, mask_right, mask_bottom]) = mask_extents(&mask) else {
        return;
    };
    detection.mask = clip_mask(&mask, [x, y, mask_right, mask_bottom]).into();
    detection.area = detection
        .mask
        .pixels
        .iter()
        .filter(|value| **value != 0)
        .count() as u32;
    detection.bbox = [
        left.min(x as f32),
        top.min(y as f32),
        right.max(mask_right as f32),
        bottom.max(mask_bottom as f32),
    ];
}

fn mask_extents(mask: &KoharuLayoutMask) -> Option<[u32; 4]> {
    if !valid_mask(mask) || mask.width == 0 {
        return None;
    }
    let mut extents: Option<[u32; 4]> = None;
    for (index, _) in mask
        .pixels
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0)
    {
        let x = mask.x + (index % mask.width as usize) as u32;
        let y = mask.y + (index / mask.width as usize) as u32;
        extents = Some(extents.map_or([x, y, x + 1, y + 1], |bounds| {
            union_bounds(bounds, [x, y, x + 1, y + 1])
        }));
    }
    extents
}

fn box_mask(bbox: [f32; 4]) -> KoharuLayoutMask {
    let x = bbox[0].floor() as u32;
    let y = bbox[1].floor() as u32;
    let width = (bbox[2].ceil() as u32).saturating_sub(x).max(1);
    let height = (bbox[3].ceil() as u32).saturating_sub(y).max(1);
    KoharuLayoutMask {
        x,
        y,
        width,
        height,
        pixels: vec![u8::MAX; (width * height) as usize],
    }
}

fn clip_bounds(bounds: [f32; 4], page: [f32; 4]) -> [f32; 4] {
    [
        bounds[0].clamp(page[0], page[2]),
        bounds[1].clamp(page[1], page[3]),
        bounds[2].clamp(page[0], page[2]),
        bounds[3].clamp(page[1], page[3]),
    ]
}

fn covering_pixels([left, top, right, bottom]: [f64; 4]) -> [u32; 4] {
    [
        left.floor() as u32,
        top.floor() as u32,
        right.ceil() as u32,
        bottom.ceil() as u32,
    ]
}

fn scope_pixels(scope: crate::Bounds) -> [u32; 4] {
    covering_pixels([
        scope.x,
        scope.y,
        scope.x + scope.width,
        scope.y + scope.height,
    ])
}

fn clip_mask(mask: &KoharuLayoutMask, bounds: [u32; 4]) -> KoharuLayoutMask {
    let left = bounds[0].max(mask.x);
    let top = bounds[1].max(mask.y);
    let right = bounds[2].min(mask.x + mask.width);
    let bottom = bounds[3].min(mask.y + mask.height);
    if left >= right || top >= bottom {
        return KoharuLayoutMask {
            x: left,
            y: top,
            width: 0,
            height: 0,
            pixels: Vec::new(),
        };
    }
    let pixels = (top..bottom)
        .flat_map(|y| {
            let row = ((y - mask.y) * mask.width) as usize;
            mask.pixels[row + (left - mask.x) as usize..row + (right - mask.x) as usize]
                .iter()
                .copied()
        })
        .collect();
    KoharuLayoutMask {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
        pixels,
    }
}

fn ellipse_mask(bbox: [f32; 4], detections: &[KoharuLayoutDetection]) -> KoharuLayoutMask {
    let x = bbox[0].floor() as u32;
    let y = bbox[1].floor() as u32;
    let width = (bbox[2].ceil() as u32).saturating_sub(x).max(1);
    let height = (bbox[3].ceil() as u32).saturating_sub(y).max(1);
    let (radius_x, radius_y) = ((bbox[2] - bbox[0]) / 2.0, (bbox[3] - bbox[1]) / 2.0);
    let (center_x, center_y) = (bbox[0] + radius_x, bbox[1] + radius_y);
    let mut mask = GrayImage::from_fn(width, height, |local_x, local_y| {
        let dx = (x as f32 + local_x as f32 + 0.5 - center_x) / radius_x;
        let dy = (y as f32 + local_y as f32 + 0.5 - center_y) / radius_y;
        Luma([if dx * dx + dy * dy <= 1.0 { u8::MAX } else { 0 }])
    });
    for text in detections.iter().filter(|detection| {
        detection.label == "text"
            && containment(bbox, detection.bbox) >= DIALOGUE_MASK_CONTAINMENT_THRESHOLD
    }) {
        stamp_mask(&mut mask, &text.mask, x, y);
    }
    KoharuLayoutMask {
        x,
        y,
        width,
        height,
        pixels: mask.into_raw(),
    }
}

enum RegionOutput {
    Bubble(EntityId),
    TextLayer(EntityId),
    Other,
}

#[derive(Clone, Copy)]
struct ImageSize {
    width: u32,
    height: u32,
}

async fn build_patch(
    input: &StageInput,
    image: &DynamicImage,
    output: Vec<Candidate>,
    generation: &Generation,
) -> Result<koharu_scene::Patch> {
    let page = input.page;
    let mut edit = input.scene.edit_as(generation.clone());
    edit.observe_subtree(page)?;
    write_page(input, &mut edit, page, image, output, generation)
        .await
        .context("failed to write detection output")?;
    finish(edit)
}

fn remove_previous_regions(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    generation: &Generation,
) -> Result<Vec<(EntityId, [f32; 4])>> {
    let scene = &input.scene;
    let mut remove = BTreeSet::new();
    let mut balloons = Vec::new();
    for entity in scene.descendants(input.page)? {
        let Some(region) = entity.component::<Region>()? else {
            continue;
        };
        let owned = entity.component::<EntityOrigin>()?.is_some_and(|origin| {
            matches!(origin.origin, Origin::Generated(ref owner) if owner.producer == generation.producer)
        });
        if !owned
            || entity
                .component::<OcrAnalysis>()?
                .is_some_and(|analysis| analysis.origin == Origin::User)
        {
            continue;
        }
        let id = entity.id();
        if input.contains_entity(id)? {
            remove.insert(id);
        }
        if input.region.is_some() && region.kind == BubbleRegion::kind() {
            balloons.push(id);
        }
    }

    let mut preserved = Vec::new();
    for id in balloons {
        if remove.contains(&id) && !balloon_has_preserved_text(input, id, &remove)? {
            continue;
        }
        remove.remove(&id);
        let geometry = scene.analysis_region(id)?.geometry()?;
        preserved.push((id, balloon_bounds(&geometry)));
    }
    for id in remove {
        edit.remove_entity(id, RemovePolicy::Cascade)?;
    }
    Ok(preserved)
}

fn balloon_has_preserved_text(
    input: &StageInput,
    balloon: EntityId,
    remove: &BTreeSet<EntityId>,
) -> Result<bool> {
    let scene = &input.scene;
    if scene
        .relations_to_as::<Inside>(balloon)
        .any(|inside| !remove.contains(&inside.value().source))
    {
        return Ok(true);
    }
    // A layer can flow into a balloon independently of its source region's Inside relation.
    for flow in scene.relations_to_as::<FlowsIn>(balloon) {
        let layer = flow.value().source;
        if let Some(frame) = scene.component::<Geometry>(layer)?
            && input
                .region
                .is_some_and(|selection| !selection.intersects(&frame))
        {
            return Ok(true);
        }
        if scene
            .text_layer(layer)?
            .content()?
            .source_region()?
            .is_none_or(|region| !remove.contains(&region.id()))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn balloon_bounds(geometry: &Geometry) -> [f32; 4] {
    let (left, top, right, bottom) =
        crate::scope::geometry_extents(geometry).expect("balloon geometry has vertices");
    [left, top, right, bottom].map(|v| v as f32)
}

async fn write_page(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    image: &DynamicImage,
    output: Vec<Candidate>,
    generation: &Generation,
) -> Result<()> {
    let size = ImageSize {
        width: image.width(),
        height: image.height(),
    };
    let mut candidates = output;
    non_maximum_suppression(&mut candidates, DUPLICATE_IOU);
    let converted;
    let image = match image {
        DynamicImage::ImageRgb8(image) => image,
        image => {
            converted = image.to_rgb8();
            &converted
        }
    };
    fit_glyph_masks(
        candidates
            .iter_mut()
            .map(|c| (c.proposal, &mut c.detection)),
        image,
    );
    let mut detections: Vec<_> = candidates.into_iter().map(|c| c.detection).collect();
    // A text block can join two balloon lobes even when only one lobe is in scope.
    merge_text_blocks(&mut detections);
    join_split_bubbles(&mut detections);
    if let Some(region) = input.region {
        detections.retain(|detection| intersects(detection.bbox, region));
    }
    write_regions(input, edit, image, &detections, generation)
        .context("failed to write detected regions")?;
    write_masks(input, edit, page, &detections, size)
        .await
        .context("failed to write detection masks")
}

fn write_regions(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    image: &RgbImage,
    detections: &[KoharuLayoutDetection],
    generation: &Generation,
) -> Result<()> {
    let mut preserved_balloons = remove_previous_regions(input, edit, generation)
        .context("failed to replace the previous detection regions")?;
    let snapshot = &input.scene;
    let page = input.page;
    let order = layout_order(detections);
    let inferred = order
        .par_iter()
        .map(|item| {
            let detection = &detections[item.index];
            (detection.label == "text")
                .then(|| infer_typography(image, detection))
                .flatten()
        })
        .collect::<Vec<_>>();
    let text_group = snapshot.page(page)?.text_group()?;
    let managed_text_group = if let Some(group) = text_group {
        snapshot
            .component::<EntityOrigin>(group.id())?
            .is_some_and(|origin| origin.origin != Origin::User)
            .then_some(group.id())
    } else {
        None
    };
    let mut bubble_entities = vec![None; detections.len()];
    for (item, inferred) in order.into_iter().zip(inferred) {
        let detection = &detections[item.index];
        let bubble = item
            .bubble
            .map(|index| bubble_entities[index].expect("balloon precedes its text"));
        match write_region(
            edit,
            input,
            detection,
            inferred,
            bubble,
            generation,
            &mut preserved_balloons,
        )
        .with_context(|| {
            format!(
                "failed to write {} detection {}",
                detection.label, item.index
            )
        })? {
            RegionOutput::Bubble(entity) => bubble_entities[item.index] = Some(entity),
            RegionOutput::TextLayer(layer) => {
                if let Some(group) = managed_text_group {
                    edit.move_entity(layer, Some(group), At::End)
                        .context("failed to place the detected text layer in its group")?;
                }
            }
            RegionOutput::Other => {}
        }
    }
    Ok(())
}

fn write_region(
    edit: &mut koharu_scene::Edit,
    input: &StageInput,
    detection: &KoharuLayoutDetection,
    inferred: Option<InferredTypography>,
    bubble: Option<EntityId>,
    generation: &Generation,
    preserved_balloons: &mut Vec<(EntityId, [f32; 4])>,
) -> Result<RegionOutput> {
    let page = input.page;
    let selection = input.region;
    let kind = region_kind(&detection.label)?;
    let geometry = if detection.label == "bubble" {
        let geometry =
            mask_geometry(&detection.mask).unwrap_or_else(|| rectangle_geometry(detection.bbox));
        if !preserved_balloons.is_empty() {
            let bounds = balloon_bounds(&geometry);
            let best = preserved_balloons
                .iter()
                .enumerate()
                .map(|(index, (_, existing))| (index, intersection_over_union(*existing, bounds)))
                .filter(|(_, overlap)| *overlap >= DUPLICATE_IOU)
                .max_by(|(_, left), (_, right)| left.total_cmp(right));
            if let Some((index, _)) = best {
                return Ok(RegionOutput::Bubble(
                    preserved_balloons.swap_remove(index).0,
                ));
            }
        }
        geometry
    } else if detection.label == "text" {
        let angle = inferred.map(|typography| typography.angle_degrees);
        let Some(geometry) = text_geometry(detection, angle, selection) else {
            return Ok(RegionOutput::Other);
        };
        geometry
    } else {
        rectangle_geometry(detection.bbox)
    };
    let entity = edit
        .add_entity(page, At::End)
        .context("failed to create a detected region")?;
    edit.set(entity, &geometry)
        .context("failed to set detected region geometry")?;
    edit.set(
        entity,
        &Region {
            origin: Origin::Generated(generation.clone()),
            kind: kind.clone(),
            label: Some(detection.label.clone()),
        },
    )
    .context("failed to set detected region metadata")?;
    edit.set(
        entity,
        &DetectionAnalysis {
            origin: Origin::Generated(generation.clone()),
            labels: vec![DetectionLabel {
                kind,
                confidence: detection.score,
            }],
        },
    )
    .context("failed to set detection analysis")?;
    if detection.label != "text" {
        return Ok(if detection.label == "bubble" {
            RegionOutput::Bubble(entity)
        } else {
            RegionOutput::Other
        });
    }

    let clipped = selection.map(|bounds| clip_mask(&detection.mask, scope_pixels(bounds)));
    let mask = clipped.as_ref().unwrap_or(&detection.mask);
    // A fully occluded text still owns an explicit empty mask. PNG has no
    // zero-sized image, so encode a single transparent-to-cleanup pixel.
    let asset = if mask.width == 0 || mask.height == 0 {
        mask_asset(
            &[0],
            1,
            1,
            Some([detection.bbox[0] as u32, detection.bbox[1] as u32]),
        )?
    } else {
        mask_asset(
            &mask.pixels,
            mask.width,
            mask.height,
            Some([mask.x, mask.y]),
        )?
    };
    edit.set_asset(entity, &AssetRole::new(GLYPH_MASK)?, asset)?;
    if let Some(typography) = inferred {
        edit.set(
            entity,
            &OcrAnalysis {
                origin: Origin::Generated(generation.clone()),
                direction: if typography.writing_mode == WritingMode::Vertical {
                    TextDirection::Vertical
                } else {
                    TextDirection::Horizontal
                },
                confidence: None,
                line_boundaries: Vec::new(),
            },
        )?;
    }
    let content = edit.add_text_content(page, At::End)?;
    let layer = edit.add_text_layer(
        page,
        At::End,
        content,
        &TextLayout {
            origin: Origin::Generated(generation.clone()),
            kind: TextLayoutKind::Paragraph,
            angle_degrees: None,
        },
    )?;
    edit.set(
        layer,
        &Typography {
            origin: Origin::Generated(generation.clone()),
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: inferred.map(|value| value.size),
            auto_fit: true,
            color: inferred.map(|value| [value.color[0], value.color[1], value.color[2], u8::MAX]),
            stroke_color: inferred
                .and_then(|value| value.stroke_color)
                .map(|color| [color[0], color[1], color[2], u8::MAX]),
            stroke_width: inferred.and_then(|value| value.stroke_width),
            alignment: None,
            writing_mode: inferred.map(|value| value.writing_mode),
            extensions: Default::default(),
        },
    )
    .context("failed to set detected typography")?;
    edit.relate::<RecognizedFrom>(content, entity)
        .context("failed to associate detected text content with its region")?;

    let role = if let Some(bubble) = bubble {
        edit.relate::<Inside>(entity, bubble)?;
        edit.relate::<FlowsIn>(layer, bubble)?;
        "dev.koharu.text.dialogue"
    } else {
        edit.relate::<FitsTo>(layer, entity)?;
        "dev.koharu.text.free-text"
    };
    edit.set(
        content,
        &TextRole {
            origin: Origin::Generated(generation.clone()),
            role: role.to_owned(),
        },
    )?;
    Ok(RegionOutput::TextLayer(layer))
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct InferredTypography {
    size: f32,
    color: [u8; 3],
    stroke_color: Option<[u8; 3]>,
    stroke_width: Option<f32>,
    angle_degrees: f32,
    writing_mode: WritingMode,
}

#[derive(Clone, Copy)]
struct MaskPixel {
    x: u32,
    y: u32,
    color: [u8; 3],
    inside_mask: bool,
}

fn infer_typography(
    image: &RgbImage,
    detection: &KoharuLayoutDetection,
) -> Option<InferredTypography> {
    let mask = &detection.mask;
    let measured = mask.typography(image)?;
    let width = image.width();
    let height = image.height();
    let [bbox_left, bbox_top, bbox_right, bbox_bottom] = detection.bbox;
    let [left, top, right, bottom] = mask_window(
        [
            bbox_left - TYPOGRAPHY_SAMPLE_MARGIN,
            bbox_top - TYPOGRAPHY_SAMPLE_MARGIN,
            bbox_right + TYPOGRAPHY_SAMPLE_MARGIN,
            bbox_bottom + TYPOGRAPHY_SAMPLE_MARGIN,
        ],
        width,
        height,
    )?;
    let local_width = right - left + 2;
    let local_height = bottom - top + 2;
    let background_margin = TYPOGRAPHY_SAMPLE_MARGIN.ceil() as u32;
    let mut pixels = Vec::new();
    let mut background = Vec::new();
    for y in top..bottom {
        for x in left..right {
            let local_x = x - left + 1;
            let local_y = y - top + 1;
            let color = image.get_pixel(x, y).0;
            let inside_mask = mask.contains(x, y);
            pixels.push(MaskPixel {
                x: local_x,
                y: local_y,
                color,
                inside_mask,
            });
            if x < left + background_margin
                || x + background_margin >= right
                || y < top + background_margin
                || y + background_margin >= bottom
            {
                background.push(color);
            }
        }
    }

    let background = if background.is_empty() {
        median_pixel_color(&pixels)
    } else {
        median_color(&background)
    };
    let (mut color, mut stroke_color, mut stroke_width) =
        infer_text_paint(&pixels, background, local_width, local_height);
    // The clustered paint distinguishes antialias bands and single glyph
    // counters. The source outline resolves the inverse case over dark art,
    // where the glyph fill and the surrounding art can share a palette entry.
    if stroke_color.is_none()
        && let (Some(ink), Some(outline)) = (measured.ink, measured.outline)
        && normalize_text_color(ink) != normalize_text_color(outline.color)
        && !color_lies_between(outline.color, ink, background)
        // A single thick island matching the outside paper is a counter.
        && (ink != background || measured.stroke_width.is_none_or(|w| w < 0.6 * measured.size))
    {
        color = normalize_text_color(ink);
        stroke_color = Some(normalize_text_color(outline.color));
        stroke_width = Some(outline.width);
    }
    Some(InferredTypography {
        size: measured.size,
        color,
        stroke_color,
        stroke_width,
        angle_degrees: measured.tilt,
        writing_mode: if measured.vertical {
            WritingMode::Vertical
        } else {
            WritingMode::Horizontal
        },
    })
}

fn mask_window([left, top, right, bottom]: [f32; 4], width: u32, height: u32) -> Option<[u32; 4]> {
    if width == 0 || height == 0 {
        return None;
    }
    let left = left.floor().clamp(0.0, width as f32) as u32;
    let top = top.floor().clamp(0.0, height as f32) as u32;
    let right = right.ceil().clamp(0.0, width as f32) as u32;
    let bottom = bottom.ceil().clamp(0.0, height as f32) as u32;
    (right > left && bottom > top).then_some([left, top, right, bottom])
}

fn histogram_value_at(histogram: &[u32; 256], mut rank: usize) -> u8 {
    for (value, count) in histogram.iter().enumerate() {
        if rank < *count as usize {
            return value as u8;
        }
        rank -= *count as usize;
    }
    u8::MAX
}

fn histogram_median(histogram: &[u32; 256], count: usize) -> u8 {
    if count == 0 {
        return 0;
    }
    let lower = histogram_value_at(histogram, (count - 1) / 2);
    let upper = histogram_value_at(histogram, count / 2);
    ((u16::from(lower) + u16::from(upper)) / 2) as u8
}

// The detector mask may trace glyphs or cover a whole text region. Color alone
// is also insufficient because a black glyph core can be identical to artwork
// outside its white outline. A real outline is therefore identified by the
// topology of a color band enclosing another color, then measured across that
// exact band. Clustering remains the fallback for unoutlined text.
fn infer_text_paint(
    pixels: &[MaskPixel],
    background_seed: [u8; 3],
    width: u32,
    height: u32,
) -> ([u8; 3], Option<[u8; 3]>, Option<f32>) {
    let clusters = color_clusters(pixels, background_seed);
    if let Some(outline) = infer_outline(&clusters, background_seed, width, height) {
        return (
            outline.fill_color,
            Some(outline.stroke_color),
            Some(outline.stroke_width),
        );
    }

    let nonempty = clusters
        .iter()
        .enumerate()
        .filter(|(_, cluster)| !cluster.pixels.is_empty())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if nonempty.len() == 1 {
        return (
            normalize_text_color(clusters[nonempty[0]].color),
            None,
            None,
        );
    }

    let background_index = nonempty
        .iter()
        .copied()
        .min_by_key(|index| color_distance_squared(clusters[*index].color, background_seed))
        .unwrap();
    let fill = nonempty
        .iter()
        .copied()
        .filter(|index| *index != background_index)
        .max_by_key(|index| cluster_ink_score(&clusters[*index], background_seed));
    let color = fill
        .map(|index| representative_ink_color(&clusters[index]))
        .unwrap_or(clusters[background_index].color);
    (normalize_text_color(color), None, None)
}

#[derive(Clone, Copy)]
struct OutlinePaint {
    fill_color: [u8; 3],
    stroke_color: [u8; 3],
    stroke_width: f32,
    score: f64,
}

fn infer_outline(
    clusters: &[ColorCluster; COLOR_CLUSTER_COUNT],
    background: [u8; 3],
    width: u32,
    height: u32,
) -> Option<OutlinePaint> {
    let mut assignments = vec![usize::MAX; width as usize * height as usize];
    for (index, cluster) in clusters.iter().enumerate() {
        for pixel in &cluster.pixels {
            assignments[pixel.y as usize * width as usize + pixel.x as usize] = index;
        }
    }

    let background_index = clusters
        .iter()
        .enumerate()
        .filter(|(_, cluster)| !cluster.pixels.is_empty())
        .min_by_key(|(_, cluster)| color_distance_squared(cluster.color, background))
        .map(|(index, _)| index)?;
    let mut best = None;
    for (stroke_index, stroke_cluster) in clusters.iter().enumerate() {
        if stroke_cluster.pixels.len() < 8 || stroke_index == background_index {
            continue;
        }
        let outside = reachable_without_cluster(&assignments, stroke_index, width, height);
        for (fill_index, fill_cluster) in clusters.iter().enumerate() {
            if fill_index == stroke_index || fill_cluster.pixels.is_empty() {
                continue;
            }
            let enclosed_fill = fill_cluster
                .pixels
                .iter()
                .copied()
                .filter(|pixel| !outside[pixel.y as usize * width as usize + pixel.x as usize])
                .collect::<Vec<_>>();
            if enclosed_fill.len() < 8 {
                continue;
            }
            let stroke_pixels = enclosing_cluster_pixels(
                &assignments,
                stroke_index,
                &stroke_cluster.pixels,
                &enclosed_fill,
                width,
                height,
            );
            if stroke_pixels.len() < 8 {
                continue;
            }

            let fill_color = median_pixel_color(&enclosed_fill);
            let stroke_color = median_pixel_color(&stroke_pixels);
            let contrast = color_distance_squared(fill_color, stroke_color);
            if contrast < COLOR_CLUSTER_MIN_DISTANCE_SQUARED
                || color_distance_squared(fill_color, background)
                    < COLOR_CLUSTER_MIN_DISTANCE_SQUARED
                || color_distance_squared(stroke_color, background)
                    < COLOR_CLUSTER_MIN_DISTANCE_SQUARED
                || color_lies_between(stroke_color, fill_color, background)
                || color_lies_between(fill_color, stroke_color, background)
            {
                continue;
            }
            let normalized_fill = normalize_text_color(fill_color);
            let normalized_stroke = normalize_text_color(stroke_color);
            if normalized_fill == normalized_stroke {
                continue;
            }

            let Some(stroke_width) =
                measured_stroke_width(&stroke_pixels, &enclosed_fill, &outside, width, height)
            else {
                continue;
            };
            // A real fill is almost entirely enclosed by its outline, while
            // the outline color is enclosed only inside glyph counters. Both
            // pairings enclose thousands of pixels on a high-resolution page,
            // so the enclosed share separates them, not the raw count.
            let enclosed = enclosed_fill.len() as f64;
            let candidate = OutlinePaint {
                fill_color: normalized_fill,
                stroke_color: normalized_stroke,
                stroke_width,
                score: f64::from(contrast) * enclosed * enclosed / fill_cluster.pixels.len() as f64,
            };
            if best.is_none_or(|current: OutlinePaint| candidate.score > current.score) {
                best = Some(candidate);
            }
        }
    }
    best
}

fn reachable_without_cluster(
    assignments: &[usize],
    barrier: usize,
    width: u32,
    height: u32,
) -> Vec<bool> {
    let mut reachable = vec![false; assignments.len()];
    let mut queue = VecDeque::new();
    for x in 0..width {
        push_reachable(
            &mut reachable,
            &mut queue,
            assignments,
            barrier,
            x,
            0,
            width,
        );
        push_reachable(
            &mut reachable,
            &mut queue,
            assignments,
            barrier,
            x,
            height - 1,
            width,
        );
    }
    for y in 0..height {
        push_reachable(
            &mut reachable,
            &mut queue,
            assignments,
            barrier,
            0,
            y,
            width,
        );
        push_reachable(
            &mut reachable,
            &mut queue,
            assignments,
            barrier,
            width - 1,
            y,
            width,
        );
    }
    while let Some((x, y)) = queue.pop_front() {
        if x > 0 {
            push_reachable(
                &mut reachable,
                &mut queue,
                assignments,
                barrier,
                x - 1,
                y,
                width,
            );
        }
        if x + 1 < width {
            push_reachable(
                &mut reachable,
                &mut queue,
                assignments,
                barrier,
                x + 1,
                y,
                width,
            );
        }
        if y > 0 {
            push_reachable(
                &mut reachable,
                &mut queue,
                assignments,
                barrier,
                x,
                y - 1,
                width,
            );
        }
        if y + 1 < height {
            push_reachable(
                &mut reachable,
                &mut queue,
                assignments,
                barrier,
                x,
                y + 1,
                width,
            );
        }
    }
    reachable
}

fn push_reachable(
    reachable: &mut [bool],
    queue: &mut VecDeque<(u32, u32)>,
    assignments: &[usize],
    barrier: usize,
    x: u32,
    y: u32,
    width: u32,
) {
    let index = y as usize * width as usize + x as usize;
    if !reachable[index] && assignments[index] != barrier {
        reachable[index] = true;
        queue.push_back((x, y));
    }
}

fn enclosing_cluster_pixels(
    assignments: &[usize],
    cluster: usize,
    cluster_pixels: &[MaskPixel],
    enclosed_fill: &[MaskPixel],
    width: u32,
    height: u32,
) -> Vec<MaskPixel> {
    let mut selected = vec![false; assignments.len()];
    let mut queue = VecDeque::new();
    for fill in enclosed_fill {
        for y in fill.y.saturating_sub(1)..=(fill.y + 1).min(height - 1) {
            for x in fill.x.saturating_sub(1)..=(fill.x + 1).min(width - 1) {
                let index = y as usize * width as usize + x as usize;
                if assignments[index] == cluster && !selected[index] {
                    selected[index] = true;
                    queue.push_back((x, y));
                }
            }
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        for next_y in y.saturating_sub(1)..=(y + 1).min(height - 1) {
            for next_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                let index = next_y as usize * width as usize + next_x as usize;
                if assignments[index] == cluster && !selected[index] {
                    selected[index] = true;
                    queue.push_back((next_x, next_y));
                }
            }
        }
    }
    cluster_pixels
        .iter()
        .copied()
        .filter(|pixel| selected[pixel.y as usize * width as usize + pixel.x as usize])
        .collect()
}

fn cluster_ink_score(cluster: &ColorCluster, background: [u8; 3]) -> u64 {
    let inside = cluster
        .pixels
        .iter()
        .filter(|pixel| pixel.inside_mask)
        .count();
    let evidence = if inside >= 8 {
        inside + cluster.pixels.len()
    } else {
        cluster.pixels.len()
    };
    u64::from(color_distance_squared(cluster.color, background)) * evidence.min(4096) as u64
}

fn representative_ink_color(cluster: &ColorCluster) -> [u8; 3] {
    let inside = cluster
        .pixels
        .iter()
        .copied()
        .filter(|pixel| pixel.inside_mask)
        .collect::<Vec<_>>();
    if inside.len() >= 8 {
        median_pixel_color(&inside)
    } else {
        median_pixel_color(&cluster.pixels)
    }
}

struct ColorCluster {
    color: [u8; 3],
    pixels: Vec<MaskPixel>,
}

fn color_clusters(
    pixels: &[MaskPixel],
    background: [u8; 3],
) -> [ColorCluster; COLOR_CLUSTER_COUNT] {
    let palette = color_palette(pixels);
    let darkest = extreme_palette_color(&palette, true);
    let lightest = extreme_palette_color(&palette, false);
    let distant = distant_palette_color(&palette, &[background, darkest, lightest]);
    let mut centers = [background, darkest, lightest, distant];
    for _ in 0..4 {
        let mut accumulators = [ColorAccumulator::default(); COLOR_CLUSTER_COUNT];
        for entry in &palette {
            let color = entry.color();
            let index = centers
                .iter()
                .enumerate()
                .min_by_key(|(_, center)| color_distance_squared(color, **center))
                .map(|(index, _)| index)
                .unwrap();
            accumulators[index].add(entry);
        }
        for (center, accumulator) in centers.iter_mut().zip(accumulators) {
            if let Some(color) = accumulator.color() {
                *center = color;
            }
        }
    }
    let mut groups: [Vec<MaskPixel>; COLOR_CLUSTER_COUNT] = std::array::from_fn(|_| Vec::new());
    for pixel in pixels {
        let index = centers
            .iter()
            .enumerate()
            .min_by_key(|(_, center)| color_distance_squared(pixel.color, **center))
            .map(|(index, _)| index)
            .unwrap();
        groups[index].push(*pixel);
    }
    std::array::from_fn(|index| ColorCluster {
        color: centers[index],
        pixels: std::mem::take(&mut groups[index]),
    })
}

fn extreme_palette_color(palette: &[ColorBin], darkest: bool) -> [u8; 3] {
    let select = |significant_only: bool| {
        let candidates = palette
            .iter()
            .filter(|bin| !significant_only || bin.count >= MIN_EXTREME_COLOR_PIXELS);
        if darkest {
            candidates.min_by_key(|bin| color_luminance(bin.color()))
        } else {
            candidates.max_by_key(|bin| color_luminance(bin.color()))
        }
    };
    select(true)
        .or_else(|| select(false))
        .map(ColorBin::color)
        .unwrap_or_default()
}

#[derive(Clone, Copy, Default)]
struct ColorBin {
    count: u32,
    sums: [u64; 3],
}

impl ColorBin {
    fn add(&mut self, color: [u8; 3]) {
        self.count += 1;
        for (sum, channel) in self.sums.iter_mut().zip(color) {
            *sum += u64::from(channel);
        }
    }

    fn color(&self) -> [u8; 3] {
        std::array::from_fn(|channel| {
            ((self.sums[channel] + u64::from(self.count / 2)) / u64::from(self.count)) as u8
        })
    }
}

#[derive(Clone, Copy, Default)]
struct ColorAccumulator {
    count: u64,
    sums: [u64; 3],
}

impl ColorAccumulator {
    fn add(&mut self, bin: &ColorBin) {
        self.count += u64::from(bin.count);
        for (sum, value) in self.sums.iter_mut().zip(bin.sums) {
            *sum += value;
        }
    }

    fn color(self) -> Option<[u8; 3]> {
        (self.count != 0).then(|| {
            std::array::from_fn(|channel| {
                ((self.sums[channel] + self.count / 2) / self.count) as u8
            })
        })
    }
}

fn color_palette(pixels: &[MaskPixel]) -> Vec<ColorBin> {
    let mut bins = vec![ColorBin::default(); 16 * 16 * 16];
    for pixel in pixels {
        let [red, green, blue] = pixel.color.map(|channel| usize::from(channel >> 4));
        bins[(red << 8) | (green << 4) | blue].add(pixel.color);
    }
    bins.retain(|bin| bin.count != 0);
    bins
}

fn distant_palette_color(palette: &[ColorBin], centers: &[[u8; 3]]) -> [u8; 3] {
    palette
        .iter()
        .max_by_key(|bin| {
            u64::from(minimum_color_distance(bin.color(), centers)) * u64::from(bin.count.min(64))
        })
        .map(ColorBin::color)
        .unwrap_or_default()
}

fn minimum_color_distance(color: [u8; 3], centers: &[[u8; 3]]) -> u32 {
    centers
        .iter()
        .map(|center| color_distance_squared(color, *center))
        .min()
        .unwrap_or_default()
}

fn median_pixel_color(pixels: &[MaskPixel]) -> [u8; 3] {
    let mut histograms = [[0_u32; 256]; 3];
    for pixel in pixels {
        for (histogram, channel) in histograms.iter_mut().zip(pixel.color) {
            histogram[usize::from(channel)] += 1;
        }
    }
    std::array::from_fn(|channel| histogram_median(&histograms[channel], pixels.len()))
}

fn measured_stroke_width(
    stroke_pixels: &[MaskPixel],
    fill_pixels: &[MaskPixel],
    outside: &[bool],
    width: u32,
    height: u32,
) -> Option<f32> {
    if stroke_pixels.len() < 8 || fill_pixels.len() < 8 {
        return None;
    }
    let mut fill_mask = GrayImage::from_pixel(width, height, Luma([0]));
    for pixel in fill_pixels {
        fill_mask.put_pixel(pixel.x, pixel.y, Luma([u8::MAX]));
    }
    let fill_distances = distance_transform(&fill_mask, Norm::L2);
    let mut histogram = [0_u32; 256];
    let mut count = 0_usize;
    for pixel in stroke_pixels {
        let mut touches_outside = false;
        for y in pixel.y.saturating_sub(1)..=(pixel.y + 1).min(height - 1) {
            for x in pixel.x.saturating_sub(1)..=(pixel.x + 1).min(width - 1) {
                if outside[y as usize * width as usize + x as usize] {
                    touches_outside = true;
                    break;
                }
            }
            if touches_outside {
                break;
            }
        }
        if !touches_outside {
            continue;
        }
        let distance = fill_distances.get_pixel(pixel.x, pixel.y).0[0];
        if distance == 0 {
            continue;
        }
        histogram[usize::from(distance)] += 1;
        count += 1;
    }
    if count < 8 {
        return None;
    }
    let width = histogram_median(&histogram, count);
    (width >= MIN_MEASURED_STROKE_WIDTH).then_some(f32::from(width))
}

fn color_lies_between(candidate: [u8; 3], start: [u8; 3], end: [u8; 3]) -> bool {
    let start = start.map(f64::from);
    let end = end.map(f64::from);
    let candidate = candidate.map(f64::from);
    let direction = std::array::from_fn::<_, 3, _>(|channel| end[channel] - start[channel]);
    let length_squared = direction.iter().map(|value| value * value).sum::<f64>();
    if length_squared <= f64::EPSILON {
        return false;
    }
    let projection = candidate
        .iter()
        .zip(start)
        .zip(direction)
        .map(|((&candidate, start), direction)| (candidate - start) * direction)
        .sum::<f64>()
        / length_squared;
    if !(0.0..=1.0).contains(&projection) {
        return false;
    }
    candidate
        .iter()
        .zip(start)
        .zip(direction)
        .map(|((&candidate, start), direction)| {
            let difference = candidate - (start + direction * projection);
            difference * difference
        })
        .sum::<f64>()
        <= 24.0_f64.powi(2) * 3.0
}

fn median_color(colors: &[[u8; 3]]) -> [u8; 3] {
    let mut histograms = [[0_u32; 256]; 3];
    for color in colors {
        for (histogram, channel) in histograms.iter_mut().zip(*color) {
            histogram[usize::from(channel)] += 1;
        }
    }
    std::array::from_fn(|channel| histogram_median(&histograms[channel], colors.len()))
}

fn color_distance_squared(left: [u8; 3], right: [u8; 3]) -> u32 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| i32::from(left) - i32::from(right))
        .map(|difference| difference.unsigned_abs().pow(2))
        .sum()
}

// Antialiasing and JPEG noise pull pure black and white ink toward the
// background, so near-neutral extremes snap back. Saturated ink such as red or
// blue keeps its measured color however dark it is.
fn normalize_text_color(color: [u8; 3]) -> [u8; 3] {
    if color.into_iter().max().unwrap() <= COLOR_SNAP_DARK {
        [0, 0, 0]
    } else if color.into_iter().min().unwrap() >= COLOR_SNAP_LIGHT {
        [u8::MAX; 3]
    } else {
        color
    }
}

fn color_luminance(color: [u8; 3]) -> u32 {
    u32::from(color[0]) * 54 + u32::from(color[1]) * 183 + u32::from(color[2]) * 19
}

/// Clips a polygon against the four sides of the rectangle. An edge or
/// corner touching the rectangle on its own has no area to recognize.
pub(super) fn clip_polygon(points: &[Point], bounds: crate::Bounds) -> Option<Cow<'_, [Point]>> {
    let mut polygon = Cow::Borrowed(points);
    for (vertical, edge, lower) in [
        (true, bounds.x, true),
        (true, bounds.x + bounds.width, false),
        (false, bounds.y, true),
        (false, bounds.y + bounds.height, false),
    ] {
        let distance = |p: Point| {
            let coordinate = if vertical { p.x } else { p.y };
            if lower {
                coordinate - edge
            } else {
                edge - coordinate
            }
        };
        if polygon.iter().all(|&p| distance(p) >= 0.0) {
            continue;
        }
        let mut clipped = Vec::new();
        let mut previous = *polygon.last()?;
        for &current in polygon.iter() {
            let (a, b) = (distance(previous), distance(current));
            if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
                let t = a / (a - b);
                clipped.push(if vertical {
                    Point {
                        x: edge,
                        y: previous.y + t * (current.y - previous.y),
                    }
                } else {
                    Point {
                        x: previous.x + t * (current.x - previous.x),
                        y: edge,
                    }
                });
            }
            if b >= 0.0 {
                clipped.push(current);
            }
            previous = current;
        }
        polygon = Cow::Owned(clipped);
    }
    let origin = polygon.first()?;
    let twice_area: f64 = polygon
        .windows(2)
        .map(|pair| {
            let [a, b] = [pair[0], pair[1]];
            (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x)
        })
        .sum();
    (twice_area.abs() > 0.0).then_some(polygon)
}

fn rectangle_geometry([left, top, right, bottom]: [f32; 4]) -> Geometry {
    Geometry::rectangle(
        f64::from(left),
        f64::from(top),
        f64::from((right - left).max(1.0)),
        f64::from((bottom - top).max(1.0)),
    )
}

pub(super) fn rotated_rectangle_geometry(
    detection: &KoharuLayoutDetection,
    angle_degrees: f32,
) -> Geometry {
    // A linear projection reaches its extrema at the ends of each occupied
    // row. Include the full pixel cells so the deskew crop encloses their edges.
    let mask = &detection.mask;
    let mut rows = mask
        .pixels
        .chunks_exact(mask.width.max(1) as usize)
        .enumerate()
        .filter_map(|(row, pixels)| {
            let first = pixels.iter().position(|p| *p != 0)?;
            let last = pixels.iter().rposition(|p| *p != 0).unwrap();
            let x = f64::from(mask.x);
            let y = f64::from(mask.y) + row as f64;
            Some([x + first as f64, y, x + last as f64 + 1., y + 1.])
        })
        .peekable();
    let fallback = rows
        .peek()
        .is_none()
        .then_some(detection.bbox.map(f64::from));
    let corners = rows.chain(fallback).flat_map(|[left, top, right, bottom]| {
        [(left, top), (right, top), (right, bottom), (left, bottom)]
    });
    oriented_rectangle(corners, angle_degrees)
}

fn text_geometry(
    detection: &KoharuLayoutDetection,
    angle_degrees: Option<f32>,
    selection: Option<crate::Bounds>,
) -> Option<Geometry> {
    let mut geometry = angle_degrees.map_or_else(
        || rectangle_geometry(detection.bbox),
        |angle| rotated_rectangle_geometry(detection, angle),
    );
    let Some(selection) = selection else {
        return Some(geometry);
    };
    let angle = angle_degrees.unwrap_or(0.0);
    if angle == 0.0 {
        for point in &mut geometry.points {
            point.x = point.x.clamp(selection.x, selection.x + selection.width);
            point.y = point.y.clamp(selection.y, selection.y + selection.height);
        }
        let (first, opposite) = (geometry.points[0], geometry.points[2]);
        return (first.x < opposite.x && first.y < opposite.y).then_some(geometry);
    }
    let Cow::Owned(clipped) = clip_polygon(&geometry.points, selection)? else {
        return Some(geometry);
    };
    // OCR and free-text placement need a rectangle on the original axes. Its
    // enclosing corners can spill into adjacent selections, so contract it
    // toward a point inside the convex clip until every corner is in scope.
    let count = clipped.len() as f64;
    let cx = clipped.iter().map(|p| p.x).sum::<f64>() / count;
    let cy = clipped.iter().map(|p| p.y).sum::<f64>() / count;
    let mut frame = oriented_rectangle(clipped.into_iter().map(|p| (p.x, p.y)), angle);
    let (left, top, right, bottom) = crate::scope::geometry_extents(&frame)?;
    let (x1, y1) = (
        selection.x + selection.width,
        selection.y + selection.height,
    );
    let scale = 1.0_f64
        .min((cx - selection.x) / (cx - left))
        .min((cy - selection.y) / (cy - top))
        .min((x1 - cx) / (right - cx))
        .min((y1 - cy) / (bottom - cy));
    for p in &mut frame.points {
        p.x = (cx + scale * (p.x - cx)).clamp(selection.x, x1);
        p.y = (cy + scale * (p.y - cy)).clamp(selection.y, y1);
    }
    Some(frame)
}

fn oriented_rectangle(points: impl Iterator<Item = (f64, f64)>, angle_degrees: f32) -> Geometry {
    let (sin, cos) = f64::from(angle_degrees).to_radians().sin_cos();
    let (mut u0, mut v0, mut u1, mut v1) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (x, y) in points {
        let (u, v) = (x * cos + y * sin, y * cos - x * sin);
        u0 = u0.min(u);
        v0 = v0.min(v);
        u1 = u1.max(u);
        v1 = v1.max(v);
    }
    Geometry {
        origin: Origin::User,
        points: [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            .map(|(u, v)| Point {
                x: u * cos - v * sin,
                y: u * sin + v * cos,
            })
            .into(),
    }
}

fn mask_geometry(mask: &KoharuLayoutMask) -> Option<Geometry> {
    let origin = (mask.x, mask.y);
    let mask = GrayImage::from_raw(mask.width, mask.height, mask.pixels.clone())?;
    let mut padded = GrayImage::new(mask.width() + 2, mask.height() + 2);
    image::imageops::replace(&mut padded, &mask, 1, 1);
    // Balloons are convex outside joins between lobes, and the bubble detector
    // boxes each lobe separately. Notches in a mask are artwork showing
    // through a translucent balloon, and would leave text only the narrow
    // spans between them, so the outline is the mask's convex hull.
    let points = find_contours_with_threshold::<i32>(&padded, 0)
        .into_iter()
        .filter(|contour| contour.border_type == BorderType::Outer)
        .flat_map(|contour| contour.points)
        .collect::<Vec<_>>();
    let points = convex_hull(points)
        .into_iter()
        .map(|point| Point {
            x: f64::from(point.x - 1) + f64::from(origin.0),
            y: f64::from(point.y - 1) + f64::from(origin.1),
        })
        .collect::<Vec<_>>();
    (points.len() >= 3).then_some(Geometry {
        origin: Origin::User,
        points,
    })
}

async fn write_masks(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    detections: &[KoharuLayoutDetection],
    size: ImageSize,
) -> Result<()> {
    write_mask(input, edit, page, detections, size).await?;
    edit.remove_asset(page, &AssetRole::new("bubble-mask")?)?;
    Ok(())
}

async fn write_mask(
    input: &StageInput,
    edit: &mut koharu_scene::Edit,
    page: EntityId,
    detections: &[KoharuLayoutDetection],
    size: ImageSize,
) -> Result<()> {
    let mut mask = mask_for(detections, "text", size);
    if let Some(bounds) = input.region {
        preserve_mask_outside_region(input, page, "text-mask", bounds, &mut mask).await?;
    }
    edit.set_asset(
        page,
        &AssetRole::new("text-mask")?,
        mask_asset(mask.as_raw(), size.width, size.height, None)?,
    )?;
    Ok(())
}

fn mask_asset(
    pixels: &[u8],
    width: u32,
    height: u32,
    origin: Option<[u32; 2]>,
) -> Result<AssetInput> {
    let mut bytes = Vec::new();
    PngEncoder::new_with_quality(&mut bytes, CompressionType::Fast, FilterType::NoFilter)
        .write_image(pixels, width, height, ExtendedColorType::L8)?;
    let attributes = origin
        .map(|[x, y]| {
            BTreeMap::from([
                ("x".to_owned(), x.to_string()),
                ("y".to_owned(), y.to_string()),
            ])
        })
        .unwrap_or_default();
    Ok(AssetInput::new(
        bytes,
        "image/png",
        AssetMetadata {
            width: Some(width),
            height: Some(height),
            attributes,
        },
    ))
}

async fn preserve_mask_outside_region(
    input: &StageInput,
    page: EntityId,
    role: &str,
    bounds: crate::Bounds,
    mask: &mut GrayImage,
) -> Result<()> {
    let previous = input
        .images
        .get(&input.scene, page, role)
        .await?
        .map(|image| image.to_luma8());
    if previous
        .as_ref()
        .is_some_and(|image| image.dimensions() != mask.dimensions())
    {
        bail!("existing {role} dimensions do not match page {page}");
    }
    let [left, top, right, bottom] = scope_pixels(bounds);
    for (x, y, pixel) in mask.enumerate_pixels_mut() {
        if x < left || y < top || x >= right || y >= bottom {
            *pixel = previous
                .as_ref()
                .map_or(Luma([0]), |image| *image.get_pixel(x, y));
        }
    }
    Ok(())
}

fn region_kind(label: &str) -> Result<RegionKind> {
    RegionKind::new(match label {
        "text" => TextRegion::KIND,
        "bubble" => BubbleRegion::KIND,
        "panel" => PanelRegion::KIND,
        _ => "dev.koharu.region.unknown",
    })
    .map_err(Into::into)
}

fn mask_for(detections: &[KoharuLayoutDetection], label: &str, size: ImageSize) -> GrayImage {
    let mut mask = GrayImage::new(size.width, size.height);
    for detection in detections.iter().filter(|value| value.label == label) {
        stamp_mask(&mut mask, &detection.mask, 0, 0);
    }
    mask
}

fn valid_mask(mask: &KoharuLayoutMask) -> bool {
    mask.width
        .checked_mul(mask.height)
        .and_then(|count| usize::try_from(count).ok())
        == Some(mask.pixels.len())
}

fn stamp_mask(target: &mut GrayImage, mask: &KoharuLayoutMask, origin_x: u32, origin_y: u32) {
    if !valid_mask(mask) {
        return;
    }
    for local_y in 0..mask.height {
        let Some(page_y) = mask.y.checked_add(local_y) else {
            continue;
        };
        let Some(target_y) = page_y.checked_sub(origin_y) else {
            continue;
        };
        if target_y >= target.height() {
            continue;
        }
        let row = local_y as usize * mask.width as usize;
        for local_x in 0..mask.width {
            let Some(page_x) = mask.x.checked_add(local_x) else {
                continue;
            };
            let Some(target_x) = page_x.checked_sub(origin_x) else {
                continue;
            };
            if target_x < target.width() && mask.pixels[row + local_x as usize] != 0 {
                target.put_pixel(target_x, target_y, Luma([u8::MAX]));
            }
        }
    }
}

fn union_bounds(left: [u32; 4], right: [u32; 4]) -> [u32; 4] {
    [
        left[0].min(right[0]),
        left[1].min(right[1]),
        left[2].max(right[2]),
        left[3].max(right[3]),
    ]
}

fn intersects([left, top, right, bottom]: [f32; 4], region: crate::Bounds) -> bool {
    f64::from(left) < region.x + region.width
        && f64::from(right) > region.x
        && f64::from(top) < region.y + region.height
        && f64::from(bottom) > region.y
}

fn detection_order(left: &KoharuLayoutDetection, right: &KoharuLayoutDetection) -> Ordering {
    left.bbox[1]
        .total_cmp(&right.bbox[1])
        .then_with(|| right.bbox[0].total_cmp(&left.bbox[0]))
        .then_with(|| left.label.cmp(&right.label))
}

fn non_maximum_suppression(detections: &mut Vec<Candidate>, threshold: f32) {
    detections.sort_by(|left, right| {
        let (left, right) = (&left.detection, &right.detection);
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| detection_order(left, right))
    });
    let mut kept = Vec::with_capacity(detections.len());
    for candidate in detections.drain(..) {
        // Text overlaps are resolved by merging their boxes and masks.
        let detection = &candidate.detection;
        let suppressed = detection.label != "text"
            && kept.iter().any(|existing: &Candidate| {
                existing.detection.label == detection.label
                    && intersection_over_union(existing.detection.bbox, detection.bbox) >= threshold
            });
        if !suppressed {
            kept.push(candidate);
        }
    }
    *detections = kept;
}

/// RF-DETR proposes a text box per column or line and, separately, boxes
/// spanning several of them. A lower-scoring box can cover ink that a
/// higher-scoring box misses. Each text region becomes one translated segment,
/// so a split paragraph is translated as unrelated fragments, some twice. Boxes
/// that mostly overlap, and neighbouring lines of one paragraph (separated by
/// less than a line's thickness and running mostly side by side), become one
/// block. Neighbouring lines in different bubbles stay apart.
fn merge_text_blocks(detections: &mut Vec<KoharuLayoutDetection>) {
    let texts = indices_with_label(detections, "text");
    let bubbles = indices_with_label(detections, "bubble");
    let container = texts
        .iter()
        .map(|&text| {
            bubbles
                .iter()
                .copied()
                .filter(|&bubble| {
                    containment(detections[bubble].bbox, detections[text].bbox)
                        >= DIALOGUE_MASK_CONTAINMENT_THRESHOLD
                })
                .min_by(|&left, &right| {
                    area(detections[left].bbox).total_cmp(&area(detections[right].bbox))
                })
        })
        .collect::<Vec<_>>();
    let mut parent = (0..texts.len()).collect::<Vec<_>>();
    fn root(parent: &mut [usize], mut index: usize) -> usize {
        while parent[index] != index {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        index
    }
    for first in 0..texts.len() {
        for second in first + 1..texts.len() {
            let (first_bounds, second_bounds) = (
                detections[texts[first]].bbox,
                detections[texts[second]].bbox,
            );
            if same_text(first_bounds, second_bounds)
                || (container[first] == container[second]
                    && neighbouring_lines(first_bounds, second_bounds))
            {
                let (first, second) = (root(&mut parent, first), root(&mut parent, second));
                parent[second] = first;
            }
        }
    }
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (index, &text) in texts.iter().enumerate() {
        groups
            .entry(root(&mut parent, index))
            .or_default()
            .push(text);
    }
    let groups = groups
        .into_values()
        .filter(|group| group.len() > 1)
        .collect::<Vec<_>>();
    replace_with_unions(detections, &groups);
}

/// The bubble detector splits some balloons in two, and `same_text` then
/// joins their text into one block that neither half contains, so the block
/// would be set as free text over the balloon's edge. Bubbles that each hold
/// part of such a block, and together hold all of it, become one bubble.
fn join_split_bubbles(detections: &mut Vec<KoharuLayoutDetection>) {
    let bubbles = indices_with_label(detections, "bubble");
    let mut used = vec![false; detections.len()];
    let mut groups = Vec::new();
    for text in indices_with_label(detections, "text") {
        let text = &detections[text];
        let contains = |bubble: &KoharuLayoutDetection| {
            mask_containment(&bubble.mask, &text.mask, text.bbox)
                >= DIALOGUE_MASK_CONTAINMENT_THRESHOLD
        };
        if bubbles.iter().any(|&bubble| contains(&detections[bubble])) {
            continue;
        }
        let parts = bubbles
            .iter()
            .copied()
            .filter(|&bubble| {
                !used[bubble]
                    && containment(detections[bubble].bbox, text.bbox) >= SPLIT_BUBBLE_MIN_SHARE
            })
            .collect::<Vec<_>>();
        if parts.len() < 2 || !contains(&union_detection(detections, &parts)) {
            continue;
        }
        for &part in &parts {
            used[part] = true;
        }
        groups.push(parts);
    }
    replace_with_unions(detections, &groups);
}

/// Replaces each group of same-label detections with one detection covering
/// their union.
fn replace_with_unions(detections: &mut Vec<KoharuLayoutDetection>, groups: &[Vec<usize>]) {
    let mut absorbed = vec![false; detections.len()];
    let mut merged = Vec::new();
    for group in groups {
        merged.push(union_detection(detections, group));
        for &index in group {
            absorbed[index] = true;
        }
    }
    let mut index = 0;
    detections.retain(|_| {
        index += 1;
        !absorbed[index - 1]
    });
    detections.extend(merged);
}

fn union_detection(detections: &[KoharuLayoutDetection], group: &[usize]) -> KoharuLayoutDetection {
    let bbox = group
        .iter()
        .fold(detections[group[0]].bbox, |bounds, &index| {
            let value = detections[index].bbox;
            [
                bounds[0].min(value[0]),
                bounds[1].min(value[1]),
                bounds[2].max(value[2]),
                bounds[3].max(value[3]),
            ]
        });
    // A predicted box can end inside its own foreground mask.
    let [x, y, right, bottom] = group
        .iter()
        .map(|&index| &detections[index].mask)
        .filter(|mask| mask.width != 0 && mask.height != 0)
        .map(|mask| [mask.x, mask.y, mask.x + mask.width, mask.y + mask.height])
        .reduce(union_bounds)
        .unwrap_or_default();
    let (width, height) = (right - x, bottom - y);
    let mut mask = GrayImage::new(width, height);
    for &index in group {
        stamp_mask(&mut mask, &detections[index].mask, x, y);
    }
    let pixels = mask.into_raw();
    KoharuLayoutDetection {
        label_id: detections[group[0]].label_id,
        label: detections[group[0]].label.clone(),
        score: group
            .iter()
            .map(|&index| detections[index].score)
            .fold(0.0, f32::max),
        bbox,
        area: pixels.iter().filter(|value| **value != 0).count() as u32,
        mask: KoharuLayoutMask {
            x,
            y,
            width,
            height,
            pixels,
        }
        .into(),
    }
}

/// Mostly overlapping boxes cover the same text, even when the bubble
/// detector splits their balloon in two.
fn same_text(first: [f32; 4], second: [f32; 4]) -> bool {
    intersection_area(first, second) >= area(first).min(area(second)) * 0.5
}

fn neighbouring_lines(first: [f32; 4], second: [f32; 4]) -> bool {
    let width = |bounds: [f32; 4]| bounds[2] - bounds[0];
    let height = |bounds: [f32; 4]| bounds[3] - bounds[1];
    let vertical = |bounds: [f32; 4]| height(bounds) > width(bounds);
    // `across` measures line thickness and the gap between lines; `along`
    // measures where each line runs. The thinner box is a single line, so its
    // thickness is the glyph size.
    let (across, along) = match (vertical(first), vertical(second)) {
        (true, true) => (0, 1),
        (false, false) => (1, 0),
        _ => return false,
    };
    let line = (first[across + 2] - first[across]).min(second[across + 2] - second[across]);
    let gap = first[across].max(second[across]) - first[across + 2].min(second[across + 2]);
    let shared = first[along + 2].min(second[along + 2]) - first[along].max(second[along]);
    let shorter = (first[along + 2] - first[along]).min(second[along + 2] - second[along]);
    gap <= line * TEXT_BLOCK_GAP && shared >= shorter * 0.5
}

struct LayoutItem {
    index: usize,
    /// The glyph-mask parent, visited before this text so its scene ID is available.
    bubble: Option<usize>,
}

/// Panel hierarchy and gutter cuts follow koharu_layout_highres/order.rs
/// (1235d5c8, f72997e4). A fused balloon groups its dialogue into one subtree.
fn layout_order(detections: &[KoharuLayoutDetection]) -> Vec<LayoutItem> {
    let panels = indices_with_label(detections, "panel");
    let bubbles = indices_with_label(detections, "bubble");
    let mut children = vec![Vec::new(); detections.len()];
    let mut roots = Vec::new();
    let mut other_regions = Vec::new();
    for (index, detection) in detections.iter().enumerate() {
        let bbox = detection.bbox;
        let parent = match detection.label.as_str() {
            "panel" => panels
                .iter()
                .copied()
                .filter(|&outer| {
                    let outer = detections[outer].bbox;
                    area(outer) > area(bbox) && containment(outer, bbox) >= 0.9
                })
                .min_by(|&a, &b| area(detections[a].bbox).total_cmp(&area(detections[b].bbox))),
            "text" => bubbles
                .iter()
                .copied()
                .filter(|&bubble| {
                    mask_containment(&detections[bubble].mask, &detection.mask, bbox)
                        >= DIALOGUE_MASK_CONTAINMENT_THRESHOLD
                })
                .min_by_key(|&bubble| detections[bubble].area)
                .or_else(|| holding_panel(detections, bbox, &panels)),
            "bubble" => holding_panel(detections, bbox, &panels),
            // Untranslated sound effects and other analysis-only regions
            // have scene geometry but do not participate in dialogue order.
            _ => {
                other_regions.push(LayoutItem {
                    index,
                    bubble: None,
                });
                continue;
            }
        };
        if let Some(parent) = parent {
            children[parent].push(index);
        } else {
            roots.push(index);
        }
    }
    let mut order = Vec::with_capacity(detections.len());
    read_siblings(detections, roots, &mut children, None, &mut order);
    order.extend(other_regions);
    order
}

fn indices_with_label(detections: &[KoharuLayoutDetection], label: &str) -> Vec<usize> {
    detections
        .iter()
        .enumerate()
        .filter_map(|(index, detection)| (detection.label == label).then_some(index))
        .collect()
}

/// Prefer the panel holding the largest share, then the smallest inset.
fn holding_panel(
    detections: &[KoharuLayoutDetection],
    bbox: [f32; 4],
    panels: &[usize],
) -> Option<usize> {
    let holding = panels
        .iter()
        .map(|&panel| (panel, containment(detections[panel].bbox, bbox)))
        .filter(|&(_, share)| share >= 0.25)
        .max_by(|&(a, a_share), &(b, b_share)| {
            a_share
                .total_cmp(&b_share)
                .then_with(|| area(detections[b].bbox).total_cmp(&area(detections[a].bbox)))
        });
    if let Some((panel, _)) = holding {
        return Some(panel);
    }
    let distance = |panel: usize| {
        let other = detections[panel].bbox;
        let dx = (other[0] - bbox[2]).max(bbox[0] - other[2]).max(0.0);
        let dy = (other[1] - bbox[3]).max(bbox[1] - other[3]).max(0.0);
        dx.hypot(dy)
    };
    let spanned = |panel: usize| {
        let other = detections[panel].bbox;
        (other[3].min(bbox[3]) - other[1].max(bbox[1])) / (bbox[3] - bbox[1]).max(f32::EPSILON)
    };
    let nearest = |a: &usize, b: &usize| distance(*a).total_cmp(&distance(*b));
    // A caption beside a panel belongs to that row. Above or below its
    // nearest panel it occupies the gutter and is read at page level.
    if spanned(panels.iter().copied().min_by(nearest)?) <= 0.0 {
        return None;
    }
    // A margin item may be separated from the panel by up to the narrower
    // box's width. A missing panel can otherwise pull dialogue across a page.
    panels
        .iter()
        .copied()
        .filter(|&panel| {
            let other = detections[panel].bbox;
            spanned(panel) >= 0.5 && distance(panel) <= (bbox[2] - bbox[0]).min(other[2] - other[0])
        })
        .min_by(nearest)
}

fn read_siblings(
    detections: &[KoharuLayoutDetection],
    items: Vec<usize>,
    children: &mut [Vec<usize>],
    bubble: Option<usize>,
    order: &mut Vec<LayoutItem>,
) {
    for index in gutter_order(detections, items) {
        order.push(LayoutItem { index, bubble });
        let nested = std::mem::take(&mut children[index]);
        if !nested.is_empty() {
            let bubble = (detections[index].label == "bubble").then_some(index);
            read_siblings(detections, nested, children, bubble, order);
        }
    }
}

/// Split at horizontal gutters before vertical gutters, recursively within
/// each row or column. Touching edges also separate adjacent panels.
fn gutter_order(detections: &[KoharuLayoutDetection], mut items: Vec<usize>) -> Vec<usize> {
    if items.len() < 2 {
        return items;
    }
    for axis in [1, 0] {
        let mut sorted = items.clone();
        sorted.sort_by(|&a, &b| detections[a].bbox[axis].total_cmp(&detections[b].bbox[axis]));
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut end = f32::NEG_INFINITY;
        for index in sorted {
            let bbox = detections[index].bbox;
            match groups.last_mut() {
                Some(group) if bbox[axis] < end => group.push(index),
                _ => groups.push(vec![index]),
            }
            end = end.max(bbox[axis + 2]);
        }
        if groups.len() > 1 {
            if axis == 0 {
                groups.reverse();
            }
            return groups
                .into_iter()
                .flat_map(|group| gutter_order(detections, group))
                .collect();
        }
    }

    // Pairwise preferences can cycle. Read the item with the fewest unread
    // predecessors, then remove its vote from the remaining items.
    let mut unread: Vec<_> = items
        .iter()
        .map(|&index| {
            let earlier = items
                .iter()
                .filter(|&&other| {
                    other != index && reads_before(detections[other].bbox, detections[index].bbox)
                })
                .count();
            (index, earlier)
        })
        .collect();
    for next in &mut items {
        let position = (0..unread.len())
            .min_by(|&a, &b| {
                let ((a, a_count), (b, b_count)) = (unread[a], unread[b]);
                let (left, right) = (detections[a].bbox, detections[b].bbox);
                a_count
                    .cmp(&b_count)
                    .then_with(|| (left[1] + left[3]).total_cmp(&(right[1] + right[3])))
                    .then_with(|| (right[0] + right[2]).total_cmp(&(left[0] + left[2])))
                    .then_with(|| a.cmp(&b))
            })
            .expect("one unread item per remaining slot");
        *next = unread.swap_remove(position).0;
        for (other, earlier) in &mut unread {
            *earlier -= usize::from(reads_before(
                detections[*next].bbox,
                detections[*other].bbox,
            ));
        }
    }
    items
}

/// Shrink overlapping boxes until an axis separates them.
fn reads_before(a: [f32; 4], b: [f32; 4]) -> bool {
    for erosion in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] {
        let erode = |[x1, y1, x2, y2]: [f32; 4]| {
            let dx = (x2 - x1).max(0.0) * erosion;
            let dy = (y2 - y1).max(0.0) * erosion;
            [x1 + dx, y1 + dy, x2 - dx, y2 - dy]
        };
        let (a, b) = (erode(a), erode(b));
        if a[3] <= b[1] || b[3] <= a[1] {
            return a[3] <= b[1];
        }
        if a[0] >= b[2] || b[0] >= a[2] {
            return a[0] >= b[2];
        }
    }
    (a[1] + a[3])
        .total_cmp(&(b[1] + b[3]))
        .then_with(|| (b[0] + b[2]).total_cmp(&(a[0] + a[2])))
        .is_lt()
}

fn containment(container: [f32; 4], value: [f32; 4]) -> f32 {
    let value_area = area(value);
    if value_area <= 0.0 {
        return 0.0;
    }
    intersection_area(container, value) / value_area
}

fn mask_containment(
    container: &KoharuLayoutMask,
    value: &KoharuLayoutMask,
    bounds: [f32; 4],
) -> f32 {
    if !valid_mask(container) || !valid_mask(value) {
        return 0.0;
    }

    if !bounds.iter().all(|value| value.is_finite()) {
        return 0.0;
    }
    let value_right = value.x.saturating_add(value.width);
    let value_bottom = value.y.saturating_add(value.height);
    let left = bounds[0]
        .floor()
        .max(value.x as f32)
        .min(value_right as f32) as u32;
    let top = bounds[1]
        .floor()
        .max(value.y as f32)
        .min(value_bottom as f32) as u32;
    let right = bounds[2].ceil().max(value.x as f32).min(value_right as f32) as u32;
    let bottom = bounds[3]
        .ceil()
        .max(value.y as f32)
        .min(value_bottom as f32) as u32;
    if left >= right || top >= bottom {
        return 0.0;
    }

    let mut value_area = 0usize;
    let mut intersection = 0usize;
    for y in top..bottom {
        for x in left..right {
            if !value.contains(x, y) {
                continue;
            }
            value_area += 1;
            intersection += usize::from(container.contains(x, y));
        }
    }
    if value_area == 0 {
        0.0
    } else {
        intersection as f32 / value_area as f32
    }
}

fn intersection_over_union(left: [f32; 4], right: [f32; 4]) -> f32 {
    let intersection = intersection_area(left, right);
    let union = area(left) + area(right) - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn intersection_area(left: [f32; 4], right: [f32; 4]) -> f32 {
    (left[2].min(right[2]) - left[0].max(right[0])).max(0.0)
        * (left[3].min(right[3]) - left[1].max(right[1])).max(0.0)
}

fn area(bounds: [f32; 4]) -> f32 {
    (bounds[2] - bounds[0]).max(0.0) * (bounds[3] - bounds[1]).max(0.0)
}

#[cfg(test)]
mod tests {
    use image::{GrayImage, Luma, Rgb, RgbImage};
    use koharu_ml::koharu_layout_rfdetr_seg_2xl::{KoharuLayoutDetection, KoharuLayoutMask};
    use koharu_scene::{
        At, Geometry, Inside, PageDraft, Session, TextRegion, Typography, WritingMode,
    };

    use super::{
        DIALOGUE_MASK_CONTAINMENT_THRESHOLD, DetectionModel, ImageSize,
        KoharuLayoutRFDetrSeg2XLConfig, MaskPixel, Processor, RegionOutput, StageInput,
        StageProcessor, color_palette, generation, infer_typography, layout_order,
        mask_containment, mask_for, mask_geometry, non_maximum_suppression, normalize_text_color,
        write_region,
    };
    use koharu_ml::{
        comic_text_bubble_detector::{ComicTextBubbleDetection, ComicTextBubbleLabel},
        koharu_layout_rfdetr_seg_2xl::KoharuLayoutDetections,
    };

    fn fuse(
        layout: KoharuLayoutDetections,
        comic: &[ComicTextBubbleDetection],
        threshold: f32,
    ) -> KoharuLayoutDetections {
        KoharuLayoutDetections {
            image_width: layout.image_width,
            image_height: layout.image_height,
            detections: super::with_comic_detections(layout, comic, threshold)
                .into_iter()
                .map(|c| c.detection)
                .collect(),
        }
    }

    #[test]
    fn out_of_range_thresholds_fall_back_to_the_model_defaults() {
        let processor = Processor::new(
            DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig {
                text_threshold: Some(15.0),
                bubble_threshold: Some(f32::NAN),
                panel_threshold: Some(0.55),
            }),
            koharu_ml::Device::cpu(),
        );

        let DetectionModel::KoharuLayoutRFDetrSeg2XL(settings) = &processor.config;
        assert_eq!(settings.text_threshold, None);
        assert_eq!(settings.bubble_threshold, None);
        assert_eq!(settings.panel_threshold, Some(0.55));
    }

    #[tokio::test]
    async fn detection_skips_a_page_with_existing_text() {
        let mut session = Session::memory().await.unwrap();
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(PageDraft::new("page", 100.0, 100.0), At::End)?;
                edit.add_analysis_region::<TextRegion>(
                    id,
                    At::End,
                    &Geometry::rectangle(10.0, 10.0, 20.0, 20.0),
                    None,
                )?;
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let page = page.unwrap();
        let input = StageInput::new(
            snapshot,
            page,
            None,
            None,
            std::sync::Arc::new(crate::ImageCache::default()),
            None,
        );
        let processor = Processor::new(
            DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig::default()),
            koharu_ml::Device::cpu(),
        );

        assert!(processor.skip(&input).unwrap());
    }

    #[tokio::test]
    async fn reading_order_and_flow_use_the_balloon_holding_the_glyphs() -> anyhow::Result<()> {
        // The text box is entirely in the left balloon's box, but its ink is
        // in the right balloon. The left balloon starts one pixel higher.
        let mut left = detection("bubble", 0.75, [5.0, 10.0, 65.0, 90.0]);
        left.mask = full_mask(5, 10, 45, 80).into();
        left.area = 45 * 80;
        let mut right = detection("bubble", 0.85, [55.0, 11.0, 95.0, 91.0]);
        right.mask = full_mask(55, 11, 40, 80).into();
        right.area = 40 * 80;
        let mut text = detection("text", 0.65, [10.0, 30.0, 65.0, 60.0]);
        text.mask = full_mask(61, 35, 3, 10).into();
        text.area = 3 * 10;
        let mut caption = detection("text", 0.55, [5.0, 93.0, 24.0, 99.0]);
        caption.mask = full_mask(5, 93, 19, 6).into();
        caption.area = 19 * 6;

        let mut session = Session::memory().await?;
        let mut page = None;
        let create = session.snapshot().patch(|edit| {
            page = Some(edit.add_page(PageDraft::new("page", 100.0, 100.0), At::End)?);
            Ok(())
        })?;
        let snapshot = session.commit(create).await?.snapshot;
        let page = page.unwrap();
        let input = StageInput::new(snapshot, page, None, None, Default::default(), None);
        let patch = super::build_patch(
            &input,
            &image::DynamicImage::new_rgb8(100, 100),
            [caption, left, text, right]
                .into_iter()
                .map(|detection| super::Candidate {
                    proposal: super::GlyphProposal::Segmented(detection.bbox),
                    detection,
                })
                .collect(),
            &generation(super::PRODUCER, super::MODEL_ID)?,
        )
        .await?;
        let snapshot = session.commit(patch).await?.snapshot;
        let regions: Vec<_> = snapshot
            .children(page)?
            .filter_map(|id| snapshot.analysis_region(id).ok())
            .collect();
        let scores: Vec<_> = regions
            .iter()
            .map(|region| region.detection().unwrap().unwrap().labels[0].confidence)
            .collect();
        assert_eq!(scores, [0.85, 0.65, 0.75, 0.55]);

        let layers: Vec<_> = snapshot
            .page(page)?
            .text_group()?
            .unwrap()
            .text_layers()?
            .collect();
        let [dialogue, free] = layers[..] else {
            panic!("expected two text layers")
        };
        assert_eq!(
            dialogue.content()?.source_region()?.unwrap().id(),
            regions[1].id()
        );
        assert_eq!(dialogue.balloon_target()?.unwrap().id(), regions[0].id());
        assert!(dialogue.fit_target()?.is_none());
        assert_eq!(
            snapshot
                .relations_from_as::<Inside>(regions[1].id())
                .next()
                .unwrap()
                .value()
                .target,
            regions[0].id()
        );
        assert_eq!(free.fit_target()?.unwrap().id(), regions[3].id());
        assert!(free.balloon_target()?.is_none());
        Ok(())
    }

    fn detection(label: &str, score: f32, bbox: [f32; 4]) -> KoharuLayoutDetection {
        KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score,
            bbox,
            area: 0,
            mask: KoharuLayoutMask {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                pixels: vec![0],
            }
            .into(),
        }
    }

    fn full_mask(x: u32, y: u32, width: u32, height: u32) -> KoharuLayoutMask {
        KoharuLayoutMask {
            x,
            y,
            width,
            height,
            pixels: vec![u8::MAX; (width * height) as usize],
        }
    }

    fn comic(label: ComicTextBubbleLabel, bbox: [f32; 4]) -> ComicTextBubbleDetection {
        ComicTextBubbleDetection {
            label,
            score: 0.9,
            bbox,
        }
    }

    #[test]
    fn detected_bubbles_replace_layout_bubbles_and_reuse_overlapping_masks() {
        // The proposal spans a second lobe to the right of the detected box.
        let mut proposal = detection("bubble", 0.1, [10.0, 10.0, 70.0, 90.0]);
        proposal.mask = full_mask(12, 12, 56, 76).into();
        proposal.area = 56 * 76;
        let unmatched = detection("bubble", 0.9, [100.0, 0.0, 120.0, 20.0]);
        let layout = KoharuLayoutDetections {
            image_width: 200,
            image_height: 100,
            detections: vec![
                proposal,
                unmatched,
                detection("panel", 0.9, [0.0, 0.0, 200.0, 100.0]),
            ],
        };

        let output = fuse(
            layout,
            &[
                comic(ComicTextBubbleLabel::Bubble, [8.0, 8.0, 52.0, 92.0]),
                comic(ComicTextBubbleLabel::TextFree, [150.0, 10.0, 160.0, 60.0]),
            ],
            0.5,
        );

        let bubbles = output
            .detections
            .iter()
            .filter(|detection| detection.label == "bubble")
            .collect::<Vec<_>>();
        assert_eq!(bubbles.len(), 1);
        assert_eq!(bubbles[0].bbox, [8.0, 8.0, 52.0, 92.0]);
        // Clipped to the box, joined with the box's inscribed ellipse.
        assert_eq!((bubbles[0].mask.x, bubbles[0].mask.width), (8, 44));
        assert!(bubbles[0].mask.contains(13, 13) && bubbles[0].mask.contains(30, 91));
        assert!(!bubbles[0].mask.contains(9, 9) && !bubbles[0].mask.contains(55, 50));
        assert_eq!(
            bubbles[0].area as usize,
            bubbles[0]
                .mask
                .pixels
                .iter()
                .filter(|value| **value != 0)
                .count()
        );
        // The panel, the bubble, and text from the uncovered text box.
        assert_eq!(output.detections.len(), 3);
    }

    #[test]
    fn unmatched_detected_bubbles_use_an_ellipse_that_contains_their_text() {
        // A tall text column whose corners fall outside the inscribed ellipse.
        let mut text = detection("text", 0.9, [20.0, 5.0, 40.0, 95.0]);
        text.mask = full_mask(20, 5, 20, 90).into();
        let layout = KoharuLayoutDetections {
            image_width: 100,
            image_height: 100,
            detections: vec![text],
        };

        let output = fuse(
            layout,
            &[comic(
                ComicTextBubbleLabel::Bubble,
                [10.0, 0.0, 50.0, 100.0],
            )],
            0.5,
        );

        let bubble = output
            .detections
            .iter()
            .find(|detection| detection.label == "bubble")
            .unwrap();
        let text = output
            .detections
            .iter()
            .find(|detection| detection.label == "text")
            .unwrap();
        assert!(bubble.mask.contains(30, 50));
        assert!(!bubble.mask.contains(11, 2));
        assert_eq!(mask_containment(&bubble.mask, &text.mask, text.bbox), 1.0);
    }

    #[test]
    fn text_boxes_grow_to_their_masks_within_comic_text_boxes() {
        // The box starts one glyph below the column's mask, and the mask runs
        // on below the column into an aside.
        let column = || {
            let mut text = detection("text", 0.9, [10.0, 30.0, 30.0, 100.0]);
            let mut mask = full_mask(10, 10, 20, 300);
            mask.pixels.fill(0);
            for y in (0..90).chain(200..300) {
                mask.pixels[y * 20..y * 20 + 20].fill(u8::MAX);
            }
            text.mask = mask.into();
            KoharuLayoutDetections {
                image_width: 100,
                image_height: 400,
                detections: vec![text],
            }
        };

        let grown = fuse(
            column(),
            &[comic(
                ComicTextBubbleLabel::TextBubble,
                [8.0, 5.0, 32.0, 105.0],
            )],
            0.5,
        );
        let kept = fuse(column(), &[], 0.5);

        let text = &grown.detections[0];
        assert_eq!(text.bbox, [10.0, 10.0, 30.0, 100.0]);
        assert!(text.mask.contains(10, 10) && !text.mask.contains(10, 250));
        assert_eq!(text.area, 20 * 90);
        assert_eq!(kept.detections[0].bbox, [10.0, 30.0, 30.0, 100.0]);
    }

    #[test]
    fn onomatopoeia_that_the_comic_detector_boxes_as_text_becomes_text() {
        let mut aside = detection("onomatopoeia", 0.5, [10.0, 10.0, 30.0, 60.0]);
        aside.mask = full_mask(10, 10, 20, 50).into();
        let mut effect = detection("onomatopoeia", 0.5, [100.0, 10.0, 160.0, 60.0]);
        effect.mask = full_mask(100, 10, 60, 50).into();
        let layout = KoharuLayoutDetections {
            image_width: 200,
            image_height: 100,
            detections: vec![aside, effect],
        };
        let mut agreement = comic(ComicTextBubbleLabel::TextFree, [8.0, 8.0, 32.0, 62.0]);
        agreement.score = 0.35;

        let output = fuse(layout, &[agreement], 0.5);

        let labels = output
            .detections
            .iter()
            .map(|detection| (detection.bbox[0], detection.label.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(labels, [(10.0, "text"), (100.0, "onomatopoeia")]);
    }

    #[test]
    fn confident_comic_text_that_layout_missed_becomes_text() {
        let mut column = detection("text", 0.9, [10.0, 10.0, 30.0, 90.0]);
        column.mask = full_mask(10, 10, 20, 80).into();
        let layout = KoharuLayoutDetections {
            image_width: 200,
            image_height: 100,
            detections: vec![column],
        };
        let mut faint = comic(ComicTextBubbleLabel::TextBubble, [150.0, 10.0, 170.0, 50.0]);
        faint.score = 0.4;

        let output = fuse(
            layout,
            &[
                comic(ComicTextBubbleLabel::TextBubble, [8.0, 8.0, 32.0, 92.0]),
                comic(ComicTextBubbleLabel::TextBubble, [100.0, 10.0, 120.0, 50.0]),
                faint,
            ],
            0.5,
        );

        let texts = output
            .detections
            .iter()
            .filter(|detection| detection.label == "text")
            .collect::<Vec<_>>();
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[1].bbox, [100.0, 10.0, 120.0, 50.0]);
        assert!(texts[1].mask.contains(100, 10) && texts[1].mask.contains(119, 49));
        assert_eq!(texts[1].area, 20 * 40);
    }

    #[test]
    fn no_ink_preserves_only_segmented_detector_evidence() {
        let page = RgbImage::from_pixel(100, 80, Rgb([210; 3]));
        let mut segmented = detection("text", 0.9, [10., 10., 30., 30.]);
        let mut strokes = full_mask(12, 12, 8, 12);
        strokes.pixels[2..5].fill(0);
        segmented.mask = strokes.clone().into();
        let mut candidates = super::with_comic_detections(
            KoharuLayoutDetections {
                image_width: 100,
                image_height: 80,
                detections: vec![segmented],
            },
            &[comic(
                ComicTextBubbleLabel::TextBubble,
                [50., 10., 70., 30.],
            )],
            0.5,
        );
        super::fit_glyph_masks(
            candidates
                .iter_mut()
                .map(|c| (c.proposal, &mut c.detection)),
            &page,
        );
        let [segmented, comic] = candidates.as_slice() else {
            panic!("expected both text regions")
        };
        assert_eq!(segmented.detection.mask.pixels, strokes.pixels);
        assert_eq!(comic.detection.area, 0);
        assert!(comic.detection.mask.pixels.is_empty());
    }

    #[test]
    fn letterbox_padding_does_not_create_scene_candidates() {
        let candidates = fuse(
            KoharuLayoutDetections {
                image_width: 800,
                image_height: 1600,
                detections: vec![
                    detection("text", 0.98, [1180., 200., 1220., 280.]),
                    detection("text", 0.9, [800., 100., 830., 140.]),
                    detection("text", 0.8, [790., 100., 830., 140.]),
                ],
            },
            &[],
            0.5,
        );
        assert_eq!(candidates.detections.len(), 1);
        assert_eq!(candidates.detections[0].bbox, [790., 100., 830., 140.]);
    }

    #[tokio::test]
    async fn detection_rerun_preserves_a_manual_direction() -> anyhow::Result<()> {
        use koharu_scene::{EntityOrigin, OcrAnalysis, Origin, TextDirection};
        let mut session = Session::memory().await?;
        let mut create = session.snapshot().edit();
        let page = create.add_page(PageDraft::new("page", 100., 80.), At::End)?;
        let snapshot = session.commit(create.finish()?).await?.snapshot;
        let mut input = StageInput::new(snapshot, page, None, None, Default::default(), None);
        let detector = generation(super::PRODUCER, super::MODEL_ID)?;
        let image = RgbImage::from_fn(100, 80, |x, y| {
            let first = (14..24).contains(&y) && ((12..17).contains(&x) || (22..27).contains(&x));
            let second = (54..64).contains(&y) && ((62..67).contains(&x) || (74..79).contains(&x));
            Rgb([if first || second { 10 } else { 245 }; 3])
        })
        .into();
        let candidates = super::with_comic_detections(
            KoharuLayoutDetections {
                image_width: 100,
                image_height: 80,
                detections: vec![],
            },
            &[
                comic(ComicTextBubbleLabel::TextBubble, [10., 10., 30., 30.]),
                comic(ComicTextBubbleLabel::TextBubble, [60., 50., 85., 70.]),
            ],
            0.5,
        );
        let patch = super::build_patch(&input, &image, candidates, &detector).await?;
        let snapshot = session.commit(patch).await?.snapshot;
        let regions: Vec<_> = snapshot
            .children(page)?
            .filter(|id| snapshot.analysis_region(*id).is_ok())
            .collect();
        let [edited, untouched] = regions[..] else {
            panic!("expected two text regions")
        };
        let mut correction = snapshot.component::<OcrAnalysis>(edited)?.unwrap();
        assert!(matches!(correction.origin, Origin::Generated(_)));
        assert_eq!(correction.direction, TextDirection::Horizontal);
        correction.origin = Origin::User;
        correction.direction = TextDirection::Vertical;
        let change = snapshot.patch(|e| e.set(edited, &correction))?;
        input.scene = session.commit(change).await?.snapshot;
        assert!(matches!(
            input
                .scene
                .component::<EntityOrigin>(edited)?
                .unwrap()
                .origin,
            Origin::Generated(_)
        ));
        let patch = super::build_patch(&input, &image, vec![], &detector).await?;
        let snapshot = session.commit(patch).await?.snapshot;
        assert_eq!(snapshot.component::<OcrAnalysis>(edited)?, Some(correction));
        assert!(snapshot.entity(untouched).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn comic_only_box_fits_ink_in_both_region_and_page_assets() -> anyhow::Result<()> {
        check_comic_glyph_assets(&[None]).await
    }

    #[tokio::test]
    async fn adjacent_scoped_detections_cover_both_halves_of_a_bisected_glyph() -> anyhow::Result<()>
    {
        let split = 49.99999999;
        check_comic_glyph_assets(&[
            Some(crate::Bounds {
                x: split,
                y: 30.,
                width: 30.,
                height: 40.,
            }),
            Some(crate::Bounds {
                x: 20.,
                y: 30.,
                width: split - 20.,
                height: 40.,
            }),
        ])
        .await
    }

    #[tokio::test]
    async fn adjacent_scoped_detections_share_the_balloon_for_both_texts() -> anyhow::Result<()> {
        check_scoped_balloon_reuse(&[
            boxed_detection("bubble", 0.9, [5., 5., 95., 75.]),
            boxed_detection("text", 0.8, [15., 30., 35., 45.]),
            boxed_detection("text", 0.8, [65., 30., 85., 45.]),
        ])
        .await
    }

    #[tokio::test]
    async fn adjacent_scoped_detections_reuse_balloons_joined_by_out_of_scope_text()
    -> anyhow::Result<()> {
        // Only the first selection sees the text spanning both lobes. The second
        // text sits entirely in the small lobe, past the first selection's edge.
        check_scoped_balloon_reuse(&[
            boxed_detection("bubble", 0.9, [2., 5., 42., 75.]),
            boxed_detection("bubble", 0.9, [31., 25., 96., 50.]),
            boxed_detection("text", 0.8, [22., 36., 49., 44.]),
            boxed_detection("text", 0.8, [65., 36., 85., 44.]),
        ])
        .await
    }

    async fn check_scoped_balloon_reuse(
        detections: &[KoharuLayoutDetection],
    ) -> anyhow::Result<()> {
        let mut session = Session::memory().await?;
        let mut create = session.snapshot().edit();
        let page = create.add_page(PageDraft::new("page", 100., 80.), At::End)?;
        let snapshot = session.commit(create.finish()?).await?.snapshot;
        let mut input = StageInput::new(snapshot, page, None, None, Default::default(), None);
        let image = image::DynamicImage::new_rgb8(100, 80);
        let detector = generation(super::PRODUCER, super::MODEL_ID)?;
        let mut original_balloon = None;
        for (run, x) in [0., 50.].into_iter().enumerate() {
            input.region = Some(crate::Bounds {
                x,
                y: 35.,
                width: 50.,
                height: 30.,
            });
            let output = detections
                .iter()
                .cloned()
                .map(|detection| super::Candidate {
                    proposal: super::GlyphProposal::Segmented(detection.bbox),
                    detection,
                })
                .collect();
            let patch = super::build_patch(&input, &image, output, &detector).await?;
            input.scene = session.commit(patch).await?.snapshot;
            let snapshot = &input.scene;
            let group = snapshot.page(page)?.text_group()?.unwrap();
            let layers: Vec<_> = group.text_layers()?.collect();
            assert_eq!(layers.len(), run + 1);
            assert_eq!(
                snapshot.entities_with::<super::Region>()?.count(),
                layers.len() + 1,
                "all text regions share one balloon"
            );
            for (layer, left) in layers.iter().zip([0., 50.]) {
                let balloon = layer.balloon_target()?.unwrap().id();
                assert_eq!(balloon, *original_balloon.get_or_insert(balloon));
                let region = layer.content()?.source_region()?.unwrap();
                assert!(
                    region
                        .geometry()?
                        .points
                        .iter()
                        .all(|p| (left..=left + 50.).contains(&p.x))
                );
                let inside = snapshot
                    .relations_from_as::<Inside>(region.id())
                    .map(|relation| relation.value().target)
                    .collect::<Vec<_>>();
                assert_eq!(inside, [balloon]);
            }
        }
        Ok(())
    }

    #[test]
    fn scoped_text_clips_keep_the_original_rectangle_axes() {
        let text = boxed_detection("text", 0.8, [20., 20., 80., 60.]);
        for ([x, y, width, height], [left, top, right, bottom]) in [
            ([0., 0., 100., 45.], [20., 20., 80., 45.]),
            ([0., 35., 100., 45.], [20., 35., 80., 60.]),
            ([30., 0., 70., 80.], [30., 20., 80., 60.]),
            ([0., 0., 70., 80.], [20., 20., 70., 60.]),
        ] {
            let selection = crate::Bounds {
                x,
                y,
                width,
                height,
            };
            for angle in [None, Some(0.)] {
                let clipped = super::text_geometry(&text, angle, Some(selection)).unwrap();
                let points: Vec<_> = clipped.points.iter().map(|p| (p.x, p.y)).collect();
                assert_eq!(
                    points,
                    [(left, top), (right, top), (right, bottom), (left, bottom)]
                );
            }
        }
    }

    #[test]
    fn scoped_tilted_text_fits_inside_the_selection_on_its_original_axes() {
        let text = boxed_detection("text", 0.8, [20., 20., 80., 60.]);
        let selection = crate::Bounds {
            x: 46.,
            y: 36.,
            width: 8.,
            height: 8.,
        };
        for (angle, expected) in [
            (45., [(50., 36.), (54., 40.), (50., 44.), (46., 40.)]),
            (-45., [(46., 40.), (50., 36.), (54., 40.), (50., 44.)]),
        ] {
            let clipped = super::text_geometry(&text, Some(angle), Some(selection)).unwrap();
            assert_eq!(clipped.points.len(), 4);
            for (p, (x, y)) in clipped.points.iter().zip(expected) {
                assert!((p.x - x).hypot(p.y - y) < 1e-9);
            }
        }
    }

    #[tokio::test]
    async fn adjacent_scoped_detections_preserve_tilted_text_in_both_selections()
    -> anyhow::Result<()> {
        let (image, text) = masked_text(52., 12., 12., [24, 80, 160]);
        let image = image::DynamicImage::from(image);
        let mut session = Session::memory().await?;
        let mut create = session.snapshot().edit();
        let page = create.add_page(PageDraft::new("page", 96., 96.), At::End)?;
        let snapshot = session.commit(create.finish()?).await?.snapshot;
        let mut input = StageInput::new(snapshot, page, None, None, Default::default(), None);
        let detector = Processor::new(
            DetectionModel::KoharuLayoutRFDetrSeg2XL(Default::default()),
            koharu_ml::Device::cpu(),
        );
        let generation = generation(super::PRODUCER, super::MODEL_ID)?;
        for x in [46., 54.] {
            input.region = Some(crate::Bounds {
                x,
                y: 44.,
                width: 8.,
                height: 8.,
            });
            assert!(
                !detector.skip(&input)?,
                "selection starting at {x} should run"
            );
            let patch = super::build_patch(
                &input,
                &image,
                vec![super::Candidate {
                    proposal: super::GlyphProposal::Segmented(text.bbox),
                    detection: text.clone(),
                }],
                &generation,
            )
            .await?;
            input.scene = session.commit(patch).await?.snapshot;
            assert!(
                detector.skip(&input)?,
                "the processed selection should skip"
            );
        }

        let group = input.scene.page(page)?.text_group()?.unwrap();
        let layers: Vec<_> = group.text_layers()?.collect();
        assert_eq!(layers.len(), 2);
        for (layer, x) in layers.iter().zip([46., 54.]) {
            let geometry = layer.content()?.source_region()?.unwrap().geometry()?;
            let [a, b, c, d] = geometry.points.as_slice() else {
                panic!("expected a rectangular OCR and free-text frame");
            };
            assert!(
                geometry
                    .points
                    .iter()
                    .all(|p| (x..=x + 8.).contains(&p.x) && (44.0..=52.0).contains(&p.y))
            );
            let top = (b.y - a.y).atan2(b.x - a.x).to_degrees();
            let side = (d.y - a.y).atan2(d.x - a.x).to_degrees();
            assert!((top - 12.).abs() < 1., "first edge has angle {top}");
            assert!((side - top - 90.).abs() < 1e-6);
            assert!((c.x - b.x - d.x + a.x).hypot(c.y - b.y - d.y + a.y) < 1e-9);
        }
        Ok(())
    }

    async fn check_comic_glyph_assets(selections: &[Option<crate::Bounds>]) -> anyhow::Result<()> {
        use image::DynamicImage;
        use koharu_scene::{AssetRole, OcrAnalysis, TextDirection};
        let ink = |x, y| (47..55).contains(&x) && ((28..40).contains(&y) || (49..61).contains(&y));
        let source = RgbImage::from_fn(120, 100, |x, y| Rgb([if ink(x, y) { 12 } else { 235 }; 3]));
        let mut session = Session::memory().await?;
        let mut page = None;
        let create = session.snapshot().patch(|edit| {
            page = Some(edit.add_page(PageDraft::new("page", 120., 100.), At::End)?);
            Ok(())
        })?;
        let snapshot = session.commit(create).await?.snapshot;
        let page = page.unwrap();
        let source = DynamicImage::from(source);
        let mut input = StageInput::new(snapshot, page, None, None, Default::default(), None);
        let detector = Processor::new(
            DetectionModel::KoharuLayoutRFDetrSeg2XL(Default::default()),
            koharu_ml::Device::cpu(),
        );
        for (run, &selection) in selections.iter().enumerate() {
            input.region = selection;
            assert!(
                !detector.skip(&input)?,
                "selection {run} should run detection"
            );
            let layout = super::with_comic_detections(
                KoharuLayoutDetections {
                    image_width: 120,
                    image_height: 100,
                    detections: vec![],
                },
                &[comic(
                    ComicTextBubbleLabel::TextBubble,
                    [40., 20., 67., 70.],
                )],
                0.5,
            );
            let patch = super::build_patch(
                &input,
                &source,
                layout,
                &generation(super::PRODUCER, super::MODEL_ID)?,
            )
            .await?;
            input.scene = session.commit(patch).await?.snapshot;
            let snapshot = &input.scene;
            let group = snapshot.page(page)?.text_group()?.unwrap();
            let layers = group.text_layers()?.collect::<Vec<_>>();
            assert_eq!(layers.len(), run + 1);
            let mut covered = GrayImage::new(source.width(), source.height());
            for layer in layers {
                let region = layer.content()?.source_region()?.unwrap().id();
                let glyphs = snapshot
                    .asset(region, &AssetRole::new(super::GLYPH_MASK)?)?
                    .unwrap();
                assert_eq!(glyphs.media_type, "image/png");
                let (x, y): (u32, u32) = (
                    glyphs.metadata.attributes["x"].parse()?,
                    glyphs.metadata.attributes["y"].parse()?,
                );
                let local = image::load_from_memory(&snapshot.read_blob(glyphs.blob).await?)?;
                let local = local.as_luma8().unwrap();
                assert_eq!(
                    (Some(local.width()), Some(local.height())),
                    (glyphs.metadata.width, glyphs.metadata.height)
                );
                for (dx, dy, pixel) in local.enumerate_pixels() {
                    if pixel[0] != 0 {
                        covered.put_pixel(x + dx, y + dy, *pixel);
                    }
                }
                let typography = snapshot.component::<Typography>(layer.id())?.unwrap();
                assert_eq!(typography.size, Some(8.0));
                assert!(typography.auto_fit);
                assert_eq!(typography.writing_mode, Some(WritingMode::Vertical));
                assert_eq!(
                    snapshot
                        .component::<OcrAnalysis>(region)?
                        .unwrap()
                        .direction,
                    TextDirection::Vertical
                );
            }
            let page_mask = input
                .images
                .get(snapshot, page, "text-mask")
                .await?
                .unwrap();
            let page_mask = page_mask.as_luma8().unwrap();
            assert_eq!(page_mask, &covered);
            for (x, y, pixel) in page_mask.enumerate_pixels() {
                // Both adjacent scopes cover column 49 at their fractional boundary.
                let selected =
                    selection.is_none() || ((30..70).contains(&y) && (run == 1 || x >= 49));
                if ink(x, y) {
                    assert_eq!(pixel[0] != 0, selected, "run {run} at ({x}, {y})");
                } else if !selected {
                    assert_eq!(pixel[0], 0, "run {run} at ({x}, {y})");
                }
            }
            assert_eq!(page_mask[(50, 45)][0], 0);
            assert_eq!(page_mask[(41, 22)][0], 0);
        }
        Ok(())
    }

    #[test]
    fn bubble_geometry_follows_the_instance_mask_polygon() {
        let width = 9;
        let height = 9;
        let mut pixels = vec![0; width * height];
        for y in 0..height {
            for x in 0..width {
                if x.abs_diff(4) + y.abs_diff(4) <= 4 {
                    pixels[y * width + x] = u8::MAX;
                }
            }
        }

        let geometry = mask_geometry(&KoharuLayoutMask {
            x: 10,
            y: 20,
            width: width as u32,
            height: height as u32,
            pixels,
        })
        .unwrap();

        assert!(geometry.points.len() >= 4);
        assert!(
            geometry
                .points
                .iter()
                .all(|point| !(point.x == 10.0 && point.y == 20.0))
        );
        assert!(
            geometry
                .points
                .iter()
                .any(|point| point.x == 14.0 && point.y == 20.0)
        );
    }

    #[test]
    fn bubble_geometry_fills_notches_in_the_mask() {
        // A balloon mask with a slit cut down its middle, as artwork behind a
        // translucent balloon leaves it.
        let (width, height) = (40, 60);
        let mut pixels = vec![u8::MAX; width * height];
        for y in 10..60 {
            for x in 18..22 {
                pixels[y * width + x] = 0;
            }
        }

        let geometry = mask_geometry(&KoharuLayoutMask {
            x: 0,
            y: 0,
            width: width as u32,
            height: height as u32,
            pixels,
        })
        .unwrap();

        let xs = geometry.points.iter().map(|point| point.x);
        let ys = geometry.points.iter().map(|point| point.y);
        assert_eq!(geometry.points.len(), 4);
        assert_eq!(xs.clone().fold(f64::MAX, f64::min), 0.0);
        assert_eq!(xs.fold(f64::MIN, f64::max), 39.0);
        assert_eq!(ys.fold(f64::MIN, f64::max), 59.0);
    }

    fn masked_text(
        local_width: f64,
        local_height: f64,
        angle_degrees: f64,
        color: [u8; 3],
    ) -> (RgbImage, KoharuLayoutDetection) {
        let width = 96;
        let height = 96;
        let center_x = f64::from(width) * 0.5;
        let center_y = f64::from(height) * 0.5;
        let (sin, cos) = angle_degrees.to_radians().sin_cos();
        let mut image = RgbImage::from_pixel(width, height, Rgb([200, 180, 160]));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in 0..height {
            for x in 0..width {
                let dx = f64::from(x) + 0.5 - center_x;
                let dy = f64::from(y) + 0.5 - center_y;
                let local_x = dx * cos + dy * sin;
                let local_y = -dx * sin + dy * cos;
                if local_x.abs() <= local_width * 0.5 && local_y.abs() <= local_height * 0.5 {
                    pixels[y as usize * width as usize + x as usize] = u8::MAX;
                    image.put_pixel(x, y, Rgb(color));
                }
            }
        }
        (
            image,
            KoharuLayoutDetection {
                label_id: 0,
                label: "text".to_owned(),
                score: 1.0,
                bbox: [0.0, 0.0, width as f32, height as f32],
                area: pixels.iter().filter(|value| **value != 0).count() as u32,
                mask: KoharuLayoutMask {
                    x: 0,
                    y: 0,
                    width,
                    height,
                    pixels,
                }
                .into(),
            },
        )
    }

    fn outlined_text(
        stroke_width: u32,
        fill: [u8; 3],
        stroke: [u8; 3],
    ) -> (RgbImage, KoharuLayoutDetection) {
        outlined_text_on_background(stroke_width, fill, stroke, [16, 24, 88])
    }

    fn outlined_text_on_background(
        stroke_width: u32,
        fill: [u8; 3],
        stroke: [u8; 3],
        background: [u8; 3],
    ) -> (RgbImage, KoharuLayoutDetection) {
        let width = 96;
        let height = 96;
        let [left, top, right, bottom] = [20, 38, 76, 58];
        let mut image = RgbImage::from_pixel(width, height, Rgb(background));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in top..bottom {
            for x in left..right {
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
                let is_fill = x >= left + stroke_width
                    && x < right - stroke_width
                    && y >= top + stroke_width
                    && y < bottom - stroke_width;
                image.put_pixel(x, y, if is_fill { Rgb(fill) } else { Rgb(stroke) });
            }
        }
        (
            image,
            KoharuLayoutDetection {
                label_id: 0,
                label: "text".to_owned(),
                score: 1.0,
                bbox: [left as f32, top as f32, right as f32, bottom as f32],
                area: pixels.iter().filter(|value| **value != 0).count() as u32,
                mask: KoharuLayoutMask {
                    x: 0,
                    y: 0,
                    width,
                    height,
                    pixels,
                }
                .into(),
            },
        )
    }

    fn region_masked_text(fill: [u8; 3], background: [u8; 3]) -> (RgbImage, KoharuLayoutDetection) {
        let width = 96;
        let height = 96;
        let [left, top, right, bottom] = [20, 20, 76, 76];
        let mut image = RgbImage::from_pixel(width, height, Rgb(background));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in top..bottom {
            for x in left..right {
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
            }
        }
        for x in [27, 35, 43, 51, 59, 67] {
            for y in 28..68 {
                for ink_x in x..x + 3 {
                    image.put_pixel(ink_x, y, Rgb(fill));
                }
            }
        }
        (
            image,
            KoharuLayoutDetection {
                label_id: 0,
                label: "text".to_owned(),
                score: 1.0,
                bbox: [left as f32, top as f32, right as f32, bottom as f32],
                area: pixels.iter().filter(|value| **value != 0).count() as u32,
                mask: KoharuLayoutMask {
                    x: 0,
                    y: 0,
                    width,
                    height,
                    pixels,
                }
                .into(),
            },
        )
    }

    fn outlined_text_on_textured_background() -> (RgbImage, KoharuLayoutDetection) {
        let width = 96;
        let height = 96;
        let [left, top, right, bottom] = [20, 20, 76, 76];
        let mut image = RgbImage::from_pixel(width, height, Rgb([24, 32, 96]));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in top..bottom {
            for x in left..right {
                pixels[y as usize * width as usize + x as usize] = u8::MAX;
                let background = match (x / 7 + y / 5) % 3 {
                    0 => [0, 0, 0],
                    1 => [38, 48, 128],
                    _ => [24, 32, 96],
                };
                image.put_pixel(x, y, Rgb(background));
            }
        }
        for glyph_left in [26, 38, 50, 62] {
            for y in 28..68 {
                for x in glyph_left..glyph_left + 9 {
                    let is_fill =
                        x >= glyph_left + 3 && x < glyph_left + 6 && (31..65).contains(&y);
                    image.put_pixel(
                        x,
                        y,
                        if is_fill {
                            Rgb([0, 0, 0])
                        } else {
                            Rgb([255, 255, 255])
                        },
                    );
                }
            }
        }
        (
            image,
            KoharuLayoutDetection {
                label_id: 0,
                label: "text".to_owned(),
                score: 1.0,
                bbox: [left as f32, top as f32, right as f32, bottom as f32],
                area: pixels.iter().filter(|value| **value != 0).count() as u32,
                mask: KoharuLayoutMask {
                    x: 0,
                    y: 0,
                    width,
                    height,
                    pixels,
                }
                .into(),
            },
        )
    }

    #[tokio::test]
    async fn detected_typography_preserves_the_measured_outline() {
        let mut session = Session::memory().await.unwrap();
        let mut page = None;
        let create = session
            .snapshot()
            .patch(|edit| {
                page = Some(edit.add_page(PageDraft::new("page", 96.0, 96.0), At::End)?);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(create).await.unwrap().snapshot;
        let page = page.unwrap();
        let (image, detection) = outlined_text(3, [0, 0, 0], [255, 255, 255]);
        let inferred = infer_typography(&image, &detection);
        let generation = generation(super::PRODUCER, super::MODEL_ID).unwrap();
        let input = StageInput::new(snapshot.clone(), page, None, None, Default::default(), None);
        let mut layer = None;
        let patch = snapshot
            .patch(|edit| {
                let output = write_region(
                    edit,
                    &input,
                    &detection,
                    inferred,
                    None,
                    &generation,
                    &mut Vec::new(),
                )
                .unwrap();
                let RegionOutput::TextLayer(entity) = output else {
                    panic!("expected a text region");
                };
                layer = Some(entity);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let typography = snapshot
            .component::<Typography>(layer.unwrap())
            .unwrap()
            .unwrap();

        assert_eq!(typography.color, Some([0, 0, 0, 255]));
        assert_eq!(typography.stroke_color, Some([255, 255, 255, 255]));
        assert_eq!(typography.stroke_width, Some(3.0));
    }

    #[test]
    fn columns_of_one_paragraph_merge_into_a_block() {
        // Boxes from 4961x7016 pages: a column, a box spanning it and its
        // neighbour, and that neighbour; a paragraph of three columns with a
        // staggered start; a column far to the left; one inside a bubble; and
        // a column box nested in a two-column box, each assigned its own
        // overlapping bubble.
        let column = |bbox: [f32; 4]| {
            let [left, top, right, bottom] = bbox.map(|value| value as u32);
            KoharuLayoutDetection {
                mask: full_mask(left, top, right - left, bottom - top).into(),
                ..detection("text", 0.9, bbox)
            }
        };
        let mut detections = vec![
            column([2713.0, 369.0, 2906.0, 1479.0]),
            column([2480.0, 376.0, 2906.0, 1706.0]),
            column([2499.0, 393.0, 2654.0, 1726.0]),
            column([3837.0, 822.0, 4030.0, 2069.0]),
            column([4050.0, 839.0, 4243.0, 2000.0]),
            column([4321.0, 640.0, 4476.0, 1486.0]),
            column([448.0, 411.0, 617.0, 1644.0]),
            column([4520.0, 830.0, 4640.0, 1400.0]),
            detection("bubble", 0.9, [4500.0, 800.0, 4700.0, 1500.0]),
            column([474.0, 3435.0, 1094.0, 4740.0]),
            column([939.0, 3452.0, 1094.0, 4754.0]),
            detection("bubble", 0.9, [367.0, 3271.0, 1259.0, 4924.0]),
            detection("bubble", 0.9, [860.0, 3281.0, 1289.0, 4894.0]),
        ];

        super::merge_text_blocks(&mut detections);

        let mut texts = detections
            .iter()
            .filter(|detection| detection.label == "text")
            .map(|detection| detection.bbox)
            .collect::<Vec<_>>();
        texts.sort_by(|left, right| left[0].total_cmp(&right[0]));
        assert_eq!(
            texts,
            [
                [448.0, 411.0, 617.0, 1644.0],
                [474.0, 3435.0, 1094.0, 4754.0],
                [2480.0, 369.0, 2906.0, 1726.0],
                [3837.0, 640.0, 4476.0, 2069.0],
                [4520.0, 830.0, 4640.0, 1400.0],
            ]
        );
        let block = detections
            .iter()
            .find(|detection| detection.bbox[0] == 3837.0)
            .unwrap();
        assert!(block.mask.contains(3900, 1000) && block.mask.contains(4400, 1000));
        assert!(!block.mask.contains(4280, 1000));
    }

    #[test]
    fn bubbles_split_across_one_text_block_join() {
        // 144304368_p6: two overlapping balloons hold one utterance, detected
        // as one text box. A block sticking out of its bubbles keeps them apart.
        let masked = |label: &str, bbox: [f32; 4]| {
            let [left, top, right, bottom] = bbox.map(|value| value as u32);
            KoharuLayoutDetection {
                mask: full_mask(left, top, right - left, bottom - top).into(),
                ..detection(label, 0.9, bbox)
            }
        };
        let mut detections = vec![
            masked("bubble", [4380.0, 4876.0, 4782.0, 6045.0]),
            masked("bubble", [3808.0, 4894.0, 4513.0, 6630.0]),
            masked("text", [3967.0, 5052.0, 4628.0, 6543.0]),
            masked("bubble", [0.0, 0.0, 180.0, 400.0]),
            masked("bubble", [170.0, 0.0, 250.0, 400.0]),
            masked("text", [100.0, 100.0, 300.0, 300.0]),
        ];

        super::join_split_bubbles(&mut detections);

        let mut bubbles = detections
            .iter()
            .filter(|detection| detection.label == "bubble")
            .map(|detection| detection.bbox)
            .collect::<Vec<_>>();
        bubbles.sort_by(|left, right| left[0].total_cmp(&right[0]));
        assert_eq!(
            bubbles,
            [
                [0.0, 0.0, 180.0, 400.0],
                [170.0, 0.0, 250.0, 400.0],
                [3808.0, 4876.0, 4782.0, 6630.0],
            ]
        );
    }

    #[test]
    fn overlapping_text_keeps_both_masks_while_nms_suppresses_duplicate_bubbles() {
        let detections = [
            KoharuLayoutDetection {
                mask: full_mask(110, 10, 30, 80).into(),
                ..detection("text", 0.8, [0.0, 0.0, 150.0, 100.0])
            },
            KoharuLayoutDetection {
                mask: full_mask(10, 10, 80, 80).into(),
                ..detection("text", 0.9, [0.0, 0.0, 100.0, 100.0])
            },
            detection("bubble", 0.7, [0.0, 0.0, 100.0, 100.0]),
            detection("bubble", 0.5, [5.0, 5.0, 105.0, 105.0]),
            detection("text", 0.6, [200.0, 0.0, 250.0, 50.0]),
        ];

        let mut candidates = detections
            .into_iter()
            .map(|detection| super::Candidate {
                proposal: super::GlyphProposal::Segmented(detection.bbox),
                detection,
            })
            .collect();
        non_maximum_suppression(&mut candidates, 0.5);
        let mut detections = candidates.into_iter().map(|c| c.detection).collect();
        super::merge_text_blocks(&mut detections);

        let text = detections
            .iter()
            .filter(|detection| detection.label == "text")
            .collect::<Vec<_>>();
        assert_eq!(text.len(), 2);
        let paragraph = text
            .iter()
            .find(|detection| detection.score == 0.9)
            .unwrap();
        assert_eq!(paragraph.bbox, [0.0, 0.0, 150.0, 100.0]);
        assert!(paragraph.mask.contains(20, 20));
        assert!(paragraph.mask.contains(120, 20));
        let bubble_scores = detections
            .iter()
            .filter(|detection| detection.label == "bubble")
            .map(|detection| detection.score)
            .collect::<Vec<_>>();
        assert_eq!(bubble_scores, [0.7]);
    }

    #[test]
    fn paragraph_union_keeps_mask_pixels_outside_the_box() {
        let mut detections = vec![
            KoharuLayoutDetection {
                mask: full_mask(7, 11, 8, 30).into(),
                ..detection("text", 0.9, [10.0, 10.0, 40.0, 50.0])
            },
            KoharuLayoutDetection {
                mask: full_mask(65, 12, 10, 28).into(),
                ..detection("text", 0.8, [40.0, 10.0, 70.0, 50.0])
            },
        ];

        super::merge_text_blocks(&mut detections);

        let [paragraph] = detections.as_slice() else {
            panic!("expected one paragraph");
        };
        assert_eq!(paragraph.bbox, [10.0, 10.0, 70.0, 50.0]);
        assert!(paragraph.mask.contains(7, 11));
        assert!(paragraph.mask.contains(74, 39));
        assert_eq!(paragraph.area, 520);
    }

    #[test]
    fn text_mask_excludes_onomatopoeia() {
        let detection = |label: &str, x: u32, value: u8| KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score: 1.0,
            bbox: [x as f32, 0.0, x as f32 + 1.0, 1.0],
            area: u32::from(value != 0),
            mask: KoharuLayoutMask {
                x,
                y: 0,
                width: 1,
                height: 1,
                pixels: vec![value],
            }
            .into(),
        };
        let detections = vec![
            detection("bubble", 0, 0),
            detection("onomatopoeia", 0, 255),
            detection("text", 1, 255),
            detection("onomatopoeia", 3, 255),
        ];

        let mask = mask_for(
            &detections,
            "text",
            ImageSize {
                width: 4,
                height: 1,
            },
        );

        assert_eq!(mask.as_raw(), &[0, 255, 0, 0]);
    }

    #[test]
    fn bubble_membership_uses_detected_ink_instead_of_text_box_corners() {
        let mask = |pixels| KoharuLayoutMask {
            x: 0,
            y: 0,
            width: 5,
            height: 5,
            pixels,
        };
        let bubble = mask(vec![
            0, 0, 1, 0, 0, //
            0, 1, 1, 1, 0, //
            1, 1, 1, 1, 1, //
            0, 1, 1, 1, 0, //
            0, 0, 1, 0, 0,
        ]);
        let dialogue = mask(vec![
            0, 0, 0, 0, 0, //
            0, 0, 1, 0, 0, //
            0, 1, 0, 1, 0, //
            0, 0, 1, 0, 0, //
            0, 0, 0, 0, 0,
        ]);
        let crossing = mask(vec![
            0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, //
            0, 0, 0, 1, 1, //
            0, 0, 0, 1, 1, //
            0, 0, 0, 1, 1,
        ]);

        let bounds = [0.0, 0.0, 5.0, 5.0];
        assert_eq!(mask_containment(&bubble, &dialogue, bounds), 1.0);
        assert!(mask_containment(&bubble, &crossing, bounds) < DIALOGUE_MASK_CONTAINMENT_THRESHOLD);
    }

    #[test]
    fn mask_containment_accepts_independent_local_extents() {
        let valid = KoharuLayoutMask {
            x: 5,
            y: 5,
            width: 2,
            height: 2,
            pixels: vec![1; 4],
        };
        let malformed = KoharuLayoutMask {
            x: 5,
            y: 5,
            width: 2,
            height: 2,
            pixels: vec![1; 3],
        };
        let different_size = KoharuLayoutMask {
            x: 5,
            y: 5,
            width: 1,
            height: 1,
            pixels: vec![1],
        };

        let bounds = [5.0, 5.0, 7.0, 7.0];
        assert_eq!(mask_containment(&valid, &malformed, bounds), 0.0);
        assert_eq!(mask_containment(&valid, &different_size, bounds), 1.0);
    }

    fn boxed_detection(label: &str, score: f32, bbox: [f32; 4]) -> KoharuLayoutDetection {
        let mask = super::box_mask(bbox);
        KoharuLayoutDetection {
            area: mask.width * mask.height,
            mask: mask.into(),
            ..detection(label, score, bbox)
        }
    }

    #[test]
    fn layout_order_follows_panels_then_bubbles_then_their_text() {
        let detections = vec![
            boxed_detection("text", 0.63, [20.0, 30.0, 70.0, 70.0]),
            boxed_detection("bubble", 0.7, [10.0, 20.0, 80.0, 80.0]),
            boxed_detection("panel", 0.9, [100.0, 0.0, 200.0, 200.0]),
            boxed_detection("text", 0.62, [130.0, 110.0, 180.0, 150.0]),
            boxed_detection("bubble", 0.8, [120.0, 20.0, 190.0, 80.0]),
            boxed_detection("panel", 0.9, [0.0, 0.0, 95.0, 200.0]),
            boxed_detection("text", 0.61, [130.0, 30.0, 180.0, 70.0]),
            boxed_detection("bubble", 0.7, [120.0, 100.0, 190.0, 160.0]),
        ];

        let text_scores = layout_order(&detections)
            .into_iter()
            .filter_map(|item| {
                let detection = &detections[item.index];
                (detection.label == "text").then_some(detection.score)
            })
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.61, 0.62, 0.63]);
    }

    #[test]
    fn layout_order_merges_containers_and_text_at_each_spatial_level() {
        let detections = vec![
            boxed_detection("panel", 0.9, [0.0, 40.0, 200.0, 240.0]),
            boxed_detection("bubble", 0.8, [100.0, 120.0, 190.0, 210.0]),
            boxed_detection("text", 0.3, [120.0, 140.0, 180.0, 190.0]),
            boxed_detection("text", 0.2, [20.0, 60.0, 180.0, 90.0]),
            boxed_detection("text", 0.1, [20.0, 0.0, 180.0, 20.0]),
        ];

        let text_scores = layout_order(&detections)
            .into_iter()
            .filter_map(|item| {
                let detection = &detections[item.index];
                (detection.label == "text").then_some(detection.score)
            })
            .collect::<Vec<_>>();

        assert_eq!(text_scores, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn staggered_balloons_read_right_to_left_within_each_row() {
        let detections = [
            detection("bubble", 0.9, [10.0, 10.0, 40.0, 40.0]),
            detection("bubble", 0.9, [60.0, 55.0, 90.0, 85.0]),
            detection("bubble", 0.9, [60.0, 11.0, 90.0, 42.0]),
            detection("bubble", 0.9, [10.0, 50.0, 40.0, 81.0]),
        ];
        let order: Vec<_> = layout_order(&detections)
            .iter()
            .map(|item| item.index)
            .collect();
        assert_eq!(order, [2, 0, 1, 3]);
    }

    #[test]
    fn overlapping_siblings_with_cyclic_preferences_are_each_read_once() {
        let detections = [
            detection("text", 0.9, [30.0, 20.0, 40.0, 40.0]),
            detection("text", 0.9, [20.0, 0.0, 30.0, 40.0]),
            detection("text", 0.9, [10.0, 20.0, 50.0, 30.0]),
        ];
        // Each has one predecessor. The highest centre breaks the cycle.
        let order: Vec<_> = layout_order(&detections)
            .iter()
            .map(|item| item.index)
            .collect();
        assert_eq!(order, [1, 2, 0]);
    }

    #[test]
    fn inset_panels_and_margin_captions_read_with_their_panel() {
        let detections = [
            detection("panel", 0.9, [0.0, 20.0, 90.0, 220.0]),
            detection("panel", 0.9, [100.0, 21.0, 200.0, 220.0]),
            detection("panel", 0.9, [110.0, 70.0, 185.0, 185.0]),
            detection("panel", 0.9, [115.0, 90.0, 145.0, 130.0]),
            detection("text", 0.8, [120.0, 100.0, 140.0, 120.0]),
            detection("text", 0.8, [150.0, 80.0, 180.0, 125.0]),
            detection("text", 0.8, [205.0, 190.0, 215.0, 215.0]),
            detection("text", 0.8, [30.0, 0.0, 60.0, 10.0]),
        ];
        // The right margin caption follows both insets, before the left panel.
        let order: Vec<_> = layout_order(&detections)
            .iter()
            .map(|item| item.index)
            .collect();
        assert_eq!(order, [7, 1, 2, 5, 3, 4, 6, 0]);
    }

    #[test]
    fn balloon_across_a_missing_panel_reads_at_page_level() {
        let detections = [
            boxed_detection("panel", 0.9, [0.0, 1.0, 513.0, 494.0]),
            boxed_detection("text", 0.9, [342.0, 63.0, 488.0, 337.0]),
            boxed_detection("panel", 0.9, [0.0, 503.0, 226.0, 933.0]),
            boxed_detection("text", 0.9, [162.0, 502.0, 199.0, 672.0]),
            boxed_detection("bubble", 0.9, [939.0, 191.0, 1148.0, 851.0]),
            boxed_detection("text", 0.9, [1009.0, 299.0, 1079.0, 778.0]),
        ];
        // The lower-left panel spans half the right balloon's height. The
        // page's vertical gutter puts the balloon before both left panels.
        let order: Vec<_> = layout_order(&detections)
            .iter()
            .map(|item| item.index)
            .collect();
        assert_eq!(order, [4, 5, 0, 1, 2, 3]);
    }

    #[test]
    fn untranslated_sound_effect_keeps_the_gutter_between_dialogue_rows_open() {
        let detections = [
            boxed_detection("panel", 0.9, [0.0, 0.0, 120.0, 100.0]),
            boxed_detection("bubble", 0.9, [10.0, 10.0, 40.0, 40.0]),
            boxed_detection("text", 0.9, [15.0, 15.0, 35.0, 35.0]),
            boxed_detection("bubble", 0.9, [60.0, 55.0, 90.0, 85.0]),
            boxed_detection("text", 0.9, [65.0, 60.0, 85.0, 80.0]),
            boxed_detection("onomatopoeia", 0.9, [95.0, 5.0, 115.0, 95.0]),
        ];
        let order = layout_order(&detections);
        let dialogue: Vec<_> = order
            .iter()
            .filter(|item| detections[item.index].label == "text")
            .map(|item| (item.index, item.bubble))
            .collect();
        assert_eq!(dialogue, [(2, Some(1)), (4, Some(3))]);
        assert_eq!(order.iter().filter(|item| item.index == 5).count(), 1);
    }

    #[test]
    fn typography_comes_from_horizontal_text_mask() {
        let (image, detection) = masked_text(52.0, 12.0, 12.0, [24, 80, 160]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert!((inferred.angle_degrees - 12.0).abs() < 1.0);
        assert_eq!(inferred.color, [24, 80, 160]);
        assert_eq!(inferred.stroke_color, None);
        assert_eq!(inferred.stroke_width, None);
        assert_eq!(inferred.writing_mode, WritingMode::Horizontal);
    }

    #[test]
    fn vertical_text_angle_is_relative_to_upright_vertical() {
        let (image, detection) = masked_text(12.0, 52.0, 9.0, [120, 80, 40]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert!((inferred.angle_degrees - 9.0).abs() < 1.0);
        assert_eq!(inferred.writing_mode, WritingMode::Vertical);
    }

    #[test]
    fn tall_multiline_mask_uses_text_lines_instead_of_block_aspect() {
        let width = 96;
        let height = 96;
        let angle_degrees = 8.0_f64;
        let (sin, cos) = angle_degrees.to_radians().sin_cos();
        let mut image = RgbImage::from_pixel(width, height, Rgb([240, 240, 240]));
        let mut pixels = vec![0; width as usize * height as usize];
        for y in 0..height {
            for x in 0..width {
                let dx = f64::from(x) + 0.5 - f64::from(width) * 0.5;
                let dy = f64::from(y) + 0.5 - f64::from(height) * 0.5;
                let local_x = dx * cos + dy * sin;
                let local_y = -dx * sin + dy * cos;
                let inside = [-24.0, 0.0, 24.0]
                    .into_iter()
                    .any(|line_y| local_x.abs() <= 15.0 && (local_y - line_y).abs() <= 2.0);
                if inside {
                    pixels[y as usize * width as usize + x as usize] = u8::MAX;
                    image.put_pixel(x, y, Rgb([8, 8, 8]));
                }
            }
        }
        let detection = KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 1.0,
            bbox: [0.0, 0.0, width as f32, height as f32],
            area: pixels.iter().filter(|value| **value != 0).count() as u32,
            mask: KoharuLayoutMask {
                x: 0,
                y: 0,
                width,
                height,
                pixels,
            }
            .into(),
        };

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.writing_mode, WritingMode::Horizontal);
        assert!((inferred.angle_degrees - 8.0).abs() < 1.0);
    }

    #[test]
    fn antialiased_dark_text_uses_the_high_contrast_core() {
        let (mut image, detection) = masked_text(52.0, 12.0, 0.0, [96, 94, 92]);
        for (index, &mask) in detection.mask.pixels.iter().enumerate() {
            if mask != 0 && index.is_multiple_of(3) {
                let x = index as u32 % image.width();
                let y = index as u32 / image.width();
                image.put_pixel(x, y, Rgb([8, 9, 7]));
            }
        }

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
    }

    #[test]
    fn text_colors_snap_only_near_black_or_white() {
        assert_eq!(normalize_text_color([20, 31, 24]), [0, 0, 0]);
        assert_eq!(normalize_text_color([230, 240, 250]), [255, 255, 255]);
        assert_eq!(normalize_text_color([255, 0, 0]), [255, 0, 0]);
        assert_eq!(normalize_text_color([140, 20, 30]), [140, 20, 30]);
        assert_eq!(normalize_text_color([0, 255, 0]), [0, 255, 0]);
        assert_eq!(normalize_text_color([20, 115, 235]), [20, 115, 235]);
        assert_eq!(normalize_text_color([40, 70, 40]), [0, 0, 0]);
    }

    #[test]
    fn outlined_text_uses_the_deep_fill_and_measures_the_border() {
        for stroke_width in [2, 3, 5] {
            let (image, detection) = outlined_text(stroke_width, [0, 0, 0], [255, 255, 255]);

            let inferred = infer_typography(&image, &detection).unwrap();

            assert_eq!(inferred.color, [0, 0, 0]);
            assert_eq!(inferred.stroke_color, Some([255, 255, 255]));
            assert_eq!(inferred.stroke_width, Some(stroke_width as f32));
        }
    }

    #[test]
    fn outlined_text_roles_do_not_flip_with_inverse_luminance() {
        let (image, detection) = outlined_text(3, [255, 255, 255], [0, 0, 0]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [255, 255, 255]);
        assert_eq!(inferred.stroke_color, Some([0, 0, 0]));
        assert_eq!(inferred.stroke_width, Some(3.0));
    }

    #[test]
    fn glyph_holes_matching_the_background_do_not_invert_the_paint() {
        let (image, detection) =
            outlined_text_on_background(3, [255, 255, 255], [0, 0, 0], [255, 255, 255]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
        assert_eq!(inferred.stroke_color, None);
        assert_eq!(inferred.stroke_width, None);
    }

    #[test]
    fn outlined_glyph_counters_do_not_invert_the_paint_at_high_resolution() {
        // 6x5 glyphs, each a black ring with a white outline outside it and a
        // white counter inside, the way a 600 dpi scan draws outlined kanji.
        let [left, top] = [8, 8];
        let (columns, rows, cell, stroke) = (6, 5, 40, 6);
        let width = left * 2 + columns * cell;
        let height = top * 2 + rows * cell;
        let mut image = RgbImage::from_pixel(width, height, Rgb([90, 140, 60]));
        for row in 0..rows {
            for column in 0..columns {
                let (x0, y0) = (left + column * cell, top + row * cell);
                for y in 0..cell {
                    for x in 0..cell {
                        let edge = x.min(y).min(cell - 1 - x).min(cell - 1 - y);
                        let black = (stroke..stroke * 2).contains(&edge);
                        let color = if black { [0, 0, 0] } else { [255, 255, 255] };
                        image.put_pixel(x0 + x, y0 + y, Rgb(color));
                    }
                }
            }
        }
        let pixels = vec![u8::MAX; width as usize * height as usize];
        let detection = KoharuLayoutDetection {
            label_id: 0,
            label: "text".to_owned(),
            score: 1.0,
            bbox: [0.0, 0.0, width as f32, height as f32],
            area: width * height,
            mask: KoharuLayoutMask {
                x: 0,
                y: 0,
                width,
                height,
                pixels,
            }
            .into(),
        };

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
        assert_eq!(inferred.stroke_color, Some([255, 255, 255]));
    }

    #[test]
    fn outlined_text_is_separated_from_matching_dark_artwork() {
        let (image, detection) = outlined_text_on_textured_background();

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
        assert_eq!(inferred.stroke_color, Some([255, 255, 255]));
        assert_eq!(inferred.stroke_width, Some(3.0));
    }

    #[test]
    fn fitted_outlined_ink_on_art_keeps_its_fill_band_and_em() {
        // The fill occurs in the artwork too. Its connected white band is
        // identified before the black dots can affect the source em or paint.
        let mut source = RgbImage::from_fn(180, 180, |x, y| {
            Rgb(if x % 3 == 0 && y % 3 == 0 {
                [0; 3]
            } else {
                [28; 3]
            })
        });
        let cross = |x: u32, y: u32| {
            let (x, y) = (x.wrapping_sub(68), y.wrapping_sub(38));
            x < 20
                && y < 104
                && y % 28 < 20
                && ((8..12).contains(&x) || (8..12).contains(&(y % 28)))
        };
        let mut mask = GrayImage::new(28, 112);
        for (x, y, p) in mask.enumerate_pixels_mut() {
            let (px, py) = (64 + x, 34 + y);
            if (py - 4..=py + 4).any(|y| (px - 4..=px + 4).any(|x| cross(x, y))) {
                *p = Luma([255]);
                source.put_pixel(px, py, Rgb(if cross(px, py) { [0; 3] } else { [255; 3] }));
            }
        }
        let mut text = detection("text", 0.9, [64., 34., 92., 146.]);
        text.mask = KoharuLayoutMask {
            x: 64,
            y: 34,
            width: 28,
            height: 112,
            pixels: mask.into_raw(),
        }
        .into();
        super::fit_glyph_masks(
            [(super::GlyphProposal::Segmented(text.bbox), &mut text)],
            &source,
        );
        let found = infer_typography(&source, &text).unwrap();
        assert_eq!(found.color, [0; 3]);
        assert_eq!(found.stroke_color, Some([255; 3]));
        assert!(
            found.stroke_width.is_some_and(|w| (3.0..=5.0).contains(&w)),
            "{found:?}"
        );
        assert!((18.0..=22.0).contains(&found.size), "{found:?}");
        assert!(text.mask.contains(78, 47) && text.mask.contains(68, 44));
    }

    #[test]
    fn colors_in_the_same_luminance_class_do_not_create_a_border() {
        let (image, detection) = outlined_text(3, [0, 0, 0], [48, 48, 48]);

        let inferred = infer_typography(&image, &detection).unwrap();

        assert_eq!(inferred.color, [0, 0, 0]);
        assert_eq!(inferred.stroke_color, None);
        assert_eq!(inferred.stroke_width, None);
    }

    #[test]
    fn wide_antialias_bands_do_not_create_borders() {
        for (fill, antialias, background) in [
            ([20, 115, 235], [133, 180, 240], [245, 245, 245]),
            ([0, 0, 0], [96, 96, 96], [245, 245, 245]),
        ] {
            let (image, detection) = outlined_text_on_background(4, fill, antialias, background);

            let inferred = infer_typography(&image, &detection).unwrap();

            assert_eq!(inferred.color, fill);
            assert_eq!(inferred.stroke_color, None);
            assert_eq!(inferred.stroke_width, None);
        }
    }

    #[test]
    fn text_region_background_is_not_mistaken_for_the_fill() {
        for (fill, background, expected) in [
            ([24, 40, 80], [225, 130, 175], [0, 0, 0]),
            ([220, 240, 255], [16, 24, 88], [255, 255, 255]),
            ([20, 115, 235], [245, 245, 245], [20, 115, 235]),
        ] {
            let (image, detection) = region_masked_text(fill, background);

            let inferred = infer_typography(&image, &detection).unwrap();

            assert_eq!(inferred.color, expected);
            assert_eq!(inferred.stroke_color, None);
            assert_eq!(inferred.stroke_width, None);
        }
    }

    #[test]
    fn color_palette_is_bounded_independently_of_crop_area() {
        let pixels = (0..65_536_u32)
            .map(|index| MaskPixel {
                x: 0,
                y: 0,
                color: [
                    index as u8,
                    (index >> 8) as u8,
                    index.wrapping_mul(31) as u8,
                ],
                inside_mask: true,
            })
            .collect::<Vec<_>>();

        assert!(color_palette(&pixels).len() <= 16 * 16 * 16);
    }
}
