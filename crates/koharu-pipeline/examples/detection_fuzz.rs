use std::{
    collections::{HashMap, HashSet},
    env,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use koharu_ml::{
    comic_text_bubble_detector::{ComicTextBubbleDetection, ComicTextBubbleLabel},
    koharu_layout_rfdetr_seg_2xl::{
        KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask,
    },
};
use koharu_pipeline::{DetectionFusionConfig, fuse_detections};
use rayon::prelude::*;
use serde::Deserialize;

const RFDETR_TEXT_LABEL_ID: usize = 0;
const RFDETR_ONOMATOPOEIA_LABEL_ID: usize = 1;
const RFDETR_BUBBLE_LABEL_ID: usize = 2;
const RFDETR_PANEL_LABEL_ID: usize = 3;
const DEFAULT_BUBBLE_THRESHOLD: f32 = 0.25;

// ---------------------------------------------------------------------------
// Deterministic PRNG (SplitMix64)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    fn chance(&mut self, p: f32) -> bool {
        self.f32() < p
    }
}

// ---------------------------------------------------------------------------
// COCO JSON Schema & RLE Decoding
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CocoDataset {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
    #[serde(default)]
    relations: Vec<CocoRelation>,
}

#[derive(Deserialize)]
struct CocoImage {
    id: u64,
    width: u32,
    height: u32,
    #[serde(default)]
    file_name: String,
}

#[derive(Deserialize)]
struct CocoAnnotation {
    id: u64,
    image_id: u64,
    category_id: u32,
    bbox: [f32; 4], // [x, y, w, h]
    #[serde(default)]
    segmentation: Option<CocoSegmentation>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CocoSegmentation {
    Rle {
        size: [u32; 2],
        counts: String,
    },
    #[allow(dead_code)]
    Polygons(serde_json::Value),
}

#[derive(Deserialize)]
struct CocoRelation {
    #[serde(rename = "type")]
    rel_type: String,
    from: u64,
    to: u64,
}

fn decode_coco_rle(size: [u32; 2], counts: &str, bbox: [f32; 4]) -> KoharuLayoutMask {
    let h = size[0] as usize;
    let w = size[1] as usize;
    if h == 0 || w == 0 {
        return rect_mask(bbox);
    }

    let bytes = counts.as_bytes();
    let mut cnts: Vec<usize> = Vec::new();
    let mut p = 0usize;
    while p < bytes.len() {
        let mut x: i64 = 0;
        let mut k = 0u32;
        let mut more = true;
        while more && p < bytes.len() {
            let c = (bytes[p] as i64) - 48;
            x |= (c & 0x1f) << (5 * k);
            more = (c & 0x20) != 0;
            p += 1;
            k += 1;
            if !more && (c & 0x10) != 0 {
                x |= !0i64 << (5 * k);
            }
        }
        if cnts.len() > 2 {
            x += cnts[cnts.len() - 2] as i64;
        }
        if x < 0 {
            return rect_mask(bbox);
        }
        cnts.push(x as usize);
    }

    let bx0 = (bbox[0].floor().max(0.0) as usize).min(w);
    let by0 = (bbox[1].floor().max(0.0) as usize).min(h);
    let bx1 = (bbox[2].ceil().max(0.0) as usize).min(w);
    let by1 = (bbox[3].ceil().max(0.0) as usize).min(h);
    if bx1 <= bx0 || by1 <= by0 {
        return rect_mask(bbox);
    }
    let mw = bx1 - bx0;
    let mh = by1 - by0;
    let mut pixels = vec![0u8; mw * mh];
    let mut idx = 0usize;
    let total = h * w;
    for (run_idx, &run) in cnts.iter().enumerate() {
        let end = (idx + run).min(total);
        if run_idx % 2 == 1 {
            // Foreground run in column-major order: idx = col * h + row.
            let mut cur = idx;
            while cur < end {
                let col = cur / h;
                let row = cur % h;
                let col_end = ((col + 1) * h).min(end);
                if col >= bx0 && col < bx1 {
                    let r_start = row.max(by0);
                    let r_end = (col_end - col * h).min(by1);
                    if r_start < r_end {
                        let lx = col - bx0;
                        for r in r_start..r_end {
                            pixels[(r - by0) * mw + lx] = u8::MAX;
                        }
                    }
                }
                cur = col_end;
            }
        }
        idx = end;
        if idx >= total {
            break;
        }
    }
    if pixels.iter().all(|&v| v == 0) {
        return rect_mask(bbox);
    }
    KoharuLayoutMask {
        x: bx0 as u32,
        y: by0 as u32,
        width: mw as u32,
        height: mh as u32,
        pixels,
    }
}

fn rect_mask(bbox: [f32; 4]) -> KoharuLayoutMask {
    let x = bbox[0].floor().max(0.0) as u32;
    let y = bbox[1].floor().max(0.0) as u32;
    let width = (bbox[2].ceil().max(0.0) as u32).saturating_sub(x).max(1);
    let height = (bbox[3].ceil().max(0.0) as u32).saturating_sub(y).max(1);
    KoharuLayoutMask {
        x,
        y,
        width,
        height,
        pixels: vec![u8::MAX; (width * height) as usize],
    }
}

fn ellipse_gt_mask(bbox: [f32; 4]) -> KoharuLayoutMask {
    let x = bbox[0].floor().max(0.0) as u32;
    let y = bbox[1].floor().max(0.0) as u32;
    let width = (bbox[2].ceil().max(0.0) as u32).saturating_sub(x).max(1);
    let height = (bbox[3].ceil().max(0.0) as u32).saturating_sub(y).max(1);
    let rx = ((bbox[2] - bbox[0]) / 2.0).max(0.5);
    let ry = ((bbox[3] - bbox[1]) / 2.0).max(0.5);
    let cx = bbox[0] + rx;
    let cy = bbox[1] + ry;
    let mut pixels = vec![0u8; (width * height) as usize];
    for ly in 0..height {
        for lx in 0..width {
            let dx = (x as f32 + lx as f32 + 0.5 - cx) / rx;
            let dy = (y as f32 + ly as f32 + 0.5 - cy) / ry;
            if dx * dx + dy * dy <= 1.0 {
                pixels[(ly * width + lx) as usize] = u8::MAX;
            }
        }
    }
    KoharuLayoutMask {
        x,
        y,
        width,
        height,
        pixels,
    }
}

// ---------------------------------------------------------------------------
// Ground-Truth Page Representation
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct GtItem {
    bbox: [f32; 4], // [x1, y1, x2, y2]
    mask: Arc<KoharuLayoutMask>,
    area: u32,
    bubble_idx: Option<usize>,
}

#[derive(Clone)]
struct GtPage {
    id: u64,
    #[allow(dead_code)]
    name: String,
    width: u32,
    height: u32,
    texts: Vec<GtItem>,
    onomatopoeia: Vec<GtItem>,
    bubbles: Vec<GtItem>,
    panels: Vec<GtItem>,
}

fn load_coco_split(path: &Path, max_pages: Option<usize>) -> Result<Vec<GtPage>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let dataset: CocoDataset = serde_json::from_reader(reader)
        .with_context(|| format!("failed to parse {}", path.display()))?;

    let mut contained_by_bubble: HashMap<u64, u64> = HashMap::new();
    for rel in &dataset.relations {
        if rel.rel_type == "contained_by_bubble" {
            contained_by_bubble.insert(rel.from, rel.to);
        }
    }

    let mut anns_by_image: HashMap<u64, Vec<&CocoAnnotation>> = HashMap::new();
    for ann in &dataset.annotations {
        anns_by_image.entry(ann.image_id).or_default().push(ann);
    }

    let images: Vec<&CocoImage> = match max_pages {
        Some(limit) => dataset.images.iter().take(limit).collect(),
        None => dataset.images.iter().collect(),
    };

    let pages: Vec<GtPage> = images
        .par_iter()
        .map(|img| {
            let empty = Vec::new();
            let anns = anns_by_image.get(&img.id).unwrap_or(&empty);
            let mut bubbles = Vec::new();
            let mut bubble_id_to_idx = HashMap::new();
            let mut panels = Vec::new();
            let mut raw_texts = Vec::new();
            let mut raw_sfx = Vec::new();

            for ann in anns {
                let [x, y, w, h] = ann.bbox;
                let bbox = [
                    x.clamp(0.0, img.width as f32),
                    y.clamp(0.0, img.height as f32),
                    (x + w).clamp(0.0, img.width as f32),
                    (y + h).clamp(0.0, img.height as f32),
                ];
                if bbox[2] - bbox[0] < 1.0 || bbox[3] - bbox[1] < 1.0 {
                    continue;
                }
                let mask = match &ann.segmentation {
                    Some(CocoSegmentation::Rle { size, counts }) => {
                        decode_coco_rle(*size, counts, bbox)
                    }
                    _ => rect_mask(bbox),
                };
                let area = mask.pixels.iter().filter(|&&v| v != 0).count() as u32;
                let item = GtItem {
                    bbox,
                    mask: Arc::new(mask),
                    area,
                    bubble_idx: None,
                };
                match ann.category_id {
                    1 => raw_texts.push((ann.id, item)),
                    2 => raw_sfx.push((ann.id, item)),
                    3 => {
                        bubble_id_to_idx.insert(ann.id, bubbles.len());
                        bubbles.push(item);
                    }
                    4 => panels.push(item),
                    _ => {}
                }
            }

            let texts = raw_texts
                .into_iter()
                .map(|(ann_id, mut item)| {
                    item.bubble_idx = contained_by_bubble
                        .get(&ann_id)
                        .and_then(|b_id| bubble_id_to_idx.get(b_id).copied());
                    item
                })
                .collect();

            let onomatopoeia = raw_sfx
                .into_iter()
                .map(|(ann_id, mut item)| {
                    item.bubble_idx = contained_by_bubble
                        .get(&ann_id)
                        .and_then(|b_id| bubble_id_to_idx.get(b_id).copied());
                    item
                })
                .collect();

            GtPage {
                id: img.id,
                name: img.file_name.clone(),
                width: img.width,
                height: img.height,
                texts,
                onomatopoeia,
                bubbles,
                panels,
            }
        })
        .collect();

    Ok(pages)
}

// ---------------------------------------------------------------------------
// Synthetic Adversarial Edge-Case Layouts
// ---------------------------------------------------------------------------

fn make_gt_item(bbox: [f32; 4], is_ellipse: bool, bubble_idx: Option<usize>) -> GtItem {
    let mask = if is_ellipse {
        ellipse_gt_mask(bbox)
    } else {
        rect_mask(bbox)
    };
    let area = mask.pixels.iter().filter(|&&v| v != 0).count() as u32;
    GtItem {
        bbox,
        mask: Arc::new(mask),
        area,
        bubble_idx,
    }
}

fn build_synthetic_adversarial_pages() -> Vec<GtPage> {
    let mut pages = Vec::new();
    let mut next_id = 9_000_000u64;

    // 1. Diagonal adjacent bubbles whose bounding boxes overlap, with tight vertical text inside each.
    for variant in 0..4 {
        let shift = variant as f32 * 8.0;
        let b1 = make_gt_item([100.0, 100.0, 260.0, 280.0], true, None);
        let b2 = make_gt_item(
            [210.0 - shift, 210.0 - shift, 370.0 - shift, 390.0 - shift],
            true,
            None,
        );
        // Texts strictly inside their respective ellipse masks, but b1/b2 bboxes overlap in [210-s..260, 210-s..280]
        let t1 = make_gt_item([145.0, 140.0, 210.0, 235.0], false, Some(0));
        let t2 = make_gt_item(
            [255.0 - shift, 250.0 - shift, 320.0 - shift, 345.0 - shift],
            false,
            Some(1),
        );
        pages.push(GtPage {
            id: next_id,
            name: format!("synth_diagonal_bubbles_{variant}"),
            width: 800,
            height: 600,
            texts: vec![t1, t2],
            onomatopoeia: vec![],
            bubbles: vec![b1, b2],
            panels: vec![make_gt_item([40.0, 40.0, 760.0, 560.0], false, None)],
        });
        next_id += 1;
    }

    // 2. Closely spaced vertical columns in two adjacent bubbles (thin wall between them).
    for gap in [4.0f32, 8.0, 12.0, 16.0] {
        let b1 = make_gt_item([80.0, 100.0, 220.0, 340.0], true, None);
        let b2 = make_gt_item([220.0 + gap, 100.0, 360.0 + gap, 340.0], true, None);
        let t1 = make_gt_item([120.0, 145.0, 198.0, 295.0], false, Some(0));
        let t2 = make_gt_item([235.0 + gap, 145.0, 315.0 + gap, 295.0], false, Some(1));
        pages.push(GtPage {
            id: next_id,
            name: format!("synth_thin_wall_bubbles_gap_{gap}"),
            width: 800,
            height: 600,
            texts: vec![t1, t2],
            onomatopoeia: vec![],
            bubbles: vec![b1, b2],
            panels: vec![make_gt_item([20.0, 20.0, 780.0, 580.0], false, None)],
        });
        next_id += 1;
    }

    // 3. Outlined free text directly beside a bubble (tests cross-bubble/free-text isolation).
    for offset in [6.0f32, 12.0, 18.0] {
        let b1 = make_gt_item([120.0, 120.0, 280.0, 320.0], true, None);
        let t_in = make_gt_item([165.0, 160.0, 240.0, 280.0], false, Some(0));
        let t_free = make_gt_item([280.0 + offset, 160.0, 330.0 + offset, 280.0], false, None);
        pages.push(GtPage {
            id: next_id,
            name: format!("synth_free_text_beside_bubble_{offset}"),
            width: 800,
            height: 600,
            texts: vec![t_in, t_free],
            onomatopoeia: vec![],
            bubbles: vec![b1],
            panels: vec![make_gt_item([30.0, 30.0, 770.0, 570.0], false, None)],
        });
        next_id += 1;
    }

    // 4. Translucent multi-lobe balloons (wide double-lobe with spanning text).
    for variant in 0..4 {
        let b_box = [100.0, 120.0, 420.0 + variant as f32 * 20.0, 320.0];
        let b = make_gt_item(b_box, true, None);
        let t = make_gt_item(
            [150.0, 160.0, 370.0 + variant as f32 * 10.0, 280.0],
            false,
            Some(0),
        );
        pages.push(GtPage {
            id: next_id,
            name: format!("synth_translucent_multilobe_{variant}"),
            width: 800,
            height: 600,
            texts: vec![t],
            onomatopoeia: vec![],
            bubbles: vec![b],
            panels: vec![make_gt_item([20.0, 20.0, 780.0, 580.0], false, None)],
        });
        next_id += 1;
    }

    // 5. SFX inside / overlapping bubbles alongside dialogue.
    for variant in 0..3 {
        let b = make_gt_item([100.0, 100.0, 320.0, 320.0], true, None);
        let t = make_gt_item([155.0, 145.0, 265.0, 275.0], false, Some(0));
        let sfx = make_gt_item(
            [
                115.0 + variant as f32 * 15.0,
                120.0,
                220.0 + variant as f32 * 15.0,
                240.0,
            ],
            false,
            Some(0),
        );
        pages.push(GtPage {
            id: next_id,
            name: format!("synth_sfx_in_bubble_{variant}"),
            width: 800,
            height: 600,
            texts: vec![t],
            onomatopoeia: vec![sfx],
            bubbles: vec![b],
            panels: vec![make_gt_item([20.0, 20.0, 780.0, 580.0], false, None)],
        });
        next_id += 1;
    }

    // 6. Nested panels with bubbles inside outer and inset panels.
    for variant in 0..2 {
        let p_outer = make_gt_item([30.0, 30.0, 770.0, 570.0], false, None);
        let p_inner = make_gt_item([400.0, 240.0, 720.0, 520.0], false, None);
        let b1 = make_gt_item([90.0, 80.0, 270.0, 260.0], true, None);
        let t1 = make_gt_item([140.0, 120.0, 220.0, 220.0], false, Some(0));
        let b2 = make_gt_item([450.0, 280.0, 660.0, 470.0], true, None);
        let t2 = make_gt_item([500.0, 320.0, 610.0, 430.0], false, Some(1));
        let margin_cap = make_gt_item(
            [730.0, 300.0 + variant as f32 * 20.0, 765.0, 450.0],
            false,
            None,
        );
        pages.push(GtPage {
            id: next_id,
            name: format!("synth_nested_panels_{variant}"),
            width: 800,
            height: 600,
            texts: vec![t1, t2, margin_cap],
            onomatopoeia: vec![],
            bubbles: vec![b1, b2],
            panels: vec![p_outer, p_inner],
        });
        next_id += 1;
    }

    pages
}

// ---------------------------------------------------------------------------
// Seeded Adversarial Pathology Generator
// ---------------------------------------------------------------------------

struct CorruptedPage {
    layout: KoharuLayoutDetections,
    comic: Vec<ComicTextBubbleDetection>,
    split_gt_bubbles: HashSet<usize>,
}

fn clamp_bbox(b: [f32; 4], w: f32, h: f32) -> [f32; 4] {
    let x1 = b[0].clamp(0.0, (w - 2.0).max(0.0));
    let y1 = b[1].clamp(0.0, (h - 2.0).max(0.0));
    let x2 = b[2].clamp(x1 + 1.0, w);
    let y2 = b[3].clamp(y1 + 1.0, h);
    [x1, y1, x2, y2]
}

fn jitter_bbox(b: [f32; 4], rng: &mut Rng, frac: f32, w: f32, h: f32) -> [f32; 4] {
    let bw = (b[2] - b[0]).max(4.0);
    let bh = (b[3] - b[1]).max(4.0);
    clamp_bbox(
        [
            b[0] + rng.range_f32(-frac, frac) * bw,
            b[1] + rng.range_f32(-frac, frac) * bh,
            b[2] + rng.range_f32(-frac, frac) * bw,
            b[3] + rng.range_f32(-frac, frac) * bh,
        ],
        w,
        h,
    )
}

fn mask_union_with_rect(base: &KoharuLayoutMask, extra: [f32; 4]) -> KoharuLayoutMask {
    let ex0 = extra[0].floor().max(0.0) as u32;
    let ey0 = extra[1].floor().max(0.0) as u32;
    let ex1 = extra[2].ceil().max(0.0) as u32;
    let ey1 = extra[3].ceil().max(0.0) as u32;
    let x0 = base.x.min(ex0);
    let y0 = base.y.min(ey0);
    let x1 = (base.x + base.width).max(ex1);
    let y1 = (base.y + base.height).max(ey1);
    let width = x1.saturating_sub(x0).max(1);
    let height = y1.saturating_sub(y0).max(1);
    let mut pixels = vec![0u8; (width * height) as usize];
    for ly in 0..base.height {
        for lx in 0..base.width {
            if base.pixels[(ly * base.width + lx) as usize] != 0 {
                let gx = base.x + lx - x0;
                let gy = base.y + ly - y0;
                if gx < width && gy < height {
                    pixels[(gy * width + gx) as usize] = u8::MAX;
                }
            }
        }
    }
    for gy in ey0.max(y0)..ey1.min(y0 + height) {
        for gx in ex0.max(x0)..ex1.min(x0 + width) {
            pixels[((gy - y0) * width + (gx - x0)) as usize] = u8::MAX;
        }
    }
    KoharuLayoutMask {
        x: x0,
        y: y0,
        width,
        height,
        pixels,
    }
}

fn erode_mask_to_bbox(mask: &KoharuLayoutMask, bbox: [f32; 4]) -> KoharuLayoutMask {
    let x0 = (bbox[0].floor().max(0.0) as u32).max(mask.x);
    let y0 = (bbox[1].floor().max(0.0) as u32).max(mask.y);
    let x1 = (bbox[2].ceil().max(0.0) as u32).min(mask.x + mask.width);
    let y1 = (bbox[3].ceil().max(0.0) as u32).min(mask.y + mask.height);
    if x1 <= x0 || y1 <= y0 {
        return rect_mask(bbox);
    }
    let width = x1 - x0;
    let height = y1 - y0;
    let mut pixels = vec![0u8; (width * height) as usize];
    for gy in y0..y1 {
        for gx in x0..x1 {
            if mask.contains(gx, gy) {
                pixels[((gy - y0) * width + (gx - x0)) as usize] = u8::MAX;
            }
        }
    }
    KoharuLayoutMask {
        x: x0,
        y: y0,
        width,
        height,
        pixels,
    }
}

fn generate_corrupted_proposals(page: &GtPage, seed: u64, stress_multiplier: f32) -> CorruptedPage {
    let mut rng = Rng::new(seed ^ page.id.wrapping_mul(0x517C_C1B7_2722_0A95));
    let w = page.width as f32;
    let h = page.height as f32;
    let mut rfdetr: Vec<KoharuLayoutDetection> = Vec::new();
    let mut comic: Vec<ComicTextBubbleDetection> = Vec::new();
    let mut split_gt_bubbles = HashSet::new();

    // 1. Panels (mostly clean with slight jitter)
    for panel in &page.panels {
        let bbox = jitter_bbox(panel.bbox, &mut rng, 0.015, w, h);
        rfdetr.push(KoharuLayoutDetection {
            label_id: RFDETR_PANEL_LABEL_ID,
            label: "panel".to_owned(),
            score: rng.range_f32(0.75, 0.98),
            bbox,
            area: panel.area,
            mask: panel.mask.clone(),
        });
    }

    // 2. Bubbles: RF-DETR proposals + ogkalu comic bubble detections
    for (b_idx, bubble) in page.bubbles.iter().enumerate() {
        let bw = bubble.bbox[2] - bubble.bbox[0];
        let bh = bubble.bbox[3] - bubble.bbox[1];
        let is_translucent_miss = rng.chance((0.25 * stress_multiplier).min(0.65));
        let is_split_lobe = bw > 60.0
            && bh > 40.0
            && page.texts.iter().any(|t| t.bubble_idx == Some(b_idx))
            && rng.chance((0.14 * stress_multiplier).min(0.45));
        let is_ogkalu_miss = !is_translucent_miss
            && !is_split_lobe
            && rng.chance((0.06 * stress_multiplier).min(0.20));

        // RF-DETR bubble proposal
        if !is_translucent_miss {
            let bbox = jitter_bbox(bubble.bbox, &mut rng, 0.04, w, h);
            let score = if is_ogkalu_miss {
                rng.range_f32(0.72, 0.94)
            } else {
                rng.range_f32(0.10, 0.95)
            };
            rfdetr.push(KoharuLayoutDetection {
                label_id: RFDETR_BUBBLE_LABEL_ID,
                label: "bubble".to_owned(),
                score,
                bbox,
                area: bubble.area,
                mask: bubble.mask.clone(),
            });
            // Occasional duplicate RF-DETR bubble query
            if rng.chance(0.10) {
                let dup_bbox = jitter_bbox(bubble.bbox, &mut rng, 0.06, w, h);
                rfdetr.push(KoharuLayoutDetection {
                    label_id: RFDETR_BUBBLE_LABEL_ID,
                    label: "bubble".to_owned(),
                    score: (score - 0.15).max(0.06),
                    bbox: dup_bbox,
                    area: bubble.area,
                    mask: bubble.mask.clone(),
                });
            }
        }

        // ogkalu bubble box(es)
        if !is_ogkalu_miss {
            if is_split_lobe {
                split_gt_bubbles.insert(b_idx);
                let split_x =
                    (bubble.bbox[0] + bubble.bbox[2]) * 0.5 + rng.range_f32(-0.08, 0.08) * bw;
                let overlap = rng.range_f32(0.02, 0.08) * bw;
                let left_lobe = clamp_bbox(
                    [
                        bubble.bbox[0],
                        bubble.bbox[1],
                        split_x + overlap,
                        bubble.bbox[3],
                    ],
                    w,
                    h,
                );
                let right_lobe = clamp_bbox(
                    [
                        split_x - overlap,
                        bubble.bbox[1],
                        bubble.bbox[2],
                        bubble.bbox[3],
                    ],
                    w,
                    h,
                );
                comic.push(ComicTextBubbleDetection {
                    label: ComicTextBubbleLabel::Bubble,
                    score: rng.range_f32(0.55, 0.92),
                    bbox: jitter_bbox(left_lobe, &mut rng, 0.02, w, h),
                });
                comic.push(ComicTextBubbleDetection {
                    label: ComicTextBubbleLabel::Bubble,
                    score: rng.range_f32(0.55, 0.92),
                    bbox: jitter_bbox(right_lobe, &mut rng, 0.02, w, h),
                });
            } else {
                let bbox = jitter_bbox(bubble.bbox, &mut rng, 0.035, w, h);
                comic.push(ComicTextBubbleDetection {
                    label: ComicTextBubbleLabel::Bubble,
                    score: rng.range_f32(0.45, 0.96),
                    bbox,
                });
            }
        }
    }

    // 3. Texts: RF-DETR text (with clipping, leaking, splitting, duplicates, label swaps, drops) + ogkalu text
    for (t_idx, text) in page.texts.iter().enumerate() {
        let tw = text.bbox[2] - text.bbox[0];
        let th = text.bbox[3] - text.bbox[1];
        let comic_label = if text.bubble_idx.is_some() {
            ComicTextBubbleLabel::TextBubble
        } else {
            ComicTextBubbleLabel::TextFree
        };

        // ogkalu text box (present ~90% of the time)
        let has_comic_text = rng.chance(0.90);
        if has_comic_text {
            let mut cbox = jitter_bbox(text.bbox, &mut rng, 0.04, w, h);
            // Occasionally widen ogkalu box slightly so it grazes a neighbor
            if rng.chance((0.10 * stress_multiplier).min(0.30)) {
                let pad_x = tw * rng.range_f32(0.05, 0.15);
                let pad_y = th * rng.range_f32(0.05, 0.15);
                cbox = clamp_bbox(
                    [
                        cbox[0] - pad_x,
                        cbox[1] - pad_y,
                        cbox[2] + pad_x,
                        cbox[3] + pad_y,
                    ],
                    w,
                    h,
                );
            }
            comic.push(ComicTextBubbleDetection {
                label: comic_label,
                score: rng.range_f32(0.42, 0.96),
                bbox: cbox,
            });
        }

        // Missing RF-DETR text (e.g. short exclamation)
        let rfdetr_missed = has_comic_text && rng.chance((0.08 * stress_multiplier).min(0.25));
        if rfdetr_missed {
            // Ensure ogkalu has high confidence when RF-DETR misses
            if let Some(last) = comic.last_mut() {
                last.score = rng.range_f32(0.58, 0.92);
            }
            continue;
        }

        // Split RF-DETR into 2 vertical columns
        let split_cols =
            tw > 24.0 && th > tw * 0.7 && rng.chance((0.18 * stress_multiplier).min(0.45));
        if split_cols {
            let mid_x = (text.bbox[0] + text.bbox[2]) * 0.5;
            let gap = rng.range_f32(0.0, (tw * 0.08).min(4.0));
            let left_b = clamp_bbox(
                [text.bbox[0], text.bbox[1], mid_x - gap * 0.5, text.bbox[3]],
                w,
                h,
            );
            let right_b = clamp_bbox(
                [mid_x + gap * 0.5, text.bbox[1], text.bbox[2], text.bbox[3]],
                w,
                h,
            );
            for sub in [left_b, right_b] {
                let m = erode_mask_to_bbox(&text.mask, sub);
                let area = m.pixels.iter().filter(|&&v| v != 0).count() as u32;
                rfdetr.push(KoharuLayoutDetection {
                    label_id: RFDETR_TEXT_LABEL_ID,
                    label: "text".to_owned(),
                    score: rng.range_f32(0.55, 0.95),
                    bbox: jitter_bbox(sub, &mut rng, 0.02, w, h),
                    area,
                    mask: Arc::new(m),
                });
            }
            continue;
        }

        // Clipped RF-DETR box (mask still has full glyph extents)
        let mut rbox = jitter_bbox(text.bbox, &mut rng, 0.03, w, h);
        if rng.chance((0.20 * stress_multiplier).min(0.50)) {
            let clip_frac = rng.range_f32(0.15, 0.30);
            if th >= tw {
                rbox[3] = (rbox[3] - th * clip_frac).max(rbox[1] + 2.0);
            } else {
                rbox[2] = (rbox[2] - tw * clip_frac).max(rbox[0] + 2.0);
            }
        }

        // Leaking RF-DETR mask into a nearby text box (different bubble or free text)
        let mut rmask = text.mask.clone();
        if rng.chance((0.16 * stress_multiplier).min(0.45)) {
            // Find nearest other text
            if let Some(neighbor) = page
                .texts
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != t_idx)
                .min_by(|(_, a), (_, b)| {
                    box_distance(text.bbox, a.bbox).total_cmp(&box_distance(text.bbox, b.bbox))
                })
                .map(|(_, n)| n)
            {
                if box_distance(text.bbox, neighbor.bbox) < 80.0 {
                    let leaked = mask_union_with_rect(&rmask, neighbor.bbox);
                    rmask = Arc::new(leaked);
                    // Also occasionally graze the neighbor box with rbox
                    if rng.chance(0.40) {
                        rbox[0] = rbox[0]
                            .min(neighbor.bbox[0] + (neighbor.bbox[2] - neighbor.bbox[0]) * 0.85);
                        rbox[2] = rbox[2]
                            .max(neighbor.bbox[0] + (neighbor.bbox[2] - neighbor.bbox[0]) * 0.15);
                    }
                }
            }
        }

        // Swapped label: RF-DETR labels handwritten dialogue as onomatopoeia
        let swapped_to_sfx = has_comic_text && rng.chance((0.08 * stress_multiplier).min(0.25));
        let (label_id, label) = if swapped_to_sfx {
            (RFDETR_ONOMATOPOEIA_LABEL_ID, "onomatopoeia".to_owned())
        } else {
            (RFDETR_TEXT_LABEL_ID, "text".to_owned())
        };
        let area = rmask.pixels.iter().filter(|&&v| v != 0).count() as u32;
        let score = rng.range_f32(0.50, 0.97);
        rfdetr.push(KoharuLayoutDetection {
            label_id,
            label: label.clone(),
            score,
            bbox: rbox,
            area,
            mask: rmask.clone(),
        });

        // Duplicated RF-DETR query
        if rng.chance((0.15 * stress_multiplier).min(0.40)) {
            let dup_box = jitter_bbox(rbox, &mut rng, 0.05, w, h);
            rfdetr.push(KoharuLayoutDetection {
                label_id,
                label,
                score: (score - 0.12).max(0.35),
                bbox: dup_box,
                area,
                mask: rmask,
            });
        }
    }

    // 4. Onomatopoeia (RF-DETR detects them; ogkalu normally does NOT box SFX)
    for sfx in &page.onomatopoeia {
        let bbox = jitter_bbox(sfx.bbox, &mut rng, 0.03, w, h);
        rfdetr.push(KoharuLayoutDetection {
            label_id: RFDETR_ONOMATOPOEIA_LABEL_ID,
            label: "onomatopoeia".to_owned(),
            score: rng.range_f32(0.50, 0.95),
            bbox,
            area: sfx.area,
            mask: sfx.mask.clone(),
        });
    }

    // 5. Ghost boxes on artwork (low-confidence noise)
    let ghosts = if rng.chance(0.5) { 2 } else { 1 };
    for _ in 0..ghosts {
        let gx = rng.range_f32(0.0, (w - 40.0).max(1.0));
        let gy = rng.range_f32(0.0, (h - 40.0).max(1.0));
        let gw = rng.range_f32(15.0, 50.0).min(w - gx);
        let gh = rng.range_f32(15.0, 50.0).min(h - gy);
        let gbox = [gx, gy, gx + gw, gy + gh];
        // Avoid accidentally landing on real GT text/bubbles
        let hits_gt = page
            .texts
            .iter()
            .any(|t| intersection_area(t.bbox, gbox) > 0.0)
            || page
                .bubbles
                .iter()
                .any(|b| intersection_area(b.bbox, gbox) > 0.0);
        if !hits_gt {
            if rng.chance(0.5) {
                comic.push(ComicTextBubbleDetection {
                    label: ComicTextBubbleLabel::TextFree,
                    score: rng.range_f32(0.31, 0.48),
                    bbox: gbox,
                });
            } else {
                let m = rect_mask(gbox);
                let area = m.width * m.height;
                rfdetr.push(KoharuLayoutDetection {
                    label_id: RFDETR_BUBBLE_LABEL_ID,
                    label: "bubble".to_owned(),
                    score: rng.range_f32(0.06, 0.30),
                    bbox: gbox,
                    area,
                    mask: Arc::new(m),
                });
            }
        }
    }

    CorruptedPage {
        layout: KoharuLayoutDetections {
            image_width: page.width,
            image_height: page.height,
            detections: rfdetr,
        },
        comic,
        split_gt_bubbles,
    }
}

// ---------------------------------------------------------------------------
// Geometry & Evaluation Metrics
// ---------------------------------------------------------------------------

fn area(b: [f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

fn intersection_area(a: [f32; 4], b: [f32; 4]) -> f32 {
    (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0)
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let inter = intersection_area(a, b);
    let u = area(a) + area(b) - inter;
    if u <= 0.0 { 0.0 } else { inter / u }
}

fn containment(container: [f32; 4], inner: [f32; 4]) -> f32 {
    let a = area(inner);
    if a <= 0.0 {
        0.0
    } else {
        intersection_area(container, inner) / a
    }
}

fn box_distance(a: [f32; 4], b: [f32; 4]) -> f32 {
    let dx = (a[0] - b[2]).max(b[0] - a[2]).max(0.0);
    let dy = (a[1] - b[3]).max(b[1] - a[3]).max(0.0);
    dx.hypot(dy)
}

#[derive(Default, Clone, Debug)]
struct Scorecard {
    // Hard invariants
    cross_bubble_merges: usize,
    total_pred_texts: usize,
    dropped_dialogue: usize,
    total_gt_texts: usize,
    bubble_text_misassignments: usize,
    total_matched_texts_for_assoc: usize,
    split_bubbles_recovered: usize,
    total_split_bubbles: usize,
    // Detection matching (IoU >= 0.5)
    text_tp: usize,
    text_fp: usize,
    text_fn: usize,
    text_iou_sum: f64,
    bubble_tp: usize,
    bubble_fp: usize,
    bubble_fn: usize,
    bubble_iou_sum: f64,
    // Association F1 (contained_by_bubble)
    assoc_tp: usize,
    assoc_fp: usize,
    assoc_fn: usize,
}

impl Scorecard {
    fn merge(&mut self, other: &Self) {
        self.cross_bubble_merges += other.cross_bubble_merges;
        self.total_pred_texts += other.total_pred_texts;
        self.dropped_dialogue += other.dropped_dialogue;
        self.total_gt_texts += other.total_gt_texts;
        self.bubble_text_misassignments += other.bubble_text_misassignments;
        self.total_matched_texts_for_assoc += other.total_matched_texts_for_assoc;
        self.split_bubbles_recovered += other.split_bubbles_recovered;
        self.total_split_bubbles += other.total_split_bubbles;
        self.text_tp += other.text_tp;
        self.text_fp += other.text_fp;
        self.text_fn += other.text_fn;
        self.text_iou_sum += other.text_iou_sum;
        self.bubble_tp += other.bubble_tp;
        self.bubble_fp += other.bubble_fp;
        self.bubble_fn += other.bubble_fn;
        self.bubble_iou_sum += other.bubble_iou_sum;
        self.assoc_tp += other.assoc_tp;
        self.assoc_fp += other.assoc_fp;
        self.assoc_fn += other.assoc_fn;
    }

    fn dropped_dialogue_rate(&self) -> f64 {
        if self.total_gt_texts == 0 {
            0.0
        } else {
            self.dropped_dialogue as f64 / self.total_gt_texts as f64
        }
    }

    fn misassignment_rate(&self) -> f64 {
        if self.total_matched_texts_for_assoc == 0 {
            0.0
        } else {
            self.bubble_text_misassignments as f64 / self.total_matched_texts_for_assoc as f64
        }
    }

    fn split_recovery_rate(&self) -> f64 {
        if self.total_split_bubbles == 0 {
            1.0
        } else {
            self.split_bubbles_recovered as f64 / self.total_split_bubbles as f64
        }
    }

    fn f1(tp: usize, fp: usize, fn_: usize) -> f64 {
        let denom = 2 * tp + fp + fn_;
        if denom == 0 {
            1.0
        } else {
            (2 * tp) as f64 / denom as f64
        }
    }

    fn text_f1(&self) -> f64 {
        Self::f1(self.text_tp, self.text_fp, self.text_fn)
    }

    fn text_mean_iou(&self) -> f64 {
        if self.text_tp == 0 {
            0.0
        } else {
            self.text_iou_sum / self.text_tp as f64
        }
    }

    fn bubble_f1(&self) -> f64 {
        Self::f1(self.bubble_tp, self.bubble_fp, self.bubble_fn)
    }

    fn bubble_mean_iou(&self) -> f64 {
        if self.bubble_tp == 0 {
            0.0
        } else {
            self.bubble_iou_sum / self.bubble_tp as f64
        }
    }

    fn assoc_f1(&self) -> f64 {
        Self::f1(self.assoc_tp, self.assoc_fp, self.assoc_fn)
    }

    /// Composite objective used to rank invariant-satisfying Pareto candidates.
    fn composite_score(&self) -> f64 {
        let merge_penalty = if self.total_gt_texts == 0 {
            0.0
        } else {
            (self.cross_bubble_merges as f64 / self.total_gt_texts as f64) * 5.0
        };
        0.30 * self.text_f1()
            + 0.25 * self.bubble_f1()
            + 0.30 * self.assoc_f1()
            + 0.15 * self.split_recovery_rate()
            - 2.0 * self.dropped_dialogue_rate()
            - 1.5 * self.misassignment_rate()
            - merge_penalty
    }

    fn pareto_dominates(&self, other: &Self) -> bool {
        // Minimize: cross_bubble_merges, dropped_dialogue_rate, misassignment_rate
        // Maximize: split_recovery_rate, text_f1, bubble_f1, assoc_f1
        let eps = 1e-6;
        let ge_all = self.cross_bubble_merges <= other.cross_bubble_merges
            && self.dropped_dialogue_rate() <= other.dropped_dialogue_rate() + eps
            && self.misassignment_rate() <= other.misassignment_rate() + eps
            && self.split_recovery_rate() + eps >= other.split_recovery_rate()
            && self.text_f1() + eps >= other.text_f1()
            && self.bubble_f1() + eps >= other.bubble_f1()
            && self.assoc_f1() + eps >= other.assoc_f1();
        let gt_any = self.cross_bubble_merges < other.cross_bubble_merges
            || self.dropped_dialogue_rate() + eps < other.dropped_dialogue_rate()
            || self.misassignment_rate() + eps < other.misassignment_rate()
            || self.split_recovery_rate() > other.split_recovery_rate() + eps
            || self.text_f1() > other.text_f1() + eps
            || self.bubble_f1() > other.bubble_f1() + eps
            || self.assoc_f1() > other.assoc_f1() + eps;
        ge_all && gt_any
    }
}

fn greedy_match(
    gt: &[[f32; 4]],
    pred: &[[f32; 4]],
    min_iou: f32,
) -> (Vec<(usize, usize, f32)>, usize, usize) {
    let mut pairs = Vec::new();
    for (gi, &gb) in gt.iter().enumerate() {
        for (pi, &pb) in pred.iter().enumerate() {
            let v = iou(gb, pb);
            if v >= min_iou {
                pairs.push((v, gi, pi));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut used_g = vec![false; gt.len()];
    let mut used_p = vec![false; pred.len()];
    let mut matches = Vec::new();
    for (v, gi, pi) in pairs {
        if !used_g[gi] && !used_p[pi] {
            used_g[gi] = true;
            used_p[pi] = true;
            matches.push((gi, pi, v));
        }
    }
    let fp = pred.len() - matches.len();
    let fn_ = gt.len() - matches.len();
    (matches, fp, fn_)
}

fn evaluate_page(
    page: &GtPage,
    corrupted: &CorruptedPage,
    config: &DetectionFusionConfig,
) -> Scorecard {
    let fused = fuse_detections(
        corrupted.layout.clone(),
        &corrupted.comic,
        DEFAULT_BUBBLE_THRESHOLD,
        config,
    );

    let mut sc = Scorecard::default();

    // Map detection index -> assigned bubble detection index from layout_order
    let mut det_to_bubble: HashMap<usize, Option<usize>> = HashMap::new();
    for item in &fused.order {
        det_to_bubble.insert(item.index, item.bubble);
    }

    let mut pred_texts: Vec<(usize, [f32; 4])> = Vec::new();
    let mut pred_bubbles: Vec<(usize, [f32; 4])> = Vec::new();
    for (idx, det) in fused.detections.iter().enumerate() {
        match det.label.as_str() {
            "text" => pred_texts.push((idx, det.bbox)),
            "bubble" => pred_bubbles.push((idx, det.bbox)),
            _ => {}
        }
    }

    sc.total_pred_texts = pred_texts.len();
    sc.total_gt_texts = page.texts.len();

    let gt_text_boxes: Vec<[f32; 4]> = page.texts.iter().map(|t| t.bbox).collect();
    let pred_text_boxes: Vec<[f32; 4]> = pred_texts.iter().map(|t| t.1).collect();
    let gt_bubble_boxes: Vec<[f32; 4]> = page.bubbles.iter().map(|b| b.bbox).collect();
    let pred_bubble_boxes: Vec<[f32; 4]> = pred_bubbles.iter().map(|b| b.1).collect();

    // 1. Text & Bubble F1/IoU (0.5 IoU)
    let (t_matches, t_fp, t_fn) = greedy_match(&gt_text_boxes, &pred_text_boxes, 0.5);
    sc.text_tp = t_matches.len();
    sc.text_fp = t_fp;
    sc.text_fn = t_fn;
    sc.text_iou_sum = t_matches.iter().map(|m| m.2 as f64).sum();

    let (b_matches, b_fp, b_fn) = greedy_match(&gt_bubble_boxes, &pred_bubble_boxes, 0.5);
    sc.bubble_tp = b_matches.len();
    sc.bubble_fp = b_fp;
    sc.bubble_fn = b_fn;
    sc.bubble_iou_sum = b_matches.iter().map(|m| m.2 as f64).sum();

    // Map pred bubble detection index -> matched gt bubble idx (using IoU >= 0.3 for association credit)
    let (b_assoc_matches, _, _) = greedy_match(&gt_bubble_boxes, &pred_bubble_boxes, 0.3);
    let mut pred_det_bubble_to_gt: HashMap<usize, usize> = HashMap::new();
    for (gb_i, pb_i, _) in b_assoc_matches {
        pred_det_bubble_to_gt.insert(pred_bubbles[pb_i].0, gb_i);
    }

    // 2. Cross-bubble text merge count
    // A predicted text box is a cross-bubble merge if it covers (containment >= 0.45)
    // GT texts belonging to >= 2 distinct GT bubble memberships (including Bubble A vs Bubble B,
    // or Bubble A vs free text).
    for &(_, pbox) in &pred_texts {
        let mut distinct_owners: HashSet<Option<usize>> = HashSet::new();
        let mut bubble_owners: HashSet<usize> = HashSet::new();
        for gt_t in &page.texts {
            if containment(pbox, gt_t.bbox) >= 0.45 {
                distinct_owners.insert(gt_t.bubble_idx);
                if let Some(b) = gt_t.bubble_idx {
                    bubble_owners.insert(b);
                }
            }
        }
        if bubble_owners.len() >= 2
            || (!bubble_owners.is_empty() && distinct_owners.contains(&None))
        {
            sc.cross_bubble_merges += 1;
        }
    }

    // 3. Dropped dialogue rate
    for gt_t in &page.texts {
        let covered = pred_texts
            .iter()
            .any(|&(_, pbox)| containment(pbox, gt_t.bbox) >= 0.5 || iou(pbox, gt_t.bbox) >= 0.3);
        if !covered {
            sc.dropped_dialogue += 1;
        }
    }

    // 4. Bubble-text misassignment rate & contained_by_bubble association F1
    let mut gt_assoc_pairs: HashSet<(usize, usize)> = HashSet::new();
    for (ti, gt_t) in page.texts.iter().enumerate() {
        if let Some(bi) = gt_t.bubble_idx {
            gt_assoc_pairs.insert((ti, bi));
        }
    }
    let mut pred_assoc_pairs: HashSet<(usize, usize)> = HashSet::new();

    // Match each GT text to best covering pred text (containment >= 0.4 or IoU >= 0.3)
    for (ti, gt_t) in page.texts.iter().enumerate() {
        let best_pred = pred_texts
            .iter()
            .enumerate()
            .map(|(pi, &(_, pbox))| (pi, iou(gt_t.bbox, pbox), containment(pbox, gt_t.bbox)))
            .filter(|&(_, v_iou, v_cont)| v_iou >= 0.3 || v_cont >= 0.5)
            .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.total_cmp(&b.2)));

        if let Some((pi, _, _)) = best_pred {
            sc.total_matched_texts_for_assoc += 1;
            let pred_det_idx = pred_texts[pi].0;
            let pred_bubble_gt = det_to_bubble
                .get(&pred_det_idx)
                .copied()
                .flatten()
                .and_then(|pb_det_idx| pred_det_bubble_to_gt.get(&pb_det_idx).copied());

            if pred_bubble_gt != gt_t.bubble_idx {
                sc.bubble_text_misassignments += 1;
            }
            if let Some(bi) = pred_bubble_gt {
                pred_assoc_pairs.insert((ti, bi));
            }
        }
    }
    // Also count false-positive associations on unmatched pred texts that got assigned to a bubble
    let matched_pred_indices: HashSet<usize> = page
        .texts
        .iter()
        .filter_map(|gt_t| {
            pred_texts
                .iter()
                .enumerate()
                .map(|(pi, &(_, pbox))| (pi, iou(gt_t.bbox, pbox), containment(pbox, gt_t.bbox)))
                .filter(|&(_, v_iou, v_cont)| v_iou >= 0.3 || v_cont >= 0.5)
                .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.total_cmp(&b.2)))
                .map(|(pi, _, _)| pi)
        })
        .collect();

    let mut extra_assoc_fp = 0usize;
    for (pi, &(pred_det_idx, _)) in pred_texts.iter().enumerate() {
        if !matched_pred_indices.contains(&pi)
            && det_to_bubble
                .get(&pred_det_idx)
                .copied()
                .flatten()
                .is_some()
        {
            extra_assoc_fp += 1;
        }
    }

    sc.assoc_tp = gt_assoc_pairs.intersection(&pred_assoc_pairs).count();
    sc.assoc_fp = pred_assoc_pairs.difference(&gt_assoc_pairs).count() + extra_assoc_fp;
    sc.assoc_fn = gt_assoc_pairs.difference(&pred_assoc_pairs).count();

    // 5. Split-bubble recovery rate
    sc.total_split_bubbles = corrupted.split_gt_bubbles.len();
    for &gb_idx in &corrupted.split_gt_bubbles {
        let gb_box = page.bubbles[gb_idx].bbox;
        let recovered = pred_bubbles
            .iter()
            .any(|&(_, pbox)| iou(gb_box, pbox) >= 0.70 || containment(pbox, gb_box) >= 0.80);
        if recovered {
            sc.split_bubbles_recovered += 1;
        }
    }

    sc
}

fn evaluate_suite(
    pages: &[GtPage],
    corrupted: &[CorruptedPage],
    config: &DetectionFusionConfig,
) -> Scorecard {
    pages
        .par_iter()
        .zip(corrupted.par_iter())
        .map(|(p, c)| evaluate_page(p, c, config))
        .reduce(Scorecard::default, |mut acc, item| {
            acc.merge(&item);
            acc
        })
}

fn print_header() {
    println!(
        "{:<36} | {:>6} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7}",
        "Candidate",
        "XMerge",
        "Drop%",
        "MisAs%",
        "Split%",
        "TxtF1",
        "TxtIoU",
        "BubF1",
        "BubIoU",
        "AscF1",
        "Score"
    );
    println!("{}", "-".repeat(130));
}

fn print_row(name: &str, sc: &Scorecard) {
    println!(
        "{:<36} | {:>6} | {:>6.2}% | {:>6.2}% | {:>6.2}% | {:>7.4} | {:>7.4} | {:>7.4} | {:>7.4} | {:>7.4} | {:>7.4}",
        name,
        sc.cross_bubble_merges,
        sc.dropped_dialogue_rate() * 100.0,
        sc.misassignment_rate() * 100.0,
        sc.split_recovery_rate() * 100.0,
        sc.text_f1(),
        sc.text_mean_iou(),
        sc.bubble_f1(),
        sc.bubble_mean_iou(),
        sc.assoc_f1(),
        sc.composite_score(),
    );
}

fn candidate_configs() -> Vec<(String, DetectionFusionConfig)> {
    let base = DetectionFusionConfig::legacy();
    let prod = DetectionFusionConfig::default();
    let mut prod_fb = prod.clone();
    prod_fb.fallback_rfdetr_bubble_threshold = Some(0.75);
    let mut list = vec![
        ("baseline (origin/master)".to_owned(), base.clone()),
        ("default (new production)".to_owned(), prod),
        ("default + fb0.75 (candidate)".to_owned(), prod_fb),
    ];

    // Individual strategy ablations
    let mut c = base.clone();
    c.fit_to_mask_min_overlap = 0.25;
    list.push(("ablation: fit_overlap=0.25".to_owned(), c));

    let mut c = base.clone();
    c.fit_to_mask_min_overlap = 0.40;
    list.push(("ablation: fit_overlap=0.40".to_owned(), c));

    let mut c = base.clone();
    c.use_mask_containment_for_merge = true;
    list.push(("ablation: mask_merge".to_owned(), c));

    let mut c = base.clone();
    c.prevent_cross_bubble_same_text = true;
    list.push(("ablation: no_xbubble_same_text".to_owned(), c));

    let mut c = base.clone();
    c.dedup_added_text_coverage = true;
    list.push(("ablation: dedup_added_cov".to_owned(), c));

    let mut c = base.clone();
    c.prefer_higher_bubble_containment = true;
    list.push(("ablation: prefer_high_bub_cont".to_owned(), c));

    for thresh in [0.65f32, 0.75, 0.85] {
        let mut c = base.clone();
        c.fallback_rfdetr_bubble_threshold = Some(thresh);
        list.push((format!("ablation: fallback_bub={thresh:.2}"), c));
    }

    // Threshold sweeps on dialogue/merge/ellipse containment
    for d_thresh in [0.70f32, 0.75, 0.80, 0.85] {
        let mut c = base.clone();
        c.dialogue_mask_containment_threshold = d_thresh;
        c.ellipse_text_containment_threshold = d_thresh;
        list.push((format!("ablation: dlg+ell_cont={d_thresh:.2}"), c));
    }

    for m_thresh in [0.50f32, 0.60, 0.70] {
        let mut c = base.clone();
        c.use_mask_containment_for_merge = true;
        c.prevent_cross_bubble_same_text = true;
        c.merge_container_containment_threshold = m_thresh;
        list.push((format!("ablation: mask_merge@{m_thresh:.2}"), c));
    }

    for split_share in [0.10f32, 0.15, 0.25] {
        let mut c = base.clone();
        c.split_bubble_min_share = split_share;
        list.push((format!("ablation: split_share={split_share:.2}"), c));
    }

    for gap in [0.40f32, 0.50, 0.75] {
        let mut c = base.clone();
        c.text_block_gap = gap;
        list.push((format!("ablation: text_gap={gap:.2}"), c));
    }

    // Combined strategy candidates (grid over top orthogonal improvements, keeping dlg_t=0.90)
    for fit_ov in [0.30f32, 0.40] {
        for gap in [0.40f32, 0.45, 0.60] {
            for m_t in [0.50f32, 0.60] {
                for split_s in [0.15f32, 0.20] {
                    for fb in [None, Some(0.75f32)] {
                        let c = DetectionFusionConfig {
                            dialogue_mask_containment_threshold: 0.90,
                            ellipse_text_containment_threshold: 0.90,
                            merge_container_containment_threshold: m_t,
                            text_block_gap: gap,
                            split_bubble_min_share: split_s,
                            fit_to_mask_min_overlap: fit_ov,
                            use_mask_containment_for_merge: true,
                            prevent_cross_bubble_same_text: true,
                            dedup_added_text_coverage: true,
                            prefer_higher_bubble_containment: true,
                            fallback_rfdetr_bubble_threshold: fb,
                            ..base.clone()
                        };
                        let fb_tag = fb.map_or("noFB".to_owned(), |v| format!("fb{v:.2}"));
                        let name = format!(
                            "combo f{fit_ov:.2} g{gap:.2} m{m_t:.2} s{split_s:.2} {fb_tag}"
                        );
                        list.push((name, c));
                    }
                }
            }
        }
    }

    list
}

// ---------------------------------------------------------------------------
// Real RTX 3080 Raw Detector Dump Loading (.jsonl / .jsonl.gz)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawRleMask {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    counts: Vec<u32>,
}

#[derive(Deserialize)]
struct RawRfdetrDet {
    label_id: usize,
    label: String,
    score: f32,
    bbox: [f32; 4],
    area: u32,
    mask: RawRleMask,
}

#[derive(Deserialize)]
struct RawOgkaluDet {
    label: String,
    score: f32,
    bbox: [f32; 4],
}

#[derive(Deserialize)]
struct RawDumpLine {
    image_id: u64,
    file_name: String,
    image_width: u32,
    image_height: u32,
    rfdetr: Vec<RawRfdetrDet>,
    ogkalu: Vec<RawOgkaluDet>,
}

struct LoadedRawPage {
    image_id: u64,
    file_name: String,
    image_width: u32,
    image_height: u32,
    corrupted: CorruptedPage,
}

fn decode_raw_rle(mask: &RawRleMask) -> KoharuLayoutMask {
    let total = (mask.width as usize).saturating_mul(mask.height as usize);
    let mut pixels = vec![0u8; total];
    let mut pos = 0usize;
    for (run_idx, &run) in mask.counts.iter().enumerate() {
        let end = (pos + run as usize).min(total);
        if run_idx % 2 == 1 {
            pixels[pos..end].fill(u8::MAX);
        }
        pos = end;
        if pos >= total {
            break;
        }
    }
    KoharuLayoutMask {
        x: mask.x,
        y: mask.y,
        width: mask.width,
        height: mask.height,
        pixels,
    }
}

fn parse_ogkalu_label(label: &str) -> Option<ComicTextBubbleLabel> {
    match label {
        "bubble" => Some(ComicTextBubbleLabel::Bubble),
        "text_bubble" => Some(ComicTextBubbleLabel::TextBubble),
        "text_free" => Some(ComicTextBubbleLabel::TextFree),
        _ => None,
    }
}

fn load_raw_dumps(path: &Path, max_pages: Option<usize>) -> Result<Vec<LoadedRawPage>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let is_gz = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gz"));

    let mut raw_lines: Vec<String> = Vec::new();
    if is_gz {
        let reader = BufReader::new(GzDecoder::new(file));
        for line in reader.lines() {
            let line = line.context("failed to read gzipped raw dump line")?;
            if !line.trim().is_empty() {
                raw_lines.push(line);
                if max_pages.is_some_and(|limit| raw_lines.len() >= limit) {
                    break;
                }
            }
        }
    } else {
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.context("failed to read raw dump line")?;
            if !line.trim().is_empty() {
                raw_lines.push(line);
                if max_pages.is_some_and(|limit| raw_lines.len() >= limit) {
                    break;
                }
            }
        }
    }

    raw_lines
        .par_iter()
        .map(|line| {
            let parsed: RawDumpLine =
                serde_json::from_str(line).context("failed to parse raw dump JSONL line")?;
            let detections = parsed
                .rfdetr
                .into_iter()
                .map(|d| KoharuLayoutDetection {
                    label_id: d.label_id,
                    label: d.label,
                    score: d.score,
                    bbox: d.bbox,
                    area: d.area,
                    mask: Arc::new(decode_raw_rle(&d.mask)),
                })
                .collect();
            let comic = parsed
                .ogkalu
                .into_iter()
                .filter_map(|d| {
                    parse_ogkalu_label(&d.label).map(|label| ComicTextBubbleDetection {
                        label,
                        score: d.score,
                        bbox: d.bbox,
                    })
                })
                .collect();
            Ok(LoadedRawPage {
                image_id: parsed.image_id,
                file_name: parsed.file_name,
                image_width: parsed.image_width,
                image_height: parsed.image_height,
                corrupted: CorruptedPage {
                    layout: KoharuLayoutDetections {
                        image_width: parsed.image_width,
                        image_height: parsed.image_height,
                        detections,
                    },
                    comic,
                    split_gt_bubbles: HashSet::new(),
                },
            })
        })
        .collect()
}

#[derive(Deserialize)]
struct HardGtEntry {
    gt: Vec<Vec<usize>>,
    ignore: Vec<usize>,
}

#[derive(Default, Clone, Debug)]
struct RawSummaryStats {
    bubbles: usize,
    texts: usize,
    assoc_texts: usize,
    free_texts: usize,
    sfx: usize,
    panels: usize,
    hi_res_pages: usize,
    hi_res_bubbles: usize,
    hi_res_texts: usize,
    hi_res_free_texts: usize,
    hard_gt_tp: usize,
    hard_gt_total: usize,
    hard_gt_fp: usize,
}

impl RawSummaryStats {
    fn hard_gt_f1(&self) -> f64 {
        let fn_ = self.hard_gt_total.saturating_sub(self.hard_gt_tp);
        Scorecard::f1(self.hard_gt_tp, self.hard_gt_fp, fn_)
    }
}

fn evaluate_raw_summary(
    raw_pages: &[LoadedRawPage],
    config: &DetectionFusionConfig,
    hard_gt: Option<&(HashMap<String, HardGtEntry>, HashMap<String, Vec<[f32; 4]>>)>,
) -> RawSummaryStats {
    let mut stats = RawSummaryStats::default();

    for page in raw_pages {
        let fused = fuse_detections(
            page.corrupted.layout.clone(),
            &page.corrupted.comic,
            DEFAULT_BUBBLE_THRESHOLD,
            config,
        );
        let mut det_to_bubble: HashMap<usize, Option<usize>> = HashMap::new();
        for item in &fused.order {
            det_to_bubble.insert(item.index, item.bubble);
        }

        let mut page_bubbles: Vec<[f32; 4]> = Vec::new();
        let mut page_texts = 0usize;
        let mut page_assoc = 0usize;
        let mut page_free = 0usize;

        for (idx, det) in fused.detections.iter().enumerate() {
            match det.label.as_str() {
                "bubble" => {
                    stats.bubbles += 1;
                    page_bubbles.push(det.bbox);
                }
                "text" => {
                    stats.texts += 1;
                    page_texts += 1;
                    if det_to_bubble.get(&idx).copied().flatten().is_some() {
                        stats.assoc_texts += 1;
                        page_assoc += 1;
                    } else {
                        stats.free_texts += 1;
                        page_free += 1;
                    }
                }
                "onomatopoeia" => stats.sfx += 1,
                "panel" => stats.panels += 1,
                _ => {}
            }
        }
        let _ = page_assoc;

        if page.image_width >= 4000 || page.image_height >= 4000 {
            stats.hi_res_pages += 1;
            stats.hi_res_bubbles += page_bubbles.len();
            stats.hi_res_texts += page_texts;
            stats.hi_res_free_texts += page_free;
        }

        if let Some((gt_map, cand_map)) = hard_gt {
            if let (Some(entry), Some(cands)) =
                (gt_map.get(&page.file_name), cand_map.get(&page.file_name))
            {
                let groups: Vec<Vec<[f32; 4]>> = entry
                    .gt
                    .iter()
                    .map(|grp| {
                        grp.iter()
                            .filter_map(|&idx| cands.get(idx).copied())
                            .collect()
                    })
                    .collect();
                let ign: Vec<[f32; 4]> = entry
                    .ignore
                    .iter()
                    .filter_map(|&idx| cands.get(idx).copied())
                    .collect();
                let mut hit = vec![false; groups.len()];
                for &b in &page_bubbles {
                    let matched: Vec<usize> = groups
                        .iter()
                        .enumerate()
                        .filter_map(|(k, grp)| grp.iter().any(|&c| iou(b, c) >= 0.5).then_some(k))
                        .collect();
                    if !matched.is_empty() {
                        for k in matched {
                            hit[k] = true;
                        }
                    } else if !ign.iter().any(|&c| iou(b, c) >= 0.5) {
                        stats.hard_gt_fp += 1;
                    }
                }
                stats.hard_gt_total += groups.len();
                stats.hard_gt_tp += hit.into_iter().filter(|&h| h).count();
            }
        }
    }

    stats
}

fn real_dump_candidates() -> Vec<(String, DetectionFusionConfig)> {
    let base = DetectionFusionConfig::legacy();
    let prod = DetectionFusionConfig::default();
    let mut list = vec![
        ("baseline (origin/master)".to_owned(), base.clone()),
        ("default (5 pure logic fixes)".to_owned(), prod.clone()),
    ];

    for fb in [0.65f32, 0.75, 0.85] {
        let mut c = prod.clone();
        c.fallback_rfdetr_bubble_threshold = Some(fb);
        list.push((format!("default + fb={fb:.2}"), c));
    }

    for gap in [0.40f32, 0.45, 0.60] {
        let mut c = prod.clone();
        c.text_block_gap = gap;
        list.push((format!("default + gap={gap:.2}"), c));
    }

    for gap in [0.40f32, 0.45, 0.60] {
        for fb in [0.65f32, 0.75, 0.85] {
            let mut c = prod.clone();
            c.text_block_gap = gap;
            c.fallback_rfdetr_bubble_threshold = Some(fb);
            list.push((format!("default + gap={gap:.2} + fb={fb:.2}"), c));
        }
    }

    // Individual ablations of the 5 pure logic fixes on real data
    let mut c = base.clone();
    c.fit_to_mask_min_overlap = prod.fit_to_mask_min_overlap;
    list.push((
        format!("ablation: fit_overlap={:.2}", prod.fit_to_mask_min_overlap),
        c,
    ));

    let mut c = base.clone();
    c.use_mask_containment_for_merge = true;
    list.push(("ablation: mask_merge".to_owned(), c));

    let mut c = base.clone();
    c.prevent_cross_bubble_same_text = true;
    list.push(("ablation: no_xbubble_same_text".to_owned(), c));

    let mut c = base.clone();
    c.dedup_added_text_coverage = true;
    list.push(("ablation: dedup_added_cov".to_owned(), c));

    let mut c = base.clone();
    c.prefer_higher_bubble_containment = true;
    list.push(("ablation: prefer_high_bub_cont".to_owned(), c));

    list
}

fn run_raw_dumps_mode(
    raw_path: &Path,
    coco_path: Option<&Path>,
    hard_gt_path: Option<&Path>,
    hard_cand_path: Option<&Path>,
    max_pages: Option<usize>,
) -> Result<()> {
    let t0 = Instant::now();
    println!("Loading raw detector dumps from {} ...", raw_path.display());
    let raw_pages = load_raw_dumps(raw_path, max_pages)?;
    println!(
        "Loaded {} raw detector pages in {:.2}s",
        raw_pages.len(),
        t0.elapsed().as_secs_f64()
    );

    let candidates = real_dump_candidates();

    if let Some(coco_path) = coco_path {
        let t_coco = Instant::now();
        println!("Loading COCO ground truth from {} ...", coco_path.display());
        let gt_pages = load_coco_split(coco_path, None)?;
        let gt_by_id: HashMap<u64, GtPage> = gt_pages.into_iter().map(|p| (p.id, p)).collect();

        let mut matched_gt: Vec<GtPage> = Vec::new();
        let mut matched_corrupted: Vec<CorruptedPage> = Vec::new();
        for raw in raw_pages {
            if let Some(gt) = gt_by_id.get(&raw.image_id) {
                matched_gt.push(gt.clone());
                matched_corrupted.push(raw.corrupted);
            }
        }
        println!(
            "Matched {}/{} raw dump pages against COCO GT in {:.2}s",
            matched_gt.len(),
            gt_by_id.len(),
            t_coco.elapsed().as_secs_f64()
        );
        anyhow::ensure!(
            !matched_gt.is_empty(),
            "no raw dump image_ids matched COCO GT {}",
            coco_path.display()
        );

        let t_sweep = Instant::now();
        println!(
            "\n=== REAL DETECTOR DUMP SWEEP ({} pages, {} configs) ===",
            matched_gt.len(),
            candidates.len()
        );
        print_header();
        let mut results: Vec<(String, DetectionFusionConfig, Scorecard)> =
            Vec::with_capacity(candidates.len());
        for (name, cfg) in candidates {
            let sc = evaluate_suite(&matched_gt, &matched_corrupted, &cfg);
            print_row(&name, &sc);
            results.push((name, cfg, sc));
        }
        println!(
            "\nSweep completed in {:.2}s",
            t_sweep.elapsed().as_secs_f64()
        );

        let baseline_sc = &results[0].2;
        let default_sc = &results[1].2;
        println!("\n--- Dominance Check vs baseline & default ---");
        for (name, _, sc) in &results[1..] {
            let dom_base = sc.pareto_dominates(baseline_sc);
            let dom_def = sc.pareto_dominates(default_sc);
            if dom_base || dom_def {
                println!(
                    "  {:<36} dom_base={:<5} dom_def={:<5} XMerge={:+} Drop={:+.2}% MisAs={:+.2}% TxtF1={:+.4} BubF1={:+.4} AscF1={:+.4}",
                    name,
                    dom_base,
                    dom_def,
                    sc.cross_bubble_merges as isize - baseline_sc.cross_bubble_merges as isize,
                    (sc.dropped_dialogue_rate() - baseline_sc.dropped_dialogue_rate()) * 100.0,
                    (sc.misassignment_rate() - baseline_sc.misassignment_rate()) * 100.0,
                    sc.text_f1() - baseline_sc.text_f1(),
                    sc.bubble_f1() - baseline_sc.bubble_f1(),
                    sc.assoc_f1() - baseline_sc.assoc_f1(),
                );
            }
        }
    } else {
        let hard_gt = match (hard_gt_path, hard_cand_path) {
            (Some(gt_p), Some(cand_p)) => {
                let gt_map: HashMap<String, HardGtEntry> =
                    serde_json::from_reader(BufReader::new(File::open(gt_p)?))
                        .with_context(|| format!("failed to parse {}", gt_p.display()))?;
                let cand_map: HashMap<String, Vec<[f32; 4]>> =
                    serde_json::from_reader(BufReader::new(File::open(cand_p)?))
                        .with_context(|| format!("failed to parse {}", cand_p.display()))?;
                Some((gt_map, cand_map))
            }
            _ => None,
        };

        println!(
            "\n=== REAL RAW DUMP SUMMARY ({} pages, {} configs) ===",
            raw_pages.len(),
            candidates.len()
        );
        println!(
            "{:<36} | {:>5} | {:>5} | {:>5} | {:>5} | {:>4} | {:>5} | {:>12} | {:>14}",
            "Candidate",
            "Bub",
            "Txt",
            "Assoc",
            "Free",
            "SFX",
            "Panel",
            "HiRes(B/T/F)",
            "HardGT(TP/FP/F1)"
        );
        println!("{}", "-".repeat(120));
        for (name, cfg) in candidates {
            let st = evaluate_raw_summary(&raw_pages, &cfg, hard_gt.as_ref());
            let hires = format!(
                "{}/{}/{}",
                st.hi_res_bubbles, st.hi_res_texts, st.hi_res_free_texts
            );
            let hgt = if st.hard_gt_total > 0 {
                format!(
                    "{}/{} fp{} {:.3}",
                    st.hard_gt_tp,
                    st.hard_gt_total,
                    st.hard_gt_fp,
                    st.hard_gt_f1()
                )
            } else {
                "-".to_owned()
            };
            println!(
                "{:<36} | {:>5} | {:>5} | {:>5} | {:>5} | {:>4} | {:>5} | {:>12} | {:>14}",
                name,
                st.bubbles,
                st.texts,
                st.assoc_texts,
                st.free_texts,
                st.sfx,
                st.panels,
                hires,
                hgt
            );
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let default_dir = env::var("HOME")
        .map(|h| PathBuf::from(h).join(".local/share/koharu-fuzz-eval/annotations"))
        .unwrap_or_else(|_| PathBuf::from("."));

    let mut val_path = default_dir.join("validation.coco.json");
    let mut test_path = default_dir.join("test.coco.json");
    let mut raw_dumps_path: Option<PathBuf> = None;
    let mut coco_path: Option<PathBuf> = None;
    let mut hard_gt_path: Option<PathBuf> = None;
    let mut hard_cand_path: Option<PathBuf> = None;
    let mut summary_only = false;
    let mut seed: u64 = 20261127;
    let mut max_pages: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--val" if i + 1 < args.len() => {
                val_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--test" if i + 1 < args.len() => {
                test_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--raw-dumps" if i + 1 < args.len() => {
                raw_dumps_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--coco" if i + 1 < args.len() => {
                coco_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--hard-gt" if i + 1 < args.len() => {
                hard_gt_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--hard-cand" if i + 1 < args.len() => {
                hard_cand_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--summary-only" => {
                summary_only = true;
                i += 1;
            }
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().context("invalid --seed")?;
                i += 2;
            }
            "--max-pages" if i + 1 < args.len() => {
                max_pages = Some(args[i + 1].parse().context("invalid --max-pages")?);
                i += 2;
            }
            _ => i += 1,
        }
    }

    if let Some(raw_path) = raw_dumps_path {
        let gt_coco = if summary_only {
            None
        } else {
            coco_path.as_deref().or(Some(val_path.as_path()))
        };
        return run_raw_dumps_mode(
            &raw_path,
            gt_coco,
            hard_gt_path.as_deref(),
            hard_cand_path.as_deref(),
            max_pages,
        );
    }

    let t0 = Instant::now();
    println!("Loading validation split from {} ...", val_path.display());
    let val_real = load_coco_split(&val_path, max_pages)?;
    let synth_pages = build_synthetic_adversarial_pages();
    println!(
        "Loaded {} validation spreads + {} synthetic adversarial layouts in {:.2}s",
        val_real.len(),
        synth_pages.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut val_pages = val_real;
    val_pages.extend(synth_pages.iter().cloned());

    let t_gen = Instant::now();
    let val_corrupted: Vec<CorruptedPage> = val_pages
        .par_iter()
        .map(|p| {
            let mult = if p.id >= 9_000_000 { 2.0 } else { 1.0 };
            generate_corrupted_proposals(p, seed, mult)
        })
        .collect();
    println!(
        "Generated seeded adversarial proposals for {} validation+synth pages in {:.2}s",
        val_corrupted.len(),
        t_gen.elapsed().as_secs_f64()
    );

    let candidates = candidate_configs();
    println!(
        "\n=== VALIDATION + SYNTHETIC ADVERSARIAL SWEEP ({} configs) ===",
        candidates.len()
    );
    let t_sweep = Instant::now();
    let mut results: Vec<(String, DetectionFusionConfig, Scorecard)> =
        Vec::with_capacity(candidates.len());
    for (name, cfg) in candidates {
        let sc = evaluate_suite(&val_pages, &val_corrupted, &cfg);
        results.push((name, cfg, sc));
    }
    println!(
        "Sweep completed in {:.2}s\n",
        t_sweep.elapsed().as_secs_f64()
    );

    let baseline_sc = results[0].2.clone();
    print_header();
    print_row(&results[0].0, &baseline_sc);
    print_row(&results[1].0, &results[1].2);
    print_row(&results[2].0, &results[2].2);

    // Show all single-factor ablations
    println!("\n--- Single-Factor Ablations ---");
    for (name, _, sc) in results
        .iter()
        .skip(1)
        .filter(|(n, _, _)| n.starts_with("ablation:"))
    {
        print_row(name, sc);
    }

    // Identify Pareto-dominating candidates over baseline (invariant-gated)
    let mut pareto_dominating: Vec<usize> = (1..results.len())
        .filter(|&idx| results[idx].2.pareto_dominates(&baseline_sc))
        .collect();
    pareto_dominating.sort_by(|&a, &b| {
        results[b]
            .2
            .composite_score()
            .total_cmp(&results[a].2.composite_score())
    });

    // Also compute non-dominated Pareto frontier among all candidates
    let mut frontier: Vec<usize> = (0..results.len())
        .filter(|&i| {
            !(0..results.len()).any(|j| j != i && results[j].2.pareto_dominates(&results[i].2))
        })
        .collect();
    frontier.sort_by(|&a, &b| {
        results[b]
            .2
            .composite_score()
            .total_cmp(&results[a].2.composite_score())
    });

    println!(
        "\n--- Top Candidates Strictly Pareto-Dominating Baseline ({} found) ---",
        pareto_dominating.len()
    );
    print_header();
    for &idx in pareto_dominating.iter().take(12) {
        print_row(&results[idx].0, &results[idx].2);
    }

    println!(
        "\n--- Non-Dominated Pareto Frontier ({} configs) ---",
        frontier.len()
    );
    print_header();
    for &idx in frontier.iter().take(10) {
        print_row(&results[idx].0, &results[idx].2);
    }

    // Synthetic-only breakdown for baseline vs top candidate
    let best_idx = pareto_dominating
        .first()
        .copied()
        .or_else(|| frontier.first().copied())
        .unwrap_or(0);

    let synth_corrupted: Vec<CorruptedPage> = synth_pages
        .par_iter()
        .map(|p| generate_corrupted_proposals(p, seed, 2.0))
        .collect();
    println!(
        "\n=== SYNTHETIC ADVERSARIAL STRESS SUITE ONLY ({} layouts) ===",
        synth_pages.len()
    );
    print_header();
    print_row(
        "baseline (synth)",
        &evaluate_suite(&synth_pages, &synth_corrupted, &results[0].1),
    );
    if best_idx != 0 {
        print_row(
            &format!("best ({})", results[best_idx].0),
            &evaluate_suite(&synth_pages, &synth_corrupted, &results[best_idx].1),
        );
    }

    // Strict unseen holdout evaluation on test.coco.json
    if test_path.exists() {
        let t_test = Instant::now();
        println!(
            "\nLoading UNSEEN test split from {} ...",
            test_path.display()
        );
        let test_pages = load_coco_split(&test_path, max_pages)?;
        let test_corrupted: Vec<CorruptedPage> = test_pages
            .par_iter()
            .map(|p| generate_corrupted_proposals(p, seed ^ 0xDEAD_BEEF, 1.0))
            .collect();
        println!(
            "Evaluated {} UNSEEN test spreads (loaded+corrupted in {:.2}s):",
            test_pages.len(),
            t_test.elapsed().as_secs_f64()
        );
        print_header();
        let mut eval_order: Vec<usize> = vec![0, 1, 2];
        for &idx in pareto_dominating
            .iter()
            .filter(|&&i| results[i].1.fallback_rfdetr_bubble_threshold.is_none())
            .take(4)
        {
            if !eval_order.contains(&idx) {
                eval_order.push(idx);
            }
        }
        for &idx in pareto_dominating
            .iter()
            .filter(|&&i| results[i].1.fallback_rfdetr_bubble_threshold.is_some())
            .take(3)
        {
            if !eval_order.contains(&idx) {
                eval_order.push(idx);
            }
        }
        if eval_order.len() == 1 {
            for &idx in frontier.iter().take(5) {
                if !eval_order.contains(&idx) {
                    eval_order.push(idx);
                }
            }
        }
        for idx in eval_order {
            let sc = evaluate_suite(&test_pages, &test_corrupted, &results[idx].1);
            print_row(&results[idx].0, &sc);
        }
    }

    Ok(())
}
