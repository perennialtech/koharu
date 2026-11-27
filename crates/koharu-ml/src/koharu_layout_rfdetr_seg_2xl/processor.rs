//! RF-DETR image preprocessing and instance segmentation postprocessing.
//!
//! https://github.com/roboflow/rf-detr/blob/4ab7c18729de9d02ffd0495795d0831b5630f01b/src/rfdetr/detr.py
//! https://github.com/roboflow/rf-detr/blob/4ab7c18729de9d02ffd0495795d0831b5630f01b/src/rfdetr/models/postprocess.py
//!
//! The HighRes checkpoint uses LongestMaxSize and top-left white padding
//! (koharu-rs/koharu 15914d76). Its normalized boxes and masks both cover
//! a square whose side is the page's longer dimension.

use std::{borrow::Cow, sync::Arc};

use anyhow::{Result, ensure};
use image::DynamicImage;
use rayon::prelude::*;
use serde::Serialize;

use super::{
    config::{KoharuLayoutRFDetrSeg2XLConfig, KoharuLayoutThresholds},
    model::{MASK_RESOLUTION, Model, NUM_CLASSES_WITH_BACKGROUND, NUM_QUERIES, Output, RESOLUTION},
};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

// Class slots in the checkpoint's classification head.
const TEXT: usize = 0;
const BUBBLE: usize = 2;

impl super::KoharuLayoutRFDetrSeg2XL {
    /// Retains the bubble mask proposal alongside a query's semantic label for
    /// fusion with a separate balloon detector.
    #[must_use]
    pub fn with_bubble_proposals(mut self) -> Self {
        self.processor.keep_bubble_proposals = true;
        self
    }
}

#[derive(Debug, Clone)]
pub struct KoharuLayoutRFDetrImageProcessor {
    num_select: usize,
    keep_bubble_proposals: bool,
    class_names: Vec<String>,
    recommended_thresholds: KoharuLayoutThresholds,
}

impl KoharuLayoutRFDetrImageProcessor {
    pub fn new(config: &KoharuLayoutRFDetrSeg2XLConfig) -> Result<Self> {
        ensure!(
            config.resolution == RESOLUTION as i64,
            "the RF-DETR export takes {RESOLUTION} px input, not {}",
            config.resolution
        );
        ensure!(
            (1..=(NUM_QUERIES * NUM_CLASSES_WITH_BACKGROUND) as i64).contains(&config.num_select),
            "RF-DETR num_select must be between 1 and the number of query/class pairs"
        );
        Ok(Self {
            num_select: config.num_select as usize,
            keep_bubble_proposals: false,
            class_names: config.class_names(),
            recommended_thresholds: config.recommended_thresholds,
        })
    }

    pub(super) fn recommended_thresholds(&self) -> KoharuLayoutThresholds {
        self.recommended_thresholds
    }

    pub(super) fn preprocess(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        ensure!(
            image.width() > 0 && image.height() > 0,
            "cannot segment an empty image"
        );
        let (samples, stride) = rgb_samples(image);
        let (width, height) = (image.width() as usize, image.height() as usize);
        let scale = RESOLUTION as f64 / width.max(height) as f64;
        let side = |length: usize| ((length as f64 * scale).round_ties_even() as usize).max(1);
        let (scaled_w, scaled_h) = (side(width), side(height));
        let rgb = |x: usize, y: usize| -> [f32; 3] {
            let i = (y * width + x) * stride;
            std::array::from_fn(|c| f32::from(samples[i + c]))
        };

        if (width, height) == (scaled_w, scaled_h) {
            return Ok(letterbox(scaled_w, scaled_h, rgb));
        }
        if scale < 1.0 {
            let rows = area_taps(height, scaled_h);
            let columns = area_taps(width, scaled_w);
            return Ok(letterbox_with_rows(
                scaled_w,
                scaled_h,
                |vrow, y| {
                    vrow.resize(width, [0.0; 3]);
                    blend_area_rows(vrow, &samples, stride, &rows[y]);
                },
                |vrow, x, _| area_pixel(vrow, &columns[x]),
            ));
        }
        let rows = bilinear_taps(height, scaled_h);
        let columns = bilinear_taps(width, scaled_w);
        Ok(letterbox(scaled_w, scaled_h, |x, y| {
            let column = columns[x];
            let interpolate_row = |sy| -> [f32; 3] {
                let low = rgb(column.low, sy);
                let high = rgb(column.high, sy);
                std::array::from_fn(|c| column.low_weight * low[c] + column.high_weight * high[c])
            };
            let row = rows[y];
            let low = interpolate_row(row.low);
            let high = interpolate_row(row.high);
            std::array::from_fn(|c| row.low_weight * low[c] + row.high_weight * high[c])
        }))
    }

    pub(super) fn postprocess(
        &self,
        model: &mut Model,
        output: &Output,
        image_width: u32,
        image_height: u32,
        thresholds: KoharuLayoutThresholds,
    ) -> Result<KoharuLayoutDetections> {
        self.decode_detections(
            &output.pred_logits,
            &output.pred_boxes,
            (image_width, image_height),
            thresholds,
            |queries| model.masks(output, queries),
        )
    }

    fn decode_detections(
        &self,
        pred_logits: &[f32],
        pred_boxes: &[f32],
        (image_width, image_height): (u32, u32),
        thresholds: KoharuLayoutThresholds,
        decode_masks: impl FnOnce(&[usize]) -> Result<Vec<f32>>,
    ) -> Result<KoharuLayoutDetections> {
        validate_thresholds(thresholds)?;
        // PostProcess ranks every query/class pair, including the checkpoint's
        // fifth background logit slot, before applying the caller threshold.
        let scores = pred_logits
            .iter()
            .map(|&logit| 1.0 / (1.0 + (-logit).exp()))
            .collect::<Vec<f32>>();
        let mut ranked = (0..scores.len()).collect::<Vec<_>>();
        ranked.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
        ranked.truncate(self.num_select);
        let thresholds = thresholds.with_background();
        ranked.retain(|&index| scores[index] > thresholds[index % NUM_CLASSES_WITH_BACKGROUND]);

        // The first passing class is the highest scoring, but a passing text
        // label takes precedence over ambiguous lettering (upstream 08193a3c).
        // In fusion mode the comic detector supplies balloon boxes, so an RF
        // bubble proposal does not compete with any of the semantic classes.
        let auxiliary = |label| self.keep_bubble_proposals && label == BUBBLE;
        let mut chosen = [None; NUM_QUERIES];
        for &index in &ranked {
            let label = index % NUM_CLASSES_WITH_BACKGROUND;
            let choice = &mut chosen[index / NUM_CLASSES_WITH_BACKGROUND];
            if !auxiliary(label) && (choice.is_none() || label == TEXT) {
                *choice = Some(index);
            }
        }
        let long_side = image_width.max(image_height);
        let selected: Vec<_> = ranked
            .into_iter()
            .filter_map(|index| {
                let query = index / NUM_CLASSES_WITH_BACKGROUND;
                if chosen[query] != Some(index) && !auxiliary(index % NUM_CLASSES_WITH_BACKGROUND) {
                    return None;
                }
                let [cx, cy, w, h]: [f32; 4] = pred_boxes[query * 4..query * 4 + 4]
                    .try_into()
                    .expect("box has four coordinates");
                let bbox = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
                    .map(|coordinate| coordinate * long_side as f32);
                // Queries wholly in the letterbox padding have no page instance to decode.
                (bbox[0].max(0.0) < bbox[2].min(image_width as f32)
                    && bbox[1].max(0.0) < bbox[3].min(image_height as f32))
                .then_some((index, bbox))
            })
            .collect();
        if selected.is_empty() {
            return Ok(KoharuLayoutDetections {
                image_width,
                image_height,
                detections: Vec::new(),
            });
        }

        let mut queries = Vec::new();
        let mut mask_slots = [None; NUM_QUERIES];
        for &(index, _) in &selected {
            let query = index / NUM_CLASSES_WITH_BACKGROUND;
            mask_slots[query].get_or_insert_with(|| {
                queries.push(query);
                queries.len() - 1
            });
        }
        let logits = decode_masks(&queries)?;
        // Project to the long-side square, then take its top-left page crop.
        // Both axes share taps, so the padded output pixels need no storage.
        let taps = bilinear_taps(MASK_RESOLUTION, long_side as usize);
        let rows = &taps[..image_height as usize];
        let columns = &taps[..image_width as usize];
        let masks: Vec<_> = logits
            .par_chunks_exact(MASK_RESOLUTION * MASK_RESOLUTION)
            .map(|logits| {
                let (area, mask) = page_mask(logits, rows, columns);
                (area, Arc::new(mask))
            })
            .collect();
        let detections = selected
            .into_iter()
            .map(|(index, bbox)| {
                let query = index / NUM_CLASSES_WITH_BACKGROUND;
                let label_id = index % NUM_CLASSES_WITH_BACKGROUND;
                let (area, mask) = &masks[mask_slots[query].expect("selected query has a mask")];
                KoharuLayoutDetection {
                    label_id,
                    label: self
                        .class_names
                        .get(label_id)
                        .cloned()
                        .unwrap_or_else(|| "__background__".to_owned()),
                    score: scores[index],
                    bbox,
                    area: *area,
                    mask: Arc::clone(mask),
                }
            })
            .collect();

        Ok(KoharuLayoutDetections {
            image_width,
            image_height,
            detections,
        })
    }
}

/// Interleaved 8-bit RGB samples of `image` and the bytes per pixel. 8-bit RGB
/// and RGBA buffers are borrowed; alpha is ignored, as `to_rgb8` drops it.
fn rgb_samples(image: &DynamicImage) -> (Cow<'_, [u8]>, usize) {
    match image {
        DynamicImage::ImageRgb8(image) => (Cow::Borrowed(image.as_raw()), 3),
        DynamicImage::ImageRgba8(image) => (Cow::Borrowed(image.as_raw()), 4),
        image => (Cow::Owned(image.to_rgb8().into_raw()), 3),
    }
}

fn letterbox(
    width: usize,
    height: usize,
    sample: impl Fn(usize, usize) -> [f32; 3] + Sync,
) -> Vec<f32> {
    letterbox_with_rows(width, height, |_, _| {}, |_, x, y| sample(x, y))
}

/// Store the top-left resized image on a white planar RGB canvas. OpenCV
/// returns 8-bit samples, so round before applying ImageNet normalization.
fn letterbox_with_rows(
    width: usize,
    height: usize,
    prepare_row: impl Fn(&mut Vec<[f32; 3]>, usize) + Sync,
    sample: impl Fn(&[[f32; 3]], usize, usize) -> [f32; 3] + Sync,
) -> Vec<f32> {
    let normalized: [[f32; 256]; 3] =
        std::array::from_fn(|c| std::array::from_fn(|v| (v as f32 / 255.0 - MEAN[c]) / STD[c]));
    let plane = RESOLUTION * RESOLUTION;
    let mut canvas = vec![0.0; 3 * plane];
    let (red, rest) = canvas.split_at_mut(plane);
    let (green, blue) = rest.split_at_mut(plane);
    red.par_chunks_exact_mut(RESOLUTION)
        .zip(green.par_chunks_exact_mut(RESOLUTION))
        .zip(blue.par_chunks_exact_mut(RESOLUTION))
        .enumerate()
        .with_min_len(8)
        .for_each_init(Vec::new, |scratch, (y, ((red, green), blue))| {
            let filled = if y < height {
                prepare_row(scratch, y);
                width
            } else {
                0
            };
            for x in 0..filled {
                let [r, g, b] = sample(scratch, x, y).map(|v| v.round_ties_even() as u8);
                red[x] = normalized[0][usize::from(r)];
                green[x] = normalized[1][usize::from(g)];
                blue[x] = normalized[2][usize::from(b)];
            }
            red[filled..].fill(normalized[0][255]);
            green[filled..].fill(normalized[1][255]);
            blue[filled..].fill(normalized[2][255]);
        });
    canvas
}

/// Accumulate each column in tap order before applying any horizontal weights.
fn blend_area_rows(
    vertical: &mut [[f32; 3]],
    samples: &[u8],
    stride: usize,
    taps: &[(usize, f32)],
) {
    // Accumulate contiguous components so both RGB and RGBA can vectorize.
    // Strip alpha only when copying the blended block to the RGB row.
    const BLOCK_PIXELS: usize = 256;
    let row_bytes = vertical.len() * stride;
    let mut block_sums = [0.0; BLOCK_PIXELS * 4];
    for (index, block) in vertical.chunks_mut(BLOCK_PIXELS).enumerate() {
        let offset = index * BLOCK_PIXELS * stride;
        let count = block.len() * stride;
        let sums = &mut block_sums[..count];
        sums.fill(0.0);
        for &(sy, weight) in taps {
            let start = sy * row_bytes + offset;
            for (sum, &value) in sums.iter_mut().zip(&samples[start..][..count]) {
                *sum += weight * f32::from(value);
            }
        }
        if stride == 3 {
            block.as_flattened_mut().copy_from_slice(sums);
        } else {
            for (rgb, &[r, g, b, _]) in block.iter_mut().zip(sums.as_chunks::<4>().0) {
                *rgb = [r, g, b];
            }
        }
    }
}

fn area_pixel(vertical: &[[f32; 3]], taps: &[(usize, f32)]) -> [f32; 3] {
    let mut pixel = [0.0; 3];
    for &(sx, weight) in taps {
        let column = vertical[sx];
        for c in 0..3 {
            pixel[c] += weight * column[c];
        }
    }
    pixel
}

/// Sparse OpenCV INTER_AREA weights, including computeResizeAreaTab's
/// cutoffs for partial pixels. A center-sampled box filter loses these
/// fractional contributions when the scale is nonintegral.
fn area_taps(input: usize, output: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = 1.0 / (output as f64 / input as f64);
    (0..output)
        .map(|dst| {
            let start = dst as f64 * scale;
            let end = start + scale;
            let cell = scale.min(input as f64 - start);
            let last = (end.floor() as usize).min(input - 1);
            let first = (start.ceil() as usize).min(last);
            let mut taps = Vec::with_capacity(last - first + 2);
            if first as f64 - start > 1e-3 {
                taps.push((first - 1, ((first as f64 - start) / cell) as f32));
            }
            taps.extend((first..last).map(|src| (src, (1.0 / cell) as f32)));
            if end - last as f64 > 1e-3 {
                taps.push((last, ((end - last as f64).min(1.0).min(cell) / cell) as f32));
            }
            taps
        })
        .collect()
}

/// One output coordinate of PyTorch's `upsample_bilinear2d` with
/// `align_corners=False` and no explicit scale.
#[derive(Debug, Clone, Copy)]
struct BilinearTap {
    low: usize,
    high: usize,
    low_weight: f32,
    high_weight: f32,
}

impl BilinearTap {
    fn interpolate(self, value: impl Fn(usize) -> f32) -> f32 {
        self.low_weight * value(self.low) + self.high_weight * value(self.high)
    }
}

fn bilinear_taps(input: usize, output: usize) -> Vec<BilinearTap> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|index| {
            let source = ((index as f32 + 0.5) * scale - 0.5).max(0.0);
            let low = (source as usize).min(input - 1);
            let high = (low + 1).min(input - 1);
            let high_weight = source - low as f32;
            BilinearTap {
                low,
                high,
                low_weight: 1.0 - high_weight,
                high_weight,
            }
        })
        .collect()
}

/// Upsamples one query's mask logits to the page and keeps positive pixels,
/// returning the pixel count and the mask cropped to its occupied bounds.
fn page_mask(
    logits: &[f32],
    rows: &[BilinearTap],
    columns: &[BilinearTap],
) -> (u32, KoharuLayoutMask) {
    let empty = KoharuLayoutMask {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
        pixels: Vec::new(),
    };
    // An interpolated value is a convex combination of its taps, so a page
    // pixel can be positive only if a tap lies inside the bounds of the
    // positive mask logits. Skipping the rest is exact.
    // Padding beyond the last interpolation neighbor cannot reach the page.
    let source_height = rows.last().map_or(0, |tap| tap.high + 1);
    let source_width = columns.last().map_or(0, |tap| tap.high + 1);
    let mut bounds: Option<[usize; 4]> = None;
    let (logit_rows, _) = logits.as_chunks::<MASK_RESOLUTION>();
    for (y, row) in logit_rows.iter().take(source_height).enumerate() {
        for (x, &value) in row[..source_width].iter().enumerate() {
            if value > 0.0 {
                let [left, top, right, bottom] = bounds.get_or_insert([x, y, x, y]);
                *left = (*left).min(x);
                *top = (*top).min(y);
                *right = (*right).max(x);
                *bottom = (*bottom).max(y);
            }
        }
    }
    let Some([left, top, right, bottom]) = bounds else {
        return (0, empty);
    };
    let touches = |tap: &BilinearTap, low: usize, high: usize| tap.high >= low && tap.low <= high;
    let Some(first_row) = rows.iter().position(|tap| touches(tap, top, bottom)) else {
        return (0, empty);
    };
    let last_row = rows
        .iter()
        .rposition(|tap| touches(tap, top, bottom))
        .unwrap();
    let Some(first_column) = columns.iter().position(|tap| touches(tap, left, right)) else {
        return (0, empty);
    };
    let last_column = columns
        .iter()
        .rposition(|tap| touches(tap, left, right))
        .unwrap();

    let window_rows = &rows[first_row..=last_row];
    let window_columns = &columns[first_column..=last_column];
    let window_width = window_columns.len();
    // Each logit row the window reads is interpolated along x once. Blending
    // two of those rows per page row is the same arithmetic in the same order
    // as interpolating every pixel from its four taps.
    let source_top = window_rows[0].low;
    let source_bottom = window_rows[window_rows.len() - 1].high;
    let mut horizontal = vec![0.0_f32; (source_bottom - source_top + 1) * window_width];
    horizontal
        .par_chunks_exact_mut(window_width)
        .zip(source_top..source_bottom + 1)
        .for_each(|(values, source)| {
            let source = &logit_rows[source];
            for (value, column) in values.iter_mut().zip(window_columns) {
                *value = column.interpolate(|x| source[x]);
            }
        });
    let mut window = vec![0_u8; window_rows.len() * window_width];
    // Per page row: the positive pixel count and the first and last positive
    // column.
    let extents = window
        .par_chunks_exact_mut(window_width)
        .zip(window_rows)
        .map(|(pixels, row)| {
            let low = &horizontal[(row.low - source_top) * window_width..][..window_width];
            let high = &horizontal[(row.high - source_top) * window_width..][..window_width];
            let mut count = 0_u32;
            for ((pixel, &low), &high) in pixels.iter_mut().zip(low).zip(high) {
                let positive = row.low_weight * low + row.high_weight * high > 0.0;
                *pixel = if positive { u8::MAX } else { 0 };
                count += u32::from(positive);
            }
            (count > 0).then(|| {
                let first = pixels.iter().position(|&pixel| pixel != 0).unwrap();
                let last = pixels.iter().rposition(|&pixel| pixel != 0).unwrap();
                (count, first, last)
            })
        })
        .collect::<Vec<_>>();
    let Some(top) = extents.iter().position(Option::is_some) else {
        return (0, empty);
    };
    let bottom = extents.iter().rposition(Option::is_some).unwrap();
    let (area, left, right) = extents.iter().flatten().fold(
        (0, usize::MAX, 0),
        |(area, left, right), &(count, first, last)| {
            (area + count, left.min(first), right.max(last))
        },
    );
    let width = right - left + 1;
    let mut pixels = Vec::with_capacity(width * (bottom - top + 1));
    for row in window.chunks_exact(window_width).take(bottom + 1).skip(top) {
        pixels.extend_from_slice(&row[left..][..width]);
    }
    (
        area,
        KoharuLayoutMask {
            x: (first_column + left) as u32,
            y: (first_row + top) as u32,
            width: width as u32,
            height: (bottom - top + 1) as u32,
            pixels,
        },
    )
}

fn validate_thresholds(thresholds: KoharuLayoutThresholds) -> Result<()> {
    for (class, threshold) in [
        ("text", thresholds.text),
        ("onomatopoeia", thresholds.onomatopoeia),
        ("bubble", thresholds.bubble),
        ("panel", thresholds.panel),
    ] {
        ensure!(
            (0.0..=1.0).contains(&threshold),
            "{class} confidence threshold must be between 0 and 1"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct KoharuLayoutDetections {
    pub image_width: u32,
    pub image_height: u32,
    pub detections: Vec<KoharuLayoutDetection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KoharuLayoutDetection {
    pub label_id: usize,
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
    pub area: u32,
    #[serde(skip_serializing)]
    pub mask: Arc<KoharuLayoutMask>,
}

#[derive(Debug, Clone, Default)]
pub struct KoharuLayoutMask {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl KoharuLayoutMask {
    #[must_use]
    pub fn contains(&self, x: u32, y: u32) -> bool {
        let Some(local_x) = x.checked_sub(self.x) else {
            return false;
        };
        let Some(local_y) = y.checked_sub(self.y) else {
            return false;
        };
        if local_x >= self.width || local_y >= self.height {
            return false;
        }
        self.pixels
            .get(local_y as usize * self.width as usize + local_x as usize)
            .is_some_and(|value| *value != 0)
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage, Rgba, RgbaImage};

    use super::{
        KoharuLayoutRFDetrImageProcessor, KoharuLayoutThresholds, MASK_RESOLUTION, RESOLUTION,
        area_pixel, area_taps, bilinear_taps, blend_area_rows, page_mask, validate_thresholds,
    };

    fn processor(
        num_select: usize,
        keep_bubble_proposals: bool,
    ) -> KoharuLayoutRFDetrImageProcessor {
        KoharuLayoutRFDetrImageProcessor {
            num_select,
            class_names: ["text", "onomatopoeia", "bubble", "panel"]
                .map(str::to_owned)
                .to_vec(),
            keep_bubble_proposals,
            recommended_thresholds: KoharuLayoutThresholds {
                text: 0.5,
                onomatopoeia: 0.5,
                bubble: 0.05,
                panel: 0.5,
            },
        }
    }

    fn detect_queries(
        logits: &[[f32; 5]],
        keep_bubble_proposals: bool,
        expected_queries: &[usize],
    ) -> Vec<super::KoharuLayoutDetection> {
        let processor = processor(logits.len() * 5, keep_bubble_proposals);
        let boxes = vec![[0.5, 0.5, 0.5, 0.5]; logits.len()];
        processor
            .decode_detections(
                logits.as_flattened(),
                boxes.as_flattened(),
                (16, 8),
                processor.recommended_thresholds,
                |queries| {
                    assert_eq!(queries, expected_queries);
                    const MASK_PIXELS: usize = MASK_RESOLUTION * MASK_RESOLUTION;
                    let mut masks = vec![-1.0; queries.len() * MASK_PIXELS];
                    let (mask_planes, _) = masks.as_chunks_mut::<MASK_PIXELS>();
                    for (&query, mask) in queries.iter().zip(mask_planes) {
                        let height = MASK_RESOLUTION / 2 * (query + 1) / (logits.len() + 1);
                        mask[..height * MASK_RESOLUTION].fill(1.0);
                    }
                    Ok(masks)
                },
            )
            .unwrap()
            .detections
    }

    #[test]
    fn a_passing_text_wins_its_query_at_its_own_score_rank() {
        // Query 0's sound effect would rank first. Its text ranks behind query
        // 1's panel, whose lower-scoring sound effect also passes.
        let detections = detect_queries(
            &[[1.0, 3.0, -8.0, -8.0, -8.0], [-8.0, 0.5, -8.0, 2.0, -8.0]],
            false,
            &[1, 0],
        );
        let labels: Vec<_> = detections
            .iter()
            .map(|detection| detection.label.as_str())
            .collect();
        assert_eq!(labels, ["panel", "text"]);
    }

    #[test]
    fn auxiliary_bubble_proposals_share_the_semantic_query_mask() {
        // A weak bubble on one text query, a bubble outranking the panel
        // on another, and a third query with only a bubble proposal.
        let logits = [
            [1.0, 1.5, -2.5, -8.0, -8.0],
            [-8.0, -8.0, 3.0, 2.0, -8.0],
            [-8.0, -8.0, -2.8, -8.0, -8.0],
        ];
        let detections = detect_queries(&logits, true, &[1, 0, 2]);
        let labels: Vec<_> = detections
            .iter()
            .map(|detection| detection.label.as_str())
            .collect();
        assert_eq!(labels, ["bubble", "panel", "text", "bubble", "bubble"]);
        // Decode slots hold queries 1, 0 and 2, with respective page heights
        // of 4, 2 and 6 pixels. Each class must retrieve its query's mask.
        for (detection, height) in detections.iter().zip([4, 4, 2, 2, 6]) {
            assert_eq!(detection.mask.height, height);
            assert_eq!(detection.area, 16 * height);
        }
        for (a, b) in [(0, 1), (2, 3)] {
            assert!(std::sync::Arc::ptr_eq(
                &detections[a].mask,
                &detections[b].mask
            ));
            assert_eq!(detections[a].bbox, [4.0, 4.0, 12.0, 12.0]);
        }

        let standalone = detect_queries(&logits, false, &[1, 0, 2]);
        let labels: Vec<_> = standalone
            .iter()
            .map(|detection| detection.label.as_str())
            .collect();
        assert_eq!(labels, ["bubble", "text", "bubble"]);
    }

    fn imagenet([r, g, b]: [u8; 3]) -> [f32; 3] {
        [
            (f32::from(r) / 255.0 - 0.485) / 0.229,
            (f32::from(g) / 255.0 - 0.456) / 0.224,
            (f32::from(b) / 255.0 - 0.406) / 0.225,
        ]
    }

    #[test]
    fn separable_area_matches_vertical_then_horizontal_sums_bit_for_bit() {
        let (width, height) = (317, 449);
        let columns = area_taps(width, 87);
        let rows = area_taps(height, 123);
        for stride in [3, 4] {
            let samples: Vec<u8> = (0..width * height * stride)
                .map(|i| (i * 71 + i / 13) as u8)
                .collect();
            let mut vertical = vec![[0.0; 3]; width];
            for (y, row) in rows.iter().enumerate() {
                blend_area_rows(&mut vertical, &samples, stride, row);
                for (x, taps) in columns.iter().enumerate() {
                    let mut expected = [0.0_f32; 3];
                    for &(sx, wx) in taps {
                        let mut column = [0.0; 3];
                        for &(sy, wy) in row {
                            let i = (sy * width + sx) * stride;
                            for c in 0..3 {
                                column[c] += wy * f32::from(samples[i + c]);
                            }
                        }
                        for c in 0..3 {
                            expected[c] += wx * column[c];
                        }
                    }
                    assert_eq!(
                        area_pixel(&vertical, taps).map(f32::to_bits),
                        expected.map(f32::to_bits),
                        "stride {stride}, pixel ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn preprocess_preserves_aspect_ratio_on_a_white_planar_canvas() {
        let processor = processor(1, false);
        let color = [13, 91, 207];
        for (width, height, scaled_w, scaled_h) in [
            (800, 1600, 960, 1920),
            (2400, 1200, 1920, 960),
            (1920, 1920, 1920, 1920),
            (400, 600, 1280, 1920),
            (1000, 3000, 640, 1920),
            (1281, 3840, 640, 1920), // 640.5 rounds to even.
            (1, 4000, 1, 1920),      // Keep at least one content pixel.
        ] {
            let image = RgbImage::from_pixel(width, height, Rgb(color));
            let values = processor.preprocess(&image.into()).unwrap();
            const PLANE: usize = RESOLUTION * RESOLUTION;
            assert_eq!(values.len(), 3 * PLANE);
            let (channels, _) = values.as_chunks::<PLANE>();
            for (c, channel) in channels.iter().enumerate() {
                let content = imagenet(color)[c];
                let white = imagenet([255; 3])[c];
                let (rows, _) = channel.as_chunks::<RESOLUTION>();
                for (y, row) in rows.iter().enumerate() {
                    let filled = if y < scaled_h { scaled_w } else { 0 };
                    assert!(
                        row[..filled].iter().all(|&v| v == content),
                        "{width}x{height}, row {y}"
                    );
                    assert!(
                        row[filled..].iter().all(|&v| v == white),
                        "{width}x{height}, row {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn preprocess_uses_area_when_shrinking_and_bilinear_when_enlarging() {
        let processor = processor(1, false);
        let plane = RESOLUTION * RESOLUTION;
        let pixel = |values: &[f32], x: usize, y: usize| -> [f32; 3] {
            std::array::from_fn(|c| values[c * plane + y * RESOLUTION + x])
        };

        // 2400x3 -> 1920x2. The first cell covers 1.25 columns and 1.5
        // rows, with different fractional contributions on the two axes.
        let image = RgbaImage::from_fn(2400, 3, |x, y| {
            Rgba([(x * 100) as u8, (y * 3) as u8, 41, (x + y) as u8])
        });
        let values = processor
            .preprocess(&DynamicImage::ImageRgba8(image))
            .unwrap();
        assert_eq!(pixel(&values, 0, 0), imagenet([20, 1, 41]));
        assert_eq!(pixel(&values, 1, 1), imagenet([140, 5, 41]));
        assert_eq!(pixel(&values, 0, 2), imagenet([255; 3]));

        // 960x1 -> 1920x2. Green interpolates to 2.5 and 3.5 before the
        // 8-bit conversion. Alpha is ignored even for transparent pixels.
        let image = RgbaImage::from_fn(960, 1, |x, _| {
            if x == 0 {
                Rgba([0, 2, 19, 0])
            } else {
                Rgba([100, 4, 19, 255])
            }
        });
        let values = processor
            .preprocess(&DynamicImage::ImageRgba8(image))
            .unwrap();
        assert_eq!(pixel(&values, 0, 0), imagenet([0, 2, 19]));
        assert_eq!(pixel(&values, 1, 0), imagenet([25, 2, 19]));
        assert_eq!(pixel(&values, 2, 1), imagenet([75, 4, 19]));
    }

    #[test]
    fn boxes_and_masks_project_from_the_long_side_square_to_the_page() {
        let processor = processor(2, false);
        let classes = [[4.0, -8.0, -8.0, -8.0, -8.0], [3.0, -8.0, -8.0, -8.0, -8.0]];
        // Mask cells x=48..52, y=72..76 have positive weight in every
        // interpolated pixel strictly inside their half-pixel support.
        for (width, height, mask_x, mask_y, mask_side) in [
            (800, 1600, 158, 238, 17),
            (2400, 1200, 238, 358, 24),
            (1920, 1920, 190, 286, 20),
            (400, 600, 59, 89, 7),
            (1000, 3000, 297, 447, 31),
        ] {
            let long_side = width.max(height) as f32;
            let page_box = [
                long_side / 8.0,
                long_side / 16.0,
                long_side / 4.0,
                long_side / 8.0,
            ];
            let [x0, y0, x1, y1] = page_box.map(|v| v / long_side);
            let boxes = [
                [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0],
                [0.5, 0.5, 1.0, 1.0],
            ];
            let output = processor
                .decode_detections(
                    classes.as_flattened(),
                    boxes.as_flattened(),
                    (width, height),
                    processor.recommended_thresholds,
                    |queries| {
                        assert_eq!(queries, [0, 1]);
                        let plane = MASK_RESOLUTION * MASK_RESOLUTION;
                        let mut logits = vec![0.0; 2 * plane];
                        for y in 72..76 {
                            logits[y * MASK_RESOLUTION + 48..y * MASK_RESOLUTION + 52].fill(1.0);
                        }
                        // The second mask has support only in right/bottom padding.
                        if width < height {
                            logits[plane + MASK_RESOLUTION - 1] = 1.0;
                        } else if height < width {
                            logits[2 * plane - MASK_RESOLUTION] = 1.0;
                        }
                        Ok(logits)
                    },
                )
                .unwrap();
            let [content, padding] = output.detections.as_slice() else {
                panic!("expected two detections")
            };
            assert_eq!(content.bbox, page_box);
            assert_eq!(padding.bbox, [0.0, 0.0, long_side, long_side]);
            let mask = &content.mask;
            assert_eq!(
                [mask.x, mask.y, mask.width, mask.height],
                [mask_x, mask_y, mask_side, mask_side],
                "{width}x{height}"
            );
            assert_eq!(content.area, mask_side * mask_side);
            assert_eq!(mask.pixels, vec![255; content.area as usize]);
            assert_eq!(padding.area, 0);
            let mask = &padding.mask;
            assert_eq!([mask.x, mask.y, mask.width, mask.height], [0; 4]);
            assert!(mask.pixels.is_empty());
        }
    }

    #[test]
    fn padding_only_boxes_are_dropped_before_decoding_masks() {
        let processor = processor(2, false);
        let classes = [[4.0, -8.0, -8.0, -8.0, -8.0]; 2];
        let crossing = [0.5, 0.5, 0.25, 0.25];
        for (size, padding) in [
            ((800, 1600), [0.75, 0.5, 0.25, 0.25]),
            ((1600, 800), [0.5, 0.75, 0.25, 0.25]),
        ] {
            let output = processor
                .decode_detections(
                    classes.as_flattened(),
                    [padding, crossing].as_flattened(),
                    size,
                    processor.recommended_thresholds,
                    |queries| {
                        assert_eq!(queries, [1]);
                        Ok(vec![-1.0; MASK_RESOLUTION * MASK_RESOLUTION])
                    },
                )
                .unwrap();
            assert_eq!(output.detections.len(), 1);
            assert_eq!(output.detections[0].bbox, [600., 600., 1000., 1000.]);

            let output = processor
                .decode_detections(
                    &classes[0],
                    &padding,
                    size,
                    processor.recommended_thresholds,
                    |_| panic!("no on-page queries"),
                )
                .unwrap();
            assert!(output.detections.is_empty());
        }
    }

    #[test]
    fn fractional_page_edge_is_cropped_after_mask_interpolation() {
        let processor = processor(1, false);
        // The page occupies 240.9 native mask columns or rows. Interpolating
        // across its edge gives the last page pixel 25% of the next logit.
        // Rounding the native crop to 241 first would repeat cell 240 there.
        for (width, height) in [(803, 1600), (1600, 803)] {
            let result = processor
                .decode_detections(
                    &[4.0, -8.0, -8.0, -8.0, -8.0],
                    &[0.5, 0.5, 1.0, 1.0],
                    (width, height),
                    processor.recommended_thresholds,
                    |_| {
                        let mut logits = vec![-7.0; MASK_RESOLUTION * MASK_RESOLUTION];
                        let (rows, _) = logits.as_chunks_mut::<MASK_RESOLUTION>();
                        for (y, row) in rows.iter_mut().enumerate() {
                            if width < height {
                                row[239..241].fill(1.0);
                            } else if (239..241).contains(&y) {
                                row.fill(1.0);
                            }
                        }
                        Ok(logits)
                    },
                )
                .unwrap();
            let detection = &result.detections[0];
            let mask = &detection.mask;
            let expected = if width < height {
                [798, 0, 4, height]
            } else {
                [0, 798, width, 4]
            };
            assert_eq!([mask.x, mask.y, mask.width, mask.height], expected);
            assert_eq!(detection.area, 4 * 1600);
            assert_eq!(mask.pixels, vec![255; detection.area as usize]);
        }
    }

    #[test]
    fn class_thresholds_must_be_probabilities() {
        let thresholds = KoharuLayoutThresholds {
            text: 0.25,
            onomatopoeia: 1.1,
            bubble: 0.5,
            panel: 0.5,
        };

        let error = validate_thresholds(thresholds).unwrap_err();
        assert!(error.to_string().contains("onomatopoeia"));
    }

    #[test]
    fn page_mask_matches_upsampling_the_whole_page() {
        // A blob, a single positive cell on the grid edge, and negative
        // values elsewhere, upsampled to pages that are not a multiple of the
        // mask grid on either axis. The smaller pages skip grid rows or
        // columns between output pixels.
        let mut logits = vec![-1.0_f32; MASK_RESOLUTION * MASK_RESOLUTION];
        for y in 100..140 {
            for x in 200..260 {
                logits[y * MASK_RESOLUTION + x] = ((x + y) % 7) as f32 - 2.5;
            }
        }
        logits[MASK_RESOLUTION - 1] = 0.3;
        for (width, height) in [(1013, 1517), (301, 700), (700, 301)] {
            let square = bilinear_taps(MASK_RESOLUTION, width.max(height));
            let rows = &square[..height];
            let columns = &square[..width];

            let mut expected = vec![false; width * height];
            for (y, row) in rows.iter().enumerate() {
                for (x, column) in columns.iter().enumerate() {
                    let value = row.interpolate(|source_y| {
                        column.interpolate(|source_x| logits[source_y * MASK_RESOLUTION + source_x])
                    });
                    expected[y * width + x] = value > 0.0;
                }
            }
            let (area, mask) = page_mask(&logits, rows, columns);

            assert_eq!(
                area as usize,
                expected.iter().filter(|&&set| set).count(),
                "{width}x{height}"
            );
            for y in 0..height {
                for x in 0..width {
                    assert_eq!(
                        mask.contains(x as u32, y as u32),
                        expected[y * width + x],
                        "{width}x{height} pixel {x}, {y}"
                    );
                }
            }
            let occupied = |coordinate: &dyn Fn(usize) -> usize| {
                let set = (0..expected.len()).filter(|&index| expected[index]);
                (
                    set.clone().map(coordinate).min().unwrap(),
                    set.map(coordinate).max().unwrap(),
                )
            };
            assert_eq!(
                occupied(&|index| index % width),
                (mask.x as usize, (mask.x + mask.width - 1) as usize),
                "{width}x{height}"
            );
            assert_eq!(
                occupied(&|index| index / width),
                (mask.y as usize, (mask.y + mask.height - 1) as usize),
                "{width}x{height}"
            );
        }
    }
}
