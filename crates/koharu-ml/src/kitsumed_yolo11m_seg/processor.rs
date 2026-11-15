//! Letterboxing, NMS and mask decoding of the Ultralytics segmentation
//! predictor, for one image and the checkpoint's two classes.
//!
//! https://github.com/ultralytics/ultralytics/blob/v8.4.211/ultralytics/data/augment.py (LetterBox)
//! https://github.com/ultralytics/ultralytics/blob/v8.4.211/ultralytics/utils/nms.py (non_max_suppression)
//! https://github.com/ultralytics/ultralytics/blob/v8.4.211/ultralytics/utils/ops.py (scale_boxes, process_mask_native)
//!
//! Python's `round` rounds half to even, so the ports use `round_ties_even`.

use image::{GrayImage, Luma, RgbImage};
use rayon::prelude::*;

use super::{Censor, CensorKind, network::Outputs};

/// `imgsz`: the long side of the letterboxed input.
pub(super) const INPUT_SIZE: u32 = 960;
/// The network's largest stride; letterboxing pads each side to a multiple.
pub(super) const STRIDE: u32 = 32;
pub(super) const PROTOTYPES: usize = 32;
const CLASSES: usize = 2;
pub(super) const OUTPUT_ROWS: usize = 4 + CLASSES + PROTOTYPES;
const PADDING: f32 = 114.0 / 255.0;
const CONFIDENCE: f32 = 0.4;
/// `PREDICT_IOU`, the predictor's default.
const IOU: f32 = 0.5;
const MAX_DETECTIONS: usize = 300;
const MAX_NMS: usize = 30_000;
/// Per-class box offset that keeps NMS within one class.
const MAX_WH: f32 = 7680.0;

/// Anchor points of the three detection strides for a `width` x `height` input.
pub(super) fn anchors(width: u32, height: u32) -> usize {
    [8, 16, 32]
        .into_iter()
        .map(|stride| (width / stride) as usize * (height / stride) as usize)
        .sum()
}

/// `LetterBox(imgsz, auto=True, stride=32)`: the predictor's minimum
/// rectangle for a dynamic ONNX model.
pub(super) struct Letterbox {
    source: (u32, u32),
    resized: (u32, u32),
    /// Network input as `(width, height)`.
    pub input: (u32, u32),
    left: u32,
    top: u32,
}

impl Letterbox {
    pub fn new(width: u32, height: u32) -> Self {
        let side = f64::from(INPUT_SIZE);
        let ratio = (side / f64::from(height)).min(side / f64::from(width));
        let resize = |extent: u32| ((f64::from(extent) * ratio).round_ties_even() as u32).max(1);
        let resized = (resize(width), resize(height));
        let pad = |extent: u32| {
            let total = f64::from((INPUT_SIZE - extent) % STRIDE) / 2.0;
            (
                (total - 0.1).round_ties_even() as u32,
                (total + 0.1).round_ties_even() as u32,
            )
        };
        let (left, right) = pad(resized.0);
        let (top, bottom) = pad(resized.1);
        Self {
            source: (width, height),
            resized,
            input: (resized.0 + left + right, resized.1 + top + bottom),
            left,
            top,
        }
    }

    /// Planar RGB in `[0, 1]`: the image resized with OpenCV's `INTER_LINEAR`
    /// sampling (half-pixel centres, no antialiasing) and rounded to 8 bits
    /// as `cv2.resize` returns it, on a border of 114.
    pub fn apply(&self, image: &RgbImage) -> Vec<f32> {
        let (width, height) = (self.input.0 as usize, self.input.1 as usize);
        let plane = width * height;
        let columns = taps(self.source.0, self.resized.0);
        let rows = taps(self.source.1, self.resized.1);
        let samples = image.as_raw();
        let stride = self.source.0 as usize * 3;
        let mut output = vec![PADDING; 3 * plane];
        let (red, rest) = output.split_at_mut(plane);
        let (green, blue) = rest.split_at_mut(plane);
        red.par_chunks_mut(width)
            .zip(green.par_chunks_mut(width))
            .zip(blue.par_chunks_mut(width))
            .enumerate()
            .for_each(|(y, ((red, green), blue))| {
                let Some(row) = (y as u32)
                    .checked_sub(self.top)
                    .and_then(|y| rows.get(y as usize))
                else {
                    return;
                };
                let (low, high) = (&samples[row.low * stride..], &samples[row.high * stride..]);
                for (x, column) in columns.iter().enumerate() {
                    let x = x + self.left as usize;
                    for (channel, plane) in
                        [&mut *red, &mut *green, &mut *blue].into_iter().enumerate()
                    {
                        let at = |line: &[u8], offset: usize| f32::from(line[offset * 3 + channel]);
                        let top = column.low_weight * at(low, column.low)
                            + column.high_weight * at(low, column.high);
                        let bottom = column.low_weight * at(high, column.low)
                            + column.high_weight * at(high, column.high);
                        let value = row.low_weight * top + row.high_weight * bottom;
                        plane[x] = value.round().clamp(0.0, 255.0) / 255.0;
                    }
                }
            });
        output
    }
}

struct Tap {
    low: usize,
    high: usize,
    low_weight: f32,
    high_weight: f32,
}

/// Half-pixel bilinear taps from `source` samples to `target`, clamped at
/// the edges, as OpenCV's and PyTorch's (align_corners=False) resizes take
/// them.
fn taps(source: u32, target: u32) -> Vec<Tap> {
    let scale = f64::from(source) / f64::from(target);
    (0..target)
        .map(|index| {
            let position = ((f64::from(index) + 0.5) * scale - 0.5).max(0.0);
            let low = (position.floor() as usize).min(source as usize - 1);
            let high = (low + 1).min(source as usize - 1);
            let fraction = if low == high {
                0.0
            } else {
                position - low as f64
            };
            Tap {
                low,
                high,
                low_weight: (1.0 - fraction) as f32,
                high_weight: fraction as f32,
            }
        })
        .collect()
}

struct Candidate {
    /// `[x1, y1, x2, y2]` in input pixels.
    bbox: [f32; 4],
    score: f32,
    class: usize,
    anchor: usize,
}

pub(super) fn postprocess(output: &Outputs, letterbox: &Letterbox) -> Vec<Censor> {
    let anchors = output.anchors;
    let value = |row: usize, anchor: usize| output.detections[row * anchors + anchor];
    // Best class only (multi_label=False); a box is a candidate when its best
    // class score exceeds the threshold.
    let mut candidates = (0..anchors)
        .filter_map(|anchor| {
            let (class, score) = (0..CLASSES)
                .map(|class| (class, value(4 + class, anchor)))
                .fold((0, f32::NEG_INFINITY), |best, next| {
                    if next.1 > best.1 { next } else { best }
                });
            (score > CONFIDENCE).then(|| {
                let (x, y) = (value(0, anchor), value(1, anchor));
                let (half_width, half_height) = (value(2, anchor) / 2.0, value(3, anchor) / 2.0);
                Candidate {
                    bbox: [
                        x - half_width,
                        y - half_height,
                        x + half_width,
                        y + half_height,
                    ],
                    score,
                    class,
                    anchor,
                }
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    candidates.truncate(MAX_NMS);
    let kept = nms(&candidates);

    let (input_width, input_height) = letterbox.input;
    let prototype_size = (input_width as usize / 4, input_height as usize / 4);
    kept.into_iter()
        .filter_map(|candidate| {
            let bbox = scale_box(candidate.bbox, letterbox);
            let coefficients = (0..PROTOTYPES)
                .map(|index| value(4 + CLASSES + index, candidate.anchor))
                .collect::<Vec<_>>();
            let (mask, mask_origin) = decode_mask(
                &output.prototypes,
                prototype_size,
                &coefficients,
                bbox,
                letterbox.source,
            )?;
            Some(Censor {
                kind: match candidate.class {
                    0 => CensorKind::Bar,
                    _ => CensorKind::Mosaic,
                },
                score: candidate.score,
                bbox,
                mask,
                mask_origin,
            })
        })
        .collect()
}

/// torchvision's greedy NMS over boxes offset by class, on candidates sorted
/// by descending score.
fn nms(candidates: &[Candidate]) -> Vec<&Candidate> {
    let offset = |candidate: &Candidate| {
        let shift = candidate.class as f32 * MAX_WH;
        candidate.bbox.map(|value| value + shift)
    };
    let area = |[x1, y1, x2, y2]: [f32; 4]| (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let mut kept: Vec<&Candidate> = Vec::new();
    for candidate in candidates {
        let bbox = offset(candidate);
        let suppressed = kept.iter().any(|other| {
            let other = offset(other);
            let width = (bbox[2].min(other[2]) - bbox[0].max(other[0])).max(0.0);
            let height = (bbox[3].min(other[3]) - bbox[1].max(other[1])).max(0.0);
            let intersection = width * height;
            intersection / (area(bbox) + area(other) - intersection) > IOU
        });
        if !suppressed {
            kept.push(candidate);
            if kept.len() == MAX_DETECTIONS {
                break;
            }
        }
    }
    kept
}

/// `scale_boxes` from the input to the source image, then `clip_boxes`.
fn scale_box(bbox: [f32; 4], letterbox: &Letterbox) -> [f32; 4] {
    let (width, height) = letterbox.source;
    let (input_width, input_height) = letterbox.input;
    let gain = (f64::from(input_height) / f64::from(height))
        .min(f64::from(input_width) / f64::from(width));
    let resized = |extent: u32| ((f64::from(extent) * gain).round_ties_even()).max(1.0);
    let (resized_width, resized_height) = (resized(width), resized(height));
    let (gain_x, gain_y) = (
        (resized_width / f64::from(width)) as f32,
        (resized_height / f64::from(height)) as f32,
    );
    let pad_x = ((f64::from(input_width) - resized_width) / 2.0 - 0.1).round_ties_even() as f32;
    let pad_y = ((f64::from(input_height) - resized_height) / 2.0 - 0.1).round_ties_even() as f32;
    let [x1, y1, x2, y2] = bbox;
    [
        ((x1 - pad_x) / gain_x).clamp(0.0, width as f32),
        ((y1 - pad_y) / gain_y).clamp(0.0, height as f32),
        ((x2 - pad_x) / gain_x).clamp(0.0, width as f32),
        ((y2 - pad_y) / gain_y).clamp(0.0, height as f32),
    ]
}

/// `process_mask_native` for one detection: the coefficient-weighted sum of
/// prototypes, upsampled by the factor that crops the letterbox padding on
/// whole pixels, cropped to the content and resized bilinearly to the
/// source, positive logits inside `bbox`. Only the box's pixel span is
/// evaluated, since everything outside it is cropped away. `None` when the
/// mask is empty, as the predictor drops those detections.
fn decode_mask(
    prototypes: &[f32],
    (prototype_width, prototype_height): (usize, usize),
    coefficients: &[f32],
    bbox: [f32; 4],
    (width, height): (u32, u32),
) -> Option<(GrayImage, (u32, u32))> {
    let plane = prototype_width * prototype_height;
    let mut logits = vec![0f32; plane];
    for (coefficient, prototype) in coefficients.iter().zip(prototypes.chunks_exact(plane)) {
        for (logit, value) in logits.iter_mut().zip(prototype) {
            *logit += coefficient * value;
        }
    }

    let input = (4 * prototype_height, 4 * prototype_width);
    let gain = (input.0 as f64 / f64::from(height)).min(input.1 as f64 / f64::from(width));
    let content = |extent: u32| ((f64::from(extent) * gain).round_ties_even() as usize).max(1);
    let pad = (input.0 - content(height)) | (input.1 - content(width));
    let factor = 4 / gcd(4, pad | (pad / 2));
    let (logits, map_width, map_height) = if factor > 1 {
        let (map_width, map_height) = (prototype_width * factor, prototype_height * factor);
        let columns = taps(prototype_width as u32, map_width as u32);
        let rows = taps(prototype_height as u32, map_height as u32);
        let mut upsampled = vec![0f32; map_width * map_height];
        for (row, target) in rows.iter().zip(upsampled.chunks_exact_mut(map_width)) {
            for (column, value) in columns.iter().zip(target) {
                *value = sample(&logits, prototype_width, row, column);
            }
        }
        (upsampled, map_width, map_height)
    } else {
        (logits, prototype_width, prototype_height)
    };

    // scale_masks: crop the letterbox padding, then resize to the source.
    let gain = (map_height as f64 / f64::from(height)).min(map_width as f64 / f64::from(width));
    let content = |extent: u32| ((f64::from(extent) * gain).round_ties_even() as usize).max(1);
    let (content_width, content_height) = (content(width), content(height));
    let left = ((map_width - content_width) as f64 / 2.0 - 0.1).round_ties_even() as usize;
    let top = ((map_height - content_height) as f64 / 2.0 - 0.1).round_ties_even() as usize;
    let cropped = logits
        .chunks_exact(map_width)
        .skip(top)
        .take(content_height)
        .flat_map(|row| &row[left..left + content_width])
        .copied()
        .collect::<Vec<_>>();

    // crop_mask keeps pixel indices i with x1 <= i < x2.
    let span = |low: f32, high: f32, extent: u32| {
        let start = (low.max(0.0).ceil() as u32).min(extent);
        let end = (high.max(0.0).ceil() as u32).min(extent);
        start..end.max(start)
    };
    let (columns_span, rows_span) = (
        span(bbox[0], bbox[2], width),
        span(bbox[1], bbox[3], height),
    );
    let columns = taps(content_width as u32, width);
    let rows = taps(content_height as u32, height);
    let mut mask = GrayImage::new(columns_span.len() as u32, rows_span.len() as u32);
    let mut any = false;
    for (y, source_row) in rows_span.clone().enumerate() {
        for (x, source_column) in columns_span.clone().enumerate() {
            let value = sample(
                &cropped,
                content_width,
                &rows[source_row as usize],
                &columns[source_column as usize],
            );
            if value > 0.0 {
                mask.put_pixel(x as u32, y as u32, Luma([u8::MAX]));
                any = true;
            }
        }
    }
    any.then_some((mask, (columns_span.start, rows_span.start)))
}

fn sample(values: &[f32], width: usize, row: &Tap, column: &Tap) -> f32 {
    let at = |y: usize, x: usize| values[y * width + x];
    let top =
        column.low_weight * at(row.low, column.low) + column.high_weight * at(row.low, column.high);
    let bottom = column.low_weight * at(row.high, column.low)
        + column.high_weight * at(row.high, column.high);
    row.low_weight * top + row.high_weight * bottom
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_pads_to_the_stride_around_the_resized_page() {
        // 2400 x 3400 -> 678 x 960, padded by 26 to 704.
        let letterbox = Letterbox::new(2400, 3400);
        assert_eq!(letterbox.resized, (678, 960));
        assert_eq!(letterbox.input, (704, 960));
        assert_eq!((letterbox.left, letterbox.top), (13, 0));
        assert_eq!(anchors(704, 960), 88 * 120 + 44 * 60 + 22 * 30);
    }

    #[test]
    fn nms_keeps_overlapping_boxes_of_different_classes() {
        let candidate = |bbox, score, class| Candidate {
            bbox,
            score,
            class,
            anchor: 0,
        };
        let candidates = [
            candidate([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            candidate([1.0, 1.0, 11.0, 11.0], 0.8, 0),
            candidate([1.0, 1.0, 11.0, 11.0], 0.7, 1),
        ];
        let kept = nms(&candidates);
        assert_eq!(
            kept.iter().map(|kept| kept.score).collect::<Vec<_>>(),
            [0.9, 0.7]
        );
    }
}
