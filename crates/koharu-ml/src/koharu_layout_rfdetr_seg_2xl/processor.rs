//! RF-DETR image preprocessing and instance segmentation postprocessing.
//!
//! https://github.com/roboflow/rf-detr/blob/4ab7c18729de9d02ffd0495795d0831b5630f01b/src/rfdetr/detr.py
//! https://github.com/roboflow/rf-detr/blob/4ab7c18729de9d02ffd0495795d0831b5630f01b/src/rfdetr/models/postprocess.py

use anyhow::{Result, ensure};
use image::DynamicImage;
use serde::Serialize;

use super::{
    config::{KoharuLayoutRFDetrSeg2XLConfig, KoharuLayoutThresholds},
    model::{MASK_RESOLUTION, Model, NUM_CLASSES_WITH_BACKGROUND, NUM_QUERIES, Output, RESOLUTION},
};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone)]
pub struct KoharuLayoutRFDetrImageProcessor {
    num_select: usize,
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
            class_names: config.class_names(),
            recommended_thresholds: config.recommended_thresholds,
        })
    }

    pub(super) fn recommended_thresholds(&self) -> KoharuLayoutThresholds {
        self.recommended_thresholds
    }

    /// Rescales to `[0, 1]`, resizes to the model resolution with bilinear
    /// interpolation, then normalizes, as planar RGB. Latest upstream disables
    /// antialiasing to match the bilinear Albumentations resize used during
    /// training.
    pub(super) fn preprocess(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        ensure!(
            image.width() > 0 && image.height() > 0,
            "cannot segment an empty image"
        );
        let image = image.to_rgb8();
        let (width, height) = (image.width() as usize, image.height() as usize);
        let rows = bilinear_taps(height, RESOLUTION);
        let columns = bilinear_taps(width, RESOLUTION);
        let source = |y: usize, x: usize, channel: usize| {
            f32::from(image.as_raw()[(y * width + x) * 3 + channel]) / 255.0
        };
        let plane = RESOLUTION * RESOLUTION;
        let mut pixel_values = vec![0.0; 3 * plane];
        for (y, row) in rows.iter().enumerate() {
            for (x, column) in columns.iter().enumerate() {
                for channel in 0..3 {
                    let value = row.interpolate(|source_y| {
                        column.interpolate(|source_x| source(source_y, source_x, channel))
                    });
                    pixel_values[channel * plane + y * RESOLUTION + x] =
                        (value - MEAN[channel]) / STD[channel];
                }
            }
        }
        Ok(pixel_values)
    }

    pub(super) fn postprocess(
        &self,
        model: &mut Model,
        output: &Output,
        image_width: u32,
        image_height: u32,
        thresholds: KoharuLayoutThresholds,
    ) -> Result<KoharuLayoutDetections> {
        validate_thresholds(thresholds)?;
        // PostProcess ranks every query/class pair, including the checkpoint's
        // fifth background logit slot, before applying the caller threshold.
        let scores = output
            .pred_logits
            .iter()
            .map(|&logit| 1.0 / (1.0 + (-logit).exp()))
            .collect::<Vec<f32>>();
        let mut ranked = (0..scores.len()).collect::<Vec<_>>();
        ranked.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
        ranked.truncate(self.num_select);
        let thresholds = thresholds.with_background();
        let selected = ranked
            .into_iter()
            .filter(|&index| scores[index] > thresholds[index % NUM_CLASSES_WITH_BACKGROUND])
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return Ok(KoharuLayoutDetections {
                image_width,
                image_height,
                detections: Vec::new(),
            });
        }

        let queries = selected
            .iter()
            .map(|index| index / NUM_CLASSES_WITH_BACKGROUND)
            .collect::<Vec<_>>();
        let masks = model.masks(output, &queries)?;
        let rows = bilinear_taps(MASK_RESOLUTION, image_height as usize);
        let columns = bilinear_taps(MASK_RESOLUTION, image_width as usize);
        let size = [
            image_width as f32,
            image_height as f32,
            image_width as f32,
            image_height as f32,
        ];
        let detections = selected
            .iter()
            .zip(&queries)
            .zip(masks.as_chunks::<{ MASK_RESOLUTION * MASK_RESOLUTION }>().0)
            .map(|((&index, &query), logits)| {
                let label_id = index % NUM_CLASSES_WITH_BACKGROUND;
                let [center_x, center_y, width, height]: [f32; 4] = output.pred_boxes
                    [query * 4..query * 4 + 4]
                    .try_into()
                    .expect("box has four coordinates");
                let half_width = width / 2.0;
                let half_height = height / 2.0;
                let corners = [
                    center_x - half_width,
                    center_y - half_height,
                    center_x + half_width,
                    center_y + half_height,
                ];
                let (area, mask) = page_mask(logits, &rows, &columns);
                KoharuLayoutDetection {
                    label_id,
                    label: self
                        .class_names
                        .get(label_id)
                        .cloned()
                        .unwrap_or_else(|| "__background__".to_owned()),
                    score: scores[index],
                    bbox: std::array::from_fn(|axis| corners[axis] * size[axis]),
                    area,
                    mask,
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
    let mut bounds: Option<[usize; 4]> = None;
    for (y, row) in logits.as_chunks::<MASK_RESOLUTION>().0.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
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
    let first_column = columns
        .iter()
        .position(|tap| touches(tap, left, right))
        .unwrap();
    let last_column = columns
        .iter()
        .rposition(|tap| touches(tap, left, right))
        .unwrap();

    let window_width = last_column - first_column + 1;
    let mut window = vec![false; (last_row - first_row + 1) * window_width];
    let (mut area, mut occupied) = (0_u32, None::<[usize; 4]>);
    for (y, row) in rows[first_row..=last_row].iter().enumerate() {
        let low = &logits[row.low * MASK_RESOLUTION..][..MASK_RESOLUTION];
        let high = &logits[row.high * MASK_RESOLUTION..][..MASK_RESOLUTION];
        for (x, column) in columns[first_column..=last_column].iter().enumerate() {
            let value = row.low_weight * column.interpolate(|source| low[source])
                + row.high_weight * column.interpolate(|source| high[source]);
            if value > 0.0 {
                window[y * window_width + x] = true;
                area += 1;
                let [left, top, right, bottom] = occupied.get_or_insert([x, y, x, y]);
                *left = (*left).min(x);
                *top = (*top).min(y);
                *right = (*right).max(x);
                *bottom = (*bottom).max(y);
            }
        }
    }
    let Some([left, top, right, bottom]) = occupied else {
        return (0, empty);
    };
    let width = right - left + 1;
    let mut pixels = Vec::with_capacity(width * (bottom - top + 1));
    for y in top..=bottom {
        pixels.extend(
            window[y * window_width + left..][..width]
                .iter()
                .map(|&set| if set { u8::MAX } else { 0 }),
        );
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
    pub mask: KoharuLayoutMask,
}

#[derive(Debug, Clone)]
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
    use super::{
        KoharuLayoutThresholds, MASK_RESOLUTION, bilinear_taps, page_mask, validate_thresholds,
    };

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
        // values elsewhere, upsampled to a page that is not a multiple of the
        // mask grid on either axis.
        let mut logits = vec![-1.0_f32; MASK_RESOLUTION * MASK_RESOLUTION];
        for y in 100..140 {
            for x in 200..260 {
                logits[y * MASK_RESOLUTION + x] = ((x + y) % 7) as f32 - 2.5;
            }
        }
        logits[MASK_RESOLUTION - 1] = 0.3;
        let (width, height) = (1013, 1517);
        let rows = bilinear_taps(MASK_RESOLUTION, height);
        let columns = bilinear_taps(MASK_RESOLUTION, width);

        let mut expected = vec![false; width * height];
        for (y, row) in rows.iter().enumerate() {
            for (x, column) in columns.iter().enumerate() {
                let value = row.interpolate(|source_y| {
                    column.interpolate(|source_x| logits[source_y * MASK_RESOLUTION + source_x])
                });
                expected[y * width + x] = value > 0.0;
            }
        }
        let (area, mask) = page_mask(&logits, &rows, &columns);

        assert_eq!(area as usize, expected.iter().filter(|&&set| set).count());
        for y in 0..height {
            for x in 0..width {
                assert_eq!(
                    mask.contains(x as u32, y as u32),
                    expected[y * width + x],
                    "pixel {x}, {y}"
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
            (mask.x as usize, (mask.x + mask.width - 1) as usize)
        );
        assert_eq!(
            occupied(&|index| index / width),
            (mask.y as usize, (mask.y + mask.height - 1) as usize)
        );
    }
}
