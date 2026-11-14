//! RTDetrImageProcessor preprocessing and `post_process_object_detection`.
//!
//! https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/rt_detr/image_processing_rt_detr.py

use anyhow::{Result, anyhow, ensure};
use fast_image_resize::{
    FilterType, ResizeAlg, ResizeOptions,
    pixels::{U8x3, U8x4},
};
use image::{DynamicImage, RgbImage, RgbaImage};
use serde::Serialize;

use super::model::{NUM_LABELS, NUM_QUERIES, Output, SIZE};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComicTextBubbleLabel {
    /// A speech, thought or narration balloon outline.
    Bubble,
    /// Text inside a balloon.
    TextBubble,
    /// Text drawn directly over the artwork.
    TextFree,
}

impl ComicTextBubbleLabel {
    fn from_index(index: usize) -> Result<Self> {
        match index {
            0 => Ok(Self::Bubble),
            1 => Ok(Self::TextBubble),
            2 => Ok(Self::TextFree),
            _ => Err(anyhow!("unexpected comic detector label {index}")),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ComicTextBubbleDetection {
    pub label: ComicTextBubbleLabel,
    pub score: f32,
    /// `left, top, right, bottom` in source image pixels.
    pub bbox: [f32; 4],
}

/// Resizes to 640x640 with PIL-compatible bilinear filtering on 8-bit RGB,
/// then rescales to `[0, 1]` as planar RGB. The checkpoint disables mean/std
/// normalization.
pub(super) fn preprocess(image: &DynamicImage) -> Result<Vec<f32>> {
    ensure!(
        image.width() > 0 && image.height() > 0,
        "cannot detect bubbles in an empty image"
    );
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
    let failed = |error| anyhow!("failed to resize image for bubble detection: {error}");
    let (resized, stride) = match image {
        // The filter runs on each channel independently, so resizing RGBA
        // without alpha weighting and then dropping alpha equals converting
        // with `to_rgb8` first.
        DynamicImage::ImageRgba8(image) => {
            let mut resized = RgbaImage::new(SIZE as u32, SIZE as u32);
            crate::resize_image::<U8x4>(image, &mut resized, &options.use_alpha(false))
                .map_err(failed)?;
            (resized.into_raw(), 4)
        }
        image => {
            let converted;
            let source = match image {
                DynamicImage::ImageRgb8(image) => image,
                image => {
                    converted = image.to_rgb8();
                    &converted
                }
            };
            let mut resized = RgbImage::new(SIZE as u32, SIZE as u32);
            crate::resize_image::<U8x3>(source, &mut resized, &options).map_err(failed)?;
            (resized.into_raw(), 3)
        }
    };
    let plane = SIZE * SIZE;
    let mut pixel_values = vec![0.0; 3 * plane];
    for (index, pixel) in resized.chunks_exact(stride).enumerate() {
        for channel in 0..3 {
            pixel_values[channel * plane + index] = f32::from(pixel[channel]) / 255.0;
        }
    }
    Ok(pixel_values)
}

/// Focal-loss postprocessing: ranks all query/class pairs and keeps those above
/// `threshold`, with boxes scaled to the source image.
pub(super) fn postprocess(
    output: &Output,
    image_width: u32,
    image_height: u32,
    threshold: f32,
) -> Result<Vec<ComicTextBubbleDetection>> {
    ensure!(
        (0.0..=1.0).contains(&threshold),
        "bubble detector threshold must be between 0 and 1"
    );
    // `topk(NUM_QUERIES)` over the flattened query/class scores, best first.
    let scores = output
        .logits
        .iter()
        .map(|&logit| 1.0 / (1.0 + (-logit).exp()))
        .collect::<Vec<f32>>();
    let mut ranked = (0..scores.len()).collect::<Vec<_>>();
    ranked.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
    ranked.truncate(NUM_QUERIES);

    let size = [
        image_width as f32,
        image_height as f32,
        image_width as f32,
        image_height as f32,
    ];
    ranked
        .into_iter()
        .filter(|&index| scores[index] > threshold)
        .map(|index| {
            let query = index / NUM_LABELS;
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
            Ok(ComicTextBubbleDetection {
                label: ComicTextBubbleLabel::from_index(index % NUM_LABELS)?,
                score: scores[index],
                bbox: std::array::from_fn(|axis| corners[axis] * size[axis]),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgba, RgbaImage};

    use super::preprocess;

    #[test]
    fn rgba_preprocessing_matches_rgb() {
        // Downsampled along y and upsampled along x, with alpha that must be
        // ignored.
        let rgba = DynamicImage::ImageRgba8(RgbaImage::from_fn(503, 1777, |x, y| {
            Rgba([
                (x * 7 + y * 3) as u8,
                (x * y) as u8,
                (x ^ y) as u8,
                (x + y) as u8,
            ])
        }));
        let rgb = DynamicImage::ImageRgb8(rgba.to_rgb8());

        assert_eq!(preprocess(&rgba).unwrap(), preprocess(&rgb).unwrap());
    }
}
