//! RTDetrImageProcessor preprocessing and `post_process_object_detection`.
//!
//! https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/rt_detr/image_processing_rt_detr.py

use anyhow::{Result, anyhow, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, pixels::U8x3};
use image::{DynamicImage, RgbImage};
use koharu_torch::{Device, IndexOp, Kind, Tensor};
use serde::Serialize;

use super::model::{NUM_LABELS, NUM_QUERIES, Output};

const SIZE: u32 = 640;

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
    fn from_index(index: i64) -> Result<Self> {
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
/// then rescales to `[0, 1]`. The checkpoint disables mean/std normalization.
pub(super) fn preprocess(image: &DynamicImage, device: Device) -> Result<Tensor> {
    ensure!(
        image.width() > 0 && image.height() > 0,
        "cannot detect bubbles in an empty image"
    );
    let image = image.to_rgb8();
    let mut resized = RgbImage::new(SIZE, SIZE);
    crate::resize_image::<U8x3>(
        &image,
        &mut resized,
        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
    )
    .map_err(|error| anyhow!("failed to resize image for bubble detection: {error}"))?;
    Ok(Tensor::from_slice(resized.as_raw())
        .view([1, i64::from(SIZE), i64::from(SIZE), 3])
        .permute([0, 3, 1, 2])
        .to_device(device)
        .to_kind(Kind::Float)
        / 255.0)
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
    ensure!(
        output.logits.size() == [1, NUM_QUERIES, NUM_LABELS],
        "unexpected comic detector logits shape {:?}",
        output.logits.size()
    );
    let (scores, indexes) = output
        .logits
        .sigmoid()
        .view([1, -1])
        .topk(NUM_QUERIES, 1, true, true);
    let keep = scores.i(0).gt(f64::from(threshold)).nonzero().view([-1]);
    let scores = scores.i(0).index_select(0, &keep);
    let indexes = indexes.i(0).index_select(0, &keep);
    let boxes = output
        .pred_boxes
        .i(0)
        .index_select(0, &indexes.floor_divide_scalar(NUM_LABELS));
    let center = boxes.i((.., 0..2));
    let half_size = boxes.i((.., 2..4)) / 2.0;
    let boxes = Tensor::cat(&[&center - &half_size, &center + &half_size], 1).to_kind(Kind::Float)
        * Tensor::from_slice(&[
            image_width as f32,
            image_height as f32,
            image_width as f32,
            image_height as f32,
        ])
        .to_device(boxes.device());

    let scores = to_vec_f32(&scores)?;
    let boxes = to_vec_f32(&boxes)?;
    let labels = to_vec_i64(&indexes.remainder(NUM_LABELS))?;
    scores
        .iter()
        .zip(labels)
        .zip(boxes.as_chunks::<4>().0)
        .map(|((&score, label), &bbox)| {
            Ok(ComicTextBubbleDetection {
                label: ComicTextBubbleLabel::from_index(label)?,
                score,
                bbox,
            })
        })
        .collect()
}

fn to_vec_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    let tensor = tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous();
    let length = tensor.numel();
    let mut values = vec![0.0; length];
    tensor.f_copy_data(&mut values, length)?;
    Ok(values)
}

fn to_vec_i64(tensor: &Tensor) -> Result<Vec<i64>> {
    let tensor = tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64)
        .contiguous();
    let length = tensor.numel();
    let mut values = vec![0; length];
    tensor.f_copy_data(&mut values, length)?;
    Ok(values)
}
