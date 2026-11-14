//! ogkalu's RT-DETRv2 (ResNet-50-vd) comic text and bubble detector as an ONNX
//! Runtime session.
//!
//! The graph is transformers v4.57.6 `RTDetrV2ForObjectDetection` in eval
//! mode, exported from the pinned checkpoint at a fixed 640x640 input:
//! https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py
//! The checkpoint's own `detector.onnx` comes from the original RT-DETR code
//! and moves boxes by up to a pixel against these weights, so it is not used.

use std::path::Path;

use anyhow::{Result, ensure};
use ort::{session::Session, value::TensorRef};

use crate::{
    Device,
    backend::{Shapes, onnx_session},
};

pub(super) const NUM_LABELS: usize = 3;
pub(super) const NUM_QUERIES: usize = 300;
pub(super) const SIZE: usize = 640;

#[derive(Debug)]
pub(super) struct Output {
    /// Final decoder layer class logits, `[NUM_QUERIES, NUM_LABELS]`.
    pub logits: Vec<f32>,
    /// Final decoder layer boxes as normalized `cx, cy, w, h`, `[NUM_QUERIES, 4]`.
    pub pred_boxes: Vec<f32>,
}

#[derive(Debug)]
pub(super) struct Model {
    session: Session,
}

impl Model {
    pub(super) fn load(device: &Device, path: &Path) -> Result<Self> {
        // FP32 on CUDA, without TF32: BF16 moves boxes by several pixels and
        // drops about 6% of the transformers FP32 detections on the hard-page
        // set, and TF32 still moves boxes by up to 7 px, while FP32 on CUDA
        // matches the CPU to 0.02 px. The 43M-parameter network is cheap at
        // FP32.
        Ok(Self {
            session: onnx_session(device, path, false, Shapes::Fixed)?,
        })
    }

    /// Runs the detector on planar `[1, 3, SIZE, SIZE]` RGB in `[0, 1]`.
    pub(super) fn forward(&mut self, pixel_values: &[f32]) -> Result<Output> {
        let input = TensorRef::from_array_view(([1, 3, SIZE, SIZE], pixel_values))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let (logits_shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        let (boxes_shape, pred_boxes) = outputs["pred_boxes"].try_extract_tensor::<f32>()?;
        ensure!(
            logits_shape.as_ref() == [1, NUM_QUERIES as i64, NUM_LABELS as i64],
            "unexpected comic detector logits shape {logits_shape:?}"
        );
        ensure!(
            boxes_shape.as_ref() == [1, NUM_QUERIES as i64, 4],
            "unexpected comic detector box shape {boxes_shape:?}"
        );
        Ok(Output {
            logits: logits.to_vec(),
            pred_boxes: pred_boxes.to_vec(),
        })
    }
}
