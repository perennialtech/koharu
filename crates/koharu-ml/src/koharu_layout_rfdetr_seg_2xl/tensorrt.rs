//! RF-DETR's network and mask head on TensorRT-RTX.
//!
//! The network graph is the ONNX Runtime export with each attention left as
//! the standard ONNX Attention node that export.py emits before its
//! MultiHeadAttention rewrite. TensorRT imports it as IAttention, which runs
//! fused only in strongly typed networks, as TensorRT-RTX's always are. Query,
//! key and value are FP16 and everything else FP32, the same split as on ONNX
//! Runtime. The mask head is the pinned ONNX Runtime graph, built with one
//! optimization profile over the number of queries.
//!
//! The spatial features stay in device memory between the two, so `masks`
//! reads the features of the latest `forward`.

use std::path::Path;

use anyhow::{Result, ensure};
use koharu_tensorrt::{Binding, DeviceBuffer, Session, ShapeRange, Stream};

use super::model::{
    HIDDEN_DIM, MASK_RESOLUTION, NUM_CLASSES_WITH_BACKGROUND, NUM_QUERIES, RESOLUTION,
    SPATIAL_RESOLUTION,
};

const PIXEL_VALUES: usize = 3 * RESOLUTION * RESOLUTION;
const SPATIAL_FEATURES: usize = HIDDEN_DIM * SPATIAL_RESOLUTION * SPATIAL_RESOLUTION;
const MASK_PIXELS: usize = MASK_RESOLUTION * MASK_RESOLUTION;
const F32: usize = std::mem::size_of::<f32>();

/// Field order drops the buffers and sessions before the stream they ran on.
pub(super) struct Network {
    network: Session,
    mask_head: Session,
    pixel_values: DeviceBuffer,
    pred_logits: DeviceBuffer,
    pred_boxes: DeviceBuffer,
    spatial_features: DeviceBuffer,
    query_features: DeviceBuffer,
    selected_queries: DeviceBuffer,
    /// Grown to the largest query count seen so far.
    mask_logits: DeviceBuffer,
    stream: Stream,
    device: u32,
}

/// Host copies of the network outputs postprocessing reads.
pub(super) struct Outputs {
    pub pred_logits: Vec<f32>,
    pub pred_boxes: Vec<f32>,
    pub query_features: Vec<f32>,
}

impl Network {
    pub(super) fn load(device: u32, network: &Path, mask_head: &Path) -> Result<Self> {
        let stream = Stream::new(device)?;
        let network =
            crate::backend::tensorrt_session(device, network, "koharu-layout-highres", true, &[])?;
        let hidden = HIDDEN_DIM as i64;
        let mask_head = crate::backend::tensorrt_session(
            device,
            mask_head,
            "koharu-layout-highres-mask-head",
            // The query count changes from page to page, and a graph would be
            // captured per count for a head that runs once a page.
            false,
            &[ShapeRange {
                name: "query_features",
                min: &[1, 1, hidden],
                opt: &[1, 32, hidden],
                max: &[1, NUM_QUERIES as i64, hidden],
            }],
        )?;
        let buffer = |floats: usize| DeviceBuffer::new(device, floats * F32);
        Ok(Self {
            network,
            mask_head,
            pixel_values: buffer(PIXEL_VALUES)?,
            pred_logits: buffer(NUM_QUERIES * NUM_CLASSES_WITH_BACKGROUND)?,
            pred_boxes: buffer(NUM_QUERIES * 4)?,
            spatial_features: buffer(SPATIAL_FEATURES)?,
            query_features: buffer(NUM_QUERIES * HIDDEN_DIM)?,
            selected_queries: buffer(NUM_QUERIES * HIDDEN_DIM)?,
            mask_logits: buffer(MASK_PIXELS)?,
            stream,
            device,
        })
    }

    pub(super) fn forward(&mut self, pixel_values: &[f32]) -> Result<Outputs> {
        ensure!(
            pixel_values.len() == PIXEL_VALUES,
            "RF-DETR expects {PIXEL_VALUES} input values, got {}",
            pixel_values.len()
        );
        self.pixel_values.upload(pixel_values, &self.stream)?;
        let shape = [1, 3, RESOLUTION as i64, RESOLUTION as i64];
        self.network.run(
            &[
                Binding::input("pixel_values", &self.pixel_values, &shape),
                Binding::output("pred_logits", &self.pred_logits),
                Binding::output("pred_boxes", &self.pred_boxes),
                Binding::output("spatial_features", &self.spatial_features),
                Binding::output("query_features", &self.query_features),
            ],
            &self.stream,
        )?;
        let mut outputs = Outputs {
            pred_logits: vec![0.0; NUM_QUERIES * NUM_CLASSES_WITH_BACKGROUND],
            pred_boxes: vec![0.0; NUM_QUERIES * 4],
            query_features: vec![0.0; NUM_QUERIES * HIDDEN_DIM],
        };
        self.pred_logits
            .download(&mut outputs.pred_logits, &self.stream)?;
        self.pred_boxes
            .download(&mut outputs.pred_boxes, &self.stream)?;
        self.query_features
            .download(&mut outputs.query_features, &self.stream)?;
        Ok(outputs)
    }

    /// Mask logits for `selected`, the packed `[K, HIDDEN_DIM]` features of
    /// the queries postprocessing keeps, against the spatial features of the
    /// latest `forward`.
    pub(super) fn masks(&mut self, selected: &[f32]) -> Result<Vec<f32>> {
        let queries = selected.len() / HIDDEN_DIM;
        ensure!(
            (1..=NUM_QUERIES).contains(&queries) && selected.len() == queries * HIDDEN_DIM,
            "RF-DETR's mask head takes 1 to {NUM_QUERIES} queries, got {} values",
            selected.len()
        );
        let bytes = queries * MASK_PIXELS * F32;
        if self.mask_logits.len() < bytes {
            self.mask_logits = DeviceBuffer::new(self.device, bytes)?;
        }
        self.selected_queries.upload(selected, &self.stream)?;
        let spatial_shape = [
            1,
            HIDDEN_DIM as i64,
            SPATIAL_RESOLUTION as i64,
            SPATIAL_RESOLUTION as i64,
        ];
        let query_shape = [1, queries as i64, HIDDEN_DIM as i64];
        self.mask_head.run(
            &[
                Binding::input("spatial_features", &self.spatial_features, &spatial_shape),
                Binding::input("query_features", &self.selected_queries, &query_shape),
                Binding::output("mask_logits", &self.mask_logits),
            ],
            &self.stream,
        )?;
        let mut masks = vec![0.0; queries * MASK_PIXELS];
        self.mask_logits.download(&mut masks, &self.stream)?;
        Ok(masks)
    }
}
