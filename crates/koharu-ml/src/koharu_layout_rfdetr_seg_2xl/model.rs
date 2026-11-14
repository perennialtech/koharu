//! RF-DETR Seg 2XL for the KoharuLayout checkpoint as ONNX Runtime sessions.
//!
//! The graphs follow RF-DETR upstream's export path (`LWDETR.forward_export`
//! and `SegmentationHead.forward_export`) at 1920x1920:
//! https://github.com/roboflow/rf-detr/blob/4ab7c18729de9d02ffd0495795d0831b5630f01b/src/rfdetr/models/lwdetr.py
//! https://github.com/roboflow/rf-detr/blob/4ab7c18729de9d02ffd0495795d0831b5630f01b/src/rfdetr/models/heads/segmentation.py
//! The segmentation head is a second graph: its spatial blocks are
//! query-independent, so only the queries postprocessing keeps are projected
//! to masks instead of all 300.

use std::path::Path;

use anyhow::{Result, ensure};
use ort::{
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{HasSelectedOutputs, OutputSelector, RunOptions, Session},
    value::{Tensor, TensorRef},
};

use crate::{
    Backend, Device,
    backend::{Shapes, onnx_session},
};

pub(super) const RESOLUTION: usize = 1920;
pub(super) const MASK_RESOLUTION: usize = RESOLUTION / 4;
pub(super) const NUM_QUERIES: usize = 300;
pub(super) const NUM_CLASSES_WITH_BACKGROUND: usize = 5;
const HIDDEN_DIM: usize = 256;
const SPATIAL_RESOLUTION: usize = RESOLUTION / 12;

#[derive(Debug)]
pub(super) struct Output {
    /// `[NUM_QUERIES, NUM_CLASSES_WITH_BACKGROUND]`
    pub pred_logits: Vec<f32>,
    /// Normalized `cx, cy, w, h`, `[NUM_QUERIES, 4]`.
    pub pred_boxes: Vec<f32>,
    /// Backbone level-0 features, `[1, HIDDEN_DIM, SPATIAL_RESOLUTION, SPATIAL_RESOLUTION]`,
    /// in device memory on CUDA.
    spatial_features: Tensor<f32>,
    /// Final decoder layer query features, `[NUM_QUERIES, HIDDEN_DIM]`.
    query_features: Vec<f32>,
}

pub(super) struct Model {
    // Declared before the sessions so the device buffer it holds is released
    // before the session that allocated it.
    network_options: RunOptions<HasSelectedOutputs>,
    network: Session,
    mask_head: Session,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Model")
            .field("network", &self.network)
            .field("mask_head", &self.mask_head)
            .finish_non_exhaustive()
    }
}

impl Model {
    pub(super) fn load(device: &Device, network: &Path, mask_head: &Path) -> Result<Self> {
        // FP32 on CUDA. Under BF16 a column box on 144304368_p1 loses a third
        // of its width, so the column is never read, and a panel on the port
        // fixture scores 0.61 instead of 0.95. TF32 is excluded too: it adds
        // and drops detections against FP32 on the hard pages at no
        // measurable speed gain.
        let network = onnx_session(device, network, false, Shapes::Fixed)?;
        // On CUDA the spatial features stay on the device for the mask head
        // instead of a round trip through host memory.
        let mut outputs = OutputSelector::default();
        if device.backend == Backend::Cuda {
            let allocator = Allocator::new(
                &network,
                MemoryInfo::new(
                    AllocationDevice::CUDA,
                    i32::try_from(device.index)?,
                    AllocatorType::Device,
                    MemoryType::Default,
                )?,
            )?;
            outputs = outputs.preallocate(
                "spatial_features",
                Tensor::<f32>::new(
                    &allocator,
                    [1, HIDDEN_DIM, SPATIAL_RESOLUTION, SPATIAL_RESOLUTION],
                )?,
            );
        }
        Ok(Self {
            network_options: RunOptions::new()?.with_outputs(outputs),
            network,
            mask_head: onnx_session(device, mask_head, false, Shapes::Dynamic)?,
        })
    }

    /// Runs the detector on normalized planar `[1, 3, RESOLUTION, RESOLUTION]` RGB.
    pub(super) fn forward(&mut self, pixel_values: &[f32]) -> Result<Output> {
        let input = TensorRef::from_array_view(([1, 3, RESOLUTION, RESOLUTION], pixel_values))?;
        let mut outputs = self
            .network
            .run_with_options(ort::inputs![input], &self.network_options)?;
        let (shape, pred_logits) = outputs["pred_logits"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1, NUM_QUERIES as i64, NUM_CLASSES_WITH_BACKGROUND as i64],
            "unexpected RF-DETR logits shape {shape:?}"
        );
        let pred_logits = pred_logits.to_vec();
        let (shape, pred_boxes) = outputs["pred_boxes"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1, NUM_QUERIES as i64, 4],
            "unexpected RF-DETR box shape {shape:?}"
        );
        let pred_boxes = pred_boxes.to_vec();
        let (shape, query_features) = outputs["query_features"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1, NUM_QUERIES as i64, HIDDEN_DIM as i64],
            "unexpected RF-DETR query feature shape {shape:?}"
        );
        let query_features = query_features.to_vec();
        let spatial_features = outputs
            .remove("spatial_features")
            .expect("network declares spatial_features")
            .downcast::<ort::value::TensorValueType<f32>>()?;
        Ok(Output {
            pred_logits,
            pred_boxes,
            spatial_features,
            query_features,
        })
    }

    /// Mask logits `[queries.len(), MASK_RESOLUTION, MASK_RESOLUTION]` for the
    /// given queries, in order. Each equals that query's mask when all
    /// queries are projected.
    pub(super) fn masks(&mut self, output: &Output, queries: &[usize]) -> Result<Vec<f32>> {
        let mut query_features = Vec::with_capacity(queries.len() * HIDDEN_DIM);
        for &query in queries {
            query_features
                .extend_from_slice(&output.query_features[query * HIDDEN_DIM..][..HIDDEN_DIM]);
        }
        let query_features =
            TensorRef::from_array_view(([1, queries.len(), HIDDEN_DIM], &*query_features))?;
        let outputs = self.mask_head.run(ort::inputs![
            "spatial_features" => &output.spatial_features,
            "query_features" => query_features,
        ])?;
        let (shape, masks) = outputs[0].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref()
                == [
                    1,
                    queries.len() as i64,
                    MASK_RESOLUTION as i64,
                    MASK_RESOLUTION as i64
                ],
            "unexpected RF-DETR mask shape {shape:?}"
        );
        Ok(masks.to_vec())
    }
}
