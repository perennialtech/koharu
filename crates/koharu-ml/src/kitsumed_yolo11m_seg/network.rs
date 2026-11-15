//! The YOLO11m-seg ONNX export on TensorRT-RTX, with one optimization
//! profile over every letterboxed input size.

use std::path::Path;

use anyhow::{Result, ensure};
use koharu_tensorrt::{Binding, DeviceBuffer, Session, ShapeRange, Stream};

use super::processor::{INPUT_SIZE, OUTPUT_ROWS, PROTOTYPES, STRIDE, anchors};

const F32: usize = std::mem::size_of::<f32>();

/// Host copies of the network outputs.
pub(super) struct Outputs {
    /// `output0`, `[OUTPUT_ROWS, anchors]`: box centre and size in input
    /// pixels, class scores, then mask coefficients.
    pub detections: Vec<f32>,
    pub anchors: usize,
    /// `output1`, `[PROTOTYPES, height / 4, width / 4]`.
    pub prototypes: Vec<f32>,
}

/// Field order drops the session and buffers before the stream they ran on.
pub(super) struct Network {
    session: Session,
    images: DeviceBuffer,
    detections: DeviceBuffer,
    prototypes: DeviceBuffer,
    stream: Stream,
}

impl Network {
    pub(super) fn load(device: u32, network: &Path) -> Result<Self> {
        let side = i64::from(INPUT_SIZE);
        let stride = i64::from(STRIDE);
        let session = crate::backend::tensorrt_session(
            device,
            network,
            "kitsumed-yolo11m-seg",
            // The input size follows each page's aspect ratio.
            false,
            &[ShapeRange {
                name: "images",
                min: &[1, 3, stride, stride],
                // A portrait manga page.
                opt: &[1, 3, side, side * 7 / 10 / stride * stride],
                max: &[1, 3, side, side],
            }],
            None,
        )?;
        let (side, prototype_side) = (INPUT_SIZE as usize, INPUT_SIZE as usize / 4);
        let buffer = |floats: usize| DeviceBuffer::new(device, floats * F32);
        Ok(Self {
            session,
            images: buffer(3 * side * side)?,
            detections: buffer(OUTPUT_ROWS * anchors(INPUT_SIZE, INPUT_SIZE))?,
            prototypes: buffer(PROTOTYPES * prototype_side * prototype_side)?,
            stream: Stream::new(device)?,
        })
    }

    /// Runs planar RGB `images` of `(width, height)`.
    pub(super) fn forward(
        &mut self,
        images: &[f32],
        (width, height): (u32, u32),
    ) -> Result<Outputs> {
        ensure!(
            images.len() == 3 * width as usize * height as usize,
            "kitsumed-yolo11m-seg expects {width}x{height} planar RGB"
        );
        self.images.upload(images, &self.stream)?;
        let shape = [1, 3, i64::from(height), i64::from(width)];
        self.session.run(
            &[
                Binding::input("images", &self.images, &shape),
                Binding::output("output0", &self.detections),
                Binding::output("output1", &self.prototypes),
            ],
            &self.stream,
        )?;
        let anchors = anchors(width, height);
        let mut outputs = Outputs {
            detections: vec![0.0; OUTPUT_ROWS * anchors],
            anchors,
            prototypes: vec![0.0; PROTOTYPES * (width as usize / 4) * (height as usize / 4)],
        };
        self.detections
            .download(&mut outputs.detections, &self.stream)?;
        self.prototypes
            .download(&mut outputs.prototypes, &self.stream)?;
        Ok(outputs)
    }
}
