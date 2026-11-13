//! LaMa's FFC generator as an ONNX Runtime session.
//!
//! The graph is advimman/lama's `FFCResNetGenerator` (big-lama.yaml) inside
//! the inference path of `DefaultInpaintingTrainingModule.forward`, exported
//! from mayocream/lama-manga's FP32 weights:
//! https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/modules/ffc.py#L49-L369
//! https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/trainers/default.py#L56-L66
//! FourierUnit's orthonormal rfftn/irfftn run as ONNX Runtime's
//! `com.microsoft` Rfft/Irfft (cuFFT). Those ops have no CPU kernel, so a CPU
//! session fails to load. The export's `export.py` explains how odd FFT widths
//! and cuFFT's handling of non-Hermitian spectra are matched to upstream.

use std::path::Path;

use anyhow::{Result, ensure};
use ort::{session::Session, value::TensorRef};

use crate::{Device, backend::onnx_session};

#[derive(Debug)]
pub(super) struct Model {
    session: Session,
}

impl Model {
    pub(super) fn load(device: &Device, path: &Path) -> Result<Self> {
        // TF32 lowers the PSNR against upstream's FP32 output from 105-120 dB
        // to 55-70 dB at no measurable speed gain.
        Ok(Self {
            session: onnx_session(device, path, false)?,
        })
    }

    /// Runs the generator on planar `[1, 3, height, width]` RGB in `[0, 1]`
    /// and a `[1, 1, height, width]` mask of zeros and ones, with both sides
    /// multiples of 8. Known pixels come back unchanged. Returns planar RGB.
    pub(super) fn forward(
        &mut self,
        image: &[f32],
        mask: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        let image = TensorRef::from_array_view(([1, 3, height, width], image))?;
        let mask = TensorRef::from_array_view(([1, 1, height, width], mask))?;
        let outputs = self.session.run(ort::inputs![image, mask])?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1, 3, height as i64, width as i64],
            "LaMa returned shape {shape:?}"
        );
        Ok(values.to_vec())
    }
}
