//! waifu2x CUNet (art, noise level 0) 2x super-resolution through ONNX Runtime.
//!
//! The model is nunif's own ONNX export, mirrored by DeepGHS:
//! https://github.com/nagadomi/nunif/blob/0.0.0/waifu2x/models/cunet.py
//! The export crops `OFFSET` input pixels from each side of its output, so
//! every tile carries at least that much context around the pixels it keeps.

use anyhow::{Context, Result, ensure};
use image::RgbImage;
use ort::{session::Session, value::TensorRef};

use crate::{Device, backend::onnx_session};

crate::model_repository!("deepghs/waifu2x_onnx" @ "333b95cc88a6a9f39abb6426ab580f0d673f1185" {
    MODEL = "20250502/onnx_models/cunet/art/noise0_scale2x.onnx"
});

pub const SCALE: u32 = 2;
/// Input pixels the network crops from each tile edge.
const OFFSET: usize = 18;
/// Input tile edge, including context. One fixed shape keeps provider
/// workspaces and future engine builds to a single profile.
const TILE: usize = 512;
/// Context kept on each tile edge; must be at least `OFFSET`.
const CONTEXT: usize = 32;
const STEP: usize = TILE - 2 * CONTEXT;

pub struct Waifu2x {
    session: Session,
}

impl Waifu2x {
    pub async fn load(device: Device) -> Result<Self> {
        let model = MODEL
            .resolve()
            .await
            .context("failed to resolve waifu2x model")?;
        let session = tokio_rayon::spawn(move || onnx_session(&device, &model, true))
            .await
            .context("failed to load waifu2x")?;
        Ok(Self { session })
    }

    /// Upscales `image` by [`SCALE`].
    pub fn inference(&mut self, image: &RgbImage) -> Result<RgbImage> {
        let (width, height) = (image.width() as usize, image.height() as usize);
        ensure!(width > 0 && height > 0, "cannot upscale an empty image");
        let scale = SCALE as usize;
        let mut output = RgbImage::new(width as u32 * SCALE, height as u32 * SCALE);
        let mut tile = vec![0.0_f32; 3 * TILE * TILE];
        let kept = STEP * scale;
        let first = (CONTEXT - OFFSET) * scale;
        let produced = (TILE - 2 * OFFSET) * scale;

        for top in (0..height).step_by(STEP) {
            for left in (0..width).step_by(STEP) {
                // Tile pixel (y, x) shows source pixel (top + y - CONTEXT,
                // left + x - CONTEXT), clamped to the image edge.
                for y in 0..TILE {
                    let sy = (top + y).saturating_sub(CONTEXT).min(height - 1);
                    for x in 0..TILE {
                        let sx = (left + x).saturating_sub(CONTEXT).min(width - 1);
                        let pixel = image.get_pixel(sx as u32, sy as u32);
                        for channel in 0..3 {
                            tile[(channel * TILE + y) * TILE + x] =
                                f32::from(pixel[channel]) / 255.0;
                        }
                    }
                }
                let input = TensorRef::from_array_view(([1, 3, TILE, TILE], tile.as_slice()))?;
                let outputs = self.session.run(ort::inputs![input])?;
                let (shape, values) = outputs[0].try_extract_tensor::<f32>()?;
                ensure!(
                    shape.as_ref() == [1, 3, produced as i64, produced as i64],
                    "waifu2x returned shape {shape:?}"
                );
                let rows = kept.min((height - top) * scale);
                let columns = kept.min((width - left) * scale);
                for y in 0..rows {
                    for x in 0..columns {
                        let at = |channel: usize| {
                            let value =
                                values[(channel * produced + first + y) * produced + first + x];
                            (value.clamp(0.0, 1.0) * 255.0).round() as u8
                        };
                        output.put_pixel(
                            (left * scale + x) as u32,
                            (top * scale + y) as u32,
                            image::Rgb([at(0), at(1), at(2)]),
                        );
                    }
                }
            }
        }
        Ok(output)
    }
}
