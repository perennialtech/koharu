//! LaMa on TensorRT-RTX: one fixed-shape engine per crop shape.
//!
//! The engines come from the same graph as the libtorch port (FFC generator
//! plus the known-pixel blend of `Model::forward`), exported with constant DFT
//! matrices because TensorRT has no FFT. A fixed shape is what makes those
//! matrices constant, so each supported shape is its own ONNX file. BF16
//! everywhere except the DFT products and the blend, which stay FP32, and the
//! upsampling deconvolutions, which run in FP16 because TensorRT-RTX 1.7 has
//! no BF16 deconvolution on Ampere.

use std::{path::Path, sync::Mutex};

use anyhow::{Result, anyhow};
use image::{GrayImage, RgbImage};
use koharu_tensorrt::{Binding, DeviceBuffer, Session, Stream};
use rayon::prelude::*;

/// Engine shapes as (width, height). Pipeline tiles are at most 768 px on a
/// side and most are wider or taller than 512, so the two oblong shapes cut
/// the pixels run per tile by about a seventh against squares alone.
const SHAPES: [(u32, u32); 4] = [(512, 512), (512, 768), (768, 512), (768, 768)];

/// The smallest engine shape that holds a `width` x `height` crop.
pub(super) fn crop_size(width: u32, height: u32) -> Option<(u32, u32)> {
    SHAPES
        .into_iter()
        .filter(|&(engine_width, engine_height)| width <= engine_width && height <= engine_height)
        .min_by_key(|&(engine_width, engine_height)| engine_width * engine_height)
}

pub(super) struct Network {
    state: Mutex<State>,
}

struct State {
    stream: Stream,
    engines: Vec<Engine>,
}

struct Engine {
    shape: (u32, u32),
    session: Session,
    image: DeviceBuffer,
    mask: DeviceBuffer,
    output: DeviceBuffer,
}

impl std::fmt::Debug for Network {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Network").finish_non_exhaustive()
    }
}

impl Network {
    /// Opens `lama-manga-<width>x<height>.onnx` from `directory` for each
    /// shape.
    pub(super) fn load(directory: &Path, device: u32) -> Result<Self> {
        let stream = Stream::new(device)?;
        let engines = SHAPES
            .iter()
            .map(|&(width, height)| {
                let onnx = directory.join(format!("lama-manga-{width}x{height}.onnx"));
                let session = crate::backend::tensorrt_session(device, &onnx, "lama")?;
                let pixels = width as usize * height as usize;
                Ok(Engine {
                    shape: (width, height),
                    session,
                    image: DeviceBuffer::new(device, pixels * 3 * 4)?,
                    mask: DeviceBuffer::new(device, pixels * 4)?,
                    output: DeviceBuffer::new(device, pixels * 3 * 4)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            state: Mutex::new(State { stream, engines }),
        })
    }

    /// Inpaints one crop, matching `InpaintModel::pad_forward` on libtorch
    /// except that the crop is mirrored out to the engine's shape rather than
    /// to the next multiple of 8.
    pub(super) fn forward(
        &self,
        image: &RgbImage,
        mask: &GrayImage,
        keep_unmasked_area: bool,
    ) -> Result<RgbImage> {
        let (width, height) = image.dimensions();
        let Some((engine_width, engine_height)) = crop_size(width, height) else {
            return self.forward_resized(image, mask, keep_unmasked_area);
        };
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("LaMa TensorRT state lock is poisoned"))?;
        let State { stream, engines } = &mut *state;
        let engine = engines
            .iter_mut()
            .find(|engine| engine.shape == (engine_width, engine_height))
            .expect("every shape has an engine");
        let (side_x, side_y) = (engine_width as usize, engine_height as usize);
        let plane = side_x * side_y;

        // Planar RGB in [0, 1] and a binary mask, with symmetric padding as
        // in pad_img_to_modulo.
        let columns = (0..side_x)
            .map(|x| super::processor::symmetric_index(x as u32, width) as usize)
            .collect::<Vec<_>>();
        let pixels = image.as_raw();
        let mask_pixels = mask.as_raw();
        let mut planes = vec![0f32; plane * 3];
        planes
            .par_chunks_mut(side_x)
            .enumerate()
            .for_each(|(row, target)| {
                let (channel, y) = (row / side_y, row % side_y);
                let source_row =
                    super::processor::symmetric_index(y as u32, height) as usize * width as usize;
                for (value, &x) in target.iter_mut().zip(&columns) {
                    *value = f32::from(pixels[(source_row + x) * 3 + channel]) / 255.0;
                }
            });
        let mut binary = vec![0f32; plane];
        binary
            .par_chunks_mut(side_x)
            .enumerate()
            .for_each(|(y, target)| {
                let source_row =
                    super::processor::symmetric_index(y as u32, height) as usize * width as usize;
                for (value, &x) in target.iter_mut().zip(&columns) {
                    *value = if mask_pixels[source_row + x] > 0 {
                        1.0
                    } else {
                        0.0
                    };
                }
            });

        engine.image.upload(&planes, stream)?;
        engine.mask.upload(&binary, stream)?;
        let image_shape = [1, 3, i64::from(engine_height), i64::from(engine_width)];
        let mask_shape = [1, 1, i64::from(engine_height), i64::from(engine_width)];
        engine.session.run(
            &[
                Binding::input("image", &engine.image, &image_shape),
                Binding::input("mask", &engine.mask, &mask_shape),
                Binding::output("output", &engine.output),
            ],
            stream,
        )?;
        engine.output.download(&mut planes, stream)?;

        // Same rounding as the libtorch path: clamp, scale, truncate.
        let mut output = RgbImage::new(width, height);
        output
            .as_flat_samples_mut()
            .samples
            .par_chunks_mut(width as usize * 3)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width as usize {
                    let source = y * side_x + x;
                    let target = y * width as usize + x;
                    let alpha = f32::from(mask_pixels[target]) / 255.0;
                    for channel in 0..3 {
                        let generated = f32::from(
                            (planes[channel * plane + source].clamp(0.0, 1.0) * 255.0) as u8,
                        );
                        row[x * 3 + channel] = if keep_unmasked_area {
                            (generated * alpha
                                + f32::from(pixels[target * 3 + channel]) * (1.0 - alpha))
                                as u8
                        } else {
                            generated as u8
                        };
                    }
                }
            });
        Ok(output)
    }

    /// Only direct callers with large HD-strategy crops get here; pipeline
    /// tiles fit an engine. Runs at the largest shape and keeps the original's
    /// unmasked pixels, as HDStrategy::Resize does.
    fn forward_resized(
        &self,
        image: &RgbImage,
        mask: &GrayImage,
        keep_unmasked_area: bool,
    ) -> Result<RgbImage> {
        let (width, height) = image.dimensions();
        let (largest, _) = SHAPES[SHAPES.len() - 1];
        let (resized_width, resized_height) =
            super::processor::resize_dimensions(width, height, largest);
        let small = self.forward(
            &super::processor::resize_rgb(image, resized_width, resized_height)?,
            &super::processor::resize_gray(mask, resized_width, resized_height)?,
            keep_unmasked_area,
        )?;
        let mut output = super::processor::resize_rgb(&small, width, height)?;
        for (index, value) in mask.as_raw().iter().enumerate() {
            if *value < 127 {
                let offset = index * 3;
                output.as_flat_samples_mut().samples[offset..offset + 3]
                    .copy_from_slice(&image.as_raw()[offset..offset + 3]);
            }
        }
        Ok(output)
    }
}
