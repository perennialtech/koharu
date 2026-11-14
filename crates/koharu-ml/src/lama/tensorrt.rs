//! LaMa on TensorRT-RTX: one fixed-shape engine per crop size.
//!
//! The engines come from the same graph as the libtorch port (FFC generator
//! plus the known-pixel blend of `Model::forward`), exported with constant DFT
//! matrices because TensorRT has no FFT. A fixed shape is what makes those
//! matrices constant, so each supported size is its own ONNX file. BF16
//! everywhere except the DFT products and the blend, which stay FP32, and the
//! upsampling deconvolutions, which run in FP16 because TensorRT-RTX 1.7 has
//! no BF16 deconvolution on Ampere.

use std::{path::Path, sync::Mutex};

use anyhow::{Result, anyhow};
use image::{GrayImage, RgbImage};
use koharu_tensorrt::{Binding, DeviceBuffer, Session, Stream};

/// Engine sizes, smallest first. Every pipeline crop fits the largest, and
/// a crop takes the smallest that fits it.
const SIZES: [u32; 2] = [512, 768];

/// The smallest engine size that holds a `width` x `height` crop.
pub(super) fn crop_size(width: u32, height: u32) -> Option<u32> {
    SIZES.into_iter().find(|size| width.max(height) <= *size)
}

pub(super) struct Network {
    state: Mutex<State>,
}

struct State {
    stream: Stream,
    engines: Vec<Engine>,
}

struct Engine {
    size: u32,
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
    /// Opens `lama-manga-<size>.onnx` from `directory` for each size.
    pub(super) fn load(directory: &Path, device: u32) -> Result<Self> {
        let stream = Stream::new(device)?;
        let engines = SIZES
            .iter()
            .map(|&size| {
                let onnx = directory.join(format!("lama-manga-{size}.onnx"));
                let session = crate::backend::tensorrt_session(device, &onnx, "lama", true, &[])?;
                let pixels = size as usize * size as usize;
                Ok(Engine {
                    size,
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
    /// except that the crop is mirrored out to the engine's size rather than
    /// to the next multiple of 8.
    pub(super) fn forward(
        &self,
        image: &RgbImage,
        mask: &GrayImage,
        keep_unmasked_area: bool,
    ) -> Result<RgbImage> {
        let (width, height) = image.dimensions();
        let largest = SIZES[SIZES.len() - 1];
        if width.max(height) > largest {
            // Only direct callers with large HD-strategy crops get here; the
            // pipeline's crops fit. Run at the largest size and keep the
            // original's unmasked pixels, as HDStrategy::Resize does.
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
            return Ok(output);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("LaMa TensorRT state lock is poisoned"))?;
        let State { stream, engines } = &mut *state;
        let size = crop_size(width, height).expect("the largest engine fits the crop");
        let engine = engines
            .iter_mut()
            .find(|engine| engine.size == size)
            .expect("every size has an engine");
        let side = size as usize;

        // Planar RGB in [0, 1] and a binary mask, with symmetric padding as
        // in pad_img_to_modulo.
        let rows = (0..side).map(|y| super::processor::symmetric_index(y as u32, height) as usize);
        let columns = (0..side)
            .map(|x| super::processor::symmetric_index(x as u32, width) as usize)
            .collect::<Vec<_>>();
        let mut planes = vec![0f32; side * side * 3];
        let mut binary = vec![0f32; side * side];
        let pixels = image.as_raw();
        let mask_pixels = mask.as_raw();
        for (y, source_y) in rows.enumerate() {
            for (x, &source_x) in columns.iter().enumerate() {
                let source = source_y * width as usize + source_x;
                let target = y * side + x;
                for channel in 0..3 {
                    planes[channel * side * side + target] =
                        f32::from(pixels[source * 3 + channel]) / 255.0;
                }
                binary[target] = if mask_pixels[source] > 0 { 1.0 } else { 0.0 };
            }
        }

        engine.image.upload(&planes, stream)?;
        engine.mask.upload(&binary, stream)?;
        let image_shape = [1, 3, i64::from(size), i64::from(size)];
        let mask_shape = [1, 1, i64::from(size), i64::from(size)];
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
        let samples = output.as_flat_samples_mut().samples;
        for y in 0..height as usize {
            for x in 0..width as usize {
                let source = y * side + x;
                let target = y * width as usize + x;
                let alpha = f32::from(mask_pixels[target]) / 255.0;
                for channel in 0..3 {
                    let generated = f32::from(
                        (planes[channel * side * side + source].clamp(0.0, 1.0) * 255.0) as u8,
                    );
                    samples[target * 3 + channel] = if keep_unmasked_area {
                        (generated * alpha
                            + f32::from(pixels[target * 3 + channel]) * (1.0 - alpha))
                            as u8
                    } else {
                        generated as u8
                    };
                }
            }
        }
        Ok(output)
    }
}
