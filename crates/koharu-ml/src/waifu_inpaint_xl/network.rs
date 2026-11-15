//! The UNet and VAE engines on TensorRT-RTX, and the denoising loop.

use anyhow::{Context, Result};
use image::RgbImage;
use koharu_tensorrt::{Binding, DeviceBuffer, Scratch, Session, ShapeRange, Stream};
use rayon::prelude::*;

use super::scheduler::EulerSchedule;

// Exports of ShinoharaHare/Waifu-Inpaint-XL; the repository holds the export
// script and the graphs' contracts. unet.onnx keeps its weights in
// unet.onnx.data beside it.
crate::model_repository!("8q0sb/koharu-onnx" @ "fda740eafb39fd7f298029531ba0e17c243c0f10" {
    UNET = "waifu-inpaint-xl/unet.onnx",
    UNET_WEIGHTS = "waifu-inpaint-xl/unet.onnx.data",
    VAE_ENCODER = "waifu-inpaint-xl/vae-encoder.onnx",
    VAE_DECODER = "waifu-inpaint-xl/vae-decoder.onnx",
});

const LATENT_CHANNELS: usize = 4;

pub(super) struct Network {
    stream: Stream,
    unet: Session,
    encoder: Session,
    decoder: Session,
    pixels: DeviceBuffer,
    mask: DeviceBuffer,
    masked_image_latents: DeviceBuffer,
    latents: DeviceBuffer,
    timestep: DeviceBuffer,
    prediction: DeviceBuffer,
}

impl Network {
    /// Opens the engines with shape ranges covering `buckets`, building them
    /// on first use.
    pub async fn load(device: u32, buckets: &[(u32, u32)]) -> Result<Self> {
        let (unet, _, encoder, decoder) = tokio::try_join!(
            UNET.resolve(),
            UNET_WEIGHTS.resolve(),
            VAE_ENCODER.resolve(),
            VAE_DECODER.resolve(),
        )
        .context("failed to resolve the Waifu-Inpaint-XL exports")?;
        let buckets = buckets.to_vec();
        tokio_rayon::spawn(move || Self::open(device, &buckets, &unet, &encoder, &decoder)).await
    }

    fn open(
        device: u32,
        buckets: &[(u32, u32)],
        unet: &std::path::Path,
        encoder: &std::path::Path,
        decoder: &std::path::Path,
    ) -> Result<Self> {
        // One profile spans every bucket: latent sides run from the smallest
        // to the largest bucket side, tuned for the square bucket.
        let sides = buckets.iter().flat_map(|&(width, height)| [width, height]);
        let min = i64::from(sides.clone().min().expect("buckets are not empty") / 8);
        let max = i64::from(sides.max().expect("buckets are not empty") / 8);
        let opt = (min + max) / 2;
        let extents = |channels: i64, scale: i64| {
            [min, opt, max].map(|side| [1, channels, side * scale, side * scale])
        };
        let latents = extents(LATENT_CHANNELS as i64, 1);
        let mask = extents(1, 1);
        let image = extents(3, 8);
        let scratch = Scratch::new(device);
        let unet = crate::backend::tensorrt_session(
            device,
            unet,
            "waifu-inpaint-xl",
            false,
            &[
                range("latents", &latents),
                range("mask", &mask),
                range("masked_image_latents", &latents),
            ],
            Some(&scratch),
        )?;
        let encoder = crate::backend::tensorrt_session(
            device,
            encoder,
            "waifu-inpaint-xl",
            false,
            &[range("image", &image)],
            Some(&scratch),
        )?;
        let decoder = crate::backend::tensorrt_session(
            device,
            decoder,
            "waifu-inpaint-xl",
            false,
            &[range("latents", &latents)],
            Some(&scratch),
        )?;

        let pixels = buckets
            .iter()
            .map(|&(width, height)| width as usize * height as usize)
            .max()
            .expect("buckets are not empty");
        let latent_pixels = pixels / 64;
        let floats = |count: usize| DeviceBuffer::new(device, count * size_of::<f32>());
        Ok(Self {
            stream: Stream::new(device)?,
            unet,
            encoder,
            decoder,
            pixels: floats(pixels * 3)?,
            mask: floats(latent_pixels)?,
            masked_image_latents: floats(latent_pixels * LATENT_CHANNELS)?,
            latents: floats(latent_pixels * LATENT_CHANNELS)?,
            timestep: floats(1)?,
            prediction: floats(latent_pixels * LATENT_CHANNELS)?,
        })
    }

    /// Generates `image` with the pixels where `masked` is set repainted,
    /// from pure noise at `strength` 1 or else from `image` noised to the
    /// schedule's step for `strength`. The image's sides are multiples of 8
    /// within the engines' ranges.
    pub fn generate(
        &mut self,
        image: &RgbImage,
        masked: &[bool],
        schedule: &EulerSchedule,
        strength: f32,
        seed: u64,
    ) -> Result<RgbImage> {
        let (width, height) = image.dimensions();
        let (width, height) = (width as usize, height as usize);
        let (latent_width, latent_height) = (width / 8, height / 8);
        let plane = width * height;
        let latent_plane = latent_width * latent_height;
        let image_shape = [1, 3, height as i64, width as i64];
        let latents_shape = [
            1,
            LATENT_CHANNELS as i64,
            latent_height as i64,
            latent_width as i64,
        ];
        let mask_shape = [1, 1, latent_height as i64, latent_width as i64];

        // Planar RGB in [0, 1]; with `mask`, masked pixels are 0 in [-1, 1],
        // 0.5 here, as in the pipeline's masked image.
        let pixels = image.as_raw();
        let mut planes = vec![0f32; plane * 3];
        let fill = |planes: &mut [f32], mask: bool| {
            planes
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(row, target)| {
                    let (channel, y) = (row / height, row % height);
                    for (x, value) in target.iter_mut().enumerate() {
                        let index = y * width + x;
                        *value = if mask && masked[index] {
                            0.5
                        } else {
                            f32::from(pixels[index * 3 + channel]) / 255.0
                        };
                    }
                });
        };
        fill(&mut planes, true);
        self.pixels.upload(&planes, &self.stream)?;
        self.encoder.run(
            &[
                Binding::input("image", &self.pixels, &image_shape),
                Binding::output("latents", &self.masked_image_latents),
            ],
            &self.stream,
        )?;
        let first = schedule.first_step(strength);
        let mut latents = vec![0f32; latent_plane * LATENT_CHANNELS];
        if first > 0 {
            fill(&mut planes, false);
            self.pixels.upload(&planes, &self.stream)?;
            self.encoder.run(
                &[
                    Binding::input("image", &self.pixels, &image_shape),
                    Binding::output("latents", &self.latents),
                ],
                &self.stream,
            )?;
            self.latents.download(&mut latents, &self.stream)?;
        }

        // Nearest-neighbour downscale, as F.interpolate's default mode.
        let mask = (0..latent_plane)
            .map(|index| {
                let (y, x) = (index / latent_width, index % latent_width);
                if masked[y * 8 * width + x * 8] {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<f32>>();
        self.mask.upload(&mask, &self.stream)?;

        let mut noise = vec![0f32; latents.len()];
        super::splitmix_normal(seed, &mut noise);
        if first == 0 {
            let init_noise_sigma = schedule.init_noise_sigma();
            for (latent, noise) in latents.iter_mut().zip(&noise) {
                *latent = noise * init_noise_sigma;
            }
        } else {
            // add_noise: the image latents plus noise at the first sigma run.
            let sigma = schedule.sigma(first);
            for (latent, noise) in latents.iter_mut().zip(&noise) {
                *latent += noise * sigma;
            }
        }
        let mut scaled = vec![0f32; latents.len()];
        let mut prediction = vec![0f32; latents.len()];
        for (index, &timestep) in schedule.timesteps.iter().enumerate().skip(first) {
            let scale = schedule.input_scale(index);
            for (scaled, &latent) in scaled.iter_mut().zip(&latents) {
                *scaled = latent * scale;
            }
            self.latents.upload(&scaled, &self.stream)?;
            self.timestep.upload(&[timestep], &self.stream)?;
            self.unet.run(
                &[
                    Binding::input("latents", &self.latents, &latents_shape),
                    Binding::input("mask", &self.mask, &mask_shape),
                    Binding::input(
                        "masked_image_latents",
                        &self.masked_image_latents,
                        &latents_shape,
                    ),
                    Binding::input("timestep", &self.timestep, &[1]),
                    Binding::output("prediction", &self.prediction),
                ],
                &self.stream,
            )?;
            self.prediction.download(&mut prediction, &self.stream)?;
            schedule.step(index, &prediction, &mut latents);
        }

        self.latents.upload(&latents, &self.stream)?;
        self.decoder.run(
            &[
                Binding::input("latents", &self.latents, &latents_shape),
                Binding::output("image", &self.pixels),
            ],
            &self.stream,
        )?;
        self.pixels.download(&mut planes, &self.stream)?;

        let mut output = RgbImage::new(width as u32, height as u32);
        output
            .as_flat_samples_mut()
            .samples
            .par_chunks_mut(width * 3)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    for channel in 0..3 {
                        let value = planes[channel * plane + y * width + x];
                        row[x * 3 + channel] = (value * 255.0).round() as u8;
                    }
                }
            });
        Ok(output)
    }
}

fn range<'a>(name: &'a str, [min, opt, max]: &'a [[i64; 4]; 3]) -> ShapeRange<'a> {
    ShapeRange {
        name,
        min,
        opt,
        max,
    }
}
