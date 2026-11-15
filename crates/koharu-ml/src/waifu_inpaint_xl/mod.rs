//! Waifu-Inpaint-XL: SDXL inpainting fine-tuned on anime illustrations.
//!
//! https://huggingface.co/ShinoharaHare/Waifu-Inpaint-XL
//!
//! Follows diffusers' `StableDiffusionXLInpaintPipeline` for a 9-channel UNet:
//! https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py
//! The UNet, guidance and prompt conditioning run as one TensorRT-RTX engine
//! (the export fixes the prompt; see the exports' README). Two divergences:
//! the VAE encoder returns the posterior mean rather than a sample, and the
//! initial noise comes from [`splitmix_normal`] so a reference run can
//! reproduce it.

mod network;
mod scheduler;

use std::sync::Mutex;

use anyhow::{Result, anyhow, ensure};
use fast_image_resize::{
    FilterType, ResizeAlg, ResizeOptions,
    pixels::{U8, U8x3},
};
use image::{GrayImage, RgbImage};

use self::{network::Network, scheduler::EulerSchedule};

/// Generation sizes as (width, height): SDXL's ~1 MP training buckets up to
/// an aspect ratio of about 3:2. Every crop is generated at ~1 MP whatever its
/// size in the page; at small native sizes the model paints shapes and glyphs
/// into masks. Wider buckets are omitted to keep the VAE's per-run activation
/// memory down.
const BUCKETS: [(u32, u32); 5] = [
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
];
/// Euler steps over the whole schedule; 28 (the model card's) looked the
/// same.
const STEPS: usize = 20;
const SEED: u64 = 42;

pub struct WaifuInpaintXl {
    network: Mutex<Network>,
}

impl std::fmt::Debug for WaifuInpaintXl {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WaifuInpaintXl")
            .finish_non_exhaustive()
    }
}

impl WaifuInpaintXl {
    /// Loads the engines on a CUDA device; TensorRT-RTX must be initialized.
    pub async fn load(device: crate::Device) -> Result<Self> {
        ensure!(
            device.backend == crate::Backend::Cuda && crate::initialized(crate::Feature::TensorRt),
            "Waifu-Inpaint-XL needs a CUDA device with TensorRT-RTX"
        );
        let network = Network::load(u32::try_from(device.index)?, &BUCKETS).await?;
        Ok(Self {
            network: Mutex::new(network),
        })
    }

    /// A crop of at least `width` x `height` with the aspect ratio of the
    /// bucket it will be generated at, so a caller with more image around
    /// the crop can grow it instead of having it stretched.
    pub fn crop_size(width: u32, height: u32) -> (u32, u32) {
        let (bucket_width, bucket_height) = bucket(width, height);
        if u64::from(width) * u64::from(bucket_height)
            >= u64::from(height) * u64::from(bucket_width)
        {
            (width, (width * bucket_height).div_ceil(bucket_width))
        } else {
            ((height * bucket_width).div_ceil(bucket_height), height)
        }
    }

    /// Repaints the pixels where `mask` is at least 127 and returns the image
    /// at its own size. At `strength` 1 they are generated from noise; below
    /// it, from the image noised partway, which keeps more of what is there.
    /// Unmasked pixels come back through the VAE, so callers composite only
    /// the masked ones.
    pub fn inference(&self, image: &RgbImage, mask: &GrayImage, strength: f32) -> Result<RgbImage> {
        ensure!(
            strength > 0.0 && strength <= 1.0,
            "strength must be in (0, 1], got {strength}"
        );
        ensure!(
            image.dimensions() == mask.dimensions(),
            "image/mask dimensions differ: image={:?}, mask={:?}",
            image.dimensions(),
            mask.dimensions()
        );
        let (width, height) = image.dimensions();
        let (bucket_width, bucket_height) = bucket(width, height);
        let options = ResizeOptions::new()
            .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
            .use_alpha(false);
        let mut resized = RgbImage::new(bucket_width, bucket_height);
        crate::resize_image::<U8x3>(image, &mut resized, &options)
            .expect("source and destination images have the same pixel type");
        let mut resized_mask = GrayImage::new(bucket_width, bucket_height);
        crate::resize_image::<U8>(mask, &mut resized_mask, &options)
            .expect("source and destination masks have the same pixel type");
        let masked = resized_mask
            .as_raw()
            .iter()
            .map(|&value| value >= 127)
            .collect::<Vec<_>>();

        let schedule = EulerSchedule::new(STEPS);
        let generated = self
            .network
            .lock()
            .map_err(|_| anyhow!("Waifu-Inpaint-XL lock is poisoned"))?
            .generate(&resized, &masked, &schedule, strength, SEED)?;

        if generated.dimensions() == (width, height) {
            return Ok(generated);
        }
        let mut output = RgbImage::new(width, height);
        crate::resize_image::<U8x3>(&generated, &mut output, &options)
            .expect("source and destination images have the same pixel type");
        Ok(output)
    }
}

/// The bucket whose aspect ratio is closest to `width` / `height`.
fn bucket(width: u32, height: u32) -> (u32, u32) {
    let aspect = (f64::from(width) / f64::from(height)).ln();
    BUCKETS
        .into_iter()
        .min_by(|a, b| {
            let distance = |(w, h): (u32, u32)| (aspect - (f64::from(w) / f64::from(h)).ln()).abs();
            distance(*a).total_cmp(&distance(*b))
        })
        .expect("BUCKETS is not empty")
}

/// Standard normals from a counter-based generator: splitmix64 outputs
/// `2i + 1` and `2i + 2` feed one Box-Muller pair, whose cosine branch is
/// element `2i` and sine branch element `2i + 1`.
fn splitmix_normal(seed: u64, output: &mut [f32]) {
    let unit = |index: u64| {
        let mut z = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * 2f64.powi(-53)
    };
    for (pair, values) in output.chunks_mut(2).enumerate() {
        let pair = pair as u64;
        let radius = (-2.0 * (-unit(2 * pair + 1)).ln_1p()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * unit(2 * pair + 2);
        values[0] = (radius * angle.cos()) as f32;
        if let Some(value) = values.get_mut(1) {
            *value = (radius * angle.sin()) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crops_take_the_nearest_bucket_aspect() {
        assert_eq!(WaifuInpaintXl::crop_size(600, 600), (600, 600));
        // 700 x 400 is closest to 1216 x 832 and grows in height.
        assert_eq!(WaifuInpaintXl::crop_size(700, 400), (700, 479));
        assert_eq!(WaifuInpaintXl::crop_size(300, 500), (343, 500));
    }

    // From numpy, as in the exports' reference.py.
    #[test]
    fn noise_matches_reference() {
        let mut noise = [0f32; 5];
        splitmix_normal(42, &mut noise);
        assert_eq!(noise, EXPECTED_NOISE);
    }

    const EXPECTED_NOISE: [f32; 5] = [
        0.882_248_9,
        1.388_473_3,
        -0.450_849_86,
        0.670_716_46,
        0.188_352_63,
    ];
}
