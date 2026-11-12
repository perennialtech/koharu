//! FLUX.2 Klein 4B generation and inpainting.
//!
//! https://github.com/huggingface/diffusers/blob/a37f6f8394ac2a7ee8360c3abea811efe54512b1/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py
//! https://github.com/huggingface/diffusers/blob/a37f6f8394ac2a7ee8360c3abea811efe54512b1/src/diffusers/pipelines/flux2/pipeline_flux2_klein_inpaint.py

mod model;
mod processor;

use anyhow::{Context, Result, ensure};
use fast_image_resize::{
    FilterType, ResizeAlg, ResizeOptions,
    pixels::{U8, U8x3},
};
use image::{DynamicImage, GenericImageView as _, RgbImage};
use koharu_diffusion::{
    GuidanceParams, ImageGenerationParams, SampleMethod, SampleParams, Scheduler,
};

use self::{
    model::{Model, ModelPaths},
    processor::Flux2ImageProcessor,
};

pub use self::processor::{Flux2KleinInpaintOptions, Flux2KleinOptions};

crate::model_repository!("unsloth/FLUX.2-klein-4B-GGUF" @ "0084d1df98e2e2137fe776d55170bc4792ec1d66" {
    TRANSFORMER_WEIGHTS = "flux-2-klein-4b-Q4_K_M.gguf"
});
crate::model_repository!("black-forest-labs/FLUX.2-small-decoder" @ "a3efc24f613ef42d9428af62fdbd6f5fd8856c4a" {
    VAE_WEIGHTS = "full_encoder_small_decoder.safetensors"
});
crate::model_repository!("unsloth/Qwen3-4B-GGUF" @ "22c9fc8a8c7700b76a1789366280a6a5a1ad1120" {
    TEXT_ENCODER_WEIGHTS = "Qwen3-4B-Q4_K_M.gguf"
});

#[derive(Debug)]
pub struct Flux2Klein {
    model: Model,
}

impl Flux2Klein {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let (transformer, text_encoder, vae) = tokio::try_join!(
            TRANSFORMER_WEIGHTS.resolve(),
            TEXT_ENCODER_WEIGHTS.resolve(),
            VAE_WEIGHTS.resolve(),
        )
        .context("failed to resolve FLUX.2 Klein model assets")?;
        let model = Model::new(
            &device,
            ModelPaths {
                transformer,
                text_encoder,
                vae,
            },
        )?;
        Ok(Self { model })
    }

    pub fn inference(
        &self,
        image: &[DynamicImage],
        prompt: &str,
        options: &Flux2KleinOptions,
    ) -> Result<Vec<RgbImage>> {
        ensure!(
            !prompt.contains('\0'),
            "prompt contains an interior NUL byte"
        );
        ensure!(
            options.num_inference_steps > 0,
            "num_inference_steps must be greater than zero"
        );
        ensure!(
            options.num_images_per_prompt > 0,
            "num_images_per_prompt must be greater than zero"
        );

        let mut condition_images = Vec::with_capacity(image.len());
        for image in image {
            Flux2ImageProcessor::check_image_input(image)?;
            let mut image = image.clone();
            if u64::from(image.width()) * u64::from(image.height()) > 1024 * 1024 {
                image = Flux2ImageProcessor::_resize_to_target_area(&image, 1024 * 1024);
            }
            let width = (image.width() / 16) * 16;
            let height = (image.height() / 16) * 16;
            ensure!(width > 0 && height > 0);
            condition_images
                .push(Flux2ImageProcessor::_resize_and_crop(&image, width, height)?.to_rgb8());
        }

        let height = options
            .height
            .or_else(|| condition_images.first().map(RgbImage::height))
            .unwrap_or(1024);
        let width = options
            .width
            .or_else(|| condition_images.first().map(RgbImage::width))
            .unwrap_or(1024);
        let height = (height / 16) * 16;
        let width = (width / 16) * 16;
        ensure!(width > 0 && height > 0);

        self.model.forward(&ImageGenerationParams {
            prompt: prompt.to_owned(),
            width: i32::try_from(width)?,
            height: i32::try_from(height)?,
            reference_images: condition_images,
            reference_image_args: Some("resize_before_vae=0".into()),
            sample: SampleParams {
                guidance: GuidanceParams {
                    text_cfg: 1.0,
                    ..GuidanceParams::default()
                },
                scheduler: Scheduler::Flux2,
                sample_method: SampleMethod::Euler,
                sample_steps: options.num_inference_steps,
                ..SampleParams::default()
            },
            seed: options.seed,
            batch_count: options.num_images_per_prompt,
            ..ImageGenerationParams::default()
        })
    }
}

#[derive(Debug)]
pub struct Flux2KleinInpaint {
    model: Model,
}

impl Flux2KleinInpaint {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let (transformer, text_encoder, vae) = tokio::try_join!(
            TRANSFORMER_WEIGHTS.resolve(),
            TEXT_ENCODER_WEIGHTS.resolve(),
            VAE_WEIGHTS.resolve(),
        )
        .context("failed to resolve FLUX.2 Klein model assets")?;
        let model = Model::new(
            &device,
            ModelPaths {
                transformer,
                text_encoder,
                vae,
            },
        )?;
        Ok(Self { model })
    }

    pub fn inference(
        &self,
        prompt: &str,
        image: &DynamicImage,
        image_reference: Option<&DynamicImage>,
        mask_image: &DynamicImage,
        options: &Flux2KleinInpaintOptions,
    ) -> Result<DynamicImage> {
        ensure!(
            image.width() == mask_image.width() && image.height() == mask_image.height(),
            "image/mask dimensions differ: image={}x{}, mask={}x{}",
            image.width(),
            image.height(),
            mask_image.width(),
            mask_image.height()
        );
        ensure!(
            !prompt.contains('\0'),
            "prompt contains an interior NUL byte"
        );
        ensure!(
            options.strength > 0.0 && options.strength <= 1.0,
            "FLUX.2 inpaint strength must be greater than zero and at most one"
        );
        ensure!(
            options.num_inference_steps > 0,
            "num_inference_steps must be greater than zero"
        );

        if let Some(reference) = image_reference {
            ensure!(
                reference.dimensions() == image.dimensions(),
                "image/reference dimensions differ: image={}x{}, reference={}x{}",
                image.width(),
                image.height(),
                reference.width(),
                reference.height()
            );
        }

        let (mut width, mut height) = image.dimensions();
        if u64::from(width) * u64::from(height) > 1024 * 1024 {
            let scale = (1024.0 * 1024.0 / (f64::from(width) * f64::from(height))).sqrt();
            width = (f64::from(width) * scale) as u32;
            height = (f64::from(height) * scale) as u32;
        }
        let width = (width / 16) * 16;
        let height = (height / 16) * 16;
        ensure!(width > 0 && height > 0);
        let resize = |image: &DynamicImage| {
            let mut image = image.clone();
            if u64::from(image.width()) * u64::from(image.height()) > 1024 * 1024 {
                image = Flux2ImageProcessor::_resize_to_target_area(&image, 1024 * 1024);
            }
            let source = image.to_rgb8();
            let mut resized = RgbImage::new(width, height);
            crate::resize_image::<U8x3>(
                &source,
                &mut resized,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                    .use_alpha(false),
            )
            .expect("source and destination images have the same pixel type");
            resized
        };
        let image = resize(image);
        let source_mask = mask_image.to_luma8();
        let mut mask_image = image::GrayImage::new(width, height);
        crate::resize_image::<U8>(
            &source_mask,
            &mut mask_image,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                .use_alpha(false),
        )
        .expect("source and destination masks have the same pixel type");

        let crop_coords = options.padding_mask_crop.and_then(|padding| {
            Flux2ImageProcessor::get_crop_region(&mask_image, width, height, padding)
        });
        let crop = |image: &RgbImage| match crop_coords {
            Some((x1, y1, x2, y2)) => Flux2ImageProcessor::_resize_and_fill(
                &DynamicImage::ImageRgb8(
                    image::imageops::crop_imm(image, x1, y1, x2 - x1, y2 - y1).to_image(),
                ),
                width,
                height,
            )
            .to_rgb8(),
            None => image.clone(),
        };
        let init_image = crop(&image);
        let mut native_mask = match crop_coords {
            Some((x1, y1, x2, y2)) => Flux2ImageProcessor::_resize_and_fill(
                &DynamicImage::ImageLuma8(
                    image::imageops::crop_imm(&mask_image, x1, y1, x2 - x1, y2 - y1).to_image(),
                ),
                width,
                height,
            )
            .to_luma8(),
            None => mask_image.clone(),
        };
        Flux2ImageProcessor::binarize(&mut native_mask);
        let reference_images = vec![match image_reference {
            Some(reference) => crop(&resize(reference)),
            None => init_image.clone(),
        }];

        let strength = if options.strength >= 1.0 {
            1.0
        } else {
            let effective_steps = options.num_inference_steps
                - (options.num_inference_steps as f64 * (1.0 - options.strength)).floor() as usize;
            let boundary = effective_steps as f32 / options.num_inference_steps as f32;
            f32::from_bits(boundary.to_bits() - 1)
        };

        let generated = self
            .model
            .forward(&ImageGenerationParams {
                prompt: prompt.to_owned(),
                width: i32::try_from(width)?,
                height: i32::try_from(height)?,
                init_image: Some(init_image),
                reference_images,
                reference_image_args: Some("resize_before_vae=0".into()),
                mask_image: Some(native_mask),
                sample: SampleParams {
                    guidance: GuidanceParams {
                        text_cfg: 1.0,
                        ..GuidanceParams::default()
                    },
                    scheduler: Scheduler::Flux2,
                    sample_method: SampleMethod::Euler,
                    sample_steps: i32::try_from(options.num_inference_steps)?,
                    ..SampleParams::default()
                },
                seed: options.seed,
                batch_count: 1,
                strength,
                ..ImageGenerationParams::default()
            })?
            .into_iter()
            .next()
            .context("FLUX.2 Klein returned no inpainted image")?;

        let generated =
            Flux2ImageProcessor::apply_overlay(&mask_image, &image, generated, crop_coords)?;
        Ok(DynamicImage::ImageRgb8(generated))
    }
}
