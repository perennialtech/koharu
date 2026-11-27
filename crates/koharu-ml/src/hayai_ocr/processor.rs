//! Hayai OCR image and text processing.
//!
//! Image preprocessing is Hayai's own `HayaiImageProcessor` with its released
//! settings and the patch budget `HayaiModel.ocr` asks for by default,
//! `max_patches="auto"`, which picks one per crop:
//! https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.7-supernova/blob/b00317ed69fccee5f787187d913fcf8a96f37f65/processing_hayai.py
//! https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.7-supernova/blob/b00317ed69fccee5f787187d913fcf8a96f37f65/preprocessor_config.json

use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result, ensure};
use fast_image_resize::{FilterType, ResizeAlg};
use image::{DynamicImage, RgbImage};
use koharu_torch::{Device, Kind, Tensor};
use serde::Deserialize;

use super::config::siglip2;

/// Pillow's bilinear filter, as convolution.
const RESIZE: ResizeAlg = ResizeAlg::Convolution(FilterType::Bilinear);
/// The fewest patches a crop is upscaled to.
const MIN_PATCHES: i64 = 64;
/// How far a crop is upscaled toward the budget before `MIN_PATCHES` rules.
const MAX_SCALE: f64 = 4.0;

#[derive(Debug)]
pub(super) struct ImageProcessor {
    device: Device,
}

/// One preprocessed crop in NaFlex layout.
#[derive(Debug)]
pub(super) struct ImageInput {
    /// Patchified pixels `(patches, channels * patch_size^2)`; generation
    /// pads them to the batch's longest crop, as upstream does.
    pub(super) pixel_values: Tensor,
    /// Patch-grid dimensions of the image `(height_patches, width_patches)`.
    pub(super) spatial_shape: [i64; 2],
}

impl ImageProcessor {
    pub(super) fn new(device: Device) -> Self {
        Self { device }
    }

    pub(super) fn preprocess(&self, image: &DynamicImage) -> Result<ImageInput> {
        if image.width() == 0 || image.height() == 0 {
            anyhow::bail!("cannot recognize an empty image");
        }

        let patch_size = siglip2::PATCH_SIZE;
        let (height, width) = (i64::from(image.height()), i64::from(image.width()));
        let (height, width) = target_size(height, width, budget(height, width));
        let patch_height = height / patch_size;
        let patch_width = width / patch_size;
        let (height, width) = (height as u32, width as u32);

        // Upstream resizes with Pillow's bilinear filter; `fast_image_resize`
        // uses the same triangle convolution but can differ at coefficient
        // rounding.
        let image = image.to_rgb8();
        let image = if image.width() == width && image.height() == height {
            image
        } else {
            // Match Pillow's horizontal pass and intermediate u8 rounding.
            let options = fast_image_resize::ResizeOptions::new()
                .resize_alg(RESIZE)
                .use_alpha(false);
            let mut horizontal = RgbImage::new(width, image.height());
            let mut resized = RgbImage::new(width, height);
            crate::resize_image::<fast_image_resize::pixels::U8x3>(
                &image,
                &mut horizontal,
                &options,
            )?;
            crate::resize_image::<fast_image_resize::pixels::U8x3>(
                &horizontal,
                &mut resized,
                &options,
            )?;
            resized
        };

        // (height, width, channels) -> (patches, patch_size^2 * channels) with
        // patch order (patch row, patch column, pixel row, pixel column), then
        // [0, 255] to [-1, 1].
        let patch = patch_size;
        let pixel_values = Tensor::f_from_slice(image.as_raw())?
            .f_to_device(self.device)?
            .f_reshape([patch_height, patch, patch_width, patch, 3])?
            .f_permute([0, 2, 1, 3, 4])?
            .f_reshape([patch_height * patch_width, -1])?
            .f_to_kind(Kind::Float)?
            .f_mul_scalar(2.0 / 255.0)?
            .f_sub_scalar(1.0)?;

        Ok(ImageInput {
            pixel_values,
            spatial_shape: [patch_height, patch_width],
        })
    }
}

/// Upstream `_select_best_max_patches`: the patch budget, of 128, 192, 256,
/// 384, and 512, that suits a crop's size and shape, with upstream's
/// thresholds. The smallest budget wins when a larger one gives the same
/// patches, as `MAX_SCALE` caps the crop; a short crop takes 128 or 192 while
/// it's still upscaled enough; a crop that isn't square takes 512 when the 384
/// budget would leave it small, by its shape; and any other crop the 256
/// budget would leave small takes 384.
fn budget(height: i64, width: i64) -> i64 {
    let patch_size = siglip2::PATCH_SIZE;
    let (crop_height, crop_width) = (height.max(1) as f64, width.max(1) as f64);
    // Each budget's patches, and the scale it upsizes the crop by.
    let tier = |budget: i64| -> (i64, f64) {
        let (height, width) = target_size(height, width, budget);
        let patches = height / patch_size * (width / patch_size);
        let scale = (height as f64 / crop_height).min(width as f64 / crop_width);
        (patches, scale)
    };
    let (n128, s128) = tier(128);
    let (n192, s192) = tier(192);
    let (n256, s256) = tier(256);
    let (n384, s384) = tier(384);
    let (n512, _) = tier(512);
    if n128 == n512 {
        return 128;
    }
    let aspect = crop_height.max(crop_width) / crop_height.min(crop_width);
    let wide = crop_width / crop_height;
    let square = (0.77..=1.30).contains(&wide);
    let horizontal = wide > 1.30;
    if !square
        && ((!horizontal && s128 >= 1.65 && aspect < 3.8)
            || (horizontal && s128 >= 2.35 && aspect < 3.2))
    {
        return 128;
    }
    // Upstream nests its 512 rule in the non-square case, so the branch it
    // writes for square crops (`s384 < 0.75`) never runs.
    if !square && (s384 < 1.05 || (aspect >= 5.5 && s384 < 1.55) || (aspect >= 7.5 && s384 < 2.00))
    {
        return 512;
    }
    if n384 > n256
        && (s256 < 1.28
            || (square && s256 < 2.20)
            || (horizontal && s256 < 1.45)
            || (aspect >= 3.8 && s256 < 1.75)
            || (aspect >= 5.0 && s256 < 2.20))
    {
        return 384;
    }
    if n192 == n128 {
        return 128;
    }
    if n256 == n192 {
        return 192;
    }
    if !square
        && ((!horizontal && s192 >= 1.55 && aspect < 3.6)
            || (horizontal && s192 >= 1.85 && aspect < 3.0))
    {
        return 192;
    }
    256
}

/// Upstream `_naflex_target_size`: the largest scale, up to `MAX_SCALE`,
/// whose size rounded up to whole patches fits `budget` patches, and for a
/// crop that leaves under `MIN_PATCHES`, the largest that fits `MIN_PATCHES`.
fn target_size(height: i64, width: i64, budget: i64) -> (i64, i64) {
    const EPS: f64 = 1e-5;
    let patch_size = siglip2::PATCH_SIZE;
    let scaled = |scale: f64, size: i64| -> i64 {
        ((size as f64 * scale / patch_size as f64).ceil() as i64 * patch_size).max(patch_size)
    };
    let search = |mut low: f64, mut high: f64, budget: i64| -> f64 {
        while high - low >= EPS {
            let scale = 0.5 * (low + high);
            let patches = scaled(scale, height) / patch_size * (scaled(scale, width) / patch_size);
            if patches <= budget {
                low = scale;
            } else {
                high = scale;
            }
        }
        low
    };

    let mut scale = search(EPS, MAX_SCALE, budget);
    if scaled(scale, height) / patch_size * (scaled(scale, width) / patch_size) < MIN_PATCHES {
        scale = search(scale, 100.0, MIN_PATCHES.min(budget));
    }
    (scaled(scale, height), scaled(scale, width))
}

#[derive(Debug)]
pub(super) struct Tokenizer {
    pieces: Vec<Vec<u8>>,
    bos_token_id: i64,
    eos_token_id: i64,
    pad_token_id: i64,
}

impl Tokenizer {
    pub(super) fn from_file(path: &Path) -> Result<Self> {
        Self::from_json(&std::fs::read(path)?)
            .with_context(|| format!("failed to parse {}", path.display()))
    }

    // Only decoding is needed. The pinned ByteLevel BPE maps each token to
    // bytes; UTF-8 code points can span any number of generated tokens.
    fn from_json(json: &[u8]) -> Result<Self> {
        #[derive(Deserialize)]
        struct File {
            model: Model,
            decoder: Decoder,
            added_tokens: Vec<AddedToken>,
        }
        #[derive(Deserialize)]
        struct Model {
            r#type: String,
            vocab: HashMap<String, usize>,
        }
        #[derive(Deserialize)]
        struct Decoder {
            r#type: String,
        }
        #[derive(Deserialize)]
        struct AddedToken {
            id: usize,
            content: String,
            special: bool,
        }

        let file: File = serde_json::from_slice(json)?;
        ensure!(
            file.model.r#type == "BPE" && file.decoder.r#type == "ByteLevel",
            "expected Hayai's byte-level BPE vocabulary"
        );
        let mut vocab = file.model.vocab;
        let mut special = HashMap::new();
        for added in file.added_tokens {
            ensure!(added.special, "unexpected non-special added token");
            if let Some(previous) = vocab.insert(added.content.clone(), added.id) {
                ensure!(previous == added.id, "conflicting ID for {}", added.content);
            }
            ensure!(
                special.insert(added.content, added.id).is_none(),
                "duplicate added token"
            );
        }
        let special_id = |token: &str| -> Result<i64> {
            Ok(i64::try_from(*special.get(token).with_context(|| {
                format!("missing Hayai special token {token}")
            })?)?)
        };
        // GPT-2 byte alphabet: printable bytes keep their code points and
        // each remaining byte is assigned a consecutive code point from 256.
        let mut escaped = 256;
        let alphabet: HashMap<char, u8> = (0..=255u8)
            .map(|byte| {
                let ch = match byte {
                    33..=126 | 161..=172 | 174..=255 => char::from(byte),
                    _ => {
                        let ch = char::from_u32(escaped).unwrap();
                        escaped += 1;
                        ch
                    }
                };
                (ch, byte)
            })
            .collect();
        let mut pieces = vec![None; vocab.len()];
        for (token, id) in vocab {
            let slot = pieces
                .get_mut(id)
                .context("non-contiguous Hayai token IDs")?;
            ensure!(slot.is_none(), "duplicate Hayai token ID {id}");
            *slot = Some(if special.contains_key(&token) {
                vec![]
            } else {
                token
                    .chars()
                    .map(|ch| {
                        alphabet
                            .get(&ch)
                            .copied()
                            .context("invalid ByteLevel vocabulary character")
                    })
                    .collect::<Result<_>>()?
            });
        }
        Ok(Self {
            bos_token_id: special_id("<bos>")?,
            eos_token_id: special_id("<eos>")?,
            pad_token_id: special_id("<pad>")?,
            pieces: pieces.into_iter().flatten().collect(),
        })
    }

    pub(super) fn len(&self) -> usize {
        self.pieces.len()
    }

    pub(super) fn bos_token_id(&self) -> i64 {
        self.bos_token_id
    }

    pub(super) fn eos_token_id(&self) -> i64 {
        self.eos_token_id
    }

    pub(super) fn pad_token_id(&self) -> i64 {
        self.pad_token_id
    }

    /// Decodes generated ids, dropping special tokens like the upstream
    /// `generate` cleanup pass.
    pub(super) fn decode(&self, token_ids: &[i64]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in token_ids {
            bytes.extend_from_slice(
                self.pieces
                    .get(usize::try_from(id)?)
                    .context("invalid Hayai token ID")?,
            );
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_decode_joins_split_utf8_and_skips_specials() -> Result<()> {
        let tokenizer = Tokenizer::from_json(
            r#"{
            "model":{"type":"BPE","vocab":{"Ġ":0,"ã":1,"ģ":2,"Ĥ":3,"[UNK]":4}},
            "decoder":{"type":"ByteLevel"},
            "added_tokens":[
                {"id":4,"content":"[UNK]","special":true},
                {"id":5,"content":"<bos>","special":true},
                {"id":6,"content":"<eos>","special":true},
                {"id":7,"content":"<pad>","special":true}
            ]
        }"#
            .as_bytes(),
        )?;
        assert_eq!(tokenizer.decode(&[5, 0, 1, 2, 3, 4, 6, 7])?, " あ");
        assert_eq!(tokenizer.decode(&[1])?, "�");
        assert!(tokenizer.decode(&[8]).is_err());
        Ok(())
    }

    // Expected sizes are upstream `_naflex_target_size`'s, and budgets
    // `_select_best_max_patches`'s, on the same inputs.

    #[test]
    fn resize_budget_matches_upstream_binary_search() {
        // A square image fills the 256-patch budget with a 16x16 patch grid.
        assert_eq!(target_size(1024, 1024, 256), (256, 256));
        // A wide line keeps its rough aspect ratio while rounding up to patch
        // size and staying within the patch budget.
        assert_eq!(target_size(120, 900, 256), (96, 672));
    }

    #[test]
    fn upscaling_stops_at_four_times() {
        assert_eq!(target_size(32, 32, 256), (128, 128));
    }

    #[test]
    fn tiny_images_are_upscaled_to_the_minimum_patch_count() {
        assert_eq!(target_size(4, 4, 256), (128, 128));
    }

    #[test]
    fn each_crop_takes_the_budget_upstream_picks_for_it() {
        // (height, width, budget, resized height, resized width)
        let crops = [
            // Capped at four times either way, so the smallest budget.
            (32, 32, 128, 128, 128),
            // Short vertical dialogue, still well upscaled at 128.
            (150, 50, 128, 288, 96),
            (200, 60, 192, 384, 128),
            (64, 180, 192, 128, 368),
            (60, 200, 256, 144, 448),
            (300, 80, 256, 480, 128),
            // A dense square block and a long strip.
            (1024, 1024, 384, 304, 304),
            (400, 100, 384, 608, 160),
            // Square crops never take 512, however small 384 leaves them.
            (2000, 2000, 384, 304, 304),
            // A wide line, a tall column, and a block 384 leaves small.
            (120, 900, 512, 128, 960),
            (900, 120, 512, 960, 128),
            (700, 300, 512, 544, 240),
        ];
        for (height, width, budget_expected, resized_height, resized_width) in crops {
            let picked = budget(height, width);
            assert_eq!(picked, budget_expected, "{height}x{width}");
            assert_eq!(
                target_size(height, width, picked),
                (resized_height, resized_width),
                "{height}x{width}"
            );
        }
    }
}
