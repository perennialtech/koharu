//! IOPaint-compatible LaMa preprocessing, crop orchestration, and postprocessing.
//!
//! Original implementations:
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/base.py#L57-L192
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/helper.py#L187-L267

use anyhow::{Context, Result, anyhow, ensure};
use fast_image_resize::{
    FilterType, ResizeAlg, ResizeOptions,
    pixels::{U8, U8x3},
};
use image::{DynamicImage, GrayImage, RgbImage};
use imageproc::contours::{BorderType, find_contours_with_threshold};

use super::{
    config::{HDStrategy, InpaintRequest},
    model::Model,
};

pub(super) fn inpaint(
    model: &mut Model,
    image: &DynamicImage,
    mask: &GrayImage,
    config: &InpaintRequest,
) -> Result<RgbImage> {
    let image = image.to_rgb8();
    ensure!(
        image.dimensions() == mask.dimensions(),
        "image and mask dimensions differ: image={:?}, mask={:?}",
        image.dimensions(),
        mask.dimensions()
    );
    ensure!(
        image.width() > 0 && image.height() > 0,
        "image dimensions must be non-zero"
    );

    match config.hd_strategy {
        HDStrategy::Crop
            if image.width().max(image.height()) > config.hd_strategy_crop_trigger_size =>
        {
            let boxes = boxes_from_mask(mask);
            let mut crop_results = Vec::with_capacity(boxes.len());
            for bounding_box in boxes {
                let crop_box = crop_box(
                    image.width(),
                    image.height(),
                    bounding_box,
                    config.hd_strategy_crop_margin,
                );
                let [left, top, right, bottom] = crop_box;
                let crop_image =
                    image::imageops::crop_imm(&image, left, top, right - left, bottom - top)
                        .to_image();
                let crop_mask =
                    image::imageops::crop_imm(mask, left, top, right - left, bottom - top)
                        .to_image();
                crop_results.push((
                    pad_forward(model, &crop_image, &crop_mask, config)?,
                    crop_box,
                ));
            }

            let mut result = image;
            for (crop_result, [left, top, _, _]) in crop_results {
                image::imageops::replace(&mut result, &crop_result, left.into(), top.into());
            }
            Ok(result)
        }
        HDStrategy::Resize
            if image.width().max(image.height()) > config.hd_strategy_resize_limit =>
        {
            let (width, height) = resize_dimensions(
                image.width(),
                image.height(),
                config.hd_strategy_resize_limit,
            );
            let resized_image = resize_rgb(&image, width, height)?;
            let resized_mask = resize_gray(mask, width, height)?;
            let resized_result = pad_forward(model, &resized_image, &resized_mask, config)?;
            let mut result = resize_rgb(&resized_result, image.width(), image.height())?;
            for (index, value) in mask.as_raw().iter().enumerate() {
                if *value < 127 {
                    let offset = index * 3;
                    result.as_flat_samples_mut().samples[offset..offset + 3]
                        .copy_from_slice(&image.as_raw()[offset..offset + 3]);
                }
            }
            Ok(result)
        }
        _ => pad_forward(model, &image, mask, config),
    }
}

fn pad_forward(
    model: &mut Model,
    image: &RgbImage,
    mask: &GrayImage,
    config: &InpaintRequest,
) -> Result<RgbImage> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let padded_height = ceil_modulo(height, 8);
    let padded_width = ceil_modulo(width, 8);
    let rows = (0..padded_height)
        .map(|y| symmetric_index(y, height))
        .collect::<Vec<_>>();
    let columns = (0..padded_width)
        .map(|x| symmetric_index(x, width))
        .collect::<Vec<_>>();
    let plane = padded_height * padded_width;
    let mut model_image = vec![0.0; 3 * plane];
    let mut model_mask = vec![0.0; plane];
    for (y, &source_y) in rows.iter().enumerate() {
        for (x, &source_x) in columns.iter().enumerate() {
            let source = source_y * width + source_x;
            let target = y * padded_width + x;
            for channel in 0..3 {
                model_image[channel * plane + target] =
                    f32::from(image.as_raw()[source * 3 + channel]) / 255.0;
            }
            model_mask[target] = if mask.as_raw()[source] > 0 { 1.0 } else { 0.0 };
        }
    }

    let output = model.forward(&model_image, &model_mask, padded_height, padded_width)?;
    let mut rgb = vec![0; height * width * 3];
    for y in 0..height {
        for x in 0..width {
            let source = y * width + x;
            for channel in 0..3 {
                // Float to uint8 conversion truncates, as torch's `.to(torch.uint8)` does.
                let generated =
                    (output[channel * plane + y * padded_width + x].clamp(0.0, 1.0) * 255.0) as u8;
                rgb[source * 3 + channel] = if config.sd_keep_unmasked_area {
                    let alpha = f32::from(mask.as_raw()[source]) / 255.0;
                    let known = f32::from(image.as_raw()[source * 3 + channel]);
                    (f32::from(generated) * alpha + known * (1.0 - alpha)) as u8
                } else {
                    generated
                };
            }
        }
    }
    RgbImage::from_raw(width as u32, height as u32, rgb)
        .context("failed to convert LaMa output to an RGB image")
}

fn resize_dimensions(width: u32, height: u32, size_limit: u32) -> (u32, u32) {
    let ratio = size_limit as f64 / width.max(height) as f64;
    (
        (width as f64 * ratio + 0.5) as u32,
        (height as f64 * ratio + 0.5) as u32,
    )
}

fn resize_rgb(image: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    let mut output = RgbImage::new(width, height);
    crate::resize_image::<U8x3>(
        image,
        &mut output,
        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom)),
    )
    .map_err(|error| anyhow!("failed to resize LaMa RGB image: {error}"))?;
    Ok(output)
}

fn resize_gray(image: &GrayImage, width: u32, height: u32) -> Result<GrayImage> {
    let mut output = GrayImage::new(width, height);
    crate::resize_image::<U8>(
        image,
        &mut output,
        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom)),
    )
    .map_err(|error| anyhow!("failed to resize LaMa mask: {error}"))?;
    Ok(output)
}

fn boxes_from_mask(mask: &GrayImage) -> Vec<[u32; 4]> {
    let width = mask.width();
    let mut left = width;
    let mut top = mask.height();
    let mut right = 0;
    let mut bottom = 0;
    for y in 0..mask.height() {
        let row = &mask.as_raw()[y as usize * width as usize..(y + 1) as usize * width as usize];
        let Some(row_left) = row.iter().position(|value| *value > 127) else {
            continue;
        };
        let row_right = row
            .iter()
            .rposition(|value| *value > 127)
            .expect("masked row must have a right edge");
        left = left.min(row_left as u32);
        top = top.min(y);
        right = right.max(row_right as u32 + 1);
        bottom = y + 1;
    }
    if right <= left || bottom <= top {
        return Vec::new();
    }

    let cropped_width = right - left;
    let cropped_height = bottom - top;
    let padded_width = cropped_width + 2;
    let mut padded = GrayImage::new(padded_width, cropped_height + 2);
    for y in 0..cropped_height as usize {
        let source_start = (top as usize + y) * width as usize + left as usize;
        let target_start = (y + 1) * padded_width as usize + 1;
        padded.as_flat_samples_mut().samples[target_start..target_start + cropped_width as usize]
            .copy_from_slice(&mask.as_raw()[source_start..source_start + cropped_width as usize]);
    }

    find_contours_with_threshold::<u32>(&padded, 127)
        .into_iter()
        .filter(|contour| contour.border_type == BorderType::Outer && contour.parent.is_none())
        .filter_map(|contour| {
            let mut points = contour.points.into_iter();
            let first = points.next()?;
            let mut contour_left = first.x;
            let mut contour_top = first.y;
            let mut contour_right = first.x;
            let mut contour_bottom = first.y;
            for point in points {
                contour_left = contour_left.min(point.x);
                contour_top = contour_top.min(point.y);
                contour_right = contour_right.max(point.x);
                contour_bottom = contour_bottom.max(point.y);
            }
            Some([
                left + contour_left.saturating_sub(1),
                top + contour_top.saturating_sub(1),
                (left + contour_right).min(mask.width()),
                (top + contour_bottom).min(mask.height()),
            ])
        })
        .filter(|[left, top, right, bottom]| right > left && bottom > top)
        .collect()
}

fn crop_box(
    image_width: u32,
    image_height: u32,
    [left, top, right, bottom]: [u32; 4],
    margin: u32,
) -> [u32; 4] {
    let image_width = i64::from(image_width);
    let image_height = i64::from(image_height);
    let crop_width = i64::from(right - left) + i64::from(margin) * 2;
    let crop_height = i64::from(bottom - top) + i64::from(margin) * 2;
    let center_x = (i64::from(left) + i64::from(right)) / 2;
    let center_y = (i64::from(top) + i64::from(bottom)) / 2;

    let raw_left = center_x - crop_width / 2;
    let raw_right = center_x + crop_width / 2;
    let raw_top = center_y - crop_height / 2;
    let raw_bottom = center_y + crop_height / 2;

    let mut left = raw_left.max(0);
    let mut right = raw_right.min(image_width);
    let mut top = raw_top.max(0);
    let mut bottom = raw_bottom.min(image_height);

    if raw_left < 0 {
        right += -raw_left;
    }
    if raw_right > image_width {
        left -= raw_right - image_width;
    }
    if raw_top < 0 {
        bottom += -raw_top;
    }
    if raw_bottom > image_height {
        top -= raw_bottom - image_height;
    }

    [
        left.clamp(0, image_width) as u32,
        top.clamp(0, image_height) as u32,
        right.clamp(0, image_width) as u32,
        bottom.clamp(0, image_height) as u32,
    ]
}

fn ceil_modulo(value: usize, modulo: usize) -> usize {
    value.div_ceil(modulo) * modulo
}

/// `numpy.pad(mode="symmetric")` source index, as IOPaint's `pad_img_to_modulo`.
fn symmetric_index(index: usize, length: usize) -> usize {
    let index = index % (length * 2);
    if index < length {
        index
    } else {
        length * 2 - index - 1
    }
}
