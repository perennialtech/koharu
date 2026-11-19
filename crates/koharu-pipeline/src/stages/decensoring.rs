use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;
use image::{
    DynamicImage, GenericImageView as _, GrayImage, ImageFormat, Luma, RgbImage, Rgba, RgbaImage,
};
use koharu_ml::{
    kitsumed_yolo11m_seg::{Censor, CensorKind, KitsumedYolo11mSeg},
    waifu_inpaint_xl::WaifuInpaintXl,
};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, Origin, RasterLayer, RasterLayerKind,
};

use super::{
    StageInput, StageProcessor, finish, generation,
    inpainting::{expand_bounds, fixed_size_crop, mask_components},
};
use crate::{Bounds, ModelCell};

const PRODUCER: &str = "dev.koharu.pipeline.decensoring";
const MODEL: &str = "kitsumed-yolo11m-seg+waifu-inpaint-xl";
/// Minimum clearance around detector edges, from the mask coverage sweep in
/// https://github.com/perennialtech/censor-detect-eval.
const MIN_MARGIN: u32 = 16;
const MOSAIC_STRENGTH: f32 = 0.8;

/// Repaints censorship already in the page source: bars and mosaics that
/// kitsumed's YOLO11m-seg finds, filled by Waifu-Inpaint-XL into a
/// decensor layer above the source.
pub(super) struct Processor {
    device: koharu_ml::Device,
    models: ModelCell<Models>,
}

struct Models {
    detector: Arc<Mutex<KitsumedYolo11mSeg>>,
    inpainter: Arc<WaifuInpaintXl>,
}

impl Processor {
    pub(super) fn new(device: koharu_ml::Device) -> Self {
        Self {
            device,
            models: ModelCell::new(),
        }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        MODEL
    }

    fn skip(&self, input: &StageInput) -> Result<bool> {
        Ok(input.decensor_layer()?.is_some())
    }

    fn unload(&self) -> bool {
        self.models.unload()
    }

    async fn load(&self) -> Result<()> {
        self.models
            .ensure(|| async {
                let detector = KitsumedYolo11mSeg::load(self.device.clone()).await?;
                let inpainter = WaifuInpaintXl::load(self.device.clone()).await?;
                Ok(Models {
                    detector: Arc::new(Mutex::new(detector)),
                    inpainter: Arc::new(inpainter),
                })
            })
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let (detector, inpainter) = {
            let models = self.models.lock().await;
            let models = models
                .as_ref()
                .ok_or_else(|| anyhow!("decensoring models are not loaded"))?;
            (models.detector.clone(), models.inpainter.clone())
        };
        let page = input.page;
        let source = input
            .images
            .get(&input.scene, page, "source")
            .await?
            .ok_or_else(|| anyhow!("page {page} has no source image"))?;
        let region = input.region;
        let overlay = tokio_rayon::spawn(move || {
            let image = source.to_rgb8();
            let censors = detector
                .lock()
                .map_err(|_| anyhow!("kitsumed-yolo11m-seg lock is poisoned"))?
                .inference(&image)?;
            repaint_regions(image, &censors, region, |image, mask, strength| {
                inpainter.inference(image, mask, strength)
            })
        })
        .await?;
        let Some(overlay) = overlay else {
            return finish(input.scene.edit());
        };
        let (width, height) = overlay.dimensions();
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(overlay).write_to(&mut bytes, ImageFormat::Png)?;

        let mut edit = input.scene.edit_as(generation(PRODUCER, MODEL)?);
        edit.observe_assets(page)?;
        // Directly over the source and under any cleanup.
        let entity = edit.add_entity(page, At::Start)?;
        edit.set(
            entity,
            &RasterLayer {
                origin: Origin::User,
                name: "Decensor".to_owned(),
                kind: RasterLayerKind::Decensor,
            },
        )?;
        edit.set_asset(
            entity,
            &AssetRole::new("source")?,
            AssetInput::new(
                Arc::<[u8]>::from(bytes.into_inner()),
                "image/png",
                AssetMetadata {
                    width: Some(width),
                    height: Some(height),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        finish(edit)
    }
}

/// Fill the gaps within one detection (including between parallel strips).
/// Its long dimension estimates the scale of the obscured area. Bars give
/// no colour cues, so give them more room than an already broad mosaic.
fn paint_region(areas: &mut GrayImage, censor: &Censor, selection: [u32; 4]) {
    use imageproc::{
        distance_transform::Norm, drawing::draw_polygon_mut, geometry::convex_hull,
        morphology::dilate_mut, point::Point,
    };

    let (w, h) = censor.mask.dimensions();
    let (x, y) = censor.mask_origin;
    // A nearby censor's margin must not start an edit in a clean selection.
    let [sx, sy, ex, ey] = selection;
    let (start_x, start_y) = (sx.max(x), sy.max(y));
    let (end_x, end_y) = (ex.min(x + w), ey.min(y + h));
    if end_x <= start_x || end_y <= start_y {
        return;
    }
    if (start_x, start_y, end_x, end_y) != (x, y, x + w, y + h)
        && !censor
            .mask
            .view(start_x - x, start_y - y, end_x - start_x, end_y - start_y)
            .pixels()
            .any(|(_, _, p)| p[0] != 0)
    {
        return;
    }
    let divisor = match censor.kind {
        CensorKind::Bar => 3,
        CensorKind::Mosaic => 8,
    };
    let margin = (w.max(h) / divisor).clamp(MIN_MARGIN, 128);
    let [left, top, right, bottom] =
        expand_bounds([x, y, x + w, y + h], margin, areas.width(), areas.height());
    let mut mask = GrayImage::new(right - left, bottom - top);
    let (dx, dy) = (x - left, y - top);
    image::imageops::replace(&mut mask, &censor.mask, dx.into(), dy.into());
    // Only the two outermost pixels in each row can be hull vertices.
    let mut points = Vec::new();
    for (row, pixels) in censor.mask.rows().enumerate() {
        let mut xs = pixels
            .enumerate()
            .filter(|(_, p)| p[0] != 0)
            .map(|(x, _)| x);
        if let Some(first) = xs.next() {
            let y = (row as u32 + dy) as i32;
            points.push(Point::new((first as u32 + dx) as i32, y));
            if let Some(last) = xs.next_back() {
                points.push(Point::new((last as u32 + dx) as i32, y));
            }
        }
    }
    let hull = convex_hull(points);
    if hull.len() >= 3 {
        draw_polygon_mut(&mut mask, &hull, Luma([255]));
    }
    dilate_mut(&mut mask, Norm::LInf, margin as u8);
    // Both values are foreground for mask_components. A bar anywhere in a
    // connected region requires generation from noise, including on overlap.
    let mode = match censor.kind {
        CensorKind::Bar => 255,
        CensorKind::Mosaic => 128,
    };
    for y in top.max(sy)..bottom.min(ey) {
        for x in left.max(sx)..right.min(ex) {
            if mask.get_pixel(x - left, y - top)[0] != 0 {
                let target = areas.get_pixel_mut(x, y);
                target[0] = target[0].max(mode);
            }
        }
    }
}

fn repaint_regions(
    mut source: RgbImage,
    censors: &[Censor],
    scope: Option<Bounds>,
    mut generate: impl FnMut(&RgbImage, &GrayImage, f32) -> Result<RgbImage>,
) -> Result<Option<RgbaImage>> {
    if censors.is_empty() {
        return Ok(None);
    }
    let (width, height) = source.dimensions();
    let mut areas = GrayImage::new(width, height);
    let selection = scope.map_or([0, 0, width, height], |b| {
        [
            b.x.floor() as u32,
            b.y.floor() as u32,
            (b.x + b.width).ceil() as u32,
            (b.y + b.height).ceil() as u32,
        ]
    });
    for censor in censors {
        paint_region(&mut areas, censor, selection);
    }
    let (labels, components) = mask_components(&areas);
    if components.is_empty() {
        return Ok(None);
    }
    let mut overlay = RgbaImage::new(width, height);
    for component in components {
        let [x0, y0, x1, y1] = component.bounds;
        // SDXL resizes the crop to a generation bucket. Even a region wider
        // than a LaMa tile fits in one pass, with landmarks at the same scale.
        let context = 128.max((x1 - x0).max(y1 - y0) / 2);
        let bounds = expand_bounds(component.bounds, context, width, height);
        let size = WaifuInpaintXl::crop_size(bounds[2] - bounds[0], bounds[3] - bounds[1]);
        let [left, top, right, bottom] = fixed_size_crop(bounds, size, width, height);
        let crop =
            image::imageops::crop_imm(&source, left, top, right - left, bottom - top).to_image();
        // Hide other pending censors that fall inside this crop's context.
        // Previously repainted regions can already condition the next crop.
        let mask = GrayImage::from_fn(crop.width(), crop.height(), |x, y| {
            Luma([if areas.get_pixel(left + x, top + y)[0] != 0 {
                255
            } else {
                0
            }])
        });
        let has_bar = (y0..y1).any(|y| {
            (x0..x1).any(|x| {
                areas.get_pixel(x, y)[0] == 255 && labels.get_pixel(x, y)[0] == component.label
            })
        });
        let strength = if has_bar { 1.0 } else { MOSAIC_STRENGTH };
        let result = generate(&crop, &mask, strength)?;
        ensure!(
            result.dimensions() == crop.dimensions(),
            "decensored crop dimensions changed"
        );
        for y in y0..y1 {
            for x in x0..x1 {
                if labels.get_pixel(x, y)[0] == component.label {
                    let rgb = *result.get_pixel(x - left, y - top);
                    source.put_pixel(x, y, rgb);
                    areas.put_pixel(x, y, Luma([0]));
                    let [r, g, b] = rgb.0;
                    overlay.put_pixel(x, y, Rgba([r, g, b, 255]));
                }
            }
        }
    }
    Ok(Some(overlay))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn censor(kind: CensorKind, x: u32, y: u32, mask: GrayImage) -> Censor {
        Censor {
            kind,
            score: 1.0,
            bbox: [
                x as f32,
                y as f32,
                (x + mask.width()) as f32,
                (y + mask.height()) as f32,
            ],
            mask,
            mask_origin: (x, y),
        }
    }

    #[test]
    fn broken_bars_and_touching_mosaic_generate_together_even_over_a_tile_wide() {
        let source = RgbImage::new(1200, 500);
        let strips = GrayImage::from_fn(620, 290, |_, y| {
            Luma([if !(4..286).contains(&y) { 255 } else { 0 }])
        });
        let censors = [
            censor(
                CensorKind::Mosaic,
                811,
                205,
                GrayImage::from_pixel(22, 22, Luma([255])),
            ),
            censor(CensorKind::Bar, 100, 150, strips),
            censor(
                CensorKind::Mosaic,
                1060,
                210,
                GrayImage::from_pixel(30, 30, Luma([255])),
            ),
        ];
        let mut strengths = Vec::new();
        let result = repaint_regions(source, &censors, None, |image, mask, strength| {
            strengths.push(strength);
            assert_eq!(mask.dimensions(), image.dimensions());
            if strength == 1.0 {
                assert!(
                    mask.rows()
                        .any(|row| row.filter(|p| p[0] == 255).count() > 620)
                );
            }
            Ok(image.clone())
        })
        .unwrap()
        .unwrap();
        assert_eq!(strengths, [1.0, MOSAIC_STRENGTH]);
        assert_eq!(result.get_pixel(450, 294)[3], 255);
        assert_eq!(result.get_pixel(820, 215)[3], 255);
        assert_eq!(result.get_pixel(950, 215)[3], 0);
        assert_eq!(result.get_pixel(1070, 220)[3], 255);
    }

    #[test]
    fn overlapping_context_hides_pending_bars_then_keeps_finished_pixels() {
        let mask = GrayImage::from_pixel(30, 4, Luma([255]));
        let censors = [
            censor(CensorKind::Bar, 40, 60, mask.clone()),
            censor(CensorKind::Bar, 140, 60, mask),
        ];
        let mut calls = 0;
        repaint_regions(RgbImage::new(220, 130), &censors, None, |crop, mask, _| {
            // Both contexts cover both bars. The left page edge is clipped
            // on the first crop; the second reaches the right page edge.
            let left = if calls == 0 { 0 } else { 220 - crop.width() };
            assert_eq!(mask.get_pixel(145 - left, 61)[0], 255);
            if calls == 0 {
                assert_eq!(mask.get_pixel(45, 61)[0], 255);
            } else {
                assert_eq!(mask.get_pixel(45 - left, 61)[0], 0);
                assert_eq!(crop.get_pixel(45 - left, 61).0, [111; 3]);
            }
            calls += 1;
            Ok(RgbImage::from_pixel(
                crop.width(),
                crop.height(),
                image::Rgb([111; 3]),
            ))
        })
        .unwrap();
        assert_eq!(calls, 2);
    }

    #[test]
    fn region_at_page_edge_respects_fractional_selection() {
        let source = RgbImage::new(200, 150);
        let censors = [censor(
            CensorKind::Mosaic,
            0,
            0,
            GrayImage::from_pixel(120, 75, Luma([255])),
        )];
        let bounds = Bounds {
            x: 10.5,
            y: 12.5,
            width: 20.0,
            height: 30.0,
        };
        let overlay = repaint_regions(
            source.clone(),
            &censors,
            Some(bounds),
            |image, _, strength| {
                assert_eq!(strength, MOSAIC_STRENGTH);
                Ok(image.clone())
            },
        )
        .unwrap()
        .unwrap();
        for (x, y, pixel) in overlay.enumerate_pixels() {
            assert_eq!(
                pixel[3] != 0,
                (10..=30).contains(&x) && (12..=42).contains(&y)
            );
        }
        // Starts beyond the detector but inside its 16 px repaint margin.
        let outside = Bounds { x: 125.0, ..bounds };
        assert!(
            repaint_regions(source, &censors, Some(outside), |_, _, _| panic!(
                "empty selection"
            ))
            .unwrap()
            .is_none()
        );
    }
}
