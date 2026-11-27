use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::{Context, Result, ensure};
use flate2::bufread::MultiGzDecoder;
use image::{GrayImage, imageops};
use koharu_ml::{
    comic_text_bubble_detector::{ComicTextBubbleDetection, ComicTextBubbleLabel},
    koharu_layout_rfdetr_seg_2xl::{
        KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct RleMask {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    counts: Vec<u32>,
}

impl RleMask {
    pub fn decode(self) -> Result<KoharuLayoutMask> {
        let total = self
            .width
            .checked_mul(self.height)
            .context("mask size overflow")?;
        ensure!(
            self.counts.iter().map(|&run| u64::from(run)).sum::<u64>() == u64::from(total),
            "RLE runs do not match the mask dimensions"
        );
        let mut pixels = Vec::with_capacity(total as usize);
        for (i, run) in self.counts.into_iter().enumerate() {
            pixels.resize(
                pixels.len() + run as usize,
                if i % 2 == 0 { 0 } else { 255 },
            );
        }
        Ok(KoharuLayoutMask {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
            pixels,
        })
    }
}

impl From<&KoharuLayoutMask> for RleMask {
    fn from(mask: &KoharuLayoutMask) -> Self {
        let mut counts = Vec::new();
        let mut foreground = false;
        let mut run = 0;
        for &value in &mask.pixels {
            if (value != 0) != foreground {
                counts.push(run);
                run = 0;
                foreground = !foreground;
            }
            run += 1;
        }
        counts.push(run);
        Self {
            x: mask.x,
            y: mask.y,
            width: mask.width,
            height: mask.height,
            counts,
        }
    }
}

#[derive(Deserialize)]
struct RawRfdetrDetection {
    label_id: usize,
    label: String,
    score: f32,
    bbox: [f32; 4],
    area: u32,
    mask: RleMask,
}

#[derive(Deserialize)]
struct RawOgkaluDetection {
    label: String,
    score: f32,
    bbox: [f32; 4],
}

#[derive(Deserialize)]
pub struct RawDumpLine {
    pub image_id: u64,
    pub file_name: String,
    pub image_width: u32,
    pub image_height: u32,
    rfdetr: Vec<RawRfdetrDetection>,
    ogkalu: Vec<RawOgkaluDetection>,
}

impl RawDumpLine {
    pub fn into_detections(
        self,
        scale: f32,
        (page_w, page_h): (u32, u32),
    ) -> Result<(KoharuLayoutDetections, Vec<ComicTextBubbleDetection>)> {
        let scale_bbox = |bbox: [f32; 4]| {
            if scale == 1.0 {
                return bbox;
            }
            let [x0, y0, x1, y1] = bbox.map(|value| value * scale);
            [
                x0.clamp(0.0, page_w as f32),
                y0.clamp(0.0, page_h as f32),
                x1.clamp(0.0, page_w as f32),
                y1.clamp(0.0, page_h as f32),
            ]
        };
        let detections = self
            .rfdetr
            .into_iter()
            .map(|raw| {
                ensure!(
                    raw.mask.x.checked_add(raw.mask.width).is_some()
                        && raw.mask.y.checked_add(raw.mask.height).is_some(),
                    "mask extent overflow"
                );
                let mut mask = raw.mask.decode()?;
                let mut area = raw.area;
                if scale != 1.0 {
                    let scaled = |value: u32| (value as f32 * scale).round() as u32;
                    let (x, y) = (scaled(mask.x), scaled(mask.y));
                    let width = scaled(mask.x + mask.width) - x;
                    let height = scaled(mask.y + mask.height) - y;
                    let (w, h) = (
                        width.min(page_w.saturating_sub(x)),
                        height.min(page_h.saturating_sub(y)),
                    );
                    mask = if w == 0 || h == 0 {
                        KoharuLayoutMask::default()
                    } else {
                        let source = GrayImage::from_raw(mask.width, mask.height, mask.pixels)
                            .context("invalid raw mask")?;
                        let resized =
                            imageops::resize(&source, width, height, imageops::FilterType::Nearest);
                        let pixels = if (w, h) == (width, height) {
                            resized.into_raw()
                        } else {
                            imageops::crop_imm(&resized, 0, 0, w, h)
                                .to_image()
                                .into_raw()
                        };
                        KoharuLayoutMask {
                            x,
                            y,
                            width: w,
                            height: h,
                            pixels,
                        }
                    };
                    area = mask.pixels.iter().filter(|&&p| p != 0).count() as u32;
                }
                Ok(KoharuLayoutDetection {
                    label_id: raw.label_id,
                    label: raw.label,
                    score: raw.score,
                    bbox: scale_bbox(raw.bbox),
                    area,
                    mask: mask.into(),
                })
            })
            .collect::<Result<_>>()?;
        let comic = self
            .ogkalu
            .into_iter()
            .filter_map(|raw| {
                let label = match raw.label.as_str() {
                    "bubble" => ComicTextBubbleLabel::Bubble,
                    "text_bubble" => ComicTextBubbleLabel::TextBubble,
                    "text_free" => ComicTextBubbleLabel::TextFree,
                    _ => return None,
                };
                Some(ComicTextBubbleDetection {
                    label,
                    score: raw.score,
                    bbox: scale_bbox(raw.bbox),
                })
            })
            .collect();
        Ok((
            KoharuLayoutDetections {
                image_width: page_w,
                image_height: page_h,
                detections,
            },
            comic,
        ))
    }
}

pub fn dump_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let stream = BufReader::new(file);
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
    {
        Ok(Box::new(BufReader::new(MultiGzDecoder::new(stream))))
    } else {
        Ok(Box::new(stream))
    }
}
