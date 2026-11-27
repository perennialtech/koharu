use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use image::{GrayImage, RgbImage};
use koharu_pipeline::ReplayedDetectionPage;
use koharu_rasterizer::{PreparedContent, PreparedSceneCommand, RasterOptions, Rasterizer};
use koharu_renderer::{LayerKind, Renderer, TypesettingConfig};
use koharu_scene::{
    At, Authored, EntityId, Geometry, LanguageTag, PageDraft, Session, SourceText, Translation,
};
use serde::Serialize;
use tokio::runtime::Handle;

pub struct RenderAuditor {
    runtime: Handle,
    engine: Mutex<(Renderer, Rasterizer)>,
}

#[derive(Serialize)]
pub struct RenderAudit {
    profiles: Profiles,
}

#[derive(Serialize)]
struct Profiles {
    short: ProfileMetrics,
    typical: ProfileMetrics,
    long: ProfileMetrics,
}

#[derive(Default, Serialize)]
struct ProfileMetrics {
    overflow_ratio_mean: f64,
    overflow_ratio_max: f64,
    overflow_regions_gt5pct: usize,
    collision_ratio_max: f64,
    collision_pairs_gt1pct: usize,
    missing_glyph_regions: usize,
    clipped_regions: usize,
    min_font_px: Option<f32>,
    min_local_contrast: Option<f64>,
}

struct Sample {
    layer: EntityId,
    content: EntityId,
    balloon: Option<Geometry>,
    typical: &'static str,
    paper: [u8; 3],
}

struct Ink {
    pixels: GrayImage,
    left: i32,
    top: i32,
    area: u32,
}

impl RenderAuditor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            runtime: Handle::current(),
            engine: Mutex::new((
                // Use production font fallback with the locally installed fonts.
                Renderer::from_config(koharu_config::Config::memory(TypesettingConfig {
                    font_families: Vec::new(),
                }))?,
                Rasterizer::new().context("render audit requires a WGPU graphics adapter")?,
            )),
        })
    }

    pub fn measure(&self, image: &RgbImage, page: &ReplayedDetectionPage) -> Result<RenderAudit> {
        let engine = self.engine.lock().map_err(|error| anyhow!("{error}"))?;
        self.runtime
            .block_on(measure_page(&engine.0, &engine.1, image, page))
    }
}

async fn measure_page(
    renderer: &Renderer,
    rasterizer: &Rasterizer,
    image: &RgbImage,
    replay: &ReplayedDetectionPage,
) -> Result<RenderAudit> {
    const LONG: &str = "I thought we could take the long way home and talk about what happened today, but then the rain started, the lights went out across the whole neighborhood, and all I could think about was whether anyone would find the note we left under the door before morning.";

    let mut metrics = [
        ProfileMetrics::default(),
        ProfileMetrics::default(),
        ProfileMetrics::default(),
    ];
    if replay
        .layout
        .iter()
        .any(|item| replay.detections[item.index].label == "text")
    {
        let (w, h) = image.dimensions();
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(
            PageDraft::new("replay", f64::from(w), f64::from(h)),
            At::End,
        )?;
        let snapshot = session.commit(edit.finish()?).await?.snapshot;
        let patch = replay.region_patch(snapshot, page)?;
        let snapshot = session.commit(patch).await?.snapshot;

        // Full scene order also defines the production writer's text layer order.
        let mut samples = Vec::new();
        let mut inferred =
            replay
                .layout
                .iter()
                .zip(&replay.inferred)
                .filter_map(|(item, inferred)| {
                    (replay.detections[item.index].label == "text").then_some(inferred)
                });
        let group = snapshot
            .page(page)?
            .text_group()?
            .context("replay has no text group")?;
        for layer in group.text_layers()? {
            let content = layer.content()?;
            let geometry = content
                .source_region()
                .context("reading the replay source")?
                .context("replay layer has no source region")?
                .geometry()?;
            let balloon = layer
                .balloon_target()?
                .map(|target| target.geometry())
                .transpose()?;
            let area = geometry
                .points
                .iter()
                .zip(geometry.points.iter().cycle().skip(1))
                .map(|(a, b)| a.x * b.y - b.x * a.y)
                .sum::<f64>()
                .abs()
                / 2.0;
            let typical = match area.sqrt() {
                ..36.0 => "Let's go home.",
                ..85.0 => "I think we should take the long way home.",
                ..175.0 => {
                    "I thought we could take the long way home and talk about what happened today."
                }
                _ => {
                    "We can take the long way home today and talk about what happened before anyone else hears the story."
                }
            };
            let paper = inferred
                .next()
                .context("replay layer has no layout item")?
                .map_or([255; 3], |typo| typo.paper_color);
            samples.push(Sample {
                layer: layer.id(),
                content: content.id(),
                balloon,
                typical,
                paper,
            });
        }

        let language = LanguageTag::new("en")?;
        // Keep the scene's geometry and paint fixed as translations change.
        let mut snapshot = snapshot;
        for (profile, metrics) in metrics.iter_mut().enumerate() {
            let mut edit = snapshot.edit();
            for sample in &samples {
                if profile == 0 {
                    // Recorded detector outputs carry no OCR transcript. An empty
                    // source satisfies the scene contract without supplying an em estimate.
                    edit.set(
                        sample.content,
                        &SourceText {
                            text: Authored::user(String::new()),
                            language: None,
                        },
                    )?;
                }
                let text = match profile {
                    0 => "OK!",
                    1 => sample.typical,
                    _ => LONG,
                };
                edit.set(
                    sample.content,
                    &Translation {
                        text: Authored::user(text.to_owned()),
                        language: Some(language.clone()),
                    },
                )?;
            }
            snapshot = session.commit(edit.finish()?).await?.snapshot;
            // Source RGB is sampled separately. The scene contains every region
            // and text layer so production typesetting still coordinates them.
            let frame = renderer.render(&snapshot, page).await
                .context("render audit could not typeset the page; install a system font with English glyphs")?;
            let mut inks = Vec::<Ink>::with_capacity(samples.len());
            for sample in &samples {
                let Some(cropped) = frame.cropped(sample.layer)? else {
                    metrics.missing_glyph_regions += 1;
                    continue;
                };
                let layer = cropped
                    .layer(sample.layer)
                    .context("missing cropped render layer")?;
                if let LayerKind::Text(text) = layer.kind() {
                    metrics.min_font_px = Some(
                        metrics
                            .min_font_px
                            .map_or(text.font_size, |min| min.min(text.font_size)),
                    );
                }
                let prepared = &cropped.prepared().frame.layers[0].content;
                let mut glyphs = match prepared {
                    PreparedContent::Vector(scene) => scene
                        .commands
                        .iter()
                        .filter_map(|command| match command {
                            PreparedSceneCommand::GlyphRun(run) => Some(&run.glyphs),
                            PreparedSceneCommand::FillPath(_) => None,
                        })
                        .flatten()
                        .peekable(),
                    PreparedContent::Raster(_) => anyhow::bail!("expected a text glyph layer"),
                };
                let missing = glyphs.peek().is_none() || glyphs.any(|glyph| glyph.id == 0);
                let raster =
                    rasterizer.rasterize(&cropped.raster_frame()?, RasterOptions::default())?;
                let (left, top) = (raster.left, raster.top);
                let mut ink = Ink {
                    pixels: GrayImage::new(raster.image.width(), raster.image.height()),
                    left,
                    top,
                    area: 0,
                };
                let (mut overflow, mut clipped, mut weighted_contrast, mut contrast_weight) =
                    (0, false, 0.0, 0.0);
                for (x, y, rgba) in raster
                    .image
                    .enumerate_pixels()
                    .filter(|(_, _, pixel)| pixel[3] > 16)
                {
                    let (px, py) = (left + x as i32, top + y as i32);
                    let on_page = px >= 0 && py >= 0 && px < w as i32 && py < h as i32;
                    clipped |= !on_page;
                    let allowed = sample.balloon.as_ref().map_or(on_page, |geometry| {
                        contains(geometry, f64::from(px) + 0.5, f64::from(py) + 0.5)
                    });
                    overflow += u32::from(!allowed);
                    ink.area += 1;
                    ink.pixels[(x, y)][0] = 255;
                    if on_page {
                        let point = (px as u32, py as u32);
                        let background = if replay.page_text_mask[point][0] == 0 {
                            image[point].0
                        } else {
                            sample.paper
                        };
                        let alpha = f64::from(rgba[3]) / 255.0;
                        let composited = [0, 1, 2].map(|c| {
                            (1.0 - alpha) * f64::from(background[c]) + alpha * f64::from(rgba[c])
                        });
                        let (a, b) = (luminance(composited), luminance(background.map(f64::from)));
                        weighted_contrast += alpha * (a.max(b) + 0.05) / (a.min(b) + 0.05);
                        contrast_weight += alpha;
                    }
                }
                metrics.missing_glyph_regions += usize::from(missing || ink.area == 0);
                metrics.clipped_regions += usize::from(clipped);
                if ink.area == 0 {
                    continue;
                }
                let outside = f64::from(overflow) / f64::from(ink.area);
                metrics.overflow_ratio_mean += outside;
                metrics.overflow_ratio_max = metrics.overflow_ratio_max.max(outside);
                metrics.overflow_regions_gt5pct += usize::from(outside > 0.05);
                if contrast_weight > 0.0 {
                    let contrast = weighted_contrast / contrast_weight;
                    metrics.min_local_contrast = Some(
                        metrics
                            .min_local_contrast
                            .map_or(contrast, |min| min.min(contrast)),
                    );
                }
                for other in &inks {
                    let overlap = ink.overlap(other);
                    metrics.collision_ratio_max = metrics.collision_ratio_max.max(overlap);
                    metrics.collision_pairs_gt1pct += usize::from(overlap > 0.01);
                }
                inks.push(ink);
            }
            if !inks.is_empty() {
                metrics.overflow_ratio_mean /= inks.len() as f64;
            }
        }
    }
    let [short, typical, long] = metrics;
    Ok(RenderAudit {
        profiles: Profiles {
            short,
            typical,
            long,
        },
    })
}

fn contains(geometry: &Geometry, x: f64, y: f64) -> bool {
    let mut inside = false;
    for (a, b) in geometry
        .points
        .iter()
        .zip(geometry.points.iter().cycle().skip(1))
    {
        if (a.y > y) != (b.y > y) && x < a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y) {
            inside = !inside;
        }
    }
    inside
}

fn luminance(rgb: [f64; 3]) -> f64 {
    let [r, g, b] = rgb.map(|channel| {
        let v = channel / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    });
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

impl Ink {
    fn overlap(&self, other: &Self) -> f64 {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        let right =
            (self.left + self.pixels.width() as i32).min(other.left + other.pixels.width() as i32);
        let bottom =
            (self.top + self.pixels.height() as i32).min(other.top + other.pixels.height() as i32);
        let mut intersection = 0;
        for y in top..bottom {
            for x in left..right {
                let a = self.pixels[((x - self.left) as u32, (y - self.top) as u32)][0];
                let b = other.pixels[((x - other.left) as u32, (y - other.top) as u32)][0];
                intersection += u32::from(a != 0 && b != 0);
            }
        }
        f64::from(intersection) / f64::from(self.area.min(other.area))
    }
}
