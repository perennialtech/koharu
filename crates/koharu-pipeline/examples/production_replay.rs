//! Stream recorded detections through the production page pipeline.

#[path = "support/raw_detections.rs"]
mod raw_detections;
#[path = "production_replay/render.rs"]
mod render;

use std::{
    fmt::Write as _,
    fs::{self, File},
    io::{BufRead, BufWriter, ErrorKind, Write},
    path::{Component, Path, PathBuf},
    sync::OnceLock,
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use flate2::{Compression, write::GzEncoder};
use image::{
    ExtendedColorType, GrayImage, ImageFormat, RgbImage, codecs::jpeg::JpegEncoder, imageops,
};
use koharu_ml::koharu_layout_rfdetr_seg_2xl::KoharuLayoutMask;
use koharu_pipeline::{DetectionFusionConfig, InferredTypography, replay_detection_page};
use koharu_scene::WritingMode;
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};

use raw_detections::{RawDumpLine, RleMask, dump_reader};
use render::{RenderAudit, RenderAuditor};

#[derive(Parser)]
#[command(about = "Stream raw detector dumps through the production detection pipeline")]
struct Args {
    /// Raw JSONL/gzip files, or directories of raw dumps in sorted order.
    #[arg(long, required = true, num_args = 1..)]
    raw_dumps: Vec<PathBuf>,
    #[arg(long)]
    images_dir: PathBuf,
    #[arg(long)]
    allow_missing_images: bool,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(u32).range(1..))]
    batch_size: u32,
    /// Cap input records, including skipped images.
    #[arg(long)]
    max_pages: Option<usize>,
    #[arg(long, default_value_t = 0.40)]
    bubble_threshold: f32,
    #[arg(long, value_enum, default_value = "default")]
    config: Fusion,
    #[arg(long, default_value = "unknown")]
    build_sha: String,
    #[arg(long)]
    emit_mask_rle: bool,
    #[arg(long, value_enum, default_value = "none")]
    perturb: Perturb,
    /// Rasterize three English samples with local fonts and a WGPU adapter.
    #[arg(long)]
    render_audit: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum Fusion {
    Default,
    Legacy,
}

#[derive(Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum Perturb {
    None,
    Jpeg75,
    Webp75,
    Scale075,
    Scale125,
}

impl Perturb {
    fn apply(self, image: RgbImage) -> Result<(RgbImage, f32)> {
        let (w, h) = image.dimensions();
        match self {
            Self::None => Ok((image, 1.0)),
            Self::Jpeg75 => {
                let mut bytes = Vec::new();
                JpegEncoder::new_with_quality(&mut bytes, 75).encode(
                    image.as_raw(),
                    w,
                    h,
                    ExtendedColorType::Rgb8,
                )?;
                Ok((
                    image::load_from_memory_with_format(&bytes, ImageFormat::Jpeg)?.into_rgb8(),
                    1.0,
                ))
            }
            Self::Webp75 => {
                let bytes = webp::Encoder::from_rgb(image.as_raw(), w, h)
                    .encode_simple(false, 75.0)
                    .map_err(|error| anyhow::anyhow!("WebP encoding failed: {error:?}"))?;
                Ok((
                    image::load_from_memory_with_format(&bytes, ImageFormat::WebP)?.into_rgb8(),
                    1.0,
                ))
            }
            Self::Scale075 | Self::Scale125 => {
                let scale = if matches!(self, Self::Scale075) {
                    0.75
                } else {
                    1.25
                };
                let [width, height] =
                    [w, h].map(|side| (side as f32 * scale).round().max(1.0) as u32);
                Ok((
                    imageops::resize(&image, width, height, imageops::FilterType::CatmullRom),
                    scale,
                ))
            }
        }
    }
}

#[derive(Serialize)]
struct PageRecord<'a> {
    schema_version: u32,
    build_sha: &'a str,
    perturb: Perturb,
    scale: f32,
    image_id: u64,
    file_name: String,
    sha256: String,
    orig_w: u32,
    orig_h: u32,
    page_w: u32,
    page_h: u32,
    codec: &'static str,
    is_color: bool,
    elapsed_ms: f64,
    page_text_mask_px: u32,
    counts: Counts,
    regions: Vec<Region>,
    #[serde(skip_serializing_if = "Option::is_none")]
    render_audit: Option<RenderAudit>,
}

#[derive(Default, Serialize)]
struct Counts {
    bubbles: usize,
    text: usize,
    associated_text: usize,
    free_text: usize,
    onomatopoeia: usize,
    panels: usize,
}

#[derive(Serialize)]
struct Region {
    id: usize,
    label: String,
    score: f32,
    bbox: [f32; 4],
    orig_bbox: [f32; 4],
    bubble_id: Option<usize>,
    reading_order: Option<usize>,
    mask_px: u32,
    empty_mask: bool,
    competing_claim_px: u32,
    cross_owner_cleanup_px: u32,
    stroke_edge_support: Option<f32>,
    non_text_edge_mass: Option<f32>,
    typo_none: bool,
    typo: Option<Typography>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mask_rle: Option<RleMask>,
}

#[derive(Serialize)]
struct Typography {
    size: f32,
    orig_size: f32,
    vertical: bool,
    tilt_deg: f32,
    ink_rgb: [u8; 3],
    paper_rgb: [u8; 3],
    has_outline: bool,
    outline_w: Option<f32>,
    outline_rgb: Option<[u8; 3]>,
}

#[tokio::main(worker_threads = 2)]
async fn main() -> Result<()> {
    run(Args::parse())
}

fn run(args: Args) -> Result<()> {
    ensure!(
        (0.0..=1.0).contains(&args.bubble_threshold),
        "bubble threshold must be between 0 and 1"
    );
    ensure!(
        args.images_dir.is_dir(),
        "{} is not an image directory",
        args.images_dir.display()
    );

    let mut dumps = Vec::new();
    for input in &args.raw_dumps {
        if input.is_dir() {
            let mut entries = Vec::new();
            for entry in fs::read_dir(input)? {
                let file = entry?.path();
                if file.file_name().is_some_and(|name| {
                    let name = name.to_string_lossy();
                    name.ends_with(".raw.jsonl") || name.ends_with(".raw.jsonl.gz")
                }) && file.is_file()
                {
                    entries.push(file);
                }
            }
            entries.sort();
            dumps.extend(entries);
        } else {
            dumps.push(input.clone());
        }
    }
    ensure!(!dumps.is_empty(), "no raw dumps found");
    if let Ok(output) = args.out.canonicalize() {
        for dump in &dumps {
            ensure!(
                dump.canonicalize()? != output,
                "output is also an input dump"
            );
        }
    }

    let config = match args.config {
        Fusion::Default => DetectionFusionConfig::default(),
        Fusion::Legacy => DetectionFusionConfig::legacy(),
    };
    let auditor = if args.render_audit {
        Some(RenderAuditor::new()?)
    } else {
        None
    };
    // Rendering submits work to the global Rayon pool from Tokio. Audit pages
    // run in a separate pool so blocking for the render cannot starve it.
    let audit_pool = auditor
        .as_ref()
        .map(|_| rayon::ThreadPoolBuilder::new().build())
        .transpose()?;
    let out = BufWriter::new(
        File::create(&args.out)
            .with_context(|| format!("failed to create {}", args.out.display()))?,
    );
    let mut writer = if args
        .out
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
    {
        Output::Gzip(Box::new(GzEncoder::new(out, Compression::fast())))
    } else {
        Output::Plain(out)
    };
    let images = ImageDirectory::new(&args.images_dir);
    let mut batch = Vec::new();
    let (mut seen, mut written, mut missing) = (0, 0, 0);
    let limit = args.max_pages.unwrap_or(usize::MAX);
    let batch_size = args.batch_size as usize;
    for dump in dumps {
        let mut lines = dump_reader(&dump)?.lines().enumerate();
        while seen < limit {
            for (index, line) in lines.by_ref() {
                let number = index + 1;
                let line = line.with_context(|| format!("{}:{number}", dump.display()))?;
                if line.trim().is_empty() {
                    continue;
                }
                batch.push((number, line));
                seen += 1;
                if batch.len() == batch_size || seen == limit {
                    break;
                }
            }
            if batch.is_empty() {
                break;
            }
            let replay_batch = || {
                batch
                    .par_iter()
                    .map(|(number, line)| {
                        replay_line(line, &images, &args, &config, auditor.as_ref())
                            .with_context(|| format!("{}:{number}", dump.display()))
                    })
                    .collect::<Vec<_>>()
            };
            let records = match &audit_pool {
                Some(pool) => pool.install(replay_batch),
                None => replay_batch(),
            };
            let out = writer.stream();
            for record in records {
                if let Some(record) = record? {
                    serde_json::to_writer(&mut *out, &record)?;
                    out.write_all(b"\n")?;
                    written += 1;
                } else {
                    missing += 1;
                }
            }
            out.flush()?;
            batch.clear();
        }
        if seen == limit {
            break;
        }
    }
    match writer {
        Output::Gzip(writer) => writer.finish()?.flush()?,
        Output::Plain(mut writer) => writer.flush()?,
    }
    eprintln!("read={seen} written={written} missing_images={missing}");
    Ok(())
}

enum Output {
    Gzip(Box<GzEncoder<BufWriter<File>>>),
    Plain(BufWriter<File>),
}

impl Output {
    fn stream(&mut self) -> &mut dyn Write {
        match self {
            Self::Gzip(writer) => writer.as_mut(),
            Self::Plain(writer) => writer,
        }
    }
}

fn replay_line<'a>(
    line: &str,
    images: &ImageDirectory<'_>,
    args: &'a Args,
    config: &DetectionFusionConfig,
    auditor: Option<&RenderAuditor>,
) -> Result<Option<PageRecord<'a>>> {
    let started = Instant::now();
    let raw: RawDumpLine = serde_json::from_str(line).context("invalid raw detection record")?;
    let file_name = raw.file_name.clone();
    let image_id = raw.image_id;
    let bytes = match images.read(Path::new(&file_name)) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound && args.allow_missing_images => {
            return Ok(None);
        }
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read image {file_name}"));
        }
    };
    let mut sha256 = String::with_capacity(64);
    for byte in Sha256::digest(&bytes) {
        write!(sha256, "{byte:02x}")?;
    }
    let format = image::guess_format(&bytes)?;
    let codec = match format {
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Png => "png",
        ImageFormat::WebP => "webp",
        _ => "other",
    };
    let image = image::load_from_memory_with_format(&bytes, format)?.into_rgb8();
    let (orig_w, orig_h) = image.dimensions();
    ensure!(
        (orig_w, orig_h) == (raw.image_width, raw.image_height),
        "{file_name} has dimensions {orig_w}x{orig_h}, dump uses {}x{}",
        raw.image_width,
        raw.image_height
    );
    let image_area = u64::from(orig_w) * u64::from(orig_h);
    let chromatic = image
        .pixels()
        .filter(|pixel| {
            let [r, g, b] = pixel.0;
            r.max(g).max(b) - r.min(g).min(b) > 18
        })
        .count() as u64;
    let is_color = chromatic * 50 > image_area;
    let (image, scale) = args.perturb.apply(image)?;
    let (page_w, page_h) = image.dimensions();
    let (layout, comic) = raw.into_detections(scale, (page_w, page_h))?;
    let page = replay_detection_page(&image, layout, &comic, args.bubble_threshold, config);

    let mut counts = Counts::default();
    let luma = page
        .detections
        .iter()
        .any(|region| matches!(region.label.as_str(), "text" | "onomatopoeia") && region.area > 0)
        .then(|| imageops::grayscale(&image));
    let regions = page
        .layout
        .iter()
        .zip(&page.inferred)
        .map(|(item, &inferred)| {
            let id = item.index;
            let detection = &page.detections[id];
            let reading_order = match detection.label.as_str() {
                "text" => {
                    let order = counts.text;
                    counts.text += 1;
                    if item.bubble.is_some() {
                        counts.associated_text += 1;
                    } else {
                        counts.free_text += 1;
                    }
                    Some(order)
                }
                "bubble" => {
                    counts.bubbles += 1;
                    None
                }
                "panel" => {
                    counts.panels += 1;
                    None
                }
                "onomatopoeia" => {
                    counts.onomatopoeia += 1;
                    None
                }
                other => bail!("unexpected final detection label {other}"),
            };
            let lettering = matches!(detection.label.as_str(), "text" | "onomatopoeia");
            let mask = &detection.mask;
            let mask_px = mask.pixels.iter().filter(|&&pixel| pixel != 0).count() as u32;
            let (stroke_edge_support, non_text_edge_mass) = if lettering && mask_px > 0 {
                edge_metrics(
                    &image,
                    luma.as_ref().expect("nonempty lettering has luminance"),
                    mask,
                    inferred,
                )
            } else {
                (None, None)
            };
            let typo = inferred.map(|value| Typography {
                size: value.size,
                orig_size: value.size / scale,
                vertical: value.writing_mode == WritingMode::Vertical,
                tilt_deg: value.angle_degrees,
                ink_rgb: value.color,
                paper_rgb: value.paper_color,
                has_outline: value.stroke_color.is_some() && value.stroke_width.is_some(),
                outline_w: value.stroke_width,
                outline_rgb: value.stroke_color,
            });
            Ok(Region {
                id,
                label: detection.label.clone(),
                score: detection.score,
                bbox: detection.bbox,
                orig_bbox: detection.bbox.map(|value| value / scale),
                bubble_id: item.bubble,
                reading_order,
                mask_px,
                empty_mask: mask_px == 0,
                competing_claim_px: page.glyph_claims[id].competing_claim_px,
                cross_owner_cleanup_px: page.glyph_claims[id].cross_owner_cleanup_px,
                stroke_edge_support,
                non_text_edge_mass,
                typo_none: typo.is_none(),
                typo,
                mask_rle: (args.emit_mask_rle && lettering).then(|| RleMask::from(mask.as_ref())),
            })
        })
        .collect::<Result<_>>()?;
    let render_audit = auditor
        .map(|auditor| auditor.measure(&image, &page))
        .transpose()?;
    Ok(Some(PageRecord {
        schema_version: 1,
        build_sha: &args.build_sha,
        perturb: args.perturb,
        scale,
        image_id,
        file_name,
        sha256,
        orig_w,
        orig_h,
        page_w,
        page_h,
        codec,
        is_color,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        page_text_mask_px: page
            .page_text_mask
            .pixels()
            .filter(|pixel| pixel[0] != 0)
            .count() as u32,
        counts,
        regions,
        render_audit,
    }))
}

struct ImageDirectory<'a> {
    root: &'a Path,
    nested: OnceLock<std::io::Result<Vec<PathBuf>>>,
}

impl<'a> ImageDirectory<'a> {
    fn new(root: &'a Path) -> Self {
        Self {
            root,
            nested: OnceLock::new(),
        }
    }

    fn read(&self, file_name: &Path) -> std::io::Result<Vec<u8>> {
        let components = file_name
            .components()
            .filter_map(|part| match part {
                Component::Normal(name) => Some(Ok(name)),
                Component::ParentDir => Some(Err(std::io::Error::new(
                    ErrorKind::InvalidInput,
                    "parent directory in image name",
                ))),
                _ => None,
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        let basename = components
            .last()
            .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidInput, "empty image name"))?;
        // Dumps may retain directory prefixes from another dataset root.
        for start in 0..components.len() {
            let suffix: PathBuf = components[start..].iter().collect();
            match fs::read(self.root.join(suffix)) {
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                result => return result,
            }
        }
        // Index directories once, only when exact/suffix lookup misses. Nested
        // pages can then be found without keeping a catalog of image paths.
        let nested = self
            .nested
            .get_or_init(|| {
                let mut nested = Vec::new();
                let mut pending = vec![self.root.to_owned()];
                while let Some(directory) = pending.pop() {
                    for entry in fs::read_dir(directory)? {
                        let entry = entry?;
                        if entry.file_type()?.is_dir() {
                            let child = entry.path();
                            pending.push(child.clone());
                            nested.push(child);
                        }
                    }
                }
                Ok(nested)
            })
            .as_ref()
            .map_err(|error| std::io::Error::new(error.kind(), error.to_string()))?;
        let mut found = None;
        for directory in nested {
            let path = directory.join(basename);
            match path.metadata() {
                Ok(metadata) if metadata.is_file() => {
                    if found.replace(path).is_some() {
                        return Err(std::io::Error::new(
                            ErrorKind::InvalidData,
                            "ambiguous nested image name",
                        ));
                    }
                }
                Ok(_) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
        fs::read(found.ok_or_else(|| std::io::Error::from(ErrorKind::NotFound))?)
    }
}

fn edge_metrics(
    image: &RgbImage,
    luma: &GrayImage,
    mask: &KoharuLayoutMask,
    inferred: Option<InferredTypography>,
) -> (Option<f32>, Option<f32>) {
    const STRONG: u16 = 160;
    let (w, h) = image.dimensions();
    if mask.x >= w || mask.y >= h {
        return (None, None);
    }
    // Two pixels cover both the 3×3 Sobel kernel and the boundary search.
    let (left, top) = (mask.x.saturating_sub(2), mask.y.saturating_sub(2));
    let right = (mask.x + mask.width).saturating_add(2).min(w);
    let bottom = (mask.y + mask.height).saturating_add(2).min(h);
    let crop = imageops::crop_imm(luma, left, top, right - left, bottom - top).to_image();
    let sobel = imageproc::gradients::sobel_gradients(&crop);
    let strong = |x, y| sobel[(x - left, y - top)][0] >= STRONG;
    let ink = inferred.map_or([0; 3], |value| value.color);
    let paper = inferred.map_or([255; 3], |value| value.paper_color);
    let color_distance = |pixel: [u8; 3], reference: [u8; 3]| {
        (0..3)
            .map(|c| u32::from(pixel[c].abs_diff(reference[c])).pow(2))
            .sum::<u32>()
    };
    let (mut boundary, mut supported, mut edges, mut stray) = (0, 0, 0, 0);
    for (offset, _) in mask.pixels.iter().enumerate().filter(|&(_, &p)| p != 0) {
        let (x, y) = (
            mask.x + offset as u32 % mask.width,
            mask.y + offset as u32 / mask.width,
        );
        if x >= w || y >= h {
            continue;
        }
        let nearby = || {
            (y.saturating_sub(1)..=(y + 1).min(h - 1))
                .flat_map(move |y| (x.saturating_sub(1)..=(x + 1).min(w - 1)).map(move |x| (x, y)))
        };
        if x == 0
            || y == 0
            || !mask.contains(x - 1, y)
            || !mask.contains(x, y - 1)
            || !mask.contains(x + 1, y)
            || !mask.contains(x, y + 1)
        {
            boundary += 1;
            supported += u32::from(nearby().any(|(x, y)| strong(x, y)));
        }
        if strong(x, y) {
            edges += 1;
            let near_ink = nearby().any(|point| {
                let pixel = image[point].0;
                let distance = color_distance(pixel, ink);
                distance <= 3 * 64 * 64 && distance < color_distance(pixel, paper)
            });
            stray += u32::from(!near_ink);
        }
    }
    (
        (boundary > 0).then(|| supported as f32 / boundary as f32),
        (edges > 0).then(|| stray as f32 / edges as f32),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[tokio::test]
    async fn streams_sorted_plain_and_gzipped_dumps_with_a_record_limit() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path();
        fs::create_dir(root.join("dumps"))?;
        fs::create_dir(root.join("nested"))?;
        let image = RgbImage::from_pixel(8, 12, image::Rgb([13, 19, 75]));
        image.save(root.join("nested/page.png"))?;
        let line = |id: u64, file_name| {
            format!(
                "{}\n",
                json!({
                    "image_id": id, "file_name": file_name, "image_width": 8, "image_height": 12,
                    "rfdetr": [], "ogkalu": [],
                })
            )
        };
        let mut gzip = GzEncoder::new(
            File::create(root.join("dumps/01.raw.jsonl.gz"))?,
            Compression::fast(),
        );
        gzip.write_all(line(1, "missing.png").as_bytes())?;
        let gzip = gzip.finish()?;
        let mut next_member = GzEncoder::new(gzip, Compression::fast());
        next_member.write_all(line(2, "dataset/nested/page.png").as_bytes())?;
        next_member.finish()?;
        fs::write(
            root.join("dumps/02.raw.jsonl"),
            format!(
                "\n{}invalid line beyond the record limit\n",
                line(3, "page.png")
            ),
        )?;
        let output = root.join("replayed.jsonl.gz");
        run(Args {
            raw_dumps: vec![root.join("dumps")],
            images_dir: root.to_owned(),
            allow_missing_images: true,
            out: output.clone(),
            batch_size: 1,
            max_pages: Some(3),
            bubble_threshold: 0.4,
            config: Fusion::Default,
            build_sha: "test".to_owned(),
            emit_mask_rle: false,
            perturb: Perturb::None,
            render_audit: false,
        })?;
        let records: Vec<Value> = dump_reader(&output)?
            .lines()
            .map(|line| Ok(serde_json::from_str(&line?)?))
            .collect::<Result<_>>()?;
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["image_id"], 2);
        assert_eq!(records[1]["image_id"], 3);
        assert_eq!(records[0]["sha256"], records[1]["sha256"]);
        assert_eq!(records[0]["is_color"], true);
        assert!(records[0].get("render_audit").is_none());
        Ok(())
    }

    #[test]
    fn scales_bounds_and_nearest_masks_and_back_projects_inference() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let image = RgbImage::from_fn(32, 24, |x, y| {
            image::Rgb(
                [if (10..14).contains(&x) && (4..16).contains(&y) {
                    0
                } else {
                    255
                }; 3],
            )
        });
        image.save(dir.path().join("page.png"))?;
        let line = json!({
            "image_id": 4, "file_name": "page.png", "image_width": 32, "image_height": 24,
            "rfdetr": [{
                "label_id": 0, "label": "text", "score": 0.95, "bbox": [8, 0, 16, 20], "area": 160,
                "mask": {"x": 8, "y": 0, "width": 8, "height": 20, "counts": [0, 160]},
            }, {
                "label_id": 3, "label": "panel", "score": 0.95, "bbox": [24, 16, 36, 24], "area": 96,
                "mask": {"x": 24, "y": 16, "width": 12, "height": 8, "counts": [0, 96]},
            }],
            "ogkalu": [{"label": "bubble", "score": 0.9, "bbox": [0, 0, 25, 22]}],
        }).to_string();
        for (perturb, scale, dims, panel_bbox, panel_mask) in [
            (Perturb::Scale075, 0.75, (24, 18), [18., 12., 24., 18.], 36),
            (Perturb::Scale125, 1.25, (40, 30), [30., 20., 40., 30.], 100),
        ] {
            let args = Args {
                raw_dumps: Vec::new(),
                images_dir: dir.path().to_owned(),
                allow_missing_images: false,
                out: PathBuf::new(),
                batch_size: 1,
                max_pages: None,
                bubble_threshold: 0.4,
                config: Fusion::Default,
                build_sha: "test".to_owned(),
                emit_mask_rle: true,
                perturb,
                render_audit: false,
            };
            let images = ImageDirectory::new(dir.path());
            let page = replay_line(
                &line,
                &images,
                &args,
                &DetectionFusionConfig::default(),
                None,
            )?
            .unwrap();
            assert_eq!(page.scale, scale);
            assert_eq!((page.page_w, page.page_h), dims);
            let panel = page.regions.iter().find(|r| r.label == "panel").unwrap();
            assert_eq!(panel.bbox, panel_bbox);
            assert_eq!(panel.orig_bbox, [24., 16., 32., 24.]);
            assert_eq!(panel.mask_px, panel_mask);
            assert!(panel.mask_rle.is_none());

            let text = page
                .regions
                .into_iter()
                .find(|r| r.label == "text")
                .unwrap();
            let typo = text
                .typo
                .context("scaled image should retain measurable source ink")?;
            assert_eq!(typo.orig_size, typo.size / scale);
            let mask = text.mask_rle.unwrap().decode()?;
            assert_eq!(
                mask.pixels.iter().filter(|&&value| value != 0).count() as u32,
                text.mask_px
            );
            assert!(mask.width > 0 && mask.height > 0);
            assert!(mask.x + mask.width <= dims.0 && mask.y + mask.height <= dims.1);
        }
        Ok(())
    }

    #[test]
    fn rle_round_trips_foreground_first_and_rejects_short_runs() -> Result<()> {
        let mask = KoharuLayoutMask {
            x: 8,
            y: 2,
            width: 3,
            height: 2,
            pixels: vec![255, 0, 0, 255, 255, 0],
        };
        let serialized = serde_json::to_value(RleMask::from(&mask))?;
        assert_eq!(serialized["counts"], json!([0, 1, 2, 2, 1]));
        let round_trip = serde_json::from_value::<RleMask>(serialized)?.decode()?;
        assert_eq!(
            (
                round_trip.x,
                round_trip.y,
                round_trip.width,
                round_trip.height
            ),
            (8, 2, 3, 2)
        );
        assert_eq!(round_trip.pixels, mask.pixels);
        let bad = serde_json::from_value::<RleMask>(json!({
            "x": 0, "y": 0, "width": 3, "height": 1, "counts": [1],
        }))?;
        assert!(bad.decode().is_err());
        Ok(())
    }
}
