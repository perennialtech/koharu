//! Evaluation probe: times and dumps both detection models over a page set.
//!
//! seg_probe <pages> <out> [reps]
//! - out/<page>.json: RF-DETR detections at LOW (default 0.05) for every class,
//!   with mask crop as RLE, and comic detections above 0.05.
//! - stdout: per-page median wall time of the production-threshold calls.

use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use koharu_ml::{
    comic_text_bubble_detector::ComicTextBubbleDetector,
    koharu_layout_rfdetr_seg_2xl::{KoharuLayoutRFDetrSeg2XL, KoharuLayoutThresholds},
};

fn rle(pixels: &[u8]) -> Vec<u32> {
    let mut runs = Vec::new();
    let mut current = false;
    let mut length = 0u32;
    for &p in pixels {
        let set = p != 0;
        if set != current {
            runs.push(length);
            length = 0;
            current = set;
        }
        length += 1;
    }
    runs.push(length);
    runs
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let mut args = std::env::args().skip(1);
    let pages = PathBuf::from(args.next().unwrap());
    let out = PathBuf::from(args.next().unwrap());
    let reps: usize = args.next().map_or(3, |v| v.parse().unwrap());
    let only = std::env::var("PAGES").ok();
    let skip_dump = std::env::var("NODUMP").is_ok();
    std::fs::create_dir_all(&out)?;
    koharu_ml::init(&[
        koharu_ml::Feature::OnnxRuntime,
        koharu_ml::Feature::TensorRt,
    ])
    .await?;
    let device = koharu_ml::device(std::env::var("CPU").is_ok());
    let t = Instant::now();
    let mut layout = KoharuLayoutRFDetrSeg2XL::load(device.clone()).await?;
    let mut comic = ComicTextBubbleDetector::load(device).await?;
    eprintln!("load_ms={}", t.elapsed().as_millis());
    let mut production = layout.recommended_thresholds();
    production.bubble = 0.05;
    let low: f32 = std::env::var("LOW")
        .ok()
        .map_or(0.05, |v| v.parse().unwrap());
    let low = KoharuLayoutThresholds {
        text: low,
        onomatopoeia: low,
        bubble: low,
        panel: low,
    };
    let mut files = std::fs::read_dir(&pages)?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()?;
    files.sort();
    let mut warmed = false;
    let (mut total_layout, mut total_comic) = (0.0, 0.0);
    for file in files {
        let name = file.file_stem().unwrap().to_string_lossy().to_string();
        if only
            .as_ref()
            .is_some_and(|only| !only.split(',').any(|p| p == name))
        {
            continue;
        }
        let image = image::open(&file)?;
        if !warmed {
            let t = Instant::now();
            layout.inference_with_thresholds(&image, production)?;
            comic.inference(&image, 0.3)?;
            eprintln!("warmup_ms={}", t.elapsed().as_millis());
            warmed = true;
        }
        let mut lt = Vec::new();
        let mut ct = Vec::new();
        let mut count = 0;
        for _ in 0..reps {
            let t = Instant::now();
            let d = layout.inference_with_thresholds(&image, production)?;
            lt.push(t.elapsed().as_secs_f64() * 1e3);
            count = d.detections.len();
            let t = Instant::now();
            comic.inference(&image, 0.3)?;
            ct.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let (l, c) = (median(lt), median(ct));
        total_layout += l;
        total_comic += c;
        println!(
            "{name} {}x{} layout_ms={l:.1} comic_ms={c:.1} dets={count}",
            image.width(),
            image.height()
        );
        if skip_dump {
            continue;
        }
        let detections = layout.inference_with_thresholds(&image, low)?;
        let boxes = comic.inference(&image, 0.05)?;
        let items = detections
            .detections
            .iter()
            .map(|d| {
                serde_json::json!({
                    "label": d.label, "score": d.score, "bbox": d.bbox, "area": d.area,
                    "mask": [d.mask.x, d.mask.y, d.mask.width, d.mask.height],
                    "rle": rle(&d.mask.pixels),
                })
            })
            .collect::<Vec<_>>();
        std::fs::write(
            out.join(format!("{name}.json")),
            serde_json::to_vec(&serde_json::json!({
                "width": image.width(), "height": image.height(),
                "rfdetr": items, "comic": boxes,
            }))?,
        )?;
    }
    println!("TOTAL layout_ms={total_layout:.1} comic_ms={total_comic:.1}");
    Ok(())
}
