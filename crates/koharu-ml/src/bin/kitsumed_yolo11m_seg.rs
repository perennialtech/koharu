use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use image::GrayImage;
use koharu_ml::kitsumed_yolo11m_seg::KitsumedYolo11mSeg;

/// Prints one JSON line of detections per image.
#[derive(Debug, Parser)]
struct Cli {
    #[arg(required = true, value_name = "FILE")]
    inputs: Vec<PathBuf>,

    /// Writes each image's union of masks here as <stem>.png.
    #[arg(long, value_name = "DIR")]
    masks: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    koharu_ml::init(&[koharu_ml::Feature::TensorRt]).await?;
    let mut model = KitsumedYolo11mSeg::load(koharu_ml::device(false)).await?;
    for input in &cli.inputs {
        let image = image::open(input)
            .with_context(|| format!("failed to open {}", input.display()))?
            .to_rgb8();
        let censors = model.inference(&image)?;
        let detections = censors
            .iter()
            .map(|censor| {
                serde_json::json!({
                    "kind": format!("{:?}", censor.kind),
                    "score": censor.score,
                    "bbox": censor.bbox,
                    "area": censor.mask.pixels().filter(|pixel| pixel[0] != 0).count(),
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::json!({ "image": input, "detections": detections })
        );
        if let Some(directory) = &cli.masks {
            let mut union = GrayImage::new(image.width(), image.height());
            for censor in &censors {
                censor.paint(&mut union);
            }
            let stem = input.file_stem().context("input has no file name")?;
            union.save(directory.join(stem).with_extension("png"))?;
        }
    }
    Ok(())
}
