use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::comic_text_bubble_detector::ComicTextBubbleDetector;

#[derive(Debug, Parser)]
struct Cli {
    #[arg(value_name = "FILE", required = true)]
    input: Vec<PathBuf>,

    #[arg(long, default_value_t = 0.05)]
    threshold: f32,

    #[arg(long, default_value_t = false)]
    cpu: bool,
}

/// Prints one JSON object per input: `{"image": ..., "detections": [...]}`.
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    koharu_ml::init(&[koharu_ml::Feature::Torch]).await?;
    let model = ComicTextBubbleDetector::load(koharu_ml::device(cli.cpu)).await?;
    for path in &cli.input {
        let image = image::open(path)?;
        let detections = model.inference(&image, cli.threshold)?;
        println!(
            "{}",
            serde_json::json!({ "image": path, "detections": detections })
        );
    }
    Ok(())
}
