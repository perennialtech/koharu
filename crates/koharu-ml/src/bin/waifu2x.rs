use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::waifu2x::Waifu2x;

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,

    #[arg(long, default_value_t = false)]
    cpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let image = image::open(&cli.input)?.to_rgb8();

    koharu_ml::init(&[koharu_ml::Feature::OnnxRuntime]).await?;

    let mut model = Waifu2x::load(koharu_ml::device(cli.cpu)).await?;
    // Every tile has one shape, so a single tile warms the provider.
    model.inference(&image::RgbImage::new(1, 1))?;
    let started = std::time::Instant::now();
    let upscaled = model.inference(&image)?;
    tracing::info!(elapsed = ?started.elapsed(), "upscaled");
    upscaled.save(cli.output)?;

    Ok(())
}
