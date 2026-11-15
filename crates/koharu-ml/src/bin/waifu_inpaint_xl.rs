use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::waifu_inpaint_xl::WaifuInpaintXl;

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    mask: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,

    #[arg(long, default_value_t = 1.0)]
    strength: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let image = image::open(&cli.input)?.to_rgb8();
    let mask = image::open(&cli.mask)?.to_luma8();

    koharu_ml::init(&[koharu_ml::Feature::TensorRt]).await?;

    let model = WaifuInpaintXl::load(koharu_ml::device(false)).await?;
    model
        .inference(&image, &mask, cli.strength)?
        .save(cli.output)?;

    Ok(())
}
