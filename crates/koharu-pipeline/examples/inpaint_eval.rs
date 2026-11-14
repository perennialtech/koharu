//! Runs detection and inpainting on every page in a directory and reports the
//! inpainting stage time per page. Cleaned pages are written as PNG.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use koharu_config::Config;
use koharu_pipeline::{
    Committer, Flux2KleinConfig, InpaintingModel, Operation, Pipeline, PipelineConfig, Progress,
    Request, Scope, Stage, StageOutput,
};
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::Renderer;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft, Session};

struct SessionCommitter<'a>(&'a mut Session);

#[async_trait::async_trait]
impl Committer for SessionCommitter<'_> {
    async fn commit(&mut self, output: StageOutput) -> Result<koharu_scene::Snapshot> {
        Ok(self.0.commit(output.patch).await?.snapshot)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .context("usage: inpaint_eval <lama|flux2-klein> <pages> <out>")?;
    let pages = PathBuf::from(args.next().context("missing pages directory")?);
    let out = PathBuf::from(args.next().context("missing output directory")?);
    std::fs::create_dir_all(&out)?;

    let mut config = PipelineConfig::default();
    config.inpainting = match model.as_str() {
        "lama" => InpaintingModel::LaMa {},
        "flux2-klein" => InpaintingModel::Flux2Klein(Flux2KleinConfig::default()),
        other => anyhow::bail!("unknown inpainting model {other}"),
    };
    let mut features = vec![
        koharu_ml::Feature::Torch,
        koharu_ml::Feature::OnnxRuntime,
        koharu_ml::Feature::TensorRt,
    ];
    if matches!(&config.inpainting, InpaintingModel::Flux2Klein(_)) {
        features.push(koharu_ml::Feature::Diffusion);
    }
    koharu_ml::init(&features).await?;
    let pipeline = Pipeline::from_config(
        Config::memory(config),
        Config::memory(Default::default()),
        koharu_ml::device(false),
    )?;
    let renderer = Renderer::new()?;
    let rasterizer = Rasterizer::new()?;

    let mut files = std::fs::read_dir(&pages)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    files.sort();
    let mut total = Duration::ZERO;
    for (index, file) in files.iter().enumerate() {
        let bytes = std::fs::read(file)?;
        let image = image::load_from_memory(&bytes)?;
        let media_type = match image::guess_format(&bytes)? {
            image::ImageFormat::Jpeg => "image/jpeg",
            _ => "image/png",
        };
        let mut session = Session::memory().await?;
        let mut page = None;
        let patch = session.snapshot().patch(|edit| {
            let id = edit.add_page(
                PageDraft::new("page", f64::from(image.width()), f64::from(image.height())),
                At::End,
            )?;
            edit.set_asset(
                id,
                &AssetRole::new("source")?,
                AssetInput::new(
                    Arc::<[u8]>::from(bytes.as_slice()),
                    media_type,
                    AssetMetadata {
                        width: Some(image.width()),
                        height: Some(image.height()),
                        attributes: BTreeMap::new(),
                    },
                ),
            )?;
            page = Some(id);
            Ok(())
        })?;
        session.commit(patch).await?;
        let page = page.context("page ID was not assigned")?;
        let elapsed = Arc::new(std::sync::Mutex::new(Duration::ZERO));
        let sink = elapsed.clone();
        let snapshot = session.snapshot();
        pipeline
            .execute(
                snapshot,
                Request {
                    operation: Operation::Stages {
                        stages: vec![Stage::Detection, Stage::Inpainting],
                    },
                    scope: Scope::Pages(vec![page]),
                    progress: Some(Arc::new(move |event| {
                        if let Progress::Finished {
                            stage: Stage::Inpainting,
                            elapsed,
                            ..
                        } = event
                        {
                            *sink.lock().unwrap() = elapsed;
                        }
                    })),
                    ..Request::default()
                },
                &mut SessionCommitter(&mut session),
            )
            .await?;
        let elapsed = *elapsed.lock().unwrap();
        if index > 0 {
            total += elapsed;
        }
        let name = file.file_stem().unwrap().to_string_lossy();
        println!("{name} inpainting={:.3}", elapsed.as_secs_f64());
        let frame = renderer.render(&session.snapshot(), page).await?;
        let raster = rasterizer.rasterize(&frame.raster_frame()?, RasterOptions::default())?;
        image::DynamicImage::ImageRgba8(raster.image)
            .to_rgb8()
            .save(out.join(format!("{name}.png")))?;
    }
    println!(
        "total_excluding_first={:.3} pages={}",
        total.as_secs_f64(),
        files.len().saturating_sub(1)
    );
    Ok(())
}
