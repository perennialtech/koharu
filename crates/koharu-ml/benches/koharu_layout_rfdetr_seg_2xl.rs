use std::{hint::black_box, path::PathBuf, time::Duration};

use anyhow::Result;
use criterion::Criterion;
use koharu_ml::koharu_layout_rfdetr_seg_2xl::KoharuLayoutRFDetrSeg2XL;

#[tokio::main]
async fn main() -> Result<()> {
    let input =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/fixtures/object_detection/1.jpg");

    koharu_ml::init().await?;
    let image = image::open(input)?;
    // Upscale the same manga page to stress page-sized mask allocations.
    let large_image = image.resize_exact(2738, 3840, image::imageops::FilterType::Triangle);
    let device = koharu_ml::device(false);
    println!("Benchmark device: {}", device.description);
    let model = KoharuLayoutRFDetrSeg2XL::load(device).await?;

    black_box(model.inference(&image)?);
    let large_warmup = model.inference(&large_image)?;
    println!("4K manga detections: {}", large_warmup.detections.len());
    black_box(large_warmup);

    let mut criterion = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10))
        .configure_from_args();
    criterion.bench_function("koharu_layout_rfdetr_seg_2xl/inference", |bencher| {
        bencher.iter(|| {
            let detections = model
                .inference(black_box(&image))
                .expect("KoharuLayout RF-DETR inference failed");
            black_box(detections);
        });
    });
    criterion.bench_function(
        "koharu_layout_rfdetr_seg_2xl/inference/2738x3840",
        |bencher| {
            bencher.iter(|| {
                let detections = model
                    .inference(black_box(&large_image))
                    .expect("KoharuLayout RF-DETR inference failed");
                black_box(detections);
            });
        },
    );
    criterion.final_summary();
    Ok(())
}
