use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{Arc, Mutex},
};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use image::{
    DynamicImage, GenericImageView as _, GrayImage, ImageBuffer, ImageFormat, Luma, Rgb, RgbImage,
    Rgba, RgbaImage,
};
use imageproc::{
    distance_transform::{Norm, distance_transform_mut},
    morphology::dilate,
    region_labelling::{Connectivity, connected_components},
};
use koharu_ml::lama::{InpaintRequest, LaMa};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, BubbleRegion, EntityOrigin, Geometry, Origin,
    RasterLayer, RasterLayerKind, Region, RegionSpec,
};

use super::{StageInput, StageProcessor, finish, generation};
use crate::{InpaintingModel, ModelCell};

const PRODUCER: &str = "dev.koharu.pipeline.inpainting";

pub(super) struct Processor {
    config: InpaintingModel,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
}

impl Processor {
    pub(super) fn new(config: InpaintingModel, device: koharu_ml::Device) -> Result<Self> {
        Ok(Self {
            config,
            device,
            model: ModelCell::new(),
        })
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        match self.config {
            InpaintingModel::LaMa {} => "lama",
        }
    }

    fn skip(&self, input: &StageInput) -> Result<bool> {
        if input.inpainting_mask.is_some() {
            return Ok(false);
        }
        let source = AssetRole::new("source")?;
        for entity in input.scene.children(input.page)? {
            if input
                .scene
                .component::<RasterLayer>(entity)?
                .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
                && input.scene.asset(entity, &source)?.is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn unload(&self) -> bool {
        self.model.unload()
    }

    async fn load(&self) -> Result<()> {
        self.model
            .ensure(|| Model::load(self.device.clone(), &self.config))
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        self.model
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| anyhow!("inpainting model is not loaded"))?
            .run(input)
            .await
    }
}

enum Model {
    LaMa(Arc<Mutex<LaMa>>),
}

impl Model {
    async fn load(device: koharu_ml::Device, config: &InpaintingModel) -> Result<Self> {
        match config {
            InpaintingModel::LaMa {} => {
                Ok(Self::LaMa(Arc::new(Mutex::new(LaMa::load(device).await?))))
            }
        }
    }

    async fn run(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let mut prepared = prepare(&input).await?;
        if prepared.mask.as_raw().iter().all(|value| *value == 0) {
            return finish(input.scene.edit());
        }
        let mask = prepared.mask.clone();
        let original = prepared.original.clone();
        let cleanup = prepared.cleanup.take();
        let cleanup_entity = prepared.cleanup_entity;
        let (model_name, image) = match self {
            Self::LaMa(model) => {
                let model = model.clone();
                (
                    "lama",
                    tokio_rayon::spawn(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("LaMa model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            PaperMasks {
                                text: &prepared.text_mask,
                                covered: prepared.unscoped_mask.as_ref().unwrap_or(&prepared.mask),
                            },
                            &prepared.flat_fill_regions,
                            LAMA_TILE_CONTEXT,
                            // A fixed-size LaMa engine gets its crop grown with
                            // more page context; mirroring a smaller crop out
                            // to the engine's size would also copy its holes.
                            |width, height| model.crop_size(width, height),
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(
                                    image,
                                    mask,
                                    &InpaintRequest::default(),
                                )?))
                            },
                        )
                    })
                    .await?,
                )
            }
        };
        let page = input.page;
        let manual = input.inpainting_mask.is_some();
        let mut edit = if manual {
            input.scene.edit()
        } else {
            input.scene.edit_as(generation(PRODUCER, model_name)?)
        };
        edit.observe_assets(page)?;
        if let Some(entity) = cleanup_entity {
            edit.observe::<RasterLayer>(entity)?;
            edit.observe_assets(entity)?;
        }
        let image = image.to_rgba8();
        if image.dimensions() != original.dimensions() || image.dimensions() != mask.dimensions() {
            bail!("inpainted image dimensions do not match page {page}");
        }
        let mut overlay = if manual {
            cleanup.unwrap_or_else(|| RgbaImage::new(image.width(), image.height()))
        } else {
            RgbaImage::new(image.width(), image.height())
        };
        for (x, y, target) in overlay.enumerate_pixels_mut() {
            if mask.get_pixel(x, y)[0] < 127 {
                continue;
            }
            let generated = image.get_pixel(x, y);
            // Keep the complete cleanup mask opaque. Transparent texels next to
            // edited pixels let linear canvas sampling reveal the artwork below.
            *target = Rgba([generated[0], generated[1], generated[2], 255]);
        }
        let mut bytes = Cursor::new(Vec::new());
        let width = overlay.width();
        let height = overlay.height();
        DynamicImage::ImageRgba8(overlay).write_to(&mut bytes, ImageFormat::Png)?;
        let cleanup_entity = if let Some(entity) = cleanup_entity {
            if manual {
                let mut layer = input
                    .scene
                    .component::<RasterLayer>(entity)?
                    .context("cleanup entity has no raster layer component")?;
                let generated = layer.origin != Origin::User
                    || input
                        .scene
                        .component::<EntityOrigin>(entity)?
                        .is_some_and(|origin| origin.origin != Origin::User);
                if generated {
                    edit.promote_entity_to_user(entity)?;
                    layer.origin = Origin::User;
                    edit.set(entity, &layer)?;
                }
            }
            entity
        } else {
            let entity = edit.add_entity(page, At::Start)?;
            edit.set(
                entity,
                &RasterLayer {
                    origin: Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: RasterLayerKind::Cleanup,
                },
            )?;
            entity
        };
        edit.set_asset(
            cleanup_entity,
            &AssetRole::new("source")?,
            AssetInput::new(
                Arc::<[u8]>::from(bytes.into_inner()),
                "image/png",
                AssetMetadata {
                    width: Some(width),
                    height: Some(height),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        finish(edit)
    }
}

#[derive(Clone, Debug)]
struct FlatFillRegion {
    bounds: [u32; 4],
    polygon: Vec<(f32, f32)>,
}

fn flat_fill_regions(input: &StageInput, width: u32, height: u32) -> Result<Vec<FlatFillRegion>> {
    let mut regions = Vec::new();
    for entity in input.scene.descendants(input.page)? {
        let id = entity.id();
        let is_bubble = input
            .scene
            .component::<Region>(id)?
            .is_some_and(|region| region.kind == BubbleRegion::kind());
        if !is_bubble {
            continue;
        }
        let Some(geometry) = input.scene.component::<Geometry>(id)? else {
            continue;
        };
        let polygon = geometry
            .points
            .iter()
            .map(|point| (point.x as f32, point.y as f32))
            .collect::<Vec<_>>();
        if polygon.len() < 3 {
            continue;
        }
        let (mut left, mut top) = (f32::INFINITY, f32::INFINITY);
        let (mut right, mut bottom) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &(x, y) in &polygon {
            left = left.min(x);
            top = top.min(y);
            right = right.max(x);
            bottom = bottom.max(y);
        }
        let bounds = [
            left.floor().clamp(0.0, width as f32) as u32,
            top.floor().clamp(0.0, height as f32) as u32,
            right.ceil().clamp(0.0, width as f32) as u32,
            bottom.ceil().clamp(0.0, height as f32) as u32,
        ];
        if bounds[2] > bounds[0] && bounds[3] > bounds[1] {
            regions.push(FlatFillRegion { bounds, polygon });
        }
    }
    Ok(regions)
}

struct InpaintInput {
    image: Arc<DynamicImage>,
    original: Arc<DynamicImage>,
    cleanup_entity: Option<koharu_scene::EntityId>,
    cleanup: Option<RgbaImage>,
    mask: GrayImage,
    text_mask: GrayImage,
    unscoped_mask: Option<GrayImage>,
    flat_fill_regions: Vec<FlatFillRegion>,
}

async fn prepare(input: &StageInput) -> Result<InpaintInput> {
    let page = input.page;
    let original = input.require_page_image().await?;
    let cleanup_entity = input.scene.children(page)?.find(|entity| {
        input
            .scene
            .component::<RasterLayer>(*entity)
            .ok()
            .flatten()
            .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
    });
    let cleanup = if let Some(entity) = cleanup_entity {
        input
            .images
            .get(&input.scene, entity, "source")
            .await?
            .map(|image| image.to_rgba8())
    } else {
        None
    };
    if cleanup
        .as_ref()
        .is_some_and(|image| image.dimensions() != original.dimensions())
    {
        bail!("cleanup layer dimensions do not match page {page}");
    }
    let source = if input.inpainting_mask.is_some() {
        if let Some(cleanup) = cleanup.as_ref() {
            let mut composite = original.to_rgba8();
            image::imageops::overlay(&mut composite, cleanup, 0, 0);
            Arc::new(DynamicImage::ImageRgba8(composite))
        } else {
            original.clone()
        }
    } else {
        original.clone()
    };
    let mut mask = GrayImage::new(source.width(), source.height());
    let mut text_mask = GrayImage::new(source.width(), source.height());
    if let Some(transient) = &input.inpainting_mask {
        let layer = image::load_from_memory(&transient.png)?.to_luma8();
        if layer.dimensions() != mask.dimensions() {
            bail!("inpainting mask dimensions do not match page {page}");
        }
        mask = layer;
    } else {
        for role in ["text-mask", "coo-mask"] {
            if let Some(image) = input.images.get(&input.scene, page, role).await? {
                let layer = image.to_luma8();
                if layer.dimensions() != mask.dimensions() {
                    bail!("{role} dimensions do not match page {page}");
                }
                for (target, source) in mask.iter_mut().zip(layer.as_raw()) {
                    *target = (*target).max(*source);
                }
                if role == "text-mask" {
                    text_mask = layer;
                }
            }
        }
    }
    let flat_fill_regions = flat_fill_regions(input, source.width(), source.height())?;
    // A regional edit limits writes. Lettering outside it is still a hole
    // when reading paper or tracing a line under a component across the cut.
    let unscoped_mask = (input.region.is_some()
        && input.inpainting_mask.is_none()
        && !flat_fill_regions.is_empty())
    .then(|| mask.clone());
    if let Some(bounds) = input.region {
        for (x, y, pixel) in mask.enumerate_pixels_mut() {
            if f64::from(x + 1) <= bounds.x
                || f64::from(y + 1) <= bounds.y
                || f64::from(x) >= bounds.x + bounds.width
                || f64::from(y) >= bounds.y + bounds.height
            {
                *pixel = Luma([0]);
            }
        }
    }
    Ok(InpaintInput {
        image: source,
        original,
        cleanup_entity,
        cleanup,
        mask,
        text_mask,
        unscoped_mask,
        flat_fill_regions,
    })
}

const TILE_SIZE: u32 = 512;
// Known pixels kept around each model crop. LaMa uses wide context to continue
// texture into the mask.
const LAMA_TILE_CONTEXT: u32 = 128;
const PAPER_RING: u8 = 4;
const PAPER_MIN_PIXELS: usize = 16;
const PLAIN_CONTRAST: u8 = 16;
const FLAT_FILL_EDGE_MARGIN: u32 = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
struct InpaintTile {
    components: Vec<u32>,
    core: [u32; 4],
    crop: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MaskComponent {
    label: u32,
    bounds: [u32; 4],
}

struct PaperMasks<'a> {
    text: &'a GrayImage,
    covered: &'a GrayImage,
}

// Fill lettering on flat balloon paper before growing the remaining model crops.
fn inpaint_tiled(
    image: &DynamicImage,
    mask: &GrayImage,
    paper_masks: PaperMasks<'_>,
    flat_fill_regions: &[FlatFillRegion],
    context: u32,
    crop_size: impl Fn(u32, u32) -> Option<(u32, u32)>,
    mut inference: impl FnMut(&DynamicImage, &GrayImage) -> Result<DynamicImage>,
) -> Result<DynamicImage> {
    ensure!(
        [mask, paper_masks.text, paper_masks.covered]
            .iter()
            .all(|mask| mask.dimensions() == image.dimensions()),
        "image and mask dimensions differ: image={:?}, mask={:?}, text_mask={:?}, covered={:?}",
        image.dimensions(),
        mask.dimensions(),
        paper_masks.text.dimensions(),
        paper_masks.covered.dimensions()
    );

    let mut output = image.to_rgb8();
    let mut pending_mask = mask.clone();
    fill_uniform_regions(
        &mut output,
        &mut pending_mask,
        paper_masks,
        flat_fill_regions,
    );
    if pending_mask.iter().all(|&value| value < 127) {
        return Ok(DynamicImage::ImageRgb8(output));
    }
    let (component_labels, mut tiles) = inpaint_tiles(&pending_mask, context);
    for tile in &mut tiles {
        let [left, top, right, bottom] = tile.crop;
        if let Some(size) = crop_size(right - left, bottom - top) {
            tile.crop = fixed_size_crop(tile.crop, size, mask.width(), mask.height());
        }
    }
    for tile in &tiles {
        let [left, top, right, bottom] = tile.crop;
        let crop_width = right - left;
        let crop_height = bottom - top;
        let crop_image =
            image::imageops::crop_imm(&output, left, top, crop_width, crop_height).to_image();
        let crop_mask = crop_tile_mask(&pending_mask, tile);

        let generated = inference(&DynamicImage::ImageRgb8(crop_image), &crop_mask)?;
        let generated = if generated.dimensions() == (crop_width, crop_height) {
            generated.to_rgb8()
        } else {
            generated
                .resize_exact(
                    crop_width,
                    crop_height,
                    image::imageops::FilterType::Lanczos3,
                )
                .to_rgb8()
        };
        composite_generated(
            &mut output,
            &mut pending_mask,
            &component_labels,
            tile,
            &generated,
        );
    }
    Ok(DynamicImage::ImageRgb8(output))
}

/// The detected balloon and its readable paper, rasterized once for all components.
struct BalloonCoverage {
    bounds: [u32; 4],
    pixels: GrayImage,
}

impl BalloonCoverage {
    fn new(region: &FlatFillRegion) -> Self {
        let [left, top, right, bottom] = region.bounds;
        let mut pixels = GrayImage::new(right - left, bottom - top);
        let mut crossings = Vec::new();
        for (row, y) in pixels
            .chunks_exact_mut((right - left) as usize)
            .zip(top..bottom)
        {
            let y = y as f32 + 0.5;
            crossings.clear();
            let mut previous = *region.polygon.last().unwrap();
            for &current in &region.polygon {
                if (previous.1 > y) != (current.1 > y) {
                    crossings.push(
                        (previous.0 - current.0) * (y - current.1) / (previous.1 - current.1)
                            + current.0,
                    );
                }
                previous = current;
            }
            crossings.sort_unstable_by(f32::total_cmp);
            // Even-odd spans contain pixel centers in [start, end).
            let offset = |x: f32| (((x - 0.5).ceil() as u32).clamp(left, right) - left) as usize;
            for &[start, end] in crossings.as_chunks::<2>().0 {
                row[offset(start)..offset(end)].fill(255);
            }
        }
        Self {
            bounds: region.bounds,
            pixels,
        }
    }

    fn holds(&self, x: u32, y: u32) -> bool {
        let [left, top, right, bottom] = self.bounds;
        (left..right).contains(&x)
            && (top..bottom).contains(&y)
            && self.pixels.get_pixel(x - left, y - top)[0] != 0
    }

    fn reads_paper(&self, x: u32, y: u32) -> bool {
        let edge = FLAT_FILL_EDGE_MARGIN;
        x >= edge
            && y >= edge
            && self.holds(x, y)
            && self.holds(x - edge, y)
            && self.holds(x.saturating_add(edge), y)
            && self.holds(x, y - edge)
            && self.holds(x, y.saturating_add(edge))
    }
}

/// Adapted from koharu-rs/koharu@43ea786d. Detection supplies the grown text
/// mask. Only its pixels inside a balloon are eligible for paper fill, while
/// every hole in the original combined mask is excluded from paper samples.
fn fill_uniform_regions(
    output: &mut RgbImage,
    pending_mask: &mut GrayImage,
    paper_masks: PaperMasks<'_>,
    regions: &[FlatFillRegion],
) {
    let PaperMasks {
        text: text_mask,
        covered,
    } = paper_masks;
    if regions.is_empty() || text_mask.iter().all(|&value| value < 127) {
        return;
    }
    let (labels, components) = mask_components(text_mask);
    let balloons: Vec<_> = regions.iter().map(BalloonCoverage::new).collect();
    for component in components {
        let [left, top, right, bottom] = component.bounds;
        let mut candidates = balloons
            .iter()
            .filter(|balloon| {
                let [x0, y0, x1, y1] = balloon.bounds;
                left < x1 && x0 < right && top < y1 && y0 < bottom
            })
            .peekable();
        if candidates.peek().is_none() {
            continue;
        }
        let points: Vec<_> = (top..bottom)
            .flat_map(|y| (left..right).map(move |x| (x, y)))
            .filter(|&(x, y)| labels.get_pixel(x, y)[0] == component.label)
            .collect();
        if !points
            .iter()
            .any(|&(x, y)| pending_mask.get_pixel(x, y)[0] >= 127)
        {
            continue;
        }
        // A lettering straddling a balloon outline belongs to the model.
        let Some((balloon, inside)) = candidates
            .map(|balloon| {
                let inside = points.iter().filter(|&&(x, y)| balloon.holds(x, y)).count();
                (balloon, inside)
            })
            .max_by_key(|&(_, inside)| inside)
        else {
            continue;
        };
        if inside < points.len() - points.len() / 10 {
            continue;
        }
        fill_paper_component(
            output,
            pending_mask,
            covered,
            balloon,
            component.bounds,
            &points,
        );
    }
}

fn fill_paper_component(
    output: &mut RgbImage,
    pending_mask: &mut GrayImage,
    covered: &GrayImage,
    balloon: &BalloonCoverage,
    bounds: [u32; 4],
    points: &[(u32, u32)],
) {
    let [left, top, right, bottom] = expand_bounds(
        bounds,
        u32::from(PAPER_RING),
        output.width(),
        output.height(),
    );
    let mut distance = GrayImage::new(right - left, bottom - top);
    for &(x, y) in points {
        distance.put_pixel(x - left, y - top, Luma([255]));
    }
    distance_transform_mut(&mut distance, Norm::LInf);
    let ring: Vec<_> = distance
        .enumerate_pixels()
        .filter(|&(x, y, d)| {
            (1..=PAPER_RING).contains(&d[0]) && covered.get_pixel(left + x, top + y)[0] < 127
        })
        .map(|(x, y, _)| (left + x, top + y))
        .collect();
    // A fully masked balloon can have a uniform exterior. Its paper must
    // come from exposed pixels inside the outline.
    let samples: Vec<_> = ring
        .iter()
        .copied()
        .filter(|&(x, y)| balloon.reads_paper(x, y))
        .collect();
    let Some(paper) = paper_median(output, &samples) else {
        return;
    };
    let plain = ring
        .iter()
        .all(|&(x, y)| contrast(*output.get_pixel(x, y), paper) <= PLAIN_CONTRAST);
    let protected = if plain && points.iter().all(|&(x, y)| balloon.holds(x, y)) {
        None
    } else {
        let mut deviations = [0u64; 3];
        for &(x, y) in &samples {
            let pixel = output.get_pixel(x, y);
            for channel in 0..3 {
                deviations[channel] += u64::from(pixel[channel].abs_diff(paper[channel]));
            }
        }
        if deviations
            .iter()
            .any(|&sum| sum >= 10 * samples.len() as u64)
        {
            return;
        }

        // Only ink touching the mask can continue underneath it. Include
        // softer edges while following the stroke through the lettering.
        let ink = GrayImage::from_fn(distance.width(), distance.height(), |x, y| {
            let d = distance.get_pixel(x, y)[0];
            let (x, y) = (left + x, top + y);
            let off_paper = contrast(*output.get_pixel(x, y), paper);
            let ink = if d == 0 {
                off_paper > PLAIN_CONTRAST / 2
            } else {
                d == 1 && covered.get_pixel(x, y)[0] < 127 && off_paper > PLAIN_CONTRAST
            };
            Luma([u8::from(ink)])
        });
        let labels = label_binary_mask(&ink);
        let label = |x, y| labels.get_pixel(x - left, y - top)[0];
        let mut contacts = ring.iter().map(|&(x, y)| label(x, y)).filter(|&id| id != 0);
        let crossing = contacts.next();
        // Separate traces can be two ends of a rule obscured by lettering.
        // Leave the intervening paper available for the model to reconnect.
        if let Some(id) = crossing
            && (contacts.any(|other| other != id)
                || !points.iter().any(|&(x, y)| label(x, y) == id))
        {
            return;
        }
        let mut keep = GrayImage::new(distance.width(), distance.height());
        for &(x, y) in points {
            if !balloon.holds(x, y) || crossing == Some(label(x, y)) {
                keep.put_pixel(x - left, y - top, Luma([255]));
            }
        }
        Some(dilate(&keep, Norm::LInf, 1))
    };
    for &(x, y) in points {
        if pending_mask.get_pixel(x, y)[0] >= 127
            && protected
                .as_ref()
                .is_none_or(|keep| keep.get_pixel(x - left, y - top)[0] == 0)
        {
            output.put_pixel(x, y, paper);
            pending_mask.put_pixel(x, y, Luma([0]));
        }
    }
}

fn paper_median(image: &RgbImage, samples: &[(u32, u32)]) -> Option<Rgb<u8>> {
    (samples.len() >= PAPER_MIN_PIXELS).then(|| {
        Rgb(std::array::from_fn(|channel| {
            let mut values: Vec<_> = samples
                .iter()
                .map(|&(x, y)| image.get_pixel(x, y)[channel])
                .collect();
            *values.select_nth_unstable(samples.len() / 2).1
        }))
    })
}

fn contrast(pixel: Rgb<u8>, paper: Rgb<u8>) -> u8 {
    (0..3)
        .map(|channel| pixel[channel].abs_diff(paper[channel]))
        .max()
        .unwrap()
}

fn inpaint_tiles(
    mask: &GrayImage,
    context: u32,
) -> (ImageBuffer<Luma<u32>, Vec<u32>>, Vec<InpaintTile>) {
    let (labels, components) = mask_components(mask);
    let mut bounded_tiles: Vec<InpaintTile> = Vec::new();
    let mut split_tiles = Vec::new();

    for component in components {
        let [left, top, right, bottom] = component.bounds;
        if right - left <= TILE_SIZE && bottom - top <= TILE_SIZE {
            let best = bounded_tiles
                .iter()
                .enumerate()
                .filter_map(|(index, tile)| {
                    let bounds = union_bounds(tile.core, component.bounds);
                    ((bounds[2] - bounds[0] <= TILE_SIZE) && (bounds[3] - bounds[1] <= TILE_SIZE))
                        .then_some((bounds_area(bounds) - bounds_area(tile.core), index, bounds))
                })
                .min_by_key(|(growth, index, _)| (*growth, *index));
            if let Some((_, index, bounds)) = best {
                let tile = &mut bounded_tiles[index];
                tile.components.push(component.label);
                tile.core = bounds;
                tile.crop = expand_bounds(bounds, context, mask.width(), mask.height());
            } else {
                bounded_tiles.push(InpaintTile {
                    components: vec![component.label],
                    core: component.bounds,
                    crop: expand_bounds(component.bounds, context, mask.width(), mask.height()),
                });
            }
            continue;
        }

        let mut core_top = top;
        while core_top < bottom {
            let core_bottom = core_top.saturating_add(TILE_SIZE).min(bottom);
            let mut core_left = left;
            while core_left < right {
                let core_right = core_left.saturating_add(TILE_SIZE).min(right);
                let core = [core_left, core_top, core_right, core_bottom];
                if let Some(owned_bounds) = component_bounds_in(&labels, component.label, core) {
                    split_tiles.push(InpaintTile {
                        components: vec![component.label],
                        core,
                        crop: expand_bounds(owned_bounds, context, mask.width(), mask.height()),
                    });
                }
                core_left = core_right;
            }
            core_top = core_bottom;
        }
    }

    bounded_tiles.append(&mut split_tiles);
    bounded_tiles.sort_by_key(|tile| (tile.core[1], tile.core[0], tile.core[3], tile.core[2]));
    (labels, bounded_tiles)
}

fn label_binary_mask(mask: &GrayImage) -> ImageBuffer<Luma<u32>, Vec<u32>> {
    // imageproc#824: a single foreground pixel needs label 1 and background 0.
    if mask.dimensions() == (1, 1) {
        return ImageBuffer::from_pixel(1, 1, Luma([u32::from(mask.get_pixel(0, 0)[0] != 0)]));
    }
    connected_components(mask, Connectivity::Eight, Luma([0]))
}

fn mask_components(mask: &GrayImage) -> (ImageBuffer<Luma<u32>, Vec<u32>>, Vec<MaskComponent>) {
    let binary = GrayImage::from_fn(mask.width(), mask.height(), |x, y| {
        Luma([(mask.get_pixel(x, y)[0] >= 127) as u8 * u8::MAX])
    });
    let labels = label_binary_mask(&binary);
    let mut bounds = Vec::<Option<[u32; 4]>>::new();
    for (x, y, pixel) in labels.enumerate_pixels() {
        let label = pixel[0] as usize;
        if label == 0 {
            continue;
        }
        if label >= bounds.len() {
            bounds.resize(label + 1, None);
        }
        if let Some(bounds) = &mut bounds[label] {
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x + 1);
            bounds[3] = bounds[3].max(y + 1);
        } else {
            bounds[label] = Some([x, y, x + 1, y + 1]);
        }
    }
    let components = bounds
        .into_iter()
        .enumerate()
        .skip(1)
        .filter_map(|(label, bounds)| {
            bounds.map(|bounds| MaskComponent {
                label: label as u32,
                bounds,
            })
        })
        .collect();
    (labels, components)
}

fn union_bounds(left: [u32; 4], right: [u32; 4]) -> [u32; 4] {
    [
        left[0].min(right[0]),
        left[1].min(right[1]),
        left[2].max(right[2]),
        left[3].max(right[3]),
    ]
}

fn bounds_area([left, top, right, bottom]: [u32; 4]) -> u64 {
    u64::from(right - left) * u64::from(bottom - top)
}

fn expand_bounds(
    [left, top, right, bottom]: [u32; 4],
    context: u32,
    width: u32,
    height: u32,
) -> [u32; 4] {
    [
        left.saturating_sub(context),
        top.saturating_sub(context),
        right.saturating_add(context).min(width),
        bottom.saturating_add(context).min(height),
    ]
}

/// Grows `crop` to `size` (width, height), or to the whole page where the page
/// is smaller, keeping it centred where the page allows. `crop` must fit in
/// `size`.
fn fixed_size_crop(crop: [u32; 4], size: (u32, u32), width: u32, height: u32) -> [u32; 4] {
    let axis = |start: u32, end: u32, length: u32, size: u32| {
        if length <= size {
            return (0, length);
        }
        let start = ((start + end) / 2)
            .saturating_sub(size / 2)
            .min(length - size);
        (start, start + size)
    };
    let (left, right) = axis(crop[0], crop[2], width, size.0);
    let (top, bottom) = axis(crop[1], crop[3], height, size.1);
    [left, top, right, bottom]
}

fn component_bounds_in(
    labels: &ImageBuffer<Luma<u32>, Vec<u32>>,
    label: u32,
    [region_left, region_top, region_right, region_bottom]: [u32; 4],
) -> Option<[u32; 4]> {
    let mut left = region_right;
    let mut top = region_bottom;
    let mut right = 0;
    let mut bottom = 0;
    for y in region_top..region_bottom {
        for x in region_left..region_right {
            if labels.get_pixel(x, y)[0] == label {
                left = left.min(x);
                top = top.min(y);
                right = right.max(x + 1);
                bottom = bottom.max(y + 1);
            }
        }
    }
    (right > left && bottom > top).then_some([left, top, right, bottom])
}

fn crop_tile_mask(mask: &GrayImage, tile: &InpaintTile) -> GrayImage {
    let [left, top, right, bottom] = tile.crop;
    let mut crop = GrayImage::new(right - left, bottom - top);
    for y in top..bottom {
        for x in left..right {
            if mask.get_pixel(x, y)[0] >= 127 {
                crop.put_pixel(x - left, y - top, Luma([u8::MAX]));
            }
        }
    }
    crop
}

/// Writes a tile's fill for the mask it owns and unmasks those pixels, so the
/// tiles that continue the same component see the fill as context instead of
/// generating their overlap again and meeting it at a seam.
fn composite_generated(
    output: &mut RgbImage,
    pending_mask: &mut GrayImage,
    labels: &ImageBuffer<Luma<u32>, Vec<u32>>,
    tile: &InpaintTile,
    generated: &RgbImage,
) {
    let [left, top, _, _] = tile.crop;
    let [core_left, core_top, core_right, core_bottom] = tile.core;
    for y in core_top..core_bottom {
        for x in core_left..core_right {
            if tile
                .components
                .binary_search(&labels.get_pixel(x, y)[0])
                .is_ok()
            {
                output.put_pixel(x, y, *generated.get_pixel(x - left, y - top));
                pending_mask.put_pixel(x, y, Luma([0]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn transient_inpainting_mask_replaces_persistent_page_masks() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let source = DynamicImage::new_rgb8(8, 8);
        let persistent = DynamicImage::ImageLuma8(GrayImage::from_pixel(8, 8, Luma([255])));
        let encode = |image: &DynamicImage| {
            let mut bytes = Cursor::new(Vec::new());
            image.write_to(&mut bytes, ImageFormat::Png).unwrap();
            Arc::<[u8]>::from(bytes.into_inner())
        };
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 8.0, 8.0),
                    koharu_scene::At::End,
                )?;
                for (role, image) in [("source", &source), ("text-mask", &persistent)] {
                    edit.set_asset(
                        id,
                        &AssetRole::new(role)?,
                        AssetInput::new(
                            encode(image),
                            "image/png",
                            AssetMetadata {
                                width: Some(8),
                                height: Some(8),
                                attributes: BTreeMap::new(),
                            },
                        ),
                    )?;
                }
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let page = page.unwrap();
        let mut transient = GrayImage::new(8, 8);
        transient.put_pixel(3, 4, Luma([255]));
        let input = StageInput::new(
            snapshot,
            page,
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            Some(crate::InpaintingMask {
                page,
                png: encode(&DynamicImage::ImageLuma8(transient)),
            }),
        );

        let prepared = prepare(&input).await.unwrap();
        assert_eq!(prepared.mask.get_pixel(3, 4), &Luma([255]));
        assert_eq!(prepared.mask.get_pixel(0, 0), &Luma([0]));
        assert!(prepared.text_mask.pixels().all(|pixel| pixel[0] == 0));
    }

    #[tokio::test]
    async fn automatic_inpainting_skips_committed_cleanup() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 8.0, 8.0),
                    koharu_scene::At::End,
                )?;
                let cleanup = edit.add_entity(id, At::Start)?;
                edit.set(
                    cleanup,
                    &RasterLayer {
                        origin: Origin::User,
                        name: "Cleanup".to_owned(),
                        kind: RasterLayerKind::Cleanup,
                    },
                )?;
                edit.set_asset(
                    cleanup,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        Arc::<[u8]>::from([0]),
                        "image/png",
                        AssetMetadata {
                            width: Some(8),
                            height: Some(8),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let page = page.unwrap();
        let automatic = StageInput::new(
            snapshot.clone(),
            page,
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            None,
        );
        let manual = StageInput::new(
            snapshot,
            page,
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            Some(crate::InpaintingMask {
                page,
                png: Arc::<[u8]>::from([]),
            }),
        );
        let processor = Processor::new(InpaintingModel::LaMa {}, koharu_ml::Device::cpu()).unwrap();

        assert!(processor.skip(&automatic).unwrap());
        assert!(!processor.skip(&manual).unwrap());
    }

    fn rectangle_region([left, top, right, bottom]: [u32; 4]) -> FlatFillRegion {
        FlatFillRegion {
            bounds: [left, top, right, bottom],
            polygon: vec![
                (left as f32, top as f32),
                (right as f32, top as f32),
                (right as f32, bottom as f32),
                (left as f32, bottom as f32),
            ],
        }
    }

    #[test]
    fn uniform_mask_background_is_filled_without_inference() {
        let mut image = RgbImage::from_pixel(96, 96, Rgb([240, 241, 242]));
        let mut mask = GrayImage::new(96, 96);
        for y in 40..56 {
            for x in 32..64 {
                image.put_pixel(x, y, Rgb([20, 20, 20]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            PaperMasks {
                text: &mask,
                covered: &mask,
            },
            &[rectangle_region([0, 0, 96, 96])],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |_, _| {
                calls += 1;
                Ok(DynamicImage::new_rgb8(1, 1))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 0);
        assert!(output.pixels().all(|pixel| *pixel == Rgb([240, 241, 242])));
    }

    #[test]
    fn crossing_rule_and_its_border_stay_pending_beside_a_flat_filled_glyph() {
        assert_crossing_rule_stays_pending(Rgb([0; 3]));
    }

    #[test]
    fn gray_rule_crossing_white_paper_stays_pending() {
        assert_crossing_rule_stays_pending(Rgb([200; 3]));
    }

    fn assert_crossing_rule_stays_pending(ink: Rgb<u8>) {
        let paper = Rgb([255; 3]);
        let mut image = RgbImage::from_pixel(96, 96, paper);
        let mask = GrayImage::from_fn(96, 96, |x, y| {
            Luma([if (32..64).contains(&x) && (38..56).contains(&y) {
                255
            } else {
                0
            }])
        });
        // Only two pixels of the rule are exposed outside the grown mask.
        for x in 31..65 {
            image.put_pixel(x, 48, ink);
        }
        image.put_pixel(41, 41, Rgb([0; 3]));
        // A soft diagonal joins the rule under the mask.
        image.put_pixel(52, 49, Rgb([210; 3]));
        image.put_pixel(53, 50, Rgb([210; 3]));
        let mut calls = 0;
        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            PaperMasks {
                text: &mask,
                covered: &mask,
            },
            &[rectangle_region([0, 0, 96, 96])],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |tile, pending| {
                calls += 1;
                for (x, y, pixel) in pending.enumerate_pixels() {
                    let rule = (32..64).contains(&x) && (47..=49).contains(&y);
                    let tip = (51..=53).contains(&x) && y == 50
                        || (52..=54).contains(&x) && (50..=51).contains(&y);
                    assert_eq!(pixel[0] != 0, rule || tip, "({x}, {y})");
                }
                assert_eq!(tile.get_pixel(41, 41), Rgba([255; 4]));
                Ok(tile.clone())
            },
        )
        .unwrap()
        .to_rgb8();
        assert_eq!(calls, 1);
        assert!((31..65).all(|x| output.get_pixel(x, 48) == &ink));
    }

    #[test]
    fn white_outlined_lettering_leaves_the_gap_in_a_crossing_rule_pending() {
        let mask = GrayImage::from_fn(64, 48, |x, y| {
            let crossed = (8..32).contains(&x) && (12..30).contains(&y);
            let separate = (44..52).contains(&x) && (16..26).contains(&y);
            Luma([if crossed || separate { 255 } else { 0 }])
        });
        let image = RgbImage::from_fn(64, 48, |x, y| {
            let rule = y == 21 && x < 38 && !(17..23).contains(&x);
            let glyph = (x == 20 || x == 47) && (17..26).contains(&y);
            Rgb([if rule || glyph { 0 } else { 255 }; 3])
        });
        let mut calls = 0;
        inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            PaperMasks {
                text: &mask,
                covered: &mask,
            },
            &[rectangle_region([0, 0, 64, 48])],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |tile, pending| {
                calls += 1;
                for x in 8..32 {
                    assert_eq!(pending.get_pixel(x, 21)[0], 255, "rule at x={x}");
                }
                for y in 16..26 {
                    for x in 44..52 {
                        assert_eq!(pending.get_pixel(x, y)[0], 0);
                        assert_eq!(tile.get_pixel(x, y), Rgba([255; 4]));
                    }
                }
                Ok(tile.clone())
            },
        )
        .unwrap();
        assert_eq!(calls, 1);
    }

    #[test]
    fn a_detached_outline_in_the_paper_ring_allows_local_fill() {
        let mut image = RgbImage::from_pixel(32, 32, Rgb([255; 3]));
        let mask = GrayImage::from_fn(32, 32, |x, y| {
            if (10..22).contains(&x) && (12..20).contains(&y) {
                image.put_pixel(x, y, Rgb([0; 3]));
                Luma([255])
            } else {
                Luma([0])
            }
        });
        image.put_pixel(7, 16, Rgb([0; 3]));
        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            PaperMasks {
                text: &mask,
                covered: &mask,
            },
            &[rectangle_region([0, 0, 32, 32])],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |_, _| panic!("a detached outline does not require inference"),
        )
        .unwrap()
        .to_rgb8();
        for (x, y, pixel) in output.enumerate_pixels() {
            let expected = if (x, y) == (7, 16) { 0 } else { 255 };
            assert_eq!(*pixel, Rgb([expected; 3]));
        }
    }

    #[test]
    fn components_in_one_balloon_use_their_own_paper_or_go_to_the_model() {
        let mut image = RgbImage::from_fn(96, 96, |x, y| {
            if x >= 64 && y >= 64 {
                Rgb([if (x + y).is_multiple_of(2) { 0 } else { 160 }; 3])
            } else if x >= 64 {
                Rgb([198, 210, 230])
            } else {
                Rgb([250; 3])
            }
        });
        let mask = GrayImage::from_fn(96, 96, |x, y| {
            let target = [(24, 24), (72, 24), (72, 72)]
                .iter()
                .any(|&(left, top)| (left..left + 8).contains(&x) && (top..top + 8).contains(&y));
            if target {
                image.put_pixel(x, y, Rgb([0; 3]));
            }
            Luma([if target { 255 } else { 0 }])
        });
        let mut calls = 0;
        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            PaperMasks {
                text: &mask,
                covered: &mask,
            },
            &[rectangle_region([0, 0, 96, 96])],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |tile, pending| {
                calls += 1;
                for (x, y, pixel) in pending.enumerate_pixels() {
                    assert_eq!(
                        pixel[0] != 0,
                        (72..80).contains(&x) && (72..80).contains(&y)
                    );
                }
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    tile.width(),
                    tile.height(),
                    Rgb([1, 2, 3]),
                )))
            },
        )
        .unwrap()
        .to_rgb8();
        assert_eq!(calls, 1);
        assert_eq!(output.get_pixel(26, 26), &Rgb([250; 3]));
        assert_eq!(output.get_pixel(76, 26), &Rgb([198, 210, 230]));
        assert_eq!(output.get_pixel(76, 76), &Rgb([1, 2, 3]));
    }

    #[test]
    fn fully_masked_balloon_cannot_use_exterior_color_as_its_paper() {
        let balloon = FlatFillRegion {
            bounds: [4, 4, 28, 28],
            polygon: vec![(16.0, 4.0), (28.0, 16.0), (16.0, 28.0), (4.0, 16.0)],
        };
        let mask = GrayImage::from_fn(32, 32, |x, y| {
            let dx = (x as f32 + 0.5 - 16.0).abs();
            let dy = (y as f32 + 0.5 - 16.0).abs();
            Luma([if dx + dy <= 12.0 { 255 } else { 0 }])
        });
        let mut pending = mask.clone();
        let mut image = RgbImage::from_pixel(32, 32, Rgb([20; 3]));
        image.put_pixel(16, 16, Rgb([220; 3]));
        fill_uniform_regions(
            &mut image,
            &mut pending,
            PaperMasks {
                text: &mask,
                covered: &mask,
            },
            &[balloon],
        );
        assert_eq!(pending, mask);
        assert_eq!(image.get_pixel(16, 16), &Rgb([220; 3]));
    }

    #[test]
    fn paper_ring_excludes_neighboring_non_text_mask_pixels() {
        let mut image = RgbImage::from_pixel(24, 24, Rgb([225; 3]));
        let mut text = GrayImage::new(24, 24);
        text.put_pixel(10, 10, Luma([255]));
        image.put_pixel(10, 10, Rgb([0; 3]));
        let mut combined = text.clone();
        for y in 6..15 {
            for x in 12..15 {
                combined.put_pixel(x, y, Luma([255]));
                image.put_pixel(x, y, Rgb([0; 3]));
            }
        }
        let mut calls = 0;
        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &combined,
            PaperMasks {
                text: &text,
                covered: &combined,
            },
            &[rectangle_region([0, 0, 24, 24])],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |tile, pending| {
                calls += 1;
                assert_eq!(pending.get_pixel(10, 10)[0], 0);
                assert_eq!(pending.iter().filter(|&&value| value != 0).count(), 27);
                Ok(tile.clone())
            },
        )
        .unwrap()
        .to_rgb8();
        assert_eq!(calls, 1);
        assert_eq!(output.get_pixel(10, 10), &Rgb([225; 3]));
    }

    #[test]
    fn regional_fill_reads_past_the_full_mask_and_writes_only_inside_the_scope() {
        let paper = Rgb([248; 3]);
        let ink = Rgb([10; 3]);
        let mut image = RgbImage::from_pixel(32, 24, paper);
        let covered = GrayImage::from_fn(32, 24, |x, y| {
            if (8..24).contains(&x) && (8..16).contains(&y) {
                image.put_pixel(x, y, ink);
                Luma([255])
            } else {
                Luma([0])
            }
        });
        let active = GrayImage::from_fn(32, 24, |x, y| {
            if x < 16 {
                *covered.get_pixel(x, y)
            } else {
                Luma([0])
            }
        });
        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &active,
            PaperMasks {
                text: &covered,
                covered: &covered,
            },
            &[rectangle_region([0, 0, 32, 24])],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |_, _| panic!("flat paper requires no inference"),
        )
        .unwrap()
        .to_rgb8();
        for (x, y, pixel) in output.enumerate_pixels() {
            let outside = (16..24).contains(&x) && (8..16).contains(&y);
            assert_eq!(*pixel, if outside { ink } else { paper });
        }
    }

    #[test]
    fn single_pixel_page_defers_to_the_model_without_paper_samples() {
        let image = DynamicImage::new_rgb8(1, 1);
        let mask = GrayImage::from_pixel(1, 1, Luma([255]));
        let mut calls = 0;
        inpaint_tiled(
            &image,
            &mask,
            PaperMasks {
                text: &mask,
                covered: &mask,
            },
            &[rectangle_region([0, 0, 1, 1])],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |tile, pending| {
                calls += 1;
                assert_eq!(pending, &mask);
                Ok(tile.clone())
            },
        )
        .unwrap();
        assert_eq!(calls, 1);
    }

    #[test]
    fn uniform_bubbles_are_filled_independently_inside_one_textured_tile() {
        let mut image = RgbImage::new(200, 100);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x + y).is_multiple_of(2) {
                Rgb([20, 80, 140])
            } else {
                Rgb([220, 80, 120])
            };
        }
        for y in 10..90 {
            for x in 10..90 {
                image.put_pixel(x, y, Rgb([245, 245, 245]));
            }
            for x in 110..190 {
                image.put_pixel(x, y, Rgb([250, 240, 220]));
            }
        }
        let mut mask = GrayImage::new(200, 100);
        for y in 40..60 {
            for x in 30..60 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
            for x in 135..165 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            PaperMasks {
                text: &mask,
                covered: &mask,
            },
            &[
                rectangle_region([10, 10, 90, 90]),
                rectangle_region([110, 10, 190, 90]),
            ],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |_, _| {
                calls += 1;
                Ok(DynamicImage::new_rgb8(1, 1))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 0);
        assert_eq!(output.get_pixel(45, 50), &Rgb([245, 245, 245]));
        assert_eq!(output.get_pixel(150, 50), &Rgb([250, 240, 220]));
    }

    #[test]
    fn connected_mask_crossing_both_page_grid_axes_is_inferred_once() {
        let image = RgbImage::from_pixel(900, 900, Rgb([20, 40, 60]));
        let mut mask = GrayImage::new(900, 900);
        for y in 500..525 {
            for x in 500..525 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            PaperMasks {
                text: &mask,
                covered: &mask,
            },
            &[],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |tile, tile_mask| {
                calls += 1;
                assert_eq!(
                    tile_mask.pixels().filter(|pixel| pixel[0] >= 127).count(),
                    25 * 25
                );
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    tile.width(),
                    tile.height(),
                    Rgb([1, 2, 3]),
                )))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 1);
        assert_eq!(output.get_pixel(500, 500), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(524, 524), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(499, 499), &Rgb([20, 40, 60]));
    }

    #[test]
    fn oversized_component_tiles_mask_their_continuation_context() {
        let mut mask = GrayImage::new(1000, 300);
        for y in 100..120 {
            for x in 50..850 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }

        let (labels, tiles) = inpaint_tiles(&mask, LAMA_TILE_CONTEXT);

        assert_eq!(tiles.len(), 2);
        for tile in &tiles {
            assert!(tile.crop[2] - tile.crop[0] <= TILE_SIZE + LAMA_TILE_CONTEXT * 2);
            assert!(tile.crop[3] - tile.crop[1] <= TILE_SIZE + LAMA_TILE_CONTEXT * 2);
            let label = tile.components[0];
            let [left, top, right, bottom] = tile.core;
            let mut owned = 0;
            for y in top..bottom {
                for x in left..right {
                    if labels.get_pixel(x, y)[0] == label {
                        owned += 1;
                    }
                }
            }
            let masked = crop_tile_mask(&mask, tile)
                .pixels()
                .filter(|pixel| pixel[0] >= 127)
                .count();
            assert!(masked > owned);
        }
    }

    #[test]
    fn continuation_tiles_use_earlier_fills_as_context() {
        let image = RgbImage::from_pixel(1000, 300, Rgb([20, 40, 60]));
        let mut mask = GrayImage::new(1000, 300);
        for y in 100..120 {
            for x in 50..850 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = Vec::new();

        inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            PaperMasks {
                text: &mask,
                covered: &mask,
            },
            &[],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |tile, tile_mask| {
                calls.push((
                    tile.to_rgb8(),
                    tile_mask.pixels().filter(|pixel| pixel[0] >= 127).count(),
                ));
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    tile.width(),
                    tile.height(),
                    Rgb([calls.len() as u8, 0, 0]),
                )))
            },
        )
        .unwrap();

        // The second tile owns x 562..850 and its crop starts 128 px earlier,
        // at the top of the page, over pixels the first tile already filled.
        assert_eq!(calls.len(), 2);
        let (second, masked) = &calls[1];
        assert_eq!(*masked, (850 - 562) * 20);
        assert_eq!(second.get_pixel(0, 100), &Rgb([1, 0, 0]));
    }

    #[test]
    fn textured_mask_regions_are_inferred_as_bounded_tiles() {
        let mut image = RgbImage::new(1200, 700);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x + y).is_multiple_of(2) {
                Rgb([0, 40, 80])
            } else {
                Rgb([255, 215, 175])
            };
        }
        let original = image.clone();
        let mut mask = GrayImage::new(1200, 700);
        mask.put_pixel(10, 10, Luma([u8::MAX]));
        mask.put_pixel(1100, 600, Luma([u8::MAX]));
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            PaperMasks {
                text: &mask,
                covered: &mask,
            },
            &[],
            LAMA_TILE_CONTEXT,
            |_, _| None,
            |tile, _| {
                calls += 1;
                assert!(tile.width() <= LAMA_TILE_CONTEXT * 2 + 1);
                assert!(tile.height() <= LAMA_TILE_CONTEXT * 2 + 1);
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    tile.width(),
                    tile.height(),
                    Rgb([1, 2, 3]),
                )))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 2);
        assert_eq!(output.get_pixel(10, 10), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(1100, 600), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(500, 300), original.get_pixel(500, 300));
    }

    #[test]
    fn lama_crops_share_one_shape_and_keep_their_context() {
        let mut mask = GrayImage::new(1700, 2400);
        for (left, top, size) in [
            (0, 0, 40),
            (900, 1200, 300),
            (1650, 2350, 50),
            (400, 100, 700),
        ] {
            for y in top..(top + size).min(2400) {
                for x in left..(left + size).min(1700) {
                    mask.put_pixel(x, y, Luma([u8::MAX]));
                }
            }
        }
        // The largest crop a tile can need, which LaMa's largest engine holds.
        let size = TILE_SIZE + 2 * LAMA_TILE_CONTEXT;
        let (_, tiles) = inpaint_tiles(&mask, LAMA_TILE_CONTEXT);
        assert!(tiles.len() > 4);
        for tile in &tiles {
            let crop = fixed_size_crop(tile.crop, (size, size), 1700, 2400);
            assert_eq!(crop[2] - crop[0], size);
            assert_eq!(crop[3] - crop[1], size);
            assert!(crop[0] <= tile.crop[0] && crop[1] <= tile.crop[1]);
            assert!(crop[2] >= tile.crop[2] && crop[3] >= tile.crop[3]);
            assert!(crop[2] <= 1700 && crop[3] <= 2400);
        }
        assert_eq!(
            fixed_size_crop([10, 20, 300, 400], (size, 512), 600, 2400),
            [0, 0, 600, 512]
        );
    }
}
