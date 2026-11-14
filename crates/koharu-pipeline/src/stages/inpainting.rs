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
use imageproc::region_labelling::{Connectivity, connected_components};
use koharu_ml::{
    flux2_klein::{Flux2KleinInpaint, Flux2KleinInpaintOptions},
    lama::{InpaintRequest, LaMa},
};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, BubbleRegion, EntityOrigin, Geometry, Origin,
    RasterLayer, RasterLayerKind, Region, RegionSpec,
};
use serde::{Deserialize, Serialize};
use specta::Type;

use super::{StageInput, StageProcessor, finish, generation};
use crate::{InpaintingModel, ModelCell};

const PRODUCER: &str = "dev.koharu.pipeline.inpainting";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct Flux2KleinConfig {
    pub prompt: String,
}

impl Default for Flux2KleinConfig {
    fn default() -> Self {
        Self {
            prompt: "Remove the text and reconstruct the background.".to_owned(),
        }
    }
}

pub(super) struct Processor {
    config: InpaintingModel,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
}

impl Processor {
    pub(super) fn new(config: InpaintingModel, device: koharu_ml::Device) -> Result<Self> {
        match &config {
            InpaintingModel::LaMa {} => {}
            InpaintingModel::Flux2Klein(settings) => {
                ensure!(
                    !settings.prompt.contains('\0'),
                    "FLUX.2 prompt contains NUL"
                );
            }
        }

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
            InpaintingModel::Flux2Klein(_) => "flux2-klein",
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
    Flux {
        lama: Arc<Mutex<LaMa>>,
        model: Arc<Mutex<Flux2KleinInpaint>>,
        config: Flux2KleinConfig,
    },
}

// FLUX.2 refines a LaMa fill instead of generating the masked region from
// noise. From full noise it often paints blobs into white balloons and needs
// all four steps. Starting partway down the schedule from the LaMa fill takes
// fewer steps and repairs LaMa's smears. The reference image keeps the
// original crop so the model still sees what it is removing.
//
// Cheaper variants measured on the hard-page set and rejected for quality:
// one step leaves smears and drops detail; a half-scale reference saves a
// quarter of the time but leaves text remnants and invents details; no
// reference, or text conditioning without its 512-token padding, paints
// unrelated photos into balloons.
//
// LaMa's fill stays plausible near known pixels and turns to blur deeper
// inside large masks, such as big free text over artwork. On the hard-page set
// every balloon and small-text tile reaches at most 63 px from known pixels and
// refines cleanly in two steps, while the large free-text tiles reach 105 px
// or more and need a third step to redraw detail.
const FLUX_DEEP_MASK: u8 = 80;

// A uniform LaMa fill, such as a plain balloon the flat fill missed, is
// already exact. Fills with any visible variation still go to FLUX.2, which
// removes the faint remnants LaMa leaves.
const UNIFORM_FILL_MAX_DEVIATION: f64 = 2.0;

fn is_uniform_fill(filled: &RgbImage, mask: &GrayImage) -> bool {
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for (pixel, mask) in filled.pixels().zip(mask.pixels()) {
        if mask[0] >= 127 {
            for channel in 0..3 {
                channels[channel].push(pixel[channel]);
            }
        }
    }
    !channels[0].is_empty()
        && channels.iter().all(|values| {
            let mean =
                values.iter().map(|&value| f64::from(value)).sum::<f64>() / values.len() as f64;
            standard_deviation(values, mean) < UNIFORM_FILL_MAX_DEVIATION
        })
}

fn flux_refine_options(mask: &GrayImage) -> Flux2KleinInpaintOptions {
    let known = GrayImage::from_fn(mask.width(), mask.height(), |x, y| {
        Luma([u8::from(mask.get_pixel(x, y)[0] < 127) * u8::MAX])
    });
    let depth = imageproc::distance_transform::distance_transform(
        &known,
        imageproc::distance_transform::Norm::L1,
    );
    let deep = depth
        .as_raw()
        .iter()
        .any(|&distance| distance > FLUX_DEEP_MASK);
    Flux2KleinInpaintOptions {
        padding_mask_crop: None,
        // Four-step schedule: 0.5 runs its last two steps, 0.75 its last three.
        strength: if deep { 0.75 } else { 0.5 },
        num_inference_steps: 4,
        seed: 42,
    }
}

impl Model {
    async fn load(device: koharu_ml::Device, config: &InpaintingModel) -> Result<Self> {
        match config {
            InpaintingModel::LaMa {} => {
                Ok(Self::LaMa(Arc::new(Mutex::new(LaMa::load(device).await?))))
            }
            InpaintingModel::Flux2Klein(config) => {
                let (lama, model) =
                    tokio::try_join!(LaMa::load(device.clone()), Flux2KleinInpaint::load(device))?;
                Ok(Self::Flux {
                    lama: Arc::new(Mutex::new(lama)),
                    model: Arc::new(Mutex::new(model)),
                    config: config.clone(),
                })
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
                            &prepared.text_mask,
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
            Self::Flux {
                lama,
                model,
                config,
            } => {
                let lama = lama.clone();
                let model = model.clone();
                let config = config.clone();
                (
                    "flux2-klein",
                    tokio_rayon::spawn(move || -> Result<DynamicImage> {
                        let lama = lama
                            .lock()
                            .map_err(|_| anyhow!("LaMa model lock is poisoned"))?;
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("FLUX model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            FLUX_TILE_CONTEXT,
                            |_, _| None,
                            |image, mask| {
                                let filled =
                                    lama.inference(image, mask, &InpaintRequest::default())?;
                                if is_uniform_fill(&filled, mask) {
                                    return Ok(DynamicImage::ImageRgb8(filled));
                                }
                                let filled = DynamicImage::ImageRgb8(filled);
                                model.inference(
                                    &config.prompt,
                                    &filled,
                                    Some(image),
                                    &DynamicImage::ImageLuma8(mask.clone()),
                                    &flux_refine_options(mask),
                                )
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
    flat_fill_regions: Vec<FlatFillRegion>,
}

async fn prepare(input: &StageInput) -> Result<InpaintInput> {
    let page = input.page;
    let original = input
        .images
        .get(&input.scene, page, "source")
        .await?
        .ok_or_else(|| anyhow!("page {page} has no source image"))?;
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
    if let Some(bounds) = input.region {
        for (x, y, pixel) in mask.enumerate_pixels_mut() {
            if f64::from(x + 1) <= bounds.x
                || f64::from(y + 1) <= bounds.y
                || f64::from(x) >= bounds.x + bounds.width
                || f64::from(y) >= bounds.y + bounds.height
            {
                *pixel = Luma([0]);
                text_mask.put_pixel(x, y, Luma([0]));
            }
        }
    }
    let flat_fill_regions = flat_fill_regions(input, source.width(), source.height())?;
    Ok(InpaintInput {
        image: source,
        original,
        cleanup_entity,
        cleanup,
        mask,
        text_mask,
        flat_fill_regions,
    })
}

const TILE_SIZE: u32 = 512;
// Known pixels kept around each model crop. LaMa uses wide context to continue
// texture into the mask. FLUX.2's time grows with crop area, and with less
// context its refinement stays closer to the LaMa fill. At 128 px it sometimes
// painted unrelated objects into large free-text masks, and at 64 px it left
// LaMa's smears in place.
const LAMA_TILE_CONTEXT: u32 = 128;
const FLUX_TILE_CONTEXT: u32 = 96;
const UNIFORM_BACKGROUND_MIN_PIXELS: usize = 16;
const FLAT_FILL_EDGE_MARGIN: f32 = 3.0;

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

// BallonsTranslator avoids model inference for a text block when the non-text
// pixels inside its balloon are nearly uniform, and otherwise sends an enlarged
// block crop to the inpainter:
// https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/modules/inpaint/base.py#L168-L200
// Koharu uses the detected bubble polygons as those blocks. Uniform bubbles are
// filled first; only the remaining mask is split into bounded model crops.
fn inpaint_tiled(
    image: &DynamicImage,
    mask: &GrayImage,
    text_mask: &GrayImage,
    flat_fill_regions: &[FlatFillRegion],
    context: u32,
    crop_size: impl Fn(u32, u32) -> Option<u32>,
    mut inference: impl FnMut(&DynamicImage, &GrayImage) -> Result<DynamicImage>,
) -> Result<DynamicImage> {
    ensure!(
        image.dimensions() == mask.dimensions() && mask.dimensions() == text_mask.dimensions(),
        "image and mask dimensions differ: image={:?}, mask={:?}, text_mask={:?}",
        image.dimensions(),
        mask.dimensions(),
        text_mask.dimensions()
    );

    let mut output = image.to_rgb8();
    let mut pending_mask = mask.clone();
    fill_uniform_regions(&mut output, &mut pending_mask, text_mask, flat_fill_regions);
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

fn fill_uniform_regions(
    output: &mut RgbImage,
    pending_mask: &mut GrayImage,
    text_mask: &GrayImage,
    regions: &[FlatFillRegion],
) {
    for region in regions {
        let [left, top, right, bottom] = region.bounds;
        let mut rows = PolygonRows::new(&region.polygon);
        let mut targets = Vec::new();
        for y in top..bottom {
            rows.set_row(y);
            for x in left..right {
                if pending_mask.get_pixel(x, y)[0] >= 127
                    && text_mask.get_pixel(x, y)[0] >= 127
                    && rows.contains(x)
                {
                    targets.push((x, y));
                }
            }
        }
        if targets.is_empty() {
            continue;
        }
        let Some(color) = uniform_region_color(output, pending_mask, region) else {
            continue;
        };
        for (x, y) in targets {
            output.put_pixel(x, y, color);
            pending_mask.put_pixel(x, y, Luma([0]));
        }
    }
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

fn mask_components(mask: &GrayImage) -> (ImageBuffer<Luma<u32>, Vec<u32>>, Vec<MaskComponent>) {
    let binary = GrayImage::from_fn(mask.width(), mask.height(), |x, y| {
        Luma([(mask.get_pixel(x, y)[0] >= 127) as u8 * u8::MAX])
    });
    let labels = connected_components(&binary, Connectivity::Eight, Luma([0]));
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

/// Grows `crop` to `size` on each axis, or to the whole page where the page is
/// smaller, keeping it centred where the page allows. `crop` must fit in `size`.
fn fixed_size_crop(crop: [u32; 4], size: u32, width: u32, height: u32) -> [u32; 4] {
    let axis = |start: u32, end: u32, length: u32| {
        if length <= size {
            return (0, length);
        }
        let start = ((start + end) / 2)
            .saturating_sub(size / 2)
            .min(length - size);
        (start, start + size)
    };
    let (left, right) = axis(crop[0], crop[2], width);
    let (top, bottom) = axis(crop[1], crop[3], height);
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

fn uniform_region_color(
    image: &RgbImage,
    mask: &GrayImage,
    region: &FlatFillRegion,
) -> Option<Rgb<u8>> {
    let [left, top, right, bottom] = region.bounds;
    let mut rows = PolygonRows::new(&region.polygon);
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for y in top..bottom {
        rows.set_row(y);
        for x in left..right {
            if mask.get_pixel(x, y)[0] >= 127 || !rows.contains(x) || rows.near_edge(x) {
                continue;
            }
            let pixel = image.get_pixel(x, y);
            for channel in 0..3 {
                channels[channel].push(pixel[channel]);
            }
        }
    }
    if channels[0].len() < UNIFORM_BACKGROUND_MIN_PIXELS {
        return None;
    }

    let medians = channels.each_mut().map(|values| median(values));
    let deviations = std::array::from_fn::<_, 3, _>(|channel| {
        standard_deviation(&channels[channel], f64::from(medians[channel]))
    });
    let mean_deviation = deviations.iter().sum::<f64>() / deviations.len() as f64;
    let channel_spread = (deviations
        .iter()
        .map(|deviation| (deviation - mean_deviation).powi(2))
        .sum::<f64>()
        / deviations.len() as f64)
        .sqrt();
    let threshold = if channel_spread > 1.0 { 7.0 } else { 10.0 };
    (deviations.iter().copied().fold(0.0, f64::max) < threshold).then_some(Rgb(medians))
}

/// Even-odd containment and edge-margin tests for the pixel centers of one
/// polygon, evaluated a row at a time. Bubble polygons trace detector masks and
/// can have hundreds of vertices, so testing every edge for every pixel of a
/// bubble dominated inpainting on large pages. Each test keeps the per-edge
/// arithmetic of a direct per-pixel test, so results are identical.
struct PolygonRows<'a> {
    polygon: &'a [(f32, f32)],
    y: f32,
    /// Sorted x coordinates where polygon edges cross the current row.
    crossings: Vec<f32>,
    /// Edges whose bounds lie within the edge margin of the current row.
    nearby: Vec<((f32, f32), (f32, f32))>,
}

impl<'a> PolygonRows<'a> {
    // Skipped edges are farther than this from the row, which keeps their
    // squared distance well above the margin despite rounding.
    const NEARBY: f32 = FLAT_FILL_EDGE_MARGIN + 1.0;

    fn new(polygon: &'a [(f32, f32)]) -> Self {
        Self {
            polygon,
            y: 0.0,
            crossings: Vec::new(),
            nearby: Vec::new(),
        }
    }

    fn edges(&self) -> impl Iterator<Item = ((f32, f32), (f32, f32))> + '_ {
        let last = self.polygon[self.polygon.len() - 1];
        std::iter::once(last)
            .chain(self.polygon.iter().copied())
            .zip(self.polygon.iter().copied())
    }

    fn set_row(&mut self, y: u32) {
        let y = y as f32 + 0.5;
        let mut crossings = std::mem::take(&mut self.crossings);
        let mut nearby = std::mem::take(&mut self.nearby);
        crossings.clear();
        nearby.clear();
        for (previous, current) in self.edges() {
            if (current.1 > y) != (previous.1 > y) {
                crossings.push(
                    (previous.0 - current.0) * (y - current.1) / (previous.1 - current.1)
                        + current.0,
                );
            }
            if previous.1.min(current.1) - Self::NEARBY <= y
                && y <= previous.1.max(current.1) + Self::NEARBY
            {
                nearby.push((previous, current));
            }
        }
        crossings.sort_unstable_by(f32::total_cmp);
        self.y = y;
        self.crossings = crossings;
        self.nearby = nearby;
    }

    fn contains(&self, x: u32) -> bool {
        let x = x as f32 + 0.5;
        let right = self.crossings.len() - self.crossings.partition_point(|&edge| edge <= x);
        right % 2 == 1
    }

    fn near_edge(&self, x: u32) -> bool {
        let point = (x as f32 + 0.5, self.y);
        self.nearby.iter().any(|&(start, end)| {
            if point.0 < start.0.min(end.0) - Self::NEARBY
                || point.0 > start.0.max(end.0) + Self::NEARBY
            {
                return false;
            }
            let segment = (end.0 - start.0, end.1 - start.1);
            let length_squared = segment.0 * segment.0 + segment.1 * segment.1;
            let projection = if length_squared > 0.0 {
                ((point.0 - start.0) * segment.0 + (point.1 - start.1) * segment.1) / length_squared
            } else {
                0.0
            }
            .clamp(0.0, 1.0);
            let closest = (
                start.0 + segment.0 * projection,
                start.1 + segment.1 * projection,
            );
            let dx = point.0 - closest.0;
            let dy = point.1 - closest.1;
            dx * dx + dy * dy < FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN
        })
    }
}

fn median(values: &mut [u8]) -> u8 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        ((u16::from(values[middle - 1]) + u16::from(values[middle])) / 2) as u8
    } else {
        values[middle]
    }
}

fn standard_deviation(values: &[u8], center: f64) -> f64 {
    (values
        .iter()
        .map(|value| (f64::from(*value) - center).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
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

    #[test]
    fn flux_refines_deep_masks_for_an_extra_step() {
        let stroke = GrayImage::from_fn(400, 400, |x, _| {
            Luma([u8::from((190..210).contains(&x)) * 255])
        });
        assert_eq!(flux_refine_options(&stroke).strength, 0.5);
        let block = GrayImage::from_fn(400, 400, |x, y| {
            Luma([u8::from((20..380).contains(&x) && (20..380).contains(&y)) * 255])
        });
        assert_eq!(flux_refine_options(&block).strength, 0.75);
    }

    #[test]
    fn only_uniform_masked_fills_skip_flux() {
        let mask = GrayImage::from_fn(64, 64, |x, _| Luma([u8::from(x < 32) * 255]));
        let flat = RgbImage::from_fn(64, 64, |x, _| {
            if x < 32 {
                Rgb([250, 250, 250])
            } else {
                Rgb([0, 0, 0])
            }
        });
        assert!(is_uniform_fill(&flat, &mask));
        let gradient = RgbImage::from_fn(64, 64, |x, _| Rgb([100 + x as u8, 200, 200]));
        assert!(!is_uniform_fill(&gradient, &mask));
        assert!(!is_uniform_fill(&flat, &GrayImage::new(64, 64)));
    }

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
            &mask,
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
        assert_eq!(output.get_pixel(48, 48), &Rgb([240, 241, 242]));
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
            &mask,
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
            &mask,
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
            &mask,
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
            &mask,
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
            let crop = fixed_size_crop(tile.crop, size, 1700, 2400);
            assert_eq!(crop[2] - crop[0], size);
            assert_eq!(crop[3] - crop[1], size);
            assert!(crop[0] <= tile.crop[0] && crop[1] <= tile.crop[1]);
            assert!(crop[2] >= tile.crop[2] && crop[3] >= tile.crop[3]);
            assert!(crop[2] <= 1700 && crop[3] <= 2400);
        }
        assert_eq!(
            fixed_size_crop([10, 20, 300, 400], size, 600, 2400),
            [0, 0, 600, size]
        );
    }
}
