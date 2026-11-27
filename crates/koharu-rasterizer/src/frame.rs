//! Retained Vello frame compiled from portable display data.

use std::{collections::HashMap, sync::Arc};

use vello::{
    DrawGlyphs, Glyph, Scene,
    kurbo::{Affine, BezPath, Join, Rect, Stroke},
    peniko::{Blob, Color, Fill, FontData, Mix},
};

use crate::{
    Bounds, CompositionCommand, Error, FillRule, LayerId, LayerKind, PathElement, PreparedContent,
    PreparedElementFrame, PreparedFrame, PreparedFrameBundle, PreparedFrameManifest, PreparedGlyph,
    PreparedGlyphRun, PreparedPath, PreparedResource, PreparedResourceStore, PreparedScene,
    PreparedSceneCommand, Presentation, RasterDraw, ResourceId, Result, Revision,
};

#[derive(Clone)]
pub struct Frame(Arc<FrameData>);

struct FrameData {
    revision: Revision,
    page: LayerId,
    width: u32,
    height: u32,
    origin: (i32, i32),
    normalization: Affine,
    layers: Arc<[Layer]>,
    layer_index: HashMap<LayerId, usize>,
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Frame")
            .field("revision", &self.revision())
            .field("page", &self.page())
            .field("size", &self.size())
            .field("origin", &self.origin())
            .field("layers", &self.layers())
            .finish_non_exhaustive()
    }
}

impl Frame {
    #[must_use]
    pub fn revision(&self) -> Revision {
        self.0.revision
    }

    #[must_use]
    pub fn page(&self) -> LayerId {
        self.0.page
    }

    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.0.width, self.0.height)
    }

    #[must_use]
    pub fn origin(&self) -> (i32, i32) {
        self.0.origin
    }

    #[must_use]
    pub fn normalization(&self) -> Affine {
        self.0.normalization
    }

    #[must_use]
    pub fn layers(&self) -> &[Layer] {
        &self.0.layers
    }

    #[must_use]
    pub fn layer(&self, id: LayerId) -> Option<&Layer> {
        self.0
            .layer_index
            .get(&id)
            .map(|index| &self.0.layers[*index])
    }

    /// Builds the ordered composition used by native headless rasterization.
    #[must_use]
    pub fn composition_commands(&self, scale: u32) -> Vec<CompositionCommand> {
        let outer = Affine::scale(f64::from(scale.max(1))) * self.normalization();
        let mut commands = Vec::new();
        let mut vectors = Scene::new();
        let mut vectors_pending = false;
        for layer in self.layers() {
            let presentation = layer.presentation();
            if let Some(image) = layer.raster_image() {
                if vectors_pending {
                    commands.push(CompositionCommand::Vector(Box::new(std::mem::replace(
                        &mut vectors,
                        Scene::new(),
                    ))));
                    vectors_pending = false;
                }
                if presentation.visible
                    && presentation.opacity.is_finite()
                    && presentation.opacity > 0.0
                {
                    commands.push(CompositionCommand::Raster(RasterDraw {
                        image: image.clone(),
                        transform: outer * layer.placement(),
                        opacity: presentation.opacity.clamp(0.0, 1.0),
                        erase: false,
                    }));
                }
            } else {
                layer.append_vector_with_presentation(&mut vectors, Some(outer), presentation);
                vectors_pending |= presentation.visible
                    && presentation.opacity.is_finite()
                    && presentation.opacity > 0.0;
            }
        }
        if vectors_pending {
            commands.push(CompositionCommand::Vector(Box::new(vectors)));
        }
        commands
    }
}

#[derive(Clone)]
pub struct Layer(Arc<LayerData>);

struct LayerData {
    id: LayerId,
    geometry: Arc<[crate::Point]>,
    bounds: Bounds,
    local_bounds: Bounds,
    presentation: Presentation,
    kind: LayerKind,
    placement: Affine,
    scene: Arc<Scene>,
    image: Option<RasterImage>,
    element_frame: Option<PreparedElementFrame>,
}

impl std::fmt::Debug for Layer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Layer")
            .field("id", &self.id())
            .field("bounds", &self.bounds())
            .field("presentation", &self.presentation())
            .field("kind", &self.kind())
            .finish_non_exhaustive()
    }
}

impl Layer {
    #[must_use]
    pub fn id(&self) -> LayerId {
        self.0.id
    }

    #[must_use]
    pub fn geometry(&self) -> &[crate::Point] {
        &self.0.geometry
    }

    #[must_use]
    pub fn bounds(&self) -> Bounds {
        self.0.bounds
    }

    #[must_use]
    pub fn presentation(&self) -> Presentation {
        self.0.presentation
    }

    #[must_use]
    pub fn kind(&self) -> LayerKind {
        self.0.kind
    }

    #[must_use]
    pub fn placement(&self) -> Affine {
        self.0.placement
    }

    #[must_use]
    pub fn raster_image(&self) -> Option<&RasterImage> {
        self.0.image.as_ref()
    }

    #[must_use]
    pub fn element_frame(&self) -> Option<PreparedElementFrame> {
        self.0.element_frame
    }

    pub fn append_vector_with_presentation(
        &self,
        scene: &mut Scene,
        transform: Option<Affine>,
        presentation: Presentation,
    ) {
        if self.raster_image().is_some()
            || !presentation.visible
            || !presentation.opacity.is_finite()
            || presentation.opacity <= 0.0
        {
            return;
        }
        let opacity = presentation.opacity.clamp(0.0, 1.0);
        let placement = transform.map_or(self.0.placement, |outer| outer * self.0.placement);
        if opacity < 1.0 {
            let bounds = self.0.local_bounds;
            scene.push_layer(
                Fill::NonZero,
                Mix::Normal,
                opacity,
                placement,
                &Rect::new(
                    f64::from(bounds.x),
                    f64::from(bounds.y),
                    f64::from(bounds.x + bounds.width),
                    f64::from(bounds.y + bounds.height),
                ),
            );
        }
        scene.append(&self.0.scene, Some(placement));
        if opacity < 1.0 {
            scene.pop_layer();
        }
    }
}

#[derive(Clone)]
pub struct RasterImage {
    source: ResourceId,
    width: u32,
    height: u32,
    tiles: Arc<[RasterTile]>,
    pixels: Option<Arc<[u8]>>,
}

#[derive(Clone)]
pub struct RasterTile {
    id: ResourceId,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    gutter: [u32; 4],
}

impl RasterImage {
    #[must_use]
    pub fn source(&self) -> ResourceId {
        self.source
    }

    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[must_use]
    pub fn tiles(&self) -> &[RasterTile] {
        &self.tiles
    }

    pub(crate) fn pixels(&self) -> Option<&Arc<[u8]>> {
        self.pixels.as_ref()
    }
}

impl RasterTile {
    #[must_use]
    pub fn id(&self) -> ResourceId {
        self.id
    }

    #[must_use]
    pub fn origin(&self) -> (u32, u32) {
        (self.x, self.y)
    }

    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[must_use]
    pub fn gutter(&self) -> [u32; 4] {
        self.gutter
    }

    #[must_use]
    pub const fn source_origin(&self) -> (u32, u32) {
        (self.x - self.gutter[0], self.y - self.gutter[1])
    }

    #[must_use]
    pub fn resource_size(&self) -> (u32, u32) {
        (
            self.width + self.gutter[0] + self.gutter[2],
            self.height + self.gutter[1] + self.gutter[3],
        )
    }
}

impl PreparedFrameBundle {
    pub fn into_frame_with_raster_sources(
        self,
        raster_sources: &HashMap<ResourceId, Arc<[u8]>>,
    ) -> Result<Frame> {
        self.validate()?;
        let Self { frame, resources } = self;
        let resources = resources
            .iter()
            .map(|resource| (resource.id(), resource))
            .collect::<HashMap<_, _>>();
        compile_frame(frame, &resources, Some(raster_sources))
    }
}

pub(crate) fn compile_manifest(
    manifest: &PreparedFrameManifest,
    store: &PreparedResourceStore,
) -> Result<Frame> {
    let mut resources = HashMap::with_capacity(manifest.resources.len());
    for expected in manifest.required_resources() {
        let packet = store.get(expected.id).ok_or_else(|| {
            Error::invalid(format!(
                "prepared resource {} is not installed",
                expected.id
            ))
        })?;
        if packet.resource().reference() != *expected {
            return Err(Error::invalid(format!(
                "installed prepared resource {} does not match its manifest reference",
                expected.id
            )));
        }
        resources.insert(expected.id, packet.resource());
    }
    compile_frame(manifest.frame.clone(), &resources, None)
}

fn compile_frame(
    frame: PreparedFrame,
    resources: &HashMap<ResourceId, &PreparedResource>,
    raster_sources: Option<&HashMap<ResourceId, Arc<[u8]>>>,
) -> Result<Frame> {
    let mut font_blobs = HashMap::new();
    let mut layers = Vec::with_capacity(frame.layers.len());
    let mut layer_index = HashMap::with_capacity(frame.layers.len());
    let normalization = Affine::new(frame.normalization);
    let viewport = Rect::from_origin_size(
        (0.0, 0.0),
        (f64::from(frame.width), f64::from(frame.height)),
    );
    for prepared in frame.layers {
        let placement = Affine::new(prepared.placement);
        let (scene, image) = match &prepared.content {
            PreparedContent::Raster(raster) => {
                let Some(PreparedResource::EncodedRaster { width, height, .. }) =
                    resources.get(&raster.source).copied()
                else {
                    return Err(Error::invalid(
                        "raster references a missing encoded source resource",
                    ));
                };
                if (*width, *height) != (raster.width, raster.height) {
                    return Err(Error::invalid(
                        "encoded source dimensions do not match the raster",
                    ));
                }
                let pixels = raster_sources
                    .map(|sources| {
                        sources.get(&raster.source).cloned().ok_or_else(|| {
                            Error::invalid(format!(
                                "native raster source {} is not installed",
                                raster.source
                            ))
                        })
                    })
                    .transpose()?;
                if let Some(pixels) = pixels.as_ref() {
                    let expected =
                        usize::try_from(u64::from(raster.width) * u64::from(raster.height) * 4)
                            .map_err(|_| {
                                Error::invalid("native raster byte length exceeds usize")
                            })?;
                    if pixels.len() != expected {
                        return Err(Error::invalid(
                            "native raster byte length does not match its dimensions",
                        ));
                    }
                }
                let mut tiles = Vec::with_capacity(raster.tiles.len());
                for prepared_tile in &raster.tiles {
                    tiles.push(RasterTile {
                        id: prepared_tile.id(raster.source),
                        x: prepared_tile.x,
                        y: prepared_tile.y,
                        width: prepared_tile.width,
                        height: prepared_tile.height,
                        gutter: prepared_tile.gutter,
                    });
                }
                (
                    Scene::new(),
                    Some(RasterImage {
                        source: raster.source,
                        width: raster.width,
                        height: raster.height,
                        tiles: tiles.into(),
                        pixels,
                    }),
                )
            }
            PreparedContent::Vector(prepared_scene) => {
                // Vello requires a clip for an opacity layer. Include the entire
                // visible frame so layout bounds cannot cut off overhanging ink.
                let bounds = prepared.local_bounds;
                let mut clip = Rect::from_origin_size(
                    (f64::from(bounds.x), f64::from(bounds.y)),
                    (f64::from(bounds.width), f64::from(bounds.height)),
                );
                let visible = (normalization * placement)
                    .inverse()
                    .transform_rect_bbox(viewport.inflate(1.0, 1.0));
                if visible.is_finite() {
                    clip = clip.union(visible);
                }
                (
                    compile_scene(prepared_scene, clip, resources, &mut font_blobs)?,
                    None,
                )
            }
        };
        let layer = Layer(Arc::new(LayerData {
            id: prepared.id,
            geometry: prepared.geometry.into(),
            bounds: prepared.bounds,
            local_bounds: prepared.local_bounds,
            presentation: prepared.presentation,
            kind: prepared.kind,
            placement,
            scene: Arc::new(scene),
            image,
            element_frame: prepared.element_frame,
        }));
        layer_index.insert(layer.id(), layers.len());
        layers.push(layer);
    }
    Ok(Frame(Arc::new(FrameData {
        revision: frame.revision,
        page: frame.page,
        width: frame.width,
        height: frame.height,
        origin: frame.origin,
        normalization,
        layers: layers.into(),
        layer_index,
    })))
}

fn compile_scene(
    prepared: &PreparedScene,
    clip: Rect,
    resources: &HashMap<ResourceId, &PreparedResource>,
    font_blobs: &mut HashMap<ResourceId, Blob<u8>>,
) -> Result<Scene> {
    let mut scene = Scene::new();
    for command in &prepared.commands {
        match command {
            PreparedSceneCommand::GlyphRun(prepared) => {
                let Some(PreparedResource::Font { bytes, .. }) =
                    resources.get(&prepared.font).copied()
                else {
                    return Err(Error::invalid(
                        "glyph run references a missing font resource",
                    ));
                };
                let blob = font_blobs
                    .entry(prepared.font)
                    .or_insert_with(|| Blob::new(Arc::new(Arc::clone(bytes))))
                    .clone();
                let font = FontData::new(blob, prepared.font_index);
                prepared.paint(&mut scene, &font, clip);
            }
            PreparedSceneCommand::FillPath(prepared) => {
                let path = compile_path(prepared);
                scene.fill(
                    match prepared.fill {
                        FillRule::NonZero => Fill::NonZero,
                        FillRule::EvenOdd => Fill::EvenOdd,
                    },
                    Affine::new(prepared.transform),
                    rgba(prepared.color),
                    None,
                    &path,
                );
            }
        }
    }
    Ok(scene)
}

impl PreparedGlyphRun {
    fn paint(&self, scene: &mut Scene, font: &FontData, clip: Rect) {
        let color = rgba(self.color);
        let fill = |scene: &mut Scene, glyphs: &[PreparedGlyph]| {
            if !glyphs.is_empty() {
                self.draw_glyphs(scene, font, color).draw(
                    Fill::NonZero,
                    glyphs.iter().map(|glyph| Glyph {
                        id: glyph.id,
                        x: glyph.x,
                        y: glyph.y,
                    }),
                );
            }
        };
        // A round stroke has one radius, so use the larger extent for an anisotropic packet.
        let radius = self.embolden[0].max(self.embolden[1]);
        if radius == 0.0 {
            fill(scene, &self.glyphs);
            return;
        }
        let stroke = Stroke::new(2.0 * f64::from(radius)).with_join(Join::Round);
        let alpha = color.components[3];
        let opaque = color.with_alpha(1.0);

        // Vello retains contiguous outlines as glyph runs and expands COLR and
        // bitmap glyphs into paths and images for this font and strike size.
        let mut filled = Scene::new();
        fill(&mut filled, &self.glyphs);
        let resources = &filled.encoding().resources;
        if resources.glyphs.is_empty() {
            scene.append(&filled, None);
            return;
        }
        let mut remaining = self.glyphs.as_slice();
        for run in &resources.glyph_runs {
            let outlines = &resources.glyphs[run.glyphs.clone()];
            let color_count = remaining
                .iter()
                .take_while(|g| g.id != outlines[0].id)
                .count();
            let (colors, rest) = remaining.split_at(color_count);
            fill(scene, colors);
            remaining = &rest[outlines.len()..];

            // Paint alpha applies once to the union of the outline fill and stroke.
            if alpha < 1.0 {
                scene.push_layer(Fill::NonZero, Mix::Normal, alpha, Affine::IDENTITY, &clip);
            }
            self.draw_glyphs(scene, font, opaque)
                .draw(Fill::NonZero, outlines.iter().copied());
            self.draw_glyphs(scene, font, opaque)
                .draw(&stroke, outlines.iter().copied());
            if alpha < 1.0 {
                scene.pop_layer();
            }
        }
        fill(scene, remaining);
    }

    fn draw_glyphs<'a>(
        &self,
        scene: &'a mut Scene,
        font: &FontData,
        color: Color,
    ) -> DrawGlyphs<'a> {
        scene
            .draw_glyphs(font)
            .font_size(self.font_size)
            .transform(Affine::new(self.transform))
            .glyph_transform(self.glyph_transform.map(Affine::new))
            .hint(self.hint)
            .normalized_coords(&self.normalized_coords)
            .brush(color)
    }
}

fn compile_path(prepared: &PreparedPath) -> BezPath {
    let mut path = BezPath::new();
    for element in &prepared.elements {
        match *element {
            PathElement::MoveTo([x, y]) => path.move_to((x, y)),
            PathElement::LineTo([x, y]) => path.line_to((x, y)),
            PathElement::QuadTo([x1, y1, x2, y2]) => path.quad_to((x1, y1), (x2, y2)),
            PathElement::CurveTo([x1, y1, x2, y2, x3, y3]) => {
                path.curve_to((x1, y1), (x2, y2), (x3, y3));
            }
            PathElement::Close => path.close_path(),
        }
    }
    path
}

fn rgba([r, g, b, a]: [u8; 4]) -> Color {
    Color::from_rgba8(r, g, b, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Point, PreparedFrame, PreparedLayer, PreparedRaster, PreparedRasterTile};

    #[test]
    fn compiled_frame_preserves_raster_identity_and_metadata() {
        let resource =
            PreparedResource::encoded_raster(1, 1, "image/png", Arc::from(&b"png"[..])).unwrap();
        let source_id = resource.id();
        let prepared_tile = PreparedRasterTile {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            gutter: [0; 4],
        };
        let bundle = PreparedFrameBundle {
            frame: PreparedFrame {
                revision: Revision::new(3),
                page: LayerId::from_bytes([9; 16]),
                width: 10,
                height: 20,
                origin: (-2, 4),
                normalization: [1.0, 0.0, 0.0, 1.0, 2.0, -4.0],
                layers: vec![PreparedLayer {
                    id: LayerId::from_bytes([4; 16]),
                    geometry: vec![
                        Point { x: 0.0, y: 0.0 },
                        Point { x: 1.0, y: 0.0 },
                        Point { x: 0.0, y: 1.0 },
                    ],
                    bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                    },
                    local_bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                    },
                    presentation: Presentation {
                        visible: true,
                        opacity: 0.5,
                    },
                    kind: LayerKind::Raster,
                    placement: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    content: PreparedContent::Raster(PreparedRaster {
                        source: source_id,
                        width: 1,
                        height: 1,
                        tiles: vec![prepared_tile],
                    }),
                    element_frame: None,
                }],
            },
            resources: vec![resource],
        };
        let frame = bundle
            .into_frame_with_raster_sources(&HashMap::from([(
                source_id,
                Arc::from(&[1, 2, 3, 4][..]),
            )]))
            .unwrap();
        assert_eq!(frame.revision(), Revision::new(3));
        assert_eq!(frame.origin(), (-2, 4));
        assert_eq!(
            frame.layers()[0].raster_image().unwrap().tiles()[0].id(),
            prepared_tile.id(source_id)
        );
        assert_eq!(
            frame.layers()[0]
                .raster_image()
                .unwrap()
                .pixels()
                .unwrap()
                .as_ref(),
            &[1, 2, 3, 4]
        );
    }
}
