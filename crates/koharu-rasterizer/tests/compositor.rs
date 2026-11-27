#![cfg(feature = "native")]

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use koharu_rasterizer::{
    Bounds, Frame, LayerId, LayerKind, Point, PreparedContent, PreparedFrame, PreparedFrameBundle,
    PreparedFrameManifest, PreparedGlyph, PreparedGlyphRun, PreparedLayer, PreparedRaster,
    PreparedRasterTile, PreparedResource, PreparedResourceStore, PreparedScene,
    PreparedSceneCommand, Presentation, RasterOptions, Rasterizer, Revision,
};
use vello::Scene;

/// One rasterizer for the whole test binary, as each server worker process
/// has. When parallel tests each create and drop their own wgpu device, NVIDIA
/// driver 615.71.09 can deadlock inside `vkDestroyDevice`; a static is never
/// dropped.
fn rasterizer() -> &'static Rasterizer {
    static RASTERIZER: LazyLock<Rasterizer> = LazyLock::new(|| Rasterizer::new().unwrap());
    &RASTERIZER
}

#[test]
fn linear_filtering_does_not_add_borders_to_any_raster_layer() {
    let background =
        PreparedResource::encoded_raster(3, 1, "image/png", Arc::from(&b"background"[..])).unwrap();
    let background_id = background.id();
    let resource =
        PreparedResource::encoded_raster(2, 1, "image/png", Arc::from(&b"foreground"[..])).unwrap();
    let resource_id = resource.id();
    let frame = PreparedFrameBundle {
        frame: PreparedFrame {
            revision: Revision::new(1),
            page: LayerId::from_bytes([1; 16]),
            width: 3,
            height: 1,
            origin: (0, 0),
            normalization: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            layers: vec![
                PreparedLayer {
                    id: LayerId::from_bytes([2; 16]),
                    geometry: vec![
                        Point { x: 0.0, y: 0.0 },
                        Point { x: 3.0, y: 0.0 },
                        Point { x: 3.0, y: 1.0 },
                        Point { x: 0.0, y: 1.0 },
                    ],
                    bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 3.0,
                        height: 1.0,
                    },
                    local_bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 3.0,
                        height: 1.0,
                    },
                    presentation: Presentation {
                        visible: true,
                        opacity: 1.0,
                    },
                    kind: LayerKind::Raster,
                    placement: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    content: PreparedContent::Raster(PreparedRaster {
                        source: background_id,
                        width: 3,
                        height: 1,
                        tiles: vec![PreparedRasterTile {
                            x: 0,
                            y: 0,
                            width: 3,
                            height: 1,
                            gutter: [0; 4],
                        }],
                    }),
                    element_frame: None,
                },
                PreparedLayer {
                    id: LayerId::from_bytes([3; 16]),
                    geometry: vec![
                        Point { x: 0.5, y: 0.0 },
                        Point { x: 2.5, y: 0.0 },
                        Point { x: 2.5, y: 1.0 },
                        Point { x: 0.5, y: 1.0 },
                    ],
                    bounds: Bounds {
                        x: 0.5,
                        y: 0.0,
                        width: 2.0,
                        height: 1.0,
                    },
                    local_bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 2.0,
                        height: 1.0,
                    },
                    presentation: Presentation {
                        visible: true,
                        opacity: 1.0,
                    },
                    kind: LayerKind::Raster,
                    placement: [1.0, 0.0, 0.0, 1.0, 0.5, 0.0],
                    content: PreparedContent::Raster(PreparedRaster {
                        source: resource_id,
                        width: 2,
                        height: 1,
                        tiles: vec![PreparedRasterTile {
                            x: 0,
                            y: 0,
                            width: 2,
                            height: 1,
                            gutter: [0; 4],
                        }],
                    }),
                    element_frame: None,
                },
            ],
        },
        resources: vec![background, resource],
    }
    .into_frame_with_raster_sources(&HashMap::from([
        (background_id, Arc::from(&[255; 12][..])),
        (resource_id, Arc::from(&[255, 0, 0, 255, 0, 0, 0, 0][..])),
    ]))
    .unwrap();

    let raster = rasterizer()
        .rasterize(&frame, RasterOptions::default())
        .unwrap();

    let edge = raster.image.get_pixel(1, 0).0;
    assert_eq!(edge[0], u8::MAX);
    assert!((126..=129).contains(&edge[1]));
    assert!((126..=129).contains(&edge[2]));
    assert_eq!(edge[3], u8::MAX);
}

#[test]
fn rasterizes_surfaces_beyond_the_default_wgpu_texture_limit() {
    // WGPU's default limits cap textures at 8192 px, far below what desktop
    // GPUs support, and tall scans or spreads exceed it.
    let image = rasterizer()
        .rasterize_scene(
            &Scene::new(),
            4,
            9000,
            [10, 20, 30, 255],
            RasterOptions::default(),
        )
        .unwrap();

    assert_eq!(image.dimensions(), (4, 9000));
    assert_eq!(image.get_pixel(3, 8999).0, [10, 20, 30, 255]);
}

fn noise(seed: u32, len: usize) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

fn tiled_layer(
    id: u8,
    source: koharu_rasterizer::ResourceId,
    width: u32,
    height: u32,
) -> PreparedLayer {
    let dimension = koharu_rasterizer::PREPARED_RASTER_TILE_DIMENSION;
    let mut tiles = Vec::new();
    for y in (0..height).step_by(dimension as usize) {
        for x in (0..width).step_by(dimension as usize) {
            let tile_width = (width - x).min(dimension);
            let tile_height = (height - y).min(dimension);
            tiles.push(PreparedRasterTile {
                x,
                y,
                width: tile_width,
                height: tile_height,
                gutter: [
                    u32::from(x > 0),
                    u32::from(y > 0),
                    u32::from(x + tile_width < width),
                    u32::from(y + tile_height < height),
                ],
            });
        }
    }
    let bounds = Bounds {
        x: 0.0,
        y: 0.0,
        width: width as f32,
        height: height as f32,
    };
    PreparedLayer {
        id: LayerId::from_bytes([id; 16]),
        geometry: vec![
            Point { x: 0.0, y: 0.0 },
            Point {
                x: f64::from(width),
                y: 0.0,
            },
            Point {
                x: f64::from(width),
                y: f64::from(height),
            },
            Point {
                x: 0.0,
                y: f64::from(height),
            },
        ],
        bounds,
        local_bounds: bounds,
        presentation: Presentation {
            visible: true,
            opacity: 1.0,
        },
        kind: LayerKind::Raster,
        placement: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        content: PreparedContent::Raster(PreparedRaster {
            source,
            width,
            height,
            tiles,
        }),
        element_frame: None,
    }
}

#[test]
fn tiled_layers_composite_like_premultiplied_source_over() {
    // Two rows and columns of tiles per layer, so the frame crosses every kind of tile seam.
    let (width, height) = (1_300, 1_100);
    let len = (width * height * 4) as usize;
    let mut below = noise(1, len);
    for pixel in below.as_chunks_mut::<4>().0 {
        pixel[3] = u8::MAX;
    }
    let above = noise(2, len);
    let below_resource =
        PreparedResource::encoded_raster(width, height, "image/png", Arc::from(&b"below"[..]))
            .unwrap();
    let above_resource =
        PreparedResource::encoded_raster(width, height, "image/png", Arc::from(&b"above"[..]))
            .unwrap();
    let (below_id, above_id) = (below_resource.id(), above_resource.id());
    let frame = PreparedFrameBundle {
        frame: PreparedFrame {
            revision: Revision::new(1),
            page: LayerId::from_bytes([1; 16]),
            width,
            height,
            origin: (0, 0),
            normalization: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            layers: vec![
                tiled_layer(2, below_id, width, height),
                tiled_layer(3, above_id, width, height),
            ],
        },
        resources: vec![below_resource, above_resource],
    }
    .into_frame_with_raster_sources(&HashMap::from([
        (below_id, Arc::from(&below[..])),
        (above_id, Arc::from(&above[..])),
    ]))
    .unwrap();

    let raster = rasterizer()
        .rasterize(&frame, RasterOptions::default())
        .unwrap();

    let output = raster.image.as_raw();
    assert_eq!(output.len(), len);
    for (index, ((output, below), above)) in output
        .as_chunks::<4>()
        .0
        .iter()
        .zip(below.as_chunks::<4>().0)
        .zip(above.as_chunks::<4>().0)
        .enumerate()
    {
        let alpha = u32::from(above[3]);
        for channel in 0..3 {
            let premultiplied = (u32::from(above[channel]) * alpha + 127) / 255;
            let expected = premultiplied + u32::from(below[channel]) * (255 - alpha) / 255;
            assert!(
                u32::from(output[channel]).abs_diff(expected) <= 1,
                "pixel {index} channel {channel}: {} != {expected}",
                output[channel]
            );
        }
        assert_eq!(output[3], u8::MAX);
    }
}

/// Vello stores straight (unpremultiplied) colour in its target, but the compositor blends
/// premultiplied. A light vector edge over a raster layer must still blend partially instead
/// of saturating every partly covered pixel to the vector colour.
#[test]
fn vector_layer_edges_blend_partially_over_raster_layers() {
    use koharu_rasterizer::{
        FillRule, PathElement, PreparedPath, PreparedScene, PreparedSceneCommand,
    };
    let (width, height) = (64u32, 64u32);
    let mut blue = vec![0u8; (width * height * 4) as usize];
    for pixel in blue.as_chunks_mut::<4>().0 {
        *pixel = [0, 0, 255, 255];
    }
    let resource =
        PreparedResource::encoded_raster(width, height, "image/png", Arc::from(&b"blue"[..]))
            .unwrap();
    let source = resource.id();
    // A white disc with fractional centre and radius, so most edge pixels are partly covered.
    let (cx, cy, radius) = (32.3, 31.7, 20.4);
    let mut elements = Vec::new();
    for step in 0..=360 {
        let angle = f64::from(step).to_radians();
        let point = [cx + radius * angle.cos(), cy + radius * angle.sin()];
        elements.push(if step == 0 {
            PathElement::MoveTo(point)
        } else {
            PathElement::LineTo(point)
        });
    }
    elements.push(PathElement::Close);
    let mut vector = tiled_layer(3, source, width, height);
    vector.kind = LayerKind::Text;
    vector.content = PreparedContent::Vector(PreparedScene {
        commands: vec![PreparedSceneCommand::FillPath(PreparedPath {
            fill: FillRule::NonZero,
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            color: [255, 255, 255, 255],
            elements,
        })],
    });
    let frame = PreparedFrameBundle {
        frame: PreparedFrame {
            revision: Revision::new(1),
            page: LayerId::from_bytes([1; 16]),
            width,
            height,
            origin: (0, 0),
            normalization: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            layers: vec![tiled_layer(2, source, width, height), vector],
        },
        resources: vec![resource],
    }
    .into_frame_with_raster_sources(&HashMap::from([(source, Arc::from(&blue[..]))]))
    .unwrap();
    let image = rasterizer()
        .rasterize(&frame, RasterOptions::default())
        .unwrap()
        .image;
    let mut partial = 0;
    for pixel in image.pixels() {
        let [red, green, blue, alpha] = pixel.0;
        assert_eq!(alpha, 255);
        assert!(blue >= 254, "white over blue keeps a full blue channel");
        // Straight-alpha white over blue: red and green both equal the coverage.
        assert!(red.abs_diff(green) <= 1);
        if (16..=239).contains(&red) {
            partial += 1;
        }
    }
    assert!(
        partial > 60,
        "the disc edge must antialias against the raster layer below: {partial} partial pixels"
    );
}

#[test]
fn native_readback_preserves_straight_alpha_for_transparent_layers() {
    use koharu_rasterizer::{
        FillRule, PathElement, PreparedPath, PreparedScene, PreparedSceneCommand,
    };
    let (width, height) = (64, 32);
    let pixels = [80, 160, 240, 128].repeat(32 * 32);
    let resource =
        PreparedResource::encoded_raster(32, 32, "image/png", Arc::from(&b"translucent"[..]))
            .unwrap();
    let source = resource.id();
    let mut raster = tiled_layer(2, source, 32, 32);
    raster.placement[4] = 32.0;
    let mut vector = tiled_layer(3, source, width, height);
    vector.kind = LayerKind::Text;
    vector.content = PreparedContent::Vector(PreparedScene {
        commands: vec![PreparedSceneCommand::FillPath(PreparedPath {
            fill: FillRule::NonZero,
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            color: [200, 100, 50, 128],
            elements: vec![
                PathElement::MoveTo([4.25, 4.25]),
                PathElement::LineTo([27.75, 4.25]),
                PathElement::LineTo([27.75, 27.75]),
                PathElement::LineTo([4.25, 27.75]),
                PathElement::Close,
            ],
        })],
    });
    let frame = PreparedFrameBundle {
        frame: PreparedFrame {
            revision: Revision::new(1),
            page: LayerId::from_bytes([1; 16]),
            width,
            height,
            origin: (0, 0),
            normalization: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            layers: vec![raster, vector],
        },
        resources: vec![resource],
    }
    .into_frame_with_raster_sources(&HashMap::from([(source, Arc::from(pixels))]))
    .unwrap();
    let image = rasterizer()
        .rasterize(&frame, RasterOptions::default())
        .unwrap()
        .image;
    for ((x, y), expected) in [
        ((8, 8), [200u8, 100, 50, 128]),
        ((40, 8), [80, 160, 240, 128]),
        ((0, 0), [0, 0, 0, 0]),
    ] {
        let actual = image.get_pixel(x, y).0;
        for channel in 0..4 {
            assert!(
                actual[channel].abs_diff(expected[channel]) <= 1,
                "straight RGBA at ({x}, {y}): {actual:?} != {expected:?}"
            );
        }
    }
}

fn glyph_frame(font: &PreparedResource, run: PreparedGlyphRun) -> Frame {
    let (width, height) = (96, 96);
    let mut layer = tiled_layer(2, font.id(), width, height);
    layer.kind = LayerKind::Text;
    layer.content = PreparedContent::Vector(PreparedScene {
        commands: vec![PreparedSceneCommand::GlyphRun(run)],
    });
    let bundle = PreparedFrameBundle {
        frame: PreparedFrame {
            revision: Revision::ZERO,
            page: LayerId::from_bytes([1; 16]),
            width,
            height,
            origin: (0, 0),
            normalization: vello::kurbo::Affine::IDENTITY.as_coeffs(),
            layers: vec![layer],
        },
        resources: vec![font.clone()],
    };
    let mut resources = PreparedResourceStore::default();
    resources.insert(bundle.resource_packet(font.id()).unwrap());
    let manifest =
        PreparedFrameManifest::decode(&bundle.manifest().unwrap().encode().unwrap()).unwrap();
    manifest.compile(&resources).unwrap()
}

#[test]
fn prepared_glyph_borders_grow_the_fill_and_apply_paint_alpha_once() {
    use vello::{kurbo::Affine, peniko::Style};

    // Glyph 1 has a curved outer contour and a counter.
    let font = PreparedResource::font(include_bytes!("fixtures/outline.ttf"));
    let render = |radius: f32, alpha: u8| {
        let frame = glyph_frame(
            &font,
            PreparedGlyphRun {
                font: font.id(),
                font_index: 0,
                font_size: 72.0,
                normalized_coords: vec![8192],
                transform: Affine::translate((4.0, 0.0)).as_coeffs(),
                glyph_transform: Some(Affine::skew(0.1, 0.0).as_coeffs()),
                hint: false,
                embolden: [radius; 2],
                color: [0, 0, 0, alpha],
                glyphs: vec![PreparedGlyph {
                    id: 1,
                    x: 8.0,
                    y: 78.0,
                }],
            },
        );
        let commands = frame.composition_commands(1);
        let [koharu_rasterizer::CompositionCommand::Vector(scene)] = commands.as_slice() else {
            panic!("expected one vector batch");
        };
        let encoding = scene.encoding();
        let runs = &encoding.resources.glyph_runs;
        assert!(matches!(runs[0].style, Style::Fill(_)));
        assert_eq!(runs.len(), if radius > 0.0 { 2 } else { 1 });
        if radius > 0.0 {
            let Style::Stroke(stroke) = &runs[1].style else {
                panic!("expected a glyph stroke")
            };
            assert_eq!(
                (stroke.width, stroke.join),
                (2.0 * f64::from(radius), vello::kurbo::Join::Round)
            );
            for run in runs {
                assert_eq!(
                    &encoding.resources.normalized_coords[run.normalized_coords.clone()],
                    &[8192]
                );
            }
            assert_eq!(runs[0].transform, runs[1].transform);
            assert_eq!(runs[0].glyph_transform, runs[1].glyph_transform);
        }
        assert!(
            runs.iter()
                .all(|run| run.font_embolden.amount == vello::kurbo::Diagonal2::new(0.0, 0.0))
        );
        assert_eq!(encoding.n_clips > 0, radius > 0.0 && alpha < 255);
        rasterizer()
            .rasterize(&frame, RasterOptions::default())
            .unwrap()
            .image
    };

    let fill = render(0.0, 255);
    let opaque = render(5.0, 255);
    let translucent = render(5.0, 128);
    let mut added_pixels = 0;
    let mut fill_pixels = 0;
    for ((fill, opaque), translucent) in
        fill.pixels().zip(opaque.pixels()).zip(translucent.pixels())
    {
        if fill[3] == 255 {
            fill_pixels += 1;
            assert_eq!(opaque[3], 255);
        }
        if fill[3] == 0 && opaque[3] == 255 {
            added_pixels += 1;
        }
        let expected = (u16::from(opaque[3]) * 128 + 127) / 255;
        assert!(u16::from(translucent[3]).abs_diff(expected) <= 1);
    }
    assert!(fill_pixels > 200);
    assert!(added_pixels > 200);
}

#[test]
fn emboldening_a_mixed_run_preserves_colr_and_bitmap_pixels() {
    let font = PreparedResource::font(include_bytes!("fixtures/mixed-glyphs.ttf"));
    let render = |radius, alpha, colr_x| {
        let frame = glyph_frame(
            &font,
            PreparedGlyphRun {
                font: font.id(),
                font_index: 0,
                font_size: 24.0,
                normalized_coords: vec![],
                transform: vello::kurbo::Affine::IDENTITY.as_coeffs(),
                glyph_transform: None,
                hint: false,
                embolden: [radius; 2],
                color: [0, 0, 0, alpha],
                glyphs: vec![
                    PreparedGlyph {
                        id: 1,
                        x: 4.0,
                        y: 48.0,
                    },
                    PreparedGlyph {
                        id: 2,
                        x: colr_x,
                        y: 48.0,
                    },
                    PreparedGlyph {
                        id: 3,
                        x: 68.0,
                        y: 48.0,
                    },
                ],
            },
        );
        rasterizer()
            .rasterize(&frame, RasterOptions::default())
            .unwrap()
            .image
    };

    for alpha in [128, 255] {
        let fill = render(0.0, alpha, 36.0);
        let bold = render(2.0, alpha, 36.0);
        let mut outline_grew = false;
        let mut colr_visible = false;
        let mut bitmap_visible = false;
        for (x, y, pixel) in fill.enumerate_pixels() {
            let emboldened = bold.get_pixel(x, y);
            if x < 32 {
                outline_grew |= emboldened[3] > pixel[3];
            } else {
                assert_eq!(emboldened, pixel, "color glyph changed at ({x}, {y})");
                colr_visible |= *pixel == image::Rgba([255, 0, 0, 128]);
                bitmap_visible |= *pixel == image::Rgba([0, 0, 255, 128]);
            }
        }
        assert!(outline_grew);
        assert!(colr_visible);
        assert!(bitmap_visible);
        assert_eq!(fill.get_pixel(24, 38).0, [0; 4]);
        assert_eq!(bold.get_pixel(24, 38).0, [0, 0, 0, alpha]);
    }

    // At this pixel only the stroke covers the background. The later COLR
    // glyph composites its half-opaque red over the opaque black stroke.
    let overlap = render(2.0, 255, 8.0);
    assert_eq!(overlap.get_pixel(24, 38).0, [128, 0, 0, 255]);
}
