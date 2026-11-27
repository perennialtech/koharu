//! The glyphs of outlined lettering. Lettering over art is set in an outline,
//! such as white around black, and the ink split only tells two tones apart:
//! where the art is nearer the glyphs' tone it takes the outline for the ink,
//! and elsewhere the art between the lines for glyphs.

use image::{GrayImage, Luma, RgbImage};
use imageproc::{
    distance_transform::{Norm, euclidean_squared_distance_transform},
    morphology::{dilate, open},
    region_labelling::{Connectivity, connected_components},
};

use super::{
    KoharuLayoutMask as Mask,
    measure::{Outline, fill_holes, outline, scaled},
};

/// The least share of a tone, inside lettering, within an outline's width of
/// what it encloses for it to be the outline. On 28 pages of four works, the
/// outlines' was at least 0.65, and the share of plain lettering's ink around
/// its counters at most 0.48.
const AROUND: f32 = 0.6;
/// How far around a glyph's part its outline must be solid, in pixels, and
/// the least share of it that must: art's specks lie too close together.
const NEAR: u8 = 3;
const SOLID: f32 = 0.7;
const NEAR_THIN: u8 = 1;
const SOLID_THIN: f32 = 0.70;

/// Outlined lettering, by its glyphs.
pub(super) struct Outlined {
    pub glyphs: GrayImage,
    pub ink: [u8; 3],
    pub outline: Outline,
}

/// The lettering in `block` by its glyphs, when its `ink`, both at `corner` on
/// `page`, is outlined or is its outline; `size` measures a lettering's lines.
pub(super) fn outlined(
    page: &RgbImage,
    block: &Mask,
    pre_closed: &Mask,
    ink: &GrayImage,
    corner: [u32; 2],
    size: impl Fn(&GrayImage) -> f32,
) -> Option<Outlined> {
    let inside = |x: u32, y: u32| block.pixels[(y * block.width + x) as usize] != 0;
    // What the ink encloses first: an outline encloses its glyphs, and glyphs
    // only their counters.
    [true, false].into_iter().find_map(|enclosed| {
        let tone = |x: u32, y: u32| (ink.get_pixel(x, y).0[0] == 0) == enclosed;
        let parts = Parts::within(
            block,
            pre_closed,
            corner,
            &tone,
            scaled(3.0, page.width(), page.height()),
        );
        let whole = parts.shape(|_| true);
        let glyphs = parts.banded(block, &tone);
        if count(&glyphs) < 16 || !enclosed && 2 * count(&glyphs) < count(ink) {
            return None;
        }
        let found = outline(page, &glyphs, corner, size(&glyphs), None);
        if found.is_none() && !enclosed {
            return None;
        }
        // The other tone, and how far it lies from the parts.
        let distance = euclidean_squared_distance_transform(&whole);
        let mut around: Vec<(f64, [u8; 3])> = distance
            .enumerate_pixels()
            .filter(|&(x, y, _)| inside(x, y) && !tone(x, y))
            .map(|(x, y, squared)| {
                let pixel = page.get_pixel(corner[0] + x, corner[1] + y).0;
                (squared.0[0].sqrt(), pixel)
            })
            .collect();
        let width = found.map_or(2.0, |found| f64::from(found.width));
        let total = around.len();
        around.retain(|(distance, _)| *distance <= width + 1.5);
        // When `enclosed` is true, `AROUND` separates an outline around glyphs
        // from plain lettering's ink around its counters. When `enclosed` is
        // false, `glyphs` is already the ink and `outline()` has verified a
        // sharp band around it, so textured art inside `block` must not veto it.
        if around.is_empty()
            || ((enclosed || found.is_none()) && (around.len() as f32) < AROUND * total as f32)
        {
            return None;
        }
        let strokes: Vec<[u8; 3]> = glyphs
            .enumerate_pixels()
            .filter(|(_, _, pixel)| pixel.0[0] != 0)
            .map(|(x, y, _)| page.get_pixel(corner[0] + x, corner[1] + y).0)
            .collect();
        let outline = found.unwrap_or_else(|| {
            // Too thin to be told from the art by its rings: the distances
            // into a band run up to its width.
            around.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
            let width = 2.0 * around[around.len() / 2].0 as f32;
            let band = around.into_iter().map(|(_, pixel)| pixel).collect();
            Outline {
                color: unlike(band, median(&strokes)),
                width,
            }
        });
        let shell_ink = unlike(strokes, outline.color);
        if enclosed
            && let Some(peeled) =
                peel_double_outline(page, &glyphs, corner, shell_ink, outline.color, &size)
        {
            return Some(peeled);
        }
        Some(Outlined {
            glyphs,
            ink: shell_ink,
            outline,
        })
    })
}

/// When `shell` is the inner outline of double-outlined lettering (for example
/// black or red glyphs inside a white inner outline inside a black outer
/// outline), every glyph sits as an enclosed hole inside `shell`. Peel one
/// layer inward so the enclosed strokes become `glyphs` and `shell` becomes
/// the inner outline.
fn peel_double_outline(
    page: &RgbImage,
    shell: &GrayImage,
    corner: [u32; 2],
    shell_color: [u8; 3],
    outer_color: [u8; 3],
    size: &impl Fn(&GrayImage) -> f32,
) -> Option<Outlined> {
    let mut filled = shell.clone();
    fill_holes(&mut filled);
    let inner = GrayImage::from_fn(shell.width(), shell.height(), |x, y| {
        Luma([
            u8::from(filled.get_pixel(x, y).0[0] != 0 && shell.get_pixel(x, y).0[0] == 0) * u8::MAX,
        ])
    });
    let shell_area = count(shell);
    let inner_area = count(&inner);
    if inner_area < 16 || inner_area * 5 < shell_area {
        return None;
    }
    let shell_parts = connected_components(shell, Connectivity::Eight, Luma([0]));
    let max_label = shell_parts.pixels().map(|p| p.0[0]).max().unwrap_or(0) as usize;
    let mut part_area = vec![0usize; max_label + 1];
    for p in shell_parts.pixels() {
        let l = p.0[0] as usize;
        if l != 0 {
            part_area[l] += 1;
        }
    }
    let mut has_hole = vec![0usize; max_label + 1];
    let (w, h) = (shell.width(), shell.height());
    for (x, y, p) in inner.enumerate_pixels() {
        if p.0[0] == 0 {
            continue;
        }
        for (nx, ny) in [
            x.checked_sub(1).map(|nx| (nx, y)),
            (x + 1 < w).then_some((x + 1, y)),
            y.checked_sub(1).map(|ny| (x, ny)),
            (y + 1 < h).then_some((x, y + 1)),
        ]
        .into_iter()
        .flatten()
        {
            let l = shell_parts.get_pixel(nx, ny).0[0] as usize;
            if l != 0 {
                has_hole[l] += 1;
                break;
            }
        }
    }
    let enclosing_area: usize = (1..=max_label)
        .filter(|&l| has_hole[l] >= 4)
        .map(|l| part_area[l])
        .sum();
    if (enclosing_area as f32) < 0.65 * shell_area as f32 {
        return None;
    }
    let inner_pixels: Vec<[u8; 3]> = inner
        .enumerate_pixels()
        .filter(|(_, _, p)| p.0[0] != 0)
        .map(|(x, y, _)| page.get_pixel(corner[0] + x, corner[1] + y).0)
        .collect();
    let inner_ink = unlike(inner_pixels, shell_color);
    let contrast = (0..3)
        .map(|c| inner_ink[c].abs_diff(shell_color[c]))
        .max()
        .unwrap_or(0);
    if contrast <= 60 {
        return None;
    }
    // Single-outlined hollow glyphs (e.g. `口`, `〇`, `0`) have enclosed counters
    // filled with the same colour as the outer outline. Those counters are solid
    // convex blobs (high bounding-box fill ratio), whereas double-outlined glyphs
    // inside an inner outline have stroke geometry with much lower fill ratio.
    let same_as_outer = (0..3)
        .map(|c| inner_ink[c].abs_diff(outer_color[c]))
        .max()
        .unwrap_or(0)
        <= 40;
    if same_as_outer {
        let inner_parts = connected_components(&inner, Connectivity::Eight, Luma([0]));
        let inner_max = inner_parts.pixels().map(|p| p.0[0]).max().unwrap_or(0) as usize;
        let mut extents: Vec<Option<[u32; 4]>> = vec![None; inner_max + 1];
        for (x, y, p) in inner_parts.enumerate_pixels() {
            let l = p.0[0] as usize;
            if l == 0 {
                continue;
            }
            let e = extents[l].get_or_insert([x, y, x, y]);
            *e = [e[0].min(x), e[1].min(y), e[2].max(x), e[3].max(y)];
        }
        let bbox_area: usize = extents
            .into_iter()
            .flatten()
            .map(|[x0, y0, x1, y1]| (x1 - x0 + 1) as usize * (y1 - y0 + 1) as usize)
            .sum();
        if bbox_area > 0 && (inner_area as f32) > 0.72 * (bbox_area as f32) {
            return None;
        }
    }
    let inner_outline = outline(page, &inner, corner, size(&inner), Some(inner_ink))
        .unwrap_or_else(|| {
            let dist = euclidean_squared_distance_transform(&inner);
            let mut d_shell: Vec<f64> = dist
                .enumerate_pixels()
                .filter(|&(x, y, _)| shell.get_pixel(x, y).0[0] != 0)
                .map(|(_, _, sq)| sq.0[0].sqrt())
                .collect();
            d_shell.sort_unstable_by(f64::total_cmp);
            let width = d_shell.get(d_shell.len() * 3 / 4).copied().unwrap_or(2.0) as f32;
            Outline {
                color: shell_color,
                width,
            }
        });
    Some(Outlined {
        glyphs: inner,
        ink: inner_ink,
        outline: inner_outline,
    })
}

/// The colour of the quarter of `pixels` most unlike `color`: a thin stroke's
/// edges blend into what's beside it.
fn unlike(mut pixels: Vec<[u8; 3]>, color: [u8; 3]) -> [u8; 3] {
    let difference = |pixel: &[u8; 3]| (0..3).map(|at| pixel[at].abs_diff(color[at])).max();
    pixels.sort_unstable_by_key(|pixel| std::cmp::Reverse(difference(pixel)));
    pixels.truncate(pixels.len().div_ceil(4));
    median(&pixels)
}

/// A tone's parts wholly inside a block.
struct Parts {
    labels: image::ImageBuffer<Luma<u32>, Vec<u32>>,
    /// By label: the part's extent and how many pixels it has; `None` for one
    /// that leaves the block or is shorter than asked.
    parts: Vec<Option<([u32; 4], u32)>>,
}

impl Parts {
    /// `tone`'s parts inside `block` that don't reach its edge, at least
    /// `least` pixels long.
    fn within(
        block: &Mask,
        pre_closed: &Mask,
        corner: [u32; 2],
        tone: &impl Fn(u32, u32) -> bool,
        least: u32,
    ) -> Self {
        let (width, height) = (block.width, block.height);
        let inside = |x: u32, y: u32| block.pixels[(y * width + x) as usize] != 0;
        let in_pre_closed = |x: u32, y: u32| pre_closed.contains(corner[0] + x, corner[1] + y);
        let shape = GrayImage::from_fn(width, height, |x, y| {
            Luma([u8::from(inside(x, y) && tone(x, y)) * u8::MAX])
        });
        let labels = connected_components(&shape, Connectivity::Eight, Luma([0]));
        let most = labels.pixels().map(|label| label.0[0]).max().unwrap_or(0) as usize;
        let mut leaves = vec![false; most + 1];
        let mut parts: Vec<Option<([u32; 4], u32)>> = vec![None; most + 1];
        for (x, y, label) in labels.enumerate_pixels() {
            let label = label.0[0] as usize;
            if label == 0 {
                continue;
            }
            leaves[label] |= x == 0
                || y == 0
                || x + 1 == width
                || y + 1 == height
                || !in_pre_closed(x, y)
                || (y - 1..=y + 1).any(|y| (x - 1..=x + 1).any(|x| !inside(x, y)));
            let ([x0, y0, x1, y1], area) = parts[label].get_or_insert(([x, y, x, y], 0));
            (*x0, *y0) = ((*x0).min(x), (*y0).min(y));
            (*x1, *y1) = ((*x1).max(x), (*y1).max(y));
            *area += 1;
        }
        for (part, leaves) in parts.iter_mut().zip(leaves) {
            *part = part.filter(|(extent, _)| !leaves && length(*extent) >= least);
        }
        Self { labels, parts }
    }

    /// The parts `keep` keeps, by label.
    fn shape(&self, keep: impl Fn(usize) -> bool) -> GrayImage {
        let kept: Vec<bool> = (0..self.parts.len())
            .map(|label| self.parts[label].is_some() && keep(label))
            .collect();
        GrayImage::from_fn(self.labels.width(), self.labels.height(), |x, y| {
            let label = self.labels.get_pixel(x, y).0[0] as usize;
            Luma([u8::from(kept[label]) * u8::MAX])
        })
    }

    /// The parts with a solid band of the other tone around them, but those
    /// much shorter than the long ones; all of them when none has.
    fn banded(&self, block: &Mask, tone: &impl Fn(u32, u32) -> bool) -> GrayImage {
        let band = GrayImage::from_fn(block.width, block.height, |x, y| {
            let inside = block.pixels[(y * block.width + x) as usize] != 0;
            Luma([u8::from(inside && !tone(x, y)) * u8::MAX])
        });
        // An outline is at least three pixels wide, and the paper between a
        // screentone's dots narrower.
        let solid = open(&band, Norm::LInf, 1);
        let mut kept: Vec<(u32, u32, usize)> = self
            .parts
            .iter()
            .enumerate()
            .filter_map(|(label, part)| {
                let (extent, area) = (*part)?;
                self.surrounded(label, extent, &solid, Norm::LInf, NEAR, SOLID)
                    .then_some((length(extent), area, label))
            })
            .collect();
        if kept.iter().map(|(_, area, _)| *area).sum::<u32>() < 16 {
            // Fall back to a tighter immediate-ring check for 2 px outlines
            // that `open(..., 1)` erases, ignoring tiny screentone specks.
            let thin: Vec<(u32, u32, usize)> = self
                .parts
                .iter()
                .enumerate()
                .filter_map(|(label, part)| {
                    let (extent, area) = (*part)?;
                    (length(extent) >= 6
                        && self.surrounded(label, extent, &band, Norm::LInf, NEAR_THIN, SOLID_THIN))
                    .then_some((length(extent), area, label))
                })
                .collect();
            let all_area: u32 = self.parts.iter().filter_map(|p| p.map(|(_, a)| a)).sum();
            let thin_area: u32 = thin.iter().map(|(_, a, _)| *a).sum();
            if thin_area >= 16 && thin_area * 2 >= all_area {
                kept = thin;
            }
        }
        kept.sort_unstable();
        let total: u32 = kept.iter().map(|(_, area, _)| area).sum();
        let mut seen = 0;
        let long = kept
            .iter()
            .find(|(_, area, _)| {
                seen += area;
                4 * seen >= 3 * total
            })
            .map_or(0, |(length, ..)| *length);
        kept.retain(|(length, ..)| 3 * length >= long);
        let shape = self.shape(|at| kept.iter().any(|(_, _, label)| *label == at));
        match count(&shape) < 16 {
            true => self.shape(|_| true),
            false => shape,
        }
    }

    /// Whether what's within `near` of part `label` is mostly `solid`.
    fn surrounded(
        &self,
        label: usize,
        [x0, y0, x1, y1]: [u32; 4],
        solid: &GrayImage,
        norm: Norm,
        near: u8,
        min_solid: f32,
    ) -> bool {
        let pad = u32::from(near);
        let (left, top) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
        let right = (x1 + pad).min(solid.width() - 1);
        let bottom = (y1 + pad).min(solid.height() - 1);
        let part = GrayImage::from_fn(right - left + 1, bottom - top + 1, |x, y| {
            let at = self.labels.get_pixel(left + x, top + y).0[0] as usize;
            Luma([u8::from(at == label) * u8::MAX])
        });
        let (mut around, mut held) = (0u32, 0u32);
        for (x, y, pixel) in dilate(&part, norm, near).enumerate_pixels() {
            if pixel.0[0] != 0 && part.get_pixel(x, y).0[0] == 0 {
                around += 1;
                held += u32::from(solid.get_pixel(left + x, top + y).0[0] != 0);
            }
        }
        around > 0 && held as f32 >= min_solid * around as f32
    }
}

/// A part's longer side.
fn length([x0, y0, x1, y1]: [u32; 4]) -> u32 {
    (x1 - x0).max(y1 - y0) + 1
}

fn count(shape: &GrayImage) -> usize {
    shape.pixels().filter(|pixel| pixel.0[0] != 0).count()
}

/// Each channel's median.
fn median(pixels: &[[u8; 3]]) -> [u8; 3] {
    std::array::from_fn(|at| {
        let mut channel: Vec<u8> = pixels.iter().map(|pixel| pixel[at]).collect();
        channel.sort_unstable();
        channel[channel.len() / 2]
    })
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::super::Typography;
    use super::*;

    const BLACK: [u8; 3] = [0; 3];
    const WHITE: [u8; 3] = [255; 3];

    /// The typography of a column of crosses, 20 pixels across in strokes of
    /// 4, each with a ring beside its strokes, in `ink` with an outline of 4
    /// pixels when there's one, on a page of `art`.
    fn column(
        ink: [u8; 3],
        outline: Option<[u8; 3]>,
        art: impl Fn(u32, u32) -> [u8; 3],
    ) -> Typography {
        let cross = |x: u32, y: u32| {
            let (x, y) = (x.wrapping_sub(60), y.wrapping_sub(60));
            x < 20
                && y < 112
                && y % 28 < 20
                && ((8..12).contains(&x)
                    || (8..12).contains(&(y % 28))
                    || (x < 8
                        && y % 28 < 8
                        && !((2..6).contains(&x) && (2..6).contains(&(y % 28)))))
        };
        let near = |x: u32, y: u32, reach: u32| {
            (y - reach..=y + reach).any(|y| (x - reach..=x + reach).any(|x| cross(x, y)))
        };
        let reach = if outline.is_some() { 4 } else { 0 };
        let page = RgbImage::from_fn(200, 240, |x, y| {
            let margin = !(40..160).contains(&x) || !(40..200).contains(&y);
            Rgb(match outline {
                _ if margin => art(x, y),
                _ if cross(x, y) => ink,
                Some(outline) if near(x, y, 4) => outline,
                _ => art(x, y),
            })
        });
        let mask = Mask {
            x: 40,
            y: 40,
            width: 120,
            height: 160,
            pixels: (40..200)
                .flat_map(|y| (40..160).map(move |x| (x, y)))
                .map(|(x, y)| u8::from(near(x, y, reach)) * u8::MAX)
                .collect(),
        };
        mask.typography(&page).expect("ink")
    }

    #[test]
    fn an_outline_isn_t_taken_for_the_ink_it_encloses() {
        // On black, on black and white art, and on a screentone.
        let grid = |x: u32, y: u32| {
            if (x / 7 + y / 5).is_multiple_of(2) {
                BLACK
            } else {
                WHITE
            }
        };
        let dots = |x: u32, y: u32| {
            if x.is_multiple_of(3) && y.is_multiple_of(3) {
                BLACK
            } else {
                WHITE
            }
        };
        for (ink, outline) in [(BLACK, WHITE), (WHITE, BLACK), ([200, 20, 40], WHITE)] {
            for found in [
                column(ink, Some(outline), |_, _| BLACK),
                column(ink, Some(outline), |_, _| [120, 120, 120]),
                column(ink, Some(outline), grid),
                column(ink, Some(outline), dots),
            ] {
                assert_eq!(found.ink, Some(ink), "{found:?}");
                assert!((18.0..=22.0).contains(&found.size), "{found:?}");
            }
        }
    }

    #[test]
    fn black_letters_on_dark_art_retain_their_white_outline_width() {
        let found = column(BLACK, Some(WHITE), |_, _| [20; 3]);
        assert_eq!(found.ink, Some(BLACK));
        let band = found.outline.expect("a white band");
        assert_eq!(band.color, WHITE);
        assert!((3.0..=5.0).contains(&band.width), "{band:?}");
        assert!((18.0..=22.0).contains(&found.size));
    }

    #[test]
    fn plain_lettering_keeps_the_ink_around_its_counters() {
        for (ink, paper) in [(BLACK, WHITE), (WHITE, BLACK), ([200, 20, 40], WHITE)] {
            let found = column(ink, None, |_, _| paper);
            assert_eq!((found.ink, found.outline), (Some(ink), None));
            assert_eq!(found.size, 20.0);
        }
    }

    #[test]
    fn high_res_outlined_columns_do_not_bridge_dark_art_gutters() {
        // Three vertical columns of white-outlined black glyphs separated by
        // 36px dark-art gutters on a 4961x7016 page, with light art bridging
        // the gutter at the top and bottom. An uncapped scaled(6.0) closing
        // (41px radius) would trap the dark-art gutters and merge all columns.
        let mut page = RgbImage::from_pixel(4961, 7016, Rgb([20, 20, 20]));
        let col_xs = [2100u32, 2280, 2460];
        let glyph_w = 120u32;
        let outline_w = 12u32;
        let row_ys = [3000u32, 3160, 3320, 3480];

        // Light art bars near top and bottom spanning across the gutters.
        for y in (2990..3010).chain(3590..3610) {
            for x in 2080..2600 {
                page.put_pixel(x, y, Rgb(WHITE));
            }
        }
        // Draw white outlines first, then black glyphs inside.
        for &cx in &col_xs {
            for &ry in &row_ys {
                for y in ry - outline_w..ry + glyph_w + outline_w {
                    for x in cx - outline_w..cx + glyph_w + outline_w {
                        page.put_pixel(x, y, Rgb(WHITE));
                    }
                }
            }
        }
        for &cx in &col_xs {
            for &ry in &row_ys {
                for y in ry..ry + glyph_w {
                    for x in cx..cx + glyph_w {
                        let stroke = x < cx + 20
                            || x >= cx + glyph_w - 20
                            || y < ry + 20
                            || y >= ry + glyph_w - 20;
                        if stroke {
                            page.put_pixel(x, y, Rgb(BLACK));
                        }
                    }
                }
            }
        }

        let (mx, my, mw, mh) = (2080u32, 2980u32, 520u32, 640u32);
        let mut pixels = vec![0u8; (mw * mh) as usize];
        for &cx in &col_xs {
            for &ry in &row_ys {
                for y in ry - outline_w..ry + glyph_w + outline_w {
                    for x in cx - outline_w..cx + glyph_w + outline_w {
                        pixels[((y - my) * mw + (x - mx)) as usize] = u8::MAX;
                    }
                }
            }
        }
        let mask = Mask {
            x: mx,
            y: my,
            width: mw,
            height: mh,
            pixels,
        };
        let found = mask.typography(&page).expect("outlined high-res columns");
        assert!(found.vertical, "expected vertical columns: {found:?}");
        assert!(
            (105.0..=145.0).contains(&found.size),
            "expected single-column size ~120px, got {}px ({found:?})",
            found.size
        );
        assert!(found.outline.is_some(), "expected white outline: {found:?}");
    }

    #[test]
    fn thin_two_pixel_outline_is_detected_without_swapping_ink() {
        let cross = |x: u32, y: u32| {
            let (x, y) = (x.wrapping_sub(60), y.wrapping_sub(60));
            x < 20
                && y < 112
                && y % 28 < 20
                && ((8..12).contains(&x) || (8..12).contains(&(y % 28)))
        };
        let near = |x: u32, y: u32, reach: u32| {
            (y.saturating_sub(reach)..=y + reach)
                .any(|yy| (x.saturating_sub(reach)..=x + reach).any(|xx| cross(xx, yy)))
        };
        let page = RgbImage::from_fn(200, 240, |x, y| {
            let bg = if (x / 8 + y / 8).is_multiple_of(2) {
                [180, 180, 180]
            } else {
                [110, 110, 110]
            };
            Rgb(if cross(x, y) {
                WHITE
            } else if near(x, y, 2) {
                BLACK
            } else {
                bg
            })
        });
        let mask = Mask {
            x: 40,
            y: 40,
            width: 120,
            height: 160,
            pixels: (40..200)
                .flat_map(|y| (40..160).map(move |x| (x, y)))
                .map(|(x, y)| u8::from(near(x, y, 2)) * u8::MAX)
                .collect(),
        };
        let found = mask.typography(&page).expect("ink");
        assert_eq!(found.ink, Some(WHITE), "{found:?}");
        let band = found.outline.expect("a 2 px black outline");
        assert_eq!(band.color, BLACK);
        assert!((1.5..=3.5).contains(&band.width), "{band:?}");
    }

    #[test]
    fn double_outlined_lettering_peels_to_inner_glyph_ink() {
        let red = [220, 20, 30];
        let cross = |x: u32, y: u32| {
            let (x, y) = (x.wrapping_sub(60), y.wrapping_sub(60));
            x < 20
                && y < 112
                && y % 28 < 20
                && ((7..13).contains(&x) || (7..13).contains(&(y % 28)))
        };
        let near = |x: u32, y: u32, reach: u32| {
            (y.saturating_sub(reach)..=y + reach)
                .any(|yy| (x.saturating_sub(reach)..=x + reach).any(|xx| cross(xx, yy)))
        };
        let page = RgbImage::from_fn(200, 240, |x, y| {
            let bg = if (x / 10 + y / 10).is_multiple_of(2) {
                [30, 30, 30]
            } else {
                [140, 140, 140]
            };
            Rgb(if cross(x, y) {
                red
            } else if near(x, y, 4) {
                WHITE
            } else if near(x, y, 7) {
                BLACK
            } else {
                bg
            })
        });
        let mask = Mask {
            x: 40,
            y: 40,
            width: 120,
            height: 160,
            pixels: (40..200)
                .flat_map(|y| (40..160).map(move |x| (x, y)))
                .map(|(x, y)| u8::from(near(x, y, 7)) * u8::MAX)
                .collect(),
        };
        let found = mask.typography(&page).expect("ink");
        assert_eq!(found.ink, Some(red), "{found:?}");
        let band = found.outline.expect("a white inner outline");
        assert_eq!(band.color, WHITE);
    }
}
