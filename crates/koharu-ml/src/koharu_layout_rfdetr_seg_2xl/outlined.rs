//! Recover glyphs from both sides of the tone split when lettering lies on art.

use std::collections::HashMap;

use image::{GrayImage, Luma, RgbImage};
use imageproc::{
    distance_transform::{Norm, euclidean_squared_distance_transform},
    morphology::open,
    region_labelling::{Connectivity, connected_components},
};

use super::{
    KoharuLayoutMask as Mask,
    measure::{Outline, fill_holes, outline, scaled},
};

const MIN_GLYPH_AREA: u32 = 16;
const SEED_TOLERANCE: u8 = 24;

/// Outlined lettering, by its glyphs.
pub(super) struct Outlined {
    pub glyphs: GrayImage,
    pub ink: [u8; 3],
    pub outline: Outline,
}

/// Recover the fill and outline of the lettering in `block` from a provisional
/// `ink` split. Segmented cleanup can recover fills beside a cut outline.
pub(super) fn outlined(
    page: &RgbImage,
    block: &Mask,
    pre_closed: &Mask,
    ink: &GrayImage,
    recover_clipped: bool,
    size: impl Fn(&GrayImage) -> f32,
) -> Option<Outlined> {
    let corner = [block.x, block.y];
    let color_at = |x, y| page[(corner[0] + x, corner[1] + y)].0;
    let least = scaled(3.0, page.width(), page.height());
    let ink_area = count(ink) as u32;
    // Bound any recovered fill by the source strokes. The detector can
    // support a wider search when its footprint resolves both strokes and
    // gaps. A nearly solid proposal has little detail to support recovery.
    let detector_area = pre_closed.pixels.iter().filter(|&&p| p != 0).count() as u32;
    let detector_detail = detector_area.min(pre_closed.pixels.len() as u32 - detector_area);
    let max_fill = (3 * ink_area).max(detector_detail / 2);
    let pad = scaled(4.0, page.width(), page.height());
    let [left, top] = corner;
    let x0 = left.saturating_sub(pad);
    let y0 = top.saturating_sub(pad);
    let x1 = (left + block.width + pad).min(page.width());
    let y1 = (top + block.height + pad).min(page.height());
    let exterior: Vec<_> = (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| (x, y)))
        .filter(|&(x, y)| !pre_closed.contains(x, y))
        .map(|(x, y)| page[(x, y)].0)
        .collect();
    let exterior_share = |color: [u8; 3], tolerance| {
        exterior
            .iter()
            .filter(|&&pixel| color_distance(pixel, color) <= tolerance)
            .count() as f32
            / exterior.len().max(1) as f32
    };
    let recover = |parts: Parts, band: &GrayImage, enclosed: bool| {
        let all: Vec<bool> = parts.parts.iter().map(Option::is_some).collect();
        let all_area = parts.area(&all);
        if all_area < MIN_GLYPH_AREA {
            return None;
        }
        let whole = parts.shape(&all);
        let opened = open(band, Norm::LInf, 1);
        let thick = parts.banded(&opened, 0, 3, 0.70);
        let thick_area = parts.area(&thick);
        let thin = (2 * thick_area < all_area).then(|| parts.banded(band, 0, 1, 0.70));
        let mut candidates: Vec<&[bool]> = [&thick[..], &all[..]].into();
        if let Some(thin) = &thin
            && thin != &thick
            && thin != &all
        {
            candidates.insert(0, thin);
        }
        candidates.dedup();
        for candidate in candidates {
            let candidate_area = parts.area(candidate);
            if !(MIN_GLYPH_AREA..=max_fill).contains(&candidate_area) {
                continue;
            }
            let glyphs = if candidate == all {
                &whole
            } else {
                &parts.shape(candidate)
            };
            let colors = |glyphs: &GrayImage| {
                glyphs
                    .enumerate_pixels()
                    .filter(|(_, _, pixel)| pixel[0] != 0)
                    .map(|(x, y, _)| color_at(x, y))
                    .collect::<Vec<_>>()
            };
            let strokes = colors(glyphs);
            let Some(found) = outline(page, glyphs, corner, size(glyphs), Some(median(&strokes)))
            else {
                continue;
            };
            let band_in_exterior =
                exterior.len() >= 16 && exterior_share(found.color, SEED_TOLERANCE) >= 0.15;
            let band_color = GrayImage::from_fn(block.width, block.height, |x, y| {
                if band_in_exterior && block.pixels[(y * block.width + x) as usize] == 0 {
                    return Luma([0]);
                }
                let error = color_distance(color_at(x, y), found.color);
                Luma([64u8.saturating_sub(error).saturating_mul(8)])
            });
            let reach = found.width.round().clamp(1.0, 3.0) as u8;
            // At least one ring beyond the ink's antialias fringe must
            // agree with the measured band, at the same Euclidean distance.
            let supported = parts.banded(&band_color, u8::from(reach > 1), reach, 0.60);
            let area = parts.area(&supported);
            if !(MIN_GLYPH_AREA..=max_fill).contains(&area)
                || 12 * area < ink_area
                || 2 * area < candidate_area
            {
                continue;
            }
            let supported_glyphs;
            let (glyphs, found, strokes) = if supported == candidate {
                (glyphs, found, strokes)
            } else {
                let glyphs = if supported == all {
                    &whole
                } else {
                    supported_glyphs = parts.shape(&supported);
                    &supported_glyphs
                };
                let strokes = colors(glyphs);
                let Some(verified) =
                    outline(page, glyphs, corner, size(glyphs), Some(median(&strokes)))
                else {
                    continue;
                };
                if color_distance(found.color, verified.color) > SEED_TOLERANCE {
                    continue;
                }
                (glyphs, verified, strokes)
            };
            if !enclosed
                && exterior.len() >= 16
                && exterior_share(found.color, SEED_TOLERANCE) >= 0.97
            {
                continue;
            }
            // The strokes and their outline must span the lettering block.
            let [x0, y0, x1, y1] = parts
                .parts
                .iter()
                .zip(&supported)
                .filter_map(|(part, &keep)| part.filter(|_| keep).map(|(extent, _)| extent))
                .fold(
                    [block.width, block.height, 0, 0],
                    |[x0, y0, x1, y1], [a, b, c, d]| [x0.min(a), y0.min(b), x1.max(c), y1.max(d)],
                );
            let diameter = 2 * found.width.ceil() as u32;
            if 5 * (x1 - x0 + 1 + diameter) < 3 * block.width
                || 5 * (y1 - y0 + 1 + diameter) < 3 * block.height
            {
                continue;
            }
            // A candidate cannot explain band-color strokes elsewhere in the
            // detector support. Account for artists using the same color in
            // the surrounding page.
            let band_share = exterior_share(found.color, SEED_TOLERANCE);
            let distance = euclidean_squared_distance_transform(glyphs);
            let reach_sq = f64::from(found.width + 2.0).powi(2);
            let (mut near_band, mut far_band, mut far_area) = (0, 0, 0);
            for (x, y, pixel) in distance.enumerate_pixels() {
                if !pre_closed.contains(corner[0] + x, corner[1] + y) {
                    continue;
                }
                let same_color = color_distance(color_at(x, y), found.color) <= SEED_TOLERANCE;
                if pixel[0] <= reach_sq {
                    near_band += u32::from(same_color);
                } else {
                    far_band += u32::from(same_color);
                    far_area += 1;
                }
            }
            let excess = far_band as f32 - band_share * far_area as f32;
            if excess * 3.0 > 2.0 * near_band as f32 {
                continue;
            }
            let shell_ink = unlike(strokes, found.color);
            if 4 * area < ink_area {
                let source_colors = colors(ink);
                let source_matches = source_colors
                    .iter()
                    .filter(|pixel| (0..3).all(|c| pixel[c].abs_diff(shell_ink[c]) <= 48))
                    .count();
                if 2 * source_matches > source_colors.len()
                    && let Some(all_band) = outline(page, ink, corner, size(ink), Some(shell_ink))
                    && (0..3).all(|c| all_band.color[c].abs_diff(found.color[c]) <= 16)
                {
                    return Some(Outlined {
                        glyphs: ink.clone(),
                        ink: shell_ink,
                        outline: all_band,
                    });
                }
            }
            if let Some(peeled) = peel_double_outline(
                page,
                glyphs,
                corner,
                [shell_ink, found.color],
                &size,
                &parts,
                &supported,
            )
            .filter(|peeled| count(&peeled.glyphs) <= max_fill as usize)
            {
                return Some(peeled);
            }
            let mut glyphs = glyphs.clone();
            if recover_clipped {
                // A cut through the outline can bring a whole fill against
                // the edge of the block. Only an already verified pair of
                // colors can identify that fill and the band around it.
                let cut = Parts::within(
                    block,
                    pre_closed,
                    &|x, y| color_distance(color_at(x, y), shell_ink) <= SEED_TOLERANCE,
                    least,
                    Some((page, found.color)),
                );
                let selected = cut.banded(&band_color, u8::from(reach > 1), reach, 0.85);
                if area + cut.area(&selected) <= max_fill {
                    for (pixel, label) in glyphs.pixels_mut().zip(cut.labels.pixels()) {
                        if selected[label[0] as usize] {
                            pixel[0] = 255;
                        }
                    }
                }
            }
            return Some(Outlined {
                glyphs,
                ink: shell_ink,
                outline: found,
            });
        }
        None
    };

    let find = |tone: &dyn Fn(u32, u32) -> bool, enclosed| {
        let parts = Parts::within(block, pre_closed, tone, least, None);
        let band = GrayImage::from_fn(block.width, block.height, |x, y| {
            let inside = block.pixels[(y * block.width + x) as usize] != 0;
            Luma([u8::from(inside && !tone(x, y)) * 255])
        });
        recover(parts, &band, enclosed)
    };
    // Use the measured strokes first when their contour verifies. Only
    // then search the opposite tone for an enclosed glyph fill.
    if let Some(found) = [false, true]
        .into_iter()
        .find_map(|enclosed| find(&|x, y| (ink[(x, y)][0] == 0) == enclosed, enclosed))
    {
        return Some(found);
    }

    // When the threshold joins a glyph to similarly light or dark art,
    // use the colors past the detector's support to seed a new split.
    if exterior.len() < 16 {
        return None;
    }
    let mut colors = HashMap::<[u8; 3], (u32, [u64; 3])>::new();
    for (i, _) in block.pixels.iter().enumerate().filter(|&(_, &p)| p != 0) {
        let color = color_at(i as u32 % block.width, i as u32 / block.width);
        let (count, sum) = colors.entry(color.map(|channel| channel / 16)).or_default();
        *count += 1;
        for (sum, channel) in sum.iter_mut().zip(color) {
            *sum += u64::from(channel);
        }
    }
    let mut seeds: Vec<_> = colors
        .into_values()
        .filter(|&(count, _)| count >= MIN_GLYPH_AREA)
        .map(|(count, sum)| (count, sum.map(|c| (c / u64::from(count)) as u8)))
        .collect();
    seeds.sort_unstable_by(|a, b| b.cmp(a));
    seeds.into_iter().take(5).find_map(|(_, seed)| {
        (exterior_share(seed, SEED_TOLERANCE) < 0.85).then_some(())?;
        find(
            &|x, y| color_distance(color_at(x, y), seed) <= SEED_TOLERANCE,
            false,
        )
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
    [shell_color, outer_color]: [[u8; 3]; 2],
    size: &impl Fn(&GrayImage) -> f32,
    parts: &Parts,
    selected: &[bool],
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
    let mut has_hole = vec![0usize; selected.len()];
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
            let l = parts.labels[(nx, ny)][0] as usize;
            if selected[l] {
                has_hole[l] += 1;
                break;
            }
        }
    }
    let enclosing_area: u32 = parts
        .parts
        .iter()
        .zip(has_hole)
        .filter_map(|(part, holes)| part.filter(|_| holes >= 4).map(|(_, area)| area))
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
    if color_distance(inner_ink, shell_color) <= 60 {
        return None;
    }
    // Single-outlined hollow glyphs (e.g. `口`, `〇`, `0`) have enclosed counters
    // filled with the same colour as the outer outline. Those counters are solid
    // convex blobs (high bounding-box fill ratio), whereas double-outlined glyphs
    // inside an inner outline have stroke geometry with much lower fill ratio.
    let same_as_outer = color_distance(inner_ink, outer_color) <= 40;
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
    pixels.sort_unstable_by_key(|&pixel| std::cmp::Reverse(color_distance(pixel, color)));
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
    /// Detector-supported parts at least `least` pixels long. With a verified
    /// band, select only parts exposed by a cut through that band.
    fn within(
        block: &Mask,
        pre_closed: &Mask,
        tone: &dyn Fn(u32, u32) -> bool,
        least: u32,
        cut_band: Option<(&RgbImage, [u8; 3])>,
    ) -> Self {
        let (width, height) = (block.width, block.height);
        let inside = |x: u32, y: u32| block.pixels[(y * width + x) as usize] != 0;
        let in_pre_closed = |x: u32, y: u32| pre_closed.contains(block.x + x, block.y + y);
        let shape = GrayImage::from_fn(width, height, |x, y| {
            Luma([u8::from(inside(x, y) && tone(x, y)) * u8::MAX])
        });
        let labels = connected_components(&shape, Connectivity::Eight, Luma([0]));
        let most = labels.pixels().map(|label| label.0[0]).max().unwrap_or(0) as usize;
        let mut touches = vec![false; most + 1];
        let mut open = vec![false; most + 1];
        let mut held = vec![0; most + 1];
        let mut parts: Vec<Option<([u32; 4], u32)>> = vec![None; most + 1];
        for (x, y, label) in labels.enumerate_pixels() {
            let label = label.0[0] as usize;
            if label == 0 {
                continue;
            }
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < width && ny < height && inside(nx, ny) {
                    continue;
                }
                touches[label] = true;
                open[label] |= cut_band.is_none_or(|(page, band)| {
                    page.get_pixel_checked(block.x.wrapping_add(nx), block.y.wrapping_add(ny))
                        .is_none_or(|pixel| color_distance(pixel.0, band) > SEED_TOLERANCE)
                });
            }
            held[label] += u32::from(in_pre_closed(x, y));
            let ([x0, y0, x1, y1], area) = parts[label].get_or_insert(([x, y, x, y], 0));
            (*x0, *y0) = ((*x0).min(x), (*y0).min(y));
            (*x1, *y1) = ((*x1).max(x), (*y1).max(y));
            *area += 1;
        }
        for (label, part) in parts.iter_mut().enumerate() {
            *part = part.filter(|(extent, area)| {
                !open[label]
                    && touches[label] == cut_band.is_some()
                    && held[label] * 4 >= area * 3
                    && length(*extent) >= least
            });
        }
        Self { labels, parts }
    }

    fn area(&self, selected: &[bool]) -> u32 {
        self.parts
            .iter()
            .zip(selected)
            .filter_map(|(part, &keep)| part.filter(|_| keep).map(|(_, area)| area))
            .sum()
    }

    fn shape(&self, selected: &[bool]) -> GrayImage {
        GrayImage::from_fn(self.labels.width(), self.labels.height(), |x, y| {
            let label = self.labels[(x, y)][0] as usize;
            Luma([u8::from(selected[label]) * 255])
        })
    }

    /// Select whole components so the outline evidence and glyph geometry
    /// refer to the same strokes.
    fn banded(&self, band: &GrayImage, fringe: u8, near: u8, solid: f32) -> Vec<bool> {
        self.parts
            .iter()
            .enumerate()
            .map(|(label, part)| {
                part.is_some_and(|(extent, _)| {
                    self.surrounded(label, extent, band, [fringe, near], solid)
                })
            })
            .collect()
    }

    /// Whether the rings around part `label` are mostly `solid`.
    fn surrounded(
        &self,
        label: usize,
        [x0, y0, x1, y1]: [u32; 4],
        solid: &GrayImage,
        [fringe, near]: [u8; 2],
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
        let (mut around, mut held) = (0u32, 0u64);
        let (inner, outer) = (f64::from(fringe).powi(2), f64::from(near).powi(2));
        for (x, y, squared) in euclidean_squared_distance_transform(&part).enumerate_pixels() {
            if squared[0] > inner && squared[0] <= outer {
                around += 1;
                held += u64::from(solid[(left + x, top + y)][0]);
            }
        }
        around > 0 && held as f32 >= min_solid * 255.0 * around as f32
    }
}

/// A part's longer side.
fn length([x0, y0, x1, y1]: [u32; 4]) -> u32 {
    (x1 - x0).max(y1 - y0) + 1
}

fn count(shape: &GrayImage) -> usize {
    shape.pixels().filter(|pixel| pixel.0[0] != 0).count()
}

pub(super) fn color_distance(a: [u8; 3], b: [u8; 3]) -> u8 {
    a[0].abs_diff(b[0])
        .max(a[1].abs_diff(b[1]))
        .max(a[2].abs_diff(b[2]))
}

/// Each channel's median.
pub(super) fn median(pixels: &[[u8; 3]]) -> [u8; 3] {
    std::array::from_fn(|at| {
        let mut channel: Vec<u8> = pixels.iter().map(|pixel| pixel[at]).collect();
        channel.sort_unstable();
        channel[channel.len() / 2]
    })
}

#[cfg(test)]
mod tests {
    use image::Rgb;
    use imageproc::morphology::dilate;

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
    fn an_outline_band_recovers_blue_glyphs_inside_a_dark_checker() {
        let ink_color = [24, 82, 215];
        let page = RgbImage::from_fn(144, 104, |x, y| {
            let dx = (x + 8) % 32;
            let inner_y = (40..66).contains(&y);
            let glyph = (30..115).contains(&x) && (9..15).contains(&dx);
            let band = (27..118).contains(&x) && (6..18).contains(&dx);
            Rgb(if glyph && inner_y {
                ink_color
            } else if band && (37..69).contains(&y) {
                WHITE
            } else if (x / 9 + y / 11).is_multiple_of(2) {
                [14, 18, 20]
            } else {
                [46, 50, 55]
            })
        });
        let block = Mask {
            x: 25,
            y: 34,
            width: 90,
            height: 39,
            pixels: vec![255; 90 * 39],
        };
        let ink = GrayImage::from_fn(block.width, block.height, |x, y| {
            Luma([u8::from(page[(x + block.x, y + block.y)].0 != WHITE) * 255])
        });
        let found = outlined(&page, &block, &block, &ink, false, |_| 26.0)
            .expect("white band around the blue glyphs");
        assert_eq!(found.ink, ink_color);
        assert_eq!(found.outline.color, WHITE);
        assert!(
            found
                .glyphs
                .enumerate_pixels()
                .all(|(x, y, p)| { p[0] == 0 || page[(x + block.x, y + block.y)].0 == ink_color })
        );
        assert!(count(&found.glyphs) >= 3 * 25 * 5);
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
    fn three_small_outlined_ellipsis_dots_keep_their_fill() {
        let dots = GrayImage::from_fn(30, 9, |x, y| {
            Luma([u8::from(
                (3..6).contains(&y) && [3, 14, 25].iter().any(|&dot| (dot..dot + 2).contains(&x)),
            ) * 255])
        });
        let shell = dilate(&dots, Norm::LInf, 3);
        let mut page = RgbImage::from_pixel(80, 64, Rgb([13, 25, 62]));
        for (x, y, _) in shell.enumerate_pixels().filter(|(_, _, p)| p[0] != 0) {
            page[(20 + x, 20 + y)] = Rgb(if dots[(x, y)][0] != 0 { BLACK } else { WHITE });
        }
        let mask = Mask {
            x: 20,
            y: 20,
            width: shell.width(),
            height: shell.height(),
            pixels: shell.into_raw(),
        };

        let found = mask.typography(&page).expect("outlined ellipsis");
        assert_eq!(found.ink, Some(BLACK), "{found:?}");
        assert_eq!(found.outline.expect("white outline").color, WHITE);
        assert_eq!(found.size, 3.0);
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

    #[test]
    fn framed_white_inset_with_strokes_is_not_recovered_as_outline_fill() {
        let page = RgbImage::from_fn(140, 140, |x, y| {
            let inset = (44..96).contains(&x) && (44..96).contains(&y);
            let frame = (42..98).contains(&x) && (42..98).contains(&y) && !inset;
            let stroke = ((58..61).contains(&x) && (54..86).contains(&y))
                || ((78..81).contains(&x) && (54..86).contains(&y));
            Rgb(if frame || stroke {
                BLACK
            } else if inset {
                WHITE
            } else {
                [210, 210, 210]
            })
        });
        let block = Mask {
            x: 38,
            y: 38,
            width: 64,
            height: 64,
            pixels: vec![255; 64 * 64],
        };
        let ink = GrayImage::from_fn(64, 64, |x, y| {
            let (px, py) = (38 + x, 38 + y);
            let inset = (44..96).contains(&px) && (44..96).contains(&py);
            let frame = (42..98).contains(&px) && (42..98).contains(&py) && !inset;
            let stroke = ((58..61).contains(&px) && (54..86).contains(&py))
                || ((78..81).contains(&px) && (54..86).contains(&py));
            Luma([u8::from(frame || stroke) * 255])
        });
        assert!(
            outlined(&page, &block, &block, &ink, false, |_| 32.0).is_none(),
            "framed caption paper must not be recovered as a glyph fill"
        );
    }

    #[test]
    fn exterior_matching_outline_color_does_not_admit_art_strokes_outside_block() {
        // White-outlined black glyphs inside a diagonal block on white-dominant
        // art, with a dark art bar right beside the block edge. Because white
        // also dominates the exterior, outside-block white paper must not count
        // as outline support for the edge art bar.
        let mut page = RgbImage::from_pixel(160, 160, Rgb(WHITE));
        for y in 20..140 {
            for x in 20..140 {
                let near_bar = (87..99).contains(&x) && (44..70).contains(&y);
                if !near_bar && (x + y) % 19 < 4 {
                    page[(x, y)] = Rgb([140, 140, 140]);
                }
            }
        }
        let mut glyphs = GrayImage::new(160, 160);
        for step in 0..3 {
            let cx = 54 + step * 20;
            let cy = 54 + step * 20;
            for y in cy..cy + 14 {
                for x in cx..cx + 14 {
                    if x < cx + 4 || y < cy + 4 || x >= cx + 10 || y >= cy + 10 {
                        glyphs[(x, y)] = Luma([255]);
                    }
                }
            }
        }
        let outline_band = dilate(&glyphs, Norm::L2, 4);
        for (x, y, p) in outline_band.enumerate_pixels() {
            if p[0] != 0 {
                page[(x, y)] = Rgb(if glyphs[(x, y)][0] != 0 { BLACK } else { WHITE });
            }
        }
        // Dark art stroke ending exactly at the block edge, well away from the
        // diagonal glyphs' own outlines and surrounded by exterior white paper
        // outside the block.
        for y in 49..65 {
            for x in 92..94 {
                page[(x, y)] = Rgb(BLACK);
            }
        }
        let mut block_img = dilate(&outline_band, Norm::L2, 2);
        for y in 49..65 {
            for x in 92..94 {
                block_img[(x, y)] = Luma([255]);
            }
        }
        let (x0, y0, w, h) = (44, 44, 72, 72);
        let block_ref = &block_img;
        let mask = Mask {
            x: x0,
            y: y0,
            width: w,
            height: h,
            pixels: (y0..y0 + h)
                .flat_map(|y| (x0..x0 + w).map(move |x| block_ref[(x, y)][0]))
                .collect(),
        };
        let ink_in_block = GrayImage::from_fn(w, h, |x, y| {
            let inside_block = block_ref[(x0 + x, y0 + y)][0] != 0;
            let dark = page[(x0 + x, y0 + y)].0 == BLACK;
            Luma([u8::from(inside_block && dark) * 255])
        });
        let found = outlined(&page, &mask, &mask, &ink_in_block, true, |_| 14.0)
            .expect("tilted outlined glyphs");
        assert_eq!(found.ink, BLACK);
        assert_eq!(found.outline.color, WHITE);
        assert_eq!(
            found.glyphs[(56 - x0, 56 - y0)][0],
            255,
            "real glyph should be recovered"
        );
        assert_eq!(
            found.glyphs[(93 - x0, 56 - y0)][0],
            0,
            "edge art bar exposed to exterior white must not be recovered into glyph mask"
        );
    }
}
