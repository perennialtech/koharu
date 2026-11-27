//! The shape and typography of lettering, on the page under its mask.
//! Ported from koharu-layout-highres at e1d22aed. Fitting runs after detector
//! fusion, while the original boxes can still resolve competing masks.

use std::{collections::HashMap, sync::Arc};

use image::{GrayImage, ImageBuffer, Luma, RgbImage, imageops};
use imageproc::{
    distance_transform::{Norm, euclidean_squared_distance_transform},
    morphology::{close, dilate},
    region_labelling::{Connectivity, connected_components},
};
use rayon::prelude::*;
use serde::Serialize;

use super::{KoharuLayoutDetection, KoharuLayoutMask as Mask, outlined::outlined};

/// What a detection's lettering shows.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Typography {
    /// Whether the lines run down the page: the lettering's block is longer
    /// that way, at its tilt, or its ink stands in columns, as a balloon of
    /// several short columns, wider than tall, has it.
    pub vertical: bool,
    /// How far the lines turn clockwise from horizontal, or from vertical for
    /// vertical lines, in degrees within ±45°: the commonest direction of the
    /// block's edges. A turn under `UPRIGHT`, or one that moves the lines'
    /// far end by less than a stroke, is 0: it can't be told from the glyphs'
    /// ragged edges, and lettering is upright far more often than not.
    pub tilt: f32,
    /// How broad the lines are across, in pixels, about a CJK glyph's em:
    /// the ink's profile across the lines splits into a run for each, and
    /// of the runs at least as broad as the one holding the median of the
    /// ink, the one holding their median is the size, so neither ruby, a
    /// stray mark, nor a line of kana narrower than the em counts.
    pub size: f32,
    pub stroke_width: Option<f32>,
    /// The commonest colour segment under the ink, and around it.
    pub ink: Option<[u8; 3]>,
    pub paper: Option<[u8; 3]>,
    /// The band of one colour around the glyphs, as lettering over art has.
    pub outline: Option<Outline>,
}

/// A lettering's outline: a band of one colour, unlike the ink's, around its
/// glyphs, with something else past it, as Photoshop's Outside stroke draws.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Outline {
    pub color: [u8; 3],
    /// How far it reaches past the glyphs, in pixels.
    pub width: f32,
}

impl Mask {
    /// Tighten the cropped mask after neighbours have claimed shared pixels.
    fn trim_to_owned_pixels(&mut self) -> u32 {
        let view =
            ImageBuffer::<Luma<u8>, _>::from_raw(self.width, self.height, self.pixels.as_slice())
                .expect("a mask holds a byte a pixel");
        let (mut x0, mut y0, mut x1, mut y1) = (self.width, self.height, 0, 0);
        let mut area = 0;
        for (x, y, pixel) in view.enumerate_pixels() {
            if pixel[0] != 0 {
                area += 1;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
        if area == 0 {
            self.width = 0;
            self.height = 0;
            self.pixels.clear();
        } else if (x0, y0, x1, y1) != (0, 0, self.width, self.height) {
            let (width, height) = (x1 - x0, y1 - y0);
            for y in 0..height {
                let from = ((y0 + y) * self.width + x0) as usize;
                self.pixels
                    .copy_within(from..from + width as usize, (y * width) as usize);
            }
            self.pixels.truncate((width * height) as usize);
            self.x += x0;
            self.y += y0;
            self.width = width;
            self.height = height;
        }
        area
    }

    /// The lettering's typography on `page`, or `None` when its block holds
    /// no ink.
    pub fn typography(&self, page: &RgbImage) -> Option<Typography> {
        let block = block(self, page.width(), page.height());
        let Ink {
            crop,
            glyphs: ink,
            corner: [left, top],
            ..
        } = ink(&block, page, 0, None)?;
        let (width, height) = (block.width, block.height);
        let inside = |x: u32, y: u32| block.contains(x + left, y + top);

        // The block's edges are the sides and ends of its lines, and their
        // commonest direction, modulo 90°, is the lines' tilt. The edges'
        // gradients are averaged at four times their angle, which folds sides
        // and ends onto one direction, then again over only the edges within
        // `TILT_REACH` of that average, until it settles. Glyphs narrower
        // than a column's widest, such as ー, っ, and !!, make its sides
        // ragged, and their leaning edges pulled the plain average of 41
        // upright texts on 8 Manga109 spreads up to 7° off; a turned block
        // turns all of its edges. Leptonica's skew finder, scoring the
        // ink's profiles, flipped direction on 14% of blocks turned by up to
        // 20° and followed only 82% of the turn; the block's hull, by its
        // minimum-area rectangle or narrowest side, tilted wherever columns
        // of uneven length step. Only the block's body counts, the parts at
        // least a tenth the size of its largest: a stray fragment of a
        // neighbour's mask is no line.
        let shape = GrayImage::from_raw(width, height, block.pixels.clone())
            .expect("a mask holds a byte a pixel");
        let parts = connected_components(&shape, Connectivity::Eight, Luma([0]));
        let mut sizes = HashMap::<u32, u32>::new();
        for part in parts.pixels().filter(|part| part.0[0] != 0) {
            *sizes.entry(part.0[0]).or_default() += 1;
        }
        let largest = sizes.values().copied().max().unwrap_or(0);
        let body = |part: u32| part != 0 && 10 * sizes[&part] >= largest;
        // Estimate the lines' tilt modulo 90° by maximizing the L2 energy of
        // the ink's 1D projections across and along candidate axes in [-45°, 45°).
        // Aligned lines concentrate ink into narrow peaks with clear gutters,
        // while isotropic shapes (such as circular dots) have nearly flat energy
        // across angles and stay upright.
        let mut points: Vec<(f32, f32)> = Vec::new();
        for (x, y, part) in parts.enumerate_pixels() {
            if body(part.0[0]) && self.contains(x + left, y + top) {
                points.push((x as f32, y as f32));
            }
        }
        // Subsample deterministically on very high-resolution masks so the angle
        // sweep stays fast.
        if points.len() > 4096 {
            let step = points.len() / 4096;
            points = points.into_iter().step_by(step).collect();
        }
        let max_dim = (width.max(height) as f32).max(1.0);
        // Scale bin size gently on very large crops so projection energy measures
        // line/glyph alignment rather than single-pixel aliasing.
        let bin_size = (max_dim / 256.0).max(1.0);
        let projection_energy = |deg: f32| -> f64 {
            if points.is_empty() {
                return 0.0;
            }
            let (sin, cos) = deg.to_radians().sin_cos();
            let (mut min_u, mut max_u) = (f32::MAX, f32::MIN);
            let (mut min_v, mut max_v) = (f32::MAX, f32::MIN);
            for &(x, y) in &points {
                let u = x * cos + y * sin;
                let v = y * cos - x * sin;
                min_u = min_u.min(u);
                max_u = max_u.max(u);
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
            let len_u = (((max_u - min_u) / bin_size).floor() as usize) + 1;
            let len_v = (((max_v - min_v) / bin_size).floor() as usize) + 1;
            let mut hist_u = vec![0u32; len_u];
            let mut hist_v = vec![0u32; len_v];
            for &(x, y) in &points {
                let u = (((x * cos + y * sin) - min_u) / bin_size).floor() as usize;
                let v = (((y * cos - x * sin) - min_v) / bin_size).floor() as usize;
                hist_u[u] += 1;
                hist_v[v] += 1;
            }
            let eu: f64 = hist_u.iter().map(|&c| f64::from(c) * f64::from(c)).sum();
            let ev: f64 = hist_v.iter().map(|&c| f64::from(c) * f64::from(c)).sum();
            eu + ev
        };
        let zero_energy = projection_energy(0.0);
        let mut best_deg = 0.0f32;
        let mut best_energy = zero_energy;
        let mut sum_energy = 0.0f64;
        for step in -45..45 {
            let deg = step as f32;
            let e = projection_energy(deg);
            sum_energy += e;
            if e > best_energy {
                best_energy = e;
                best_deg = deg;
            }
        }
        // Refine around the coarse peak in 0.25° steps.
        let coarse = best_deg;
        for q in -4..=4 {
            let deg = coarse + q as f32 * 0.25;
            let folded = (deg + 45.0).rem_euclid(90.0) - 45.0;
            let e = projection_energy(folded);
            if e > best_energy {
                best_energy = e;
                best_deg = folded;
            }
        }
        let mean_energy = sum_energy / 90.0;
        let angle = if best_energy <= zero_energy * MIN_TILT_GAIN_OVER_UPRIGHT
            || best_energy <= mean_energy * MIN_TILT_GAIN_OVER_MEAN
        {
            0.0
        } else {
            best_deg
        };

        // The lines run along the body's longer extent at that angle, or
        // down its columns.
        let (sin, cos) = angle.to_radians().sin_cos();
        let (mut along_x, mut along_y) = ((f32::MAX, f32::MIN), (f32::MAX, f32::MIN));
        for (x, y, part) in parts.enumerate_pixels() {
            if body(part.0[0]) {
                let (x, y) = (x as f32, y as f32);
                let (u, v) = (x * cos + y * sin, y * cos - x * sin);
                along_x = (along_x.0.min(u), along_x.1.max(u));
                along_y = (along_y.0.min(v), along_y.1.max(v));
            }
        }
        let (extent_x, extent_y) = (along_x.1 - along_x.0 + 1.0, along_y.1 - along_y.0 + 1.0);
        // The fork also groups horizontal lines into tall paragraphs. Check
        // for separated lines along the shorter axis in either orientation.
        let longer_down = if (extent_x - extent_y).abs() < 1.0 {
            height > width
        } else {
            extent_y > extent_x
        };
        let upright = if angle.abs() < UPRIGHT { 0.0 } else { angle };
        let vertical = if longer_down {
            !separated_lines(&ink, upright, false)
        } else {
            separated_lines(&ink, upright, true)
        };
        let solid =
            ink.as_raw().iter().filter(|&&v| v != 0).count() as f32 > 0.8 * extent_x * extent_y;
        let length = if vertical { extent_y } else { extent_x };
        let strokes = |ink: &GrayImage| {
            let stroke_width = find_stroke_width(ink, 0.15);
            // A solid mark has straight sides even when it is as thick as its stroke.
            let resolution = (stroke_width.unwrap_or(1.0) / length).atan().to_degrees();
            let tilt = if angle.abs() < UPRIGHT || (!solid && angle.abs() < resolution) {
                0.0
            } else {
                angle
            };
            (stroke_width, tilt)
        };
        let outlined = outlined(page, &block, self, &ink, [left, top], |ink| {
            breadth(ink, vertical, strokes(ink).1)
        });
        let ink = outlined.as_ref().map_or(&ink, |outlined| &outlined.glyphs);
        let (stroke_width, tilt) = strokes(ink);
        let size = breadth(ink, vertical, tilt);

        // The fork has no Leptonica dependency. Estimate each tone within
        // the fitted body using its most populated RGB bin, so an isolated
        // screentone dot cannot tint the dominant lettering colour.
        let commonest = |wanted: bool| {
            let mut bins = HashMap::<[u8; 3], (u32, [u64; 3])>::new();
            for (x, y, pixel) in crop.enumerate_pixels() {
                if inside(x, y) && (ink.get_pixel(x, y)[0] != 0) == wanted {
                    let (count, sum) = bins.entry(pixel.0.map(|v| v / 16)).or_default();
                    *count += 1;
                    for (sum, channel) in sum.iter_mut().zip(pixel.0) {
                        *sum += u64::from(channel);
                    }
                }
            }
            let (_, (count, sum)) = bins
                .into_iter()
                .max_by_key(|(bin, (count, _))| (*count, *bin))?;
            Some(sum.map(|v| (v / u64::from(count)) as u8))
        };
        let (ink_color, paper_color) = (commonest(true), commonest(false));

        Some(Typography {
            vertical,
            tilt,
            size,
            stroke_width,
            ink: outlined
                .as_ref()
                .map_or(ink_color, |outlined| Some(outlined.ink)),
            paper: paper_color,
            outline: match &outlined {
                Some(outlined) => Some(outlined.outline),
                None => outline(page, ink, [left, top], size, ink_color),
            },
        })
    }
}

/// How broad `ink`'s lines, not all blank, are across at `tilt`: the median of
/// those at least as broad as the median line. A line is as broad as its
/// broadest glyph, and kana are narrower than the em kanji fill: in a text
/// of two kana columns and one with kanji, the median of the three read it a
/// fifth small, which made its stroke read bold.
fn breadth(ink: &GrayImage, vertical: bool, tilt: f32) -> f32 {
    let lines = lines(ink, vertical, tilt);
    let middle = median(&lines).expect("the profile's peak is a run").breadth;
    let broad: Vec<Line> = (lines.into_iter())
        .filter(|line| line.breadth >= middle)
        .collect();
    median(&broad).expect("the median line is broad").breadth as f32
}

/// A line of ink: where it starts across the lines, in pixels from the
/// ink's first, how broad it is, how far its ink reaches along it, and how
/// much ink it holds.
struct Line {
    start: usize,
    breadth: usize,
    length: f32,
    ink: u32,
}

/// `ink`'s lines when they run down the page or across it, at `tilt`, in
/// order across them: the ink's profile across the lines splits into a run
/// for each.
fn lines(ink: &GrayImage, vertical: bool, tilt: f32) -> Vec<Line> {
    let (sin, cos) = tilt.to_radians().sin_cos();
    // Each pixel of ink, across the lines and along them.
    let points: Vec<(f32, f32)> = ink
        .enumerate_pixels()
        .filter(|(_, _, pixel)| pixel.0[0] != 0)
        .map(|(x, y, _)| {
            let (x, y) = (x as f32, y as f32);
            let (u, v) = (x * cos + y * sin, y * cos - x * sin);
            if vertical { (u, v) } else { (v, u) }
        })
        .collect();
    let low = points
        .iter()
        .map(|(across, _)| *across)
        .fold(f32::MAX, f32::min);
    let mut profile = Vec::<u32>::new();
    for (across, _) in &points {
        let bin = (across - low) as usize;
        if profile.len() <= bin {
            profile.resize(bin + 1, 0);
        }
        profile[bin] += 1;
    }
    // Close lines' anti-aliased edges can meet, so a gap is where the
    // profile falls below an eighth of its peak.
    let gap = profile.iter().copied().max().unwrap_or(0) / 8;
    let mut lines = Vec::new();
    let mut start = None;
    for (bin, &count) in profile.iter().chain([&0]).enumerate() {
        match (count > gap, start) {
            (true, None) => start = Some(bin),
            (false, Some(from)) => {
                lines.push(Line {
                    start: from,
                    breadth: bin - from,
                    length: 0.0,
                    ink: profile[from..bin].iter().sum(),
                });
                start = None;
            }
            _ => {}
        }
    }
    let mut line_of = vec![None; profile.len()];
    for (index, line) in lines.iter().enumerate() {
        line_of[line.start..line.start + line.breadth].fill(Some(index));
    }
    let mut reach = vec![(f32::MAX, f32::MIN); lines.len()];
    for (across, along) in &points {
        if let Some(index) = line_of[(across - low) as usize] {
            reach[index] = (reach[index].0.min(*along), reach[index].1.max(*along));
        }
    }
    for (line, (first, last)) in lines.iter_mut().zip(reach) {
        line.length = last - first + 1.0;
    }
    lines
}

/// The line holding the median of the ink, the lines ordered by breadth,
/// which stands for them all: ruby or a stray mark doesn't.
fn median(lines: &[Line]) -> Option<&Line> {
    let mut ordered: Vec<&Line> = lines.iter().collect();
    ordered.sort_unstable_by_key(|line| (line.breadth, line.ink));
    let total: u32 = lines.iter().map(|line| line.ink).sum();
    let mut seen = 0;
    ordered.into_iter().find(|line| {
        seen += line.ink;
        2 * seen >= total
    })
}

/// Whether `ink` runs along the candidate axis at `tilt`: its lines are at
/// least three times as long as broad, so each holds a few glyphs, and the
/// gaps between them are wider by half than those between its rows across,
/// as the space between lines is wider than between the glyphs of a line.
/// A single column's parts, such as い's two strokes, lie closer than its
/// glyphs. A line no more than half as broad as the median one is gap: ruby,
/// half its base's size, stands between columns, and its sparse glyphs split
/// into slivers a pixel or two broad, whose gaps had read three short
/// columns with ruby as rows.
fn separated_lines(ink: &GrayImage, tilt: f32, vertical: bool) -> bool {
    let broad = |lines: Vec<Line>| -> Vec<Line> {
        let Some(middle) = median(&lines).map(|line| line.breadth) else {
            return lines;
        };
        (lines.into_iter())
            .filter(|line| 2 * line.breadth > middle)
            .collect()
    };
    let (along, across) = (
        broad(lines(ink, vertical, tilt)),
        broad(lines(ink, !vertical, tilt)),
    );
    let Some(line) = median(&along) else {
        return false;
    };
    // The median gap between neighbouring lines, or 0 for one line.
    let between = |lines: &[Line]| {
        let mut gaps: Vec<f32> = lines
            .windows(2)
            .map(|pair| (pair[1].start - pair[0].start - pair[0].breadth) as f32)
            .collect();
        gaps.sort_unstable_by(f32::total_cmp);
        match gaps.len() {
            0 => 0.0,
            count if count % 2 == 0 => (gaps[count / 2 - 1] + gaps[count / 2]) / 2.0,
            count => gaps[count / 2],
        }
    };
    let (line_gap, glyph_gap) = (between(&along), between(&across));
    line.length >= 3.0 * line.breadth as f32 && line_gap > glyph_gap && line_gap >= 1.5 * glyph_gap
}

/// How far from the band's median colour, in every channel, an outline's
/// pixel lies: JPEG rings next to dark ink.
const OUTLINE_TOLERANCE: u8 = 48;

/// The outline around `glyphs`' set pixels, whose corner is `corner` on
/// `page`, of lettering `size` pixels across its lines and coloured `ink`.
/// The rings of pixels one, two, and more pixels from the glyphs are read
/// outside the lettering's body, the glyphs closed over the gaps between
/// them, which an outline fills too: past the antialiased first ring the
/// band is the second ring's colour, and it ends where that colour falls
/// away most sharply, by at least 0.15 of a ring within two rings, while the
/// rings before keep it. Lettering beside a balloon of the band's colour
/// keeps it past the band on that side, so how much stays past doesn't
/// count. On paper of the band's colour it fades over many rings, toward a
/// panel's border or the art, or never leaves, and the lettering has
/// none.
pub(super) fn outline(
    page: &RgbImage,
    glyphs: &GrayImage,
    [left, top]: [u32; 2],
    size: f32,
    ink: Option<[u8; 3]>,
) -> Option<Outline> {
    let far = (0.4 * size).ceil().max(4.0) as u32;
    let radius = (0.3 * size).round().clamp(2.0, 127.0) as u32;
    // Past the farthest ring, and twice the closing radius, so the closing
    // never reaches the canvas's edge, where imageproc's erosion would keep
    // it.
    let margin = (far + 6).max(2 * radius);
    let (x0, y0) = (left.saturating_sub(margin), top.saturating_sub(margin));
    let x1 = (left + glyphs.width() + margin).min(page.width());
    let y1 = (top + glyphs.height() + margin).min(page.height());
    let mut shape = GrayImage::new(x1 - x0, y1 - y0);
    for (x, y, pixel) in glyphs.enumerate_pixels() {
        if pixel.0[0] != 0 {
            shape.put_pixel(x + left - x0, y + top - y0, Luma([255]));
        }
    }
    let mut body = close(&shape, Norm::LInf, radius as u8);
    fill_holes(&mut body);
    let distance = euclidean_squared_distance_transform(&shape);
    let mut rings: Vec<Vec<[u8; 3]>> = vec![Vec::new(); far as usize + 7];
    for (x, y, squared) in distance.enumerate_pixels() {
        let ring = squared.0[0].sqrt().ceil() as usize;
        if body.get_pixel(x, y).0[0] == 0 && (1..rings.len()).contains(&ring) {
            rings[ring].push(page.get_pixel(x0 + x, y0 + y).0);
        }
    }
    let share = |ring: &[[u8; 3]], color: [u8; 3]| {
        let near = ring
            .iter()
            .filter(|pixel| (0..3).all(|at| pixel[at].abs_diff(color[at]) <= OUTLINE_TOLERANCE))
            .count();
        near as f32 / ring.len().max(1) as f32
    };
    let ring_median = |ring: &[[u8; 3]]| -> [u8; 3] {
        std::array::from_fn(|at| {
            let mut channel: Vec<u8> = ring.iter().map(|pixel| pixel[at]).collect();
            channel.sort_unstable();
            channel[channel.len() / 2]
        })
    };
    let try_ring = |start: usize, min_drop: f32, guard_aa: bool| -> Option<Outline> {
        let band = rings.get(start).filter(|ring| ring.len() >= 16)?;
        let color = ring_median(band);
        let unlike_ink = ink.is_none_or(|ink| (0..3).any(|at| ink[at].abs_diff(color[at]) > 80));
        if share(band, color) < 0.8 || !unlike_ink {
            return None;
        }
        let shares: Vec<f32> = rings.iter().map(|ring| share(ring, color)).collect();
        let drop = |ring: usize| shares[ring] - shares[ring + 2];
        let limit = if guard_aa {
            (far as usize).min(2)
        } else {
            far as usize
        };
        if start > limit {
            return None;
        }
        let end =
            (start..=limit).reduce(|end, ring| if drop(ring) > drop(end) { ring } else { end })?;
        let held = shares[start..=end].iter().sum::<f32>() / (end - start + 1) as f32;
        if drop(end) < min_drop || held < 0.8 || rings[end + 2].len() < 16 {
            return None;
        }
        // Reject a thin border, screentone dot, or hatching line on uniform
        // paper: the paper colour dips briefly at `end + 2` and bounces right
        // back to >= 0.85 outside.
        let bounces_back = ((end + 3)..rings.len().min(end + 7)).any(|r| {
            rings[r].len() >= 16 && shares[r] >= 0.85 && shares[r] >= shares[end + 2] + 0.15
        });
        if bounces_back {
            return None;
        }

        if guard_aa {
            // Reject an antialiased transition ring between ink and a uniform background.
            if let Some(ink_c) = ink {
                let outer = ring_median(&rings[end + 2]);
                if share(&rings[end + 2], outer) >= 0.65 {
                    let diff = [
                        f32::from(outer[0]) - f32::from(ink_c[0]),
                        f32::from(outer[1]) - f32::from(ink_c[1]),
                        f32::from(outer[2]) - f32::from(ink_c[2]),
                    ];
                    let norm_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                    if norm_sq > 1.0 {
                        let t = ((f32::from(color[0]) - f32::from(ink_c[0])) * diff[0]
                            + (f32::from(color[1]) - f32::from(ink_c[1])) * diff[1]
                            + (f32::from(color[2]) - f32::from(ink_c[2])) * diff[2])
                            / norm_sq;
                        if (0.12..=0.88).contains(&t) {
                            let blend_err = (0..3)
                                .map(|c| {
                                    (f32::from(color[c]) - (f32::from(ink_c[c]) + t * diff[c]))
                                        .abs()
                                })
                                .fold(0.0f32, f32::max);
                            if blend_err <= 28.0 {
                                return None;
                            }
                        }
                    }
                }
            }
        }
        let width = if end == 1 && shares.get(2).copied().unwrap_or(0.0) >= 0.45 {
            2.0
        } else {
            end as f32
        };
        Some(Outline { color, width })
    };
    try_ring(2, 0.15, false).or_else(|| try_ring(1, 0.15, true))
}

/// Detector evidence available before a proposal is fitted to source ink.
#[derive(Clone, Copy)]
pub enum GlyphProposal {
    /// The original RF-DETR box, before fusion grows its segmented search area.
    Segmented([f32; 4]),
    /// A comic-only text box without a segmentation.
    Box,
}

impl GlyphProposal {
    fn without_ink(self, mask: Mask) -> Mask {
        match self {
            Self::Segmented(_) => mask,
            Self::Box => Mask::default(),
        }
    }
}

/// Fit the fused lettering proposals to the page. Keep their original boxes
/// until every shared pixel has an owner: growing one box over a cut glyph
/// must not give it priority over a neighbouring text.
pub fn fit_glyph_masks<'a>(
    proposals: impl IntoIterator<Item = (GlyphProposal, &'a mut KoharuLayoutDetection)>,
    page: &RgbImage,
) {
    let mut letters: Vec<_> = proposals
        .into_iter()
        .filter(|(_, detection)| matches!(detection.label.as_str(), "text" | "onomatopoeia"))
        .map(|(proposal, detection)| {
            let original = match proposal {
                GlyphProposal::Segmented(original) => original,
                GlyphProposal::Box => detection.bbox,
            };
            (original, proposal, Mask::default(), detection)
        })
        .collect();
    letters
        .par_iter_mut()
        .for_each(|(_, proposal, core, detection)| {
            let input_mask = Arc::clone(&detection.mask);
            let (mask, core_mask, grown) = glyph_mask(&input_mask, detection.bbox, page, *proposal);
            // For segmented proposals, anchor ownership tie-breaking to core
            // ink supported by the detector's own mask so plain-paper box
            // expansion cannot pull a tilted neighbour's glyphs into our core.
            let anchored = if matches!(proposal, GlyphProposal::Segmented(_))
                && core_mask.width > 0
                && core_mask.height > 0
            {
                let mut anchored = core_mask.clone();
                let mut any = false;
                for y in 0..anchored.height {
                    for x in 0..anchored.width {
                        let idx = (y * anchored.width + x) as usize;
                        if anchored.pixels[idx] != 0 {
                            if input_mask.contains(anchored.x + x, anchored.y + y) {
                                any = true;
                            } else {
                                anchored.pixels[idx] = 0;
                            }
                        }
                    }
                }
                if any { anchored } else { core_mask }
            } else {
                core_mask
            };
            *core = anchored;
            detection.mask = Arc::new(mask);
            detection.bbox = grown;
        });
    let area = |[left, top, right, bottom]: [f32; 4]| (right - left) * (bottom - top);
    letters.sort_by(|(a, ..), (b, ..)| area(*a).total_cmp(&area(*b)));
    // Resolve shared pixels by Euclidean distance to each region's uncontested
    // pre-dilation core ink before falling back to original-box distance and
    // smaller-box tie-breaking.
    for i in 0..letters.len() {
        let (earlier, later) = letters.split_at_mut(i + 1);
        let (bbox, _, core_a, first) = &mut earlier[i];
        for (other, _, core_b, second) in later {
            let (a, b) = (&first.mask, &second.mask);
            let (x0, y0) = (a.x.max(b.x), a.y.max(b.y));
            let (x1, y1) = (
                (a.x + a.width).min(b.x + b.width),
                (a.y + a.height).min(b.y + b.height),
            );
            if x0 >= x1 || y0 >= y1 {
                continue;
            }
            let has_overlap = (y0..y1).any(|y| {
                (x0..x1).any(|x| {
                    let ai = ((y - a.y) * a.width + x - a.x) as usize;
                    let bi = ((y - b.y) * b.width + x - b.x) as usize;
                    a.pixels[ai] != 0 && b.pixels[bi] != 0
                })
            });
            if !has_overlap {
                continue;
            }
            let ux0 = x0.min(core_a.x).min(core_b.x);
            let uy0 = y0.min(core_a.y).min(core_b.y);
            let ux1 = x1.max(core_a.x + core_a.width).max(core_b.x + core_b.width);
            let uy1 = y1
                .max(core_a.y + core_a.height)
                .max(core_b.y + core_b.height);
            let (uw, uh) = (ux1 - ux0, uy1 - uy0);
            let mut only_a = GrayImage::new(uw, uh);
            let mut only_b = GrayImage::new(uw, uh);
            let (mut any_a, mut any_b) = (false, false);
            for (index, &value) in core_a.pixels.iter().enumerate() {
                if value != 0 {
                    let px = core_a.x + index as u32 % core_a.width;
                    let py = core_a.y + index as u32 / core_a.width;
                    if !core_b.contains(px, py) {
                        only_a.put_pixel(px - ux0, py - uy0, Luma([255]));
                        any_a = true;
                    }
                }
            }
            for (index, &value) in core_b.pixels.iter().enumerate() {
                if value != 0 {
                    let px = core_b.x + index as u32 % core_b.width;
                    let py = core_b.y + index as u32 / core_b.width;
                    if !core_a.contains(px, py) {
                        only_b.put_pixel(px - ux0, py - uy0, Luma([255]));
                        any_b = true;
                    }
                }
            }
            let core_distances = (any_a && any_b).then(|| {
                (
                    euclidean_squared_distance_transform(&only_a),
                    euclidean_squared_distance_transform(&only_b),
                )
            });
            let a = Arc::make_mut(&mut first.mask);
            let b = Arc::make_mut(&mut second.mask);
            for y in y0..y1 {
                for x in x0..x1 {
                    let ai = ((y - a.y) * a.width + x - a.x) as usize;
                    let bi = ((y - b.y) * b.width + x - b.x) as usize;
                    if a.pixels[ai] != 0 && b.pixels[bi] != 0 {
                        let second_wins = match &core_distances {
                            Some((dist_a, dist_b)) => {
                                let da = dist_a.get_pixel(x - ux0, y - uy0).0[0];
                                let db = dist_b.get_pixel(x - ux0, y - uy0).0[0];
                                if db != da {
                                    db < da
                                } else {
                                    box_distance(*other, x, y) < box_distance(*bbox, x, y)
                                }
                            }
                            None => box_distance(*other, x, y) < box_distance(*bbox, x, y),
                        };
                        if second_wins {
                            a.pixels[ai] = 0;
                        } else {
                            b.pixels[bi] = 0;
                        }
                    }
                }
            }
        }
        first.area = Arc::make_mut(&mut first.mask).trim_to_owned_pixels();
    }
}

fn box_distance([left, top, right, bottom]: [f32; 4], x: u32, y: u32) -> f32 {
    let (x, y) = (x as f32 + 0.5, y as f32 + 0.5);
    let across = (left - x).max(x - right).max(0.0);
    let down = (top - y).max(y - bottom).max(0.0);
    across * across + down * down
}

/// A text or onomatopoeia mask fitted to its glyphs: RF-DETR's mask within
/// its box, which hugs each stroke and its outline but misses counters and
/// parts of strokes, with its block's ink, widened by 2 pixels per 1024 of the
/// page's longer side for anti-aliased edges. The paper or art between glyphs
/// stays out, so inpainting redraws only the lettering. And the box, grown
/// over the glyphs it cuts that the block holds, so that what's cut to the
/// box after keeps them whole.
fn glyph_mask(
    mask: &Mask,
    bbox: [f32; 4],
    page: &RgbImage,
    proposal: GlyphProposal,
) -> (Mask, Mask, [f32; 4]) {
    let (page_width, page_height) = page.dimensions();
    // imageproc 0.27.0 labels the first foreground pixel 1 even when its
    // union-find table has just one slot (image-rs/imageproc#824).
    if mask.pixels.len() <= 1 || page_width == 0 || page_height == 0 {
        let fallback = proposal.without_ink(mask.clone());
        return (fallback.clone(), fallback, bbox);
    }
    // RF-DETR's mask can stop a few pixels short of a glyph's ends, and LaMa
    // grows even a sliver of ink left outside into a stroke, so the ink may
    // reach as far past the block as the block's closing radius.
    let reach = scaled(6.0, page_width, page_height).min(MAX_BLOCK_CLOSING_RADIUS);
    // RF-DETR predicts each query's mask over the whole page, and a lettering
    // mask can take in specks of other lettering far from its box, whose
    // surroundings then sway the ink split. Its own glyphs can run past the
    // box's ends too, but by less than a glyph, the box's narrower side, and
    // past its sides by no more than the closing radius: a glyph's width
    // further across is the next line, often another lettering's, whose
    // clean-up would cover that lettering's translation.
    let [left, top, right, bottom] = bbox;
    let (width, height) = (right - left, bottom - top);
    let (across, down) = match height >= width {
        true => (reach as f32, width),
        false => (height, reach as f32),
    };
    let clipped = GrayImage::from_fn(mask.width, mask.height, |x, y| {
        let (page_x, page_y) = ((mask.x + x) as f32, (mask.y + y) as f32);
        let kept = mask.pixels[(y * mask.width + x) as usize] != 0
            && (left - across..right + across).contains(&page_x)
            && (top - down..bottom + down).contains(&page_y);
        Luma([if kept { 255 } else { 0 }])
    });
    // A part of the mask that never comes within the closing radius of the
    // box is a speck of other lettering too, such as a balloon's glyph above
    // a tall sound effect, whose clean-up would cover that balloon's
    // translation; a glyph of its own that the box cuts off reaches into it.
    let parts = connected_components(&clipped, Connectivity::Eight, Luma([0]));
    let labels = parts.pixels().map(|part| part.0[0]).max().unwrap_or(0) as usize;
    // Part 0 is the background, never near.
    let mut near = vec![false; labels + 1];
    let mut part_area = vec![0u32; labels + 1];
    let mut part_extents: Vec<Option<[u32; 4]>> = vec![None; labels + 1];
    let radius = reach as f32;
    for (x, y, part) in parts.enumerate_pixels() {
        let l = part.0[0] as usize;
        if l == 0 {
            continue;
        }
        part_area[l] += 1;
        let e = part_extents[l].get_or_insert([x, y, x, y]);
        *e = [e[0].min(x), e[1].min(y), e[2].max(x), e[3].max(y)];
        let (page_x, page_y) = ((mask.x + x) as f32, (mask.y + y) as f32);
        if (left - radius..right + radius).contains(&page_x)
            && (top - radius..bottom + radius).contains(&page_y)
        {
            near[l] = true;
        }
    }
    let largest_near = (1..=labels)
        .filter(|&l| near[l])
        .map(|l| part_area[l])
        .max()
        .unwrap_or(0);
    if largest_near >= 64 {
        let speck_side = scaled(5.0, page_width, page_height).max(4);
        let speck_candidates: Vec<usize> = (1..=labels)
            .filter(|&l| {
                near[l]
                    && part_extents[l].is_some_and(|[x0, y0, x1, y1]| {
                        let side = (x1 - x0 + 1).max(y1 - y0 + 1);
                        side <= speck_side && part_area[l] * 25 < largest_near
                    })
            })
            .collect();
        // Drop stray specks outside `bbox` or when many tiny specks appear
        // (screentone fields), while preserving isolated small kana/punctuation
        // inside `bbox` on small pages.
        let many_specks = speck_candidates.len() >= 4;
        for l in speck_candidates {
            if let Some([x0, y0, x1, y1]) = part_extents[l] {
                let cx = (mask.x + (x0 + x1) / 2) as f32;
                let cy = (mask.y + (y0 + y1) / 2) as f32;
                let inside_box = (left..right).contains(&cx) && (top..bottom).contains(&cy);
                if !inside_box || many_specks || part_area[l] <= 12 {
                    near[l] = false;
                }
            }
        }
    }
    let clipped = GrayImage::from_fn(mask.width, mask.height, |x, y| {
        Luma([if near[parts.get_pixel(x, y).0[0] as usize] {
            255
        } else {
            0
        }])
    });
    let mask = crop(
        &clipped,
        i64::from(mask.x),
        i64::from(mask.y),
        page_width,
        page_height,
    );
    let block = block(&mask, page_width, page_height);
    let Some(Ink {
        glyphs: ink,
        corner: [left, top],
        held,
        paper,
        borders,
        ..
    }) = ink(&block, page, reach, Some(bbox))
    else {
        let fallback = proposal.without_ink(mask);
        return (fallback.clone(), fallback, bbox);
    };
    let bbox = held.map_or(bbox, |held| {
        let [x0, y0, x1, y1] = held.map(|edge| edge as f32);
        [
            bbox[0].min(x0),
            bbox[1].min(y0),
            bbox[2].max(x1),
            bbox[3].max(y1),
        ]
    });
    let margin = scaled(2.0, page_width, page_height);
    let (left, top) = (
        i64::from(left) - i64::from(margin),
        i64::from(top) - i64::from(margin),
    );
    let mut canvas = GrayImage::new(ink.width() + 2 * margin, ink.height() + 2 * margin);
    for (x, y, pixel) in ink.enumerate_pixels() {
        if pixel.0[0] != 0 {
            canvas.put_pixel(x + margin, y + margin, Luma([255]));
        }
    }
    // A solid proposal (including an ogkalu text box) is a search window,
    // with no stroke evidence of its own. Keep only its source ink. For a
    // segmented proposal retain the detector's stroke and outline evidence.
    if matches!(proposal, GlyphProposal::Segmented(_)) && mask.pixels.contains(&0) {
        // When `plain(boxed)` returned `None` (for example a translucent bubble
        // over textured or gradient art), check whether the ink is outlined.
        // If not, build a background palette from non-ink pixels outside `mask`
        // and inside `bbox` away from ink strokes so coarse segmented blobs do
        // not copy translucent background into the mask.
        let bg_palette: Option<Vec<[u8; 3]>> = if paper.is_none() {
            let approx_size = (bbox[2] - bbox[0]).min(bbox[3] - bbox[1]).max(12.0);
            let ink_corner = [
                (left + i64::from(margin)) as u32,
                (top + i64::from(margin)) as u32,
            ];
            let block_ink = GrayImage::from_fn(block.width, block.height, |x, y| {
                let ix = block.x + x;
                let iy = block.y + y;
                let set = ix >= ink_corner[0]
                    && iy >= ink_corner[1]
                    && ix - ink_corner[0] < ink.width()
                    && iy - ink_corner[1] < ink.height()
                    && ink.get_pixel(ix - ink_corner[0], iy - ink_corner[1]).0[0] != 0;
                Luma([u8::from(set) * 255])
            });
            let has_outline = outline(page, &ink, ink_corner, approx_size, None).is_some()
                || outlined(page, &block, &mask, &block_ink, [block.x, block.y], |_| {
                    approx_size
                })
                .is_some();
            if has_outline {
                None
            } else {
                Some(sample_background_palette(
                    page, &mask, &ink, ink_corner, bbox, reach,
                ))
            }
        } else {
            None
        };
        for (index, _) in mask
            .pixels
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
        {
            let px = mask.x + index as u32 % mask.width;
            let py = mask.y + index as u32 / mask.width;
            let (ix, iy) = (
                i64::from(px) - (left + i64::from(margin)),
                i64::from(py) - (top + i64::from(margin)),
            );
            if let Some(ref borders) = borders
                && (0..i64::from(borders.width())).contains(&ix)
                && (0..i64::from(borders.height())).contains(&iy)
                && borders.get_pixel(ix as u32, iy as u32).0[0] != 0
            {
                continue;
            }
            let pixel = page[(px, py)].0;
            // A coarse RF-DETR blob can bridge open paper between glyphs.
            // Its pixels carry no extra stroke evidence on uniform paper or
            // translucent bubble backgrounds.
            if paper
                .is_some_and(|paper| (0..3).all(|c| pixel[c].abs_diff(paper[c]) <= PAPER_TOLERANCE))
            {
                continue;
            }
            if let Some(ref palette) = bg_palette
                && palette
                    .iter()
                    .any(|bg| (0..3).all(|c| pixel[c].abs_diff(bg[c]) <= PAPER_TOLERANCE))
            {
                continue;
            }
            let (x, y) = (i64::from(px) - left, i64::from(py) - top);
            if (0..i64::from(canvas.width())).contains(&x)
                && (0..i64::from(canvas.height())).contains(&y)
            {
                canvas.put_pixel(x as u32, y as u32, Luma([255]));
            }
        }
    }
    // Inside the box, what the glyphs enclose is a counter or, in lettering
    // outlined in a lighter colour, the glyph's own fill; the ink split takes
    // only the outline there.
    let [box_left, box_top, box_right, box_bottom] = bbox;
    let mut enclosed = canvas.clone();
    fill_holes(&mut enclosed);
    let holes = GrayImage::from_fn(canvas.width(), canvas.height(), |x, y| {
        Luma([
            u8::from(enclosed.get_pixel(x, y).0[0] != 0 && canvas.get_pixel(x, y).0[0] == 0) * 255,
        ])
    });
    let hole_parts = connected_components(&holes, Connectivity::Four, Luma([0]));
    let hole_labels = hole_parts.pixels().map(|p| p.0[0]).max().unwrap_or(0) as usize;
    let mut hole_extents: Vec<Option<[u32; 4]>> = vec![None; hole_labels + 1];
    for (x, y, p) in hole_parts.enumerate_pixels() {
        let l = p.0[0] as usize;
        if l == 0 {
            continue;
        }
        let e = hole_extents[l].get_or_insert([x, y, x, y]);
        *e = [e[0].min(x), e[1].min(y), e[2].max(x), e[3].max(y)];
    }
    let box_area = ((box_right - box_left).max(1.0) * (box_bottom - box_top).max(1.0)) as u32;
    let keep_hole: Vec<bool> = hole_extents
        .into_iter()
        .map(|ext| {
            ext.is_some_and(|[x0, y0, x1, y1]| {
                let (hw, hh) = (x1 - x0 + 1, y1 - y0 + 1);
                // A glyph counter or outlined glyph fill is never nearly as
                // large as an entire frame-enclosed text box.
                hw * hh * 4 < 3 * box_area
            })
        })
        .collect();
    for (x, y, p) in hole_parts.enumerate_pixels() {
        let l = p.0[0] as usize;
        if l != 0 && keep_hole[l] {
            let (page_x, page_y) = ((left + i64::from(x)) as f32, (top + i64::from(y)) as f32);
            if (box_left..box_right).contains(&page_x) && (box_top..box_bottom).contains(&page_y) {
                canvas.put_pixel(x, y, Luma([255]));
            }
        }
    }
    let core = crop(&canvas, left, top, page_width, page_height);
    let fitted = dilate(&canvas, Norm::L2, margin as u8);
    (
        crop(&fitted, left, top, page_width, page_height),
        core,
        bbox,
    )
}

/// Fills what `canvas`'s marked pixels enclose: unmarked areas its edge
/// can't reach.
pub(super) fn fill_holes(canvas: &mut GrayImage) {
    let (width, height) = (canvas.width() as usize, canvas.height() as usize);
    if width == 0 || height == 0 {
        return;
    }
    // The background is every pixel that isn't 255, its parts joined 4 ways
    // where their values match; a part reaching the edge is outside, and the
    // rest are holes. A flood from the edge finds the parts outside.
    let pixels: &mut [u8] = canvas.as_mut();
    let mut outside = vec![false; width * height];
    let mut queue = Vec::new();
    let edge = (0..width)
        .flat_map(|x| [x, (height - 1) * width + x])
        .chain((0..height).flat_map(|y| [y * width, y * width + width - 1]));
    for at in edge {
        if pixels[at] != u8::MAX && !outside[at] {
            outside[at] = true;
            queue.push(at);
        }
    }
    while let Some(at) = queue.pop() {
        let (x, y) = (at % width, at / width);
        let neighbours = [
            (x > 0).then(|| at - 1),
            (x + 1 < width).then(|| at + 1),
            (y > 0).then(|| at - width),
            (y + 1 < height).then(|| at + width),
        ];
        for next in neighbours.into_iter().flatten() {
            if !outside[next] && pixels[next] == pixels[at] {
                outside[next] = true;
                queue.push(next);
            }
        }
    }
    for (pixel, outside) in pixels.iter_mut().zip(outside) {
        if *pixel != u8::MAX && !outside {
            *pixel = u8::MAX;
        }
    }
}

/// The lettering's block: `mask` closed and with its holes filled, so its
/// lines are solid and its outline is their sides and ends. The radius is
/// the pipeline's text-mask rule, 6 pixels per 1024 of the page's longer
/// side.
fn block(mask: &Mask, page_width: u32, page_height: u32) -> Mask {
    if mask.width == 0 || mask.height == 0 {
        return mask.clone();
    }
    let radius = scaled(6.0, page_width, page_height).min(MAX_BLOCK_CLOSING_RADIUS);
    // Twice the radius, so the dilation never reaches the canvas's edge,
    // where imageproc's erosion would keep it.
    let pad = 2 * radius;
    let mut canvas = GrayImage::new(mask.width + 2 * pad, mask.height + 2 * pad);
    for (index, _) in mask
        .pixels
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0)
    {
        let (x, y) = (index as u32 % mask.width, index as u32 / mask.width);
        canvas.put_pixel(x + pad, y + pad, Luma([255]));
    }
    let mut closed = close(&canvas, Norm::L2, radius as u8);
    fill_holes(&mut closed);
    let (left, top) = (
        i64::from(mask.x) - i64::from(pad),
        i64::from(mask.y) - i64::from(pad),
    );
    crop(&closed, left, top, page_width, page_height)
}

/// A crop of the page, its ink, its place on the page, and the extent of the
/// glyphs the block holds inside the box, as `ink` finds them.
struct Ink {
    crop: RgbImage,
    glyphs: GrayImage,
    corner: [u32; 2],
    held: Option<[u32; 4]>,
    paper: Option<[u8; 3]>,
    borders: Option<GrayImage>,
}

/// The page's crop around the block and the detection's box, `reach` pixels
/// wider than the block and than the box widened by its line on every side,
/// its ink, the crop's place on the page, and the extent of the glyphs the
/// block holds inside the box. The ink is the side of one Otsu split of the
/// crop that isn't the paper: inside the block, in the glyph ends it cut off,
/// and on plain paper in the glyph-sized parts wholly inside the box, held by
/// the block inside it, or in line with it. The paper is what the crop shows
/// around the block, or with none around, the side covering more of the
/// block, so light lettering on dark counts too. `None` when the block holds
/// no ink.
fn ink(block: &Mask, page: &RgbImage, reach: u32, bbox: Option<[f32; 4]>) -> Option<Ink> {
    if block.pixels.len() <= 1 {
        return None;
    }
    let boxed = bbox.map(|[left, top, right, bottom]| {
        [
            left.floor().max(0.0) as u32,
            top.floor().max(0.0) as u32,
            (right.ceil().max(0.0) as u32).min(page.width()),
            (bottom.ceil().max(0.0) as u32).min(page.height()),
        ]
    });
    // A glyph is no longer than its line is wide, the box's narrower side.
    let line = boxed.map_or(0, |[box_left, box_top, box_right, box_bottom]| {
        box_right
            .saturating_sub(box_left)
            .min(box_bottom.saturating_sub(box_top))
    });
    // The box, lengthened at both ends of its line by the line's width: a
    // glyph the box cuts off lies in line with the others.
    let lined = boxed.map(|[box_left, box_top, box_right, box_bottom]| {
        if box_bottom.saturating_sub(box_top) >= box_right.saturating_sub(box_left) {
            [
                box_left,
                box_top.saturating_sub(line),
                box_right,
                (box_bottom + line).min(page.height()),
            ]
        } else {
            [
                box_left.saturating_sub(line),
                box_top,
                (box_right + line).min(page.width()),
                box_bottom,
            ]
        }
    });
    // Include the immediate paper around a tightly cropped solid glyph,
    // even for typography, where no stroke ends are to be attached.
    let surround = reach.max(1);
    let (mut left, mut top) = (
        block.x.saturating_sub(surround),
        block.y.saturating_sub(surround),
    );
    let (mut right, mut bottom) = (
        (block.x + block.width + surround).min(page.width()),
        (block.y + block.height + surround).min(page.height()),
    );
    // `reach` past the box widened by its line on every side too, so ink
    // crossing the lengthened box's edge shows outside it, and a glyph the
    // box cuts shows whole: one in a line the box ends inside reaches past
    // its side.
    if let Some([box_left, box_top, box_right, box_bottom]) = boxed {
        let around = line + reach;
        (left, top) = (
            left.min(box_left.saturating_sub(around)),
            top.min(box_top.saturating_sub(around)),
        );
        (right, bottom) = (
            right.max((box_right + around).min(page.width())),
            bottom.max((box_bottom + around).min(page.height())),
        );
    }
    let (width, height) = (right - left, bottom - top);
    let crop = imageops::crop_imm(page, left, top, width, height).to_image();
    let (offset_x, offset_y) = (block.x - left, block.y - top);
    let inside = |x: u32, y: u32| {
        (offset_x..offset_x + block.width).contains(&x)
            && (offset_y..offset_y + block.height).contains(&y)
            && block.pixels[((y - offset_y) * block.width + x - offset_x) as usize] != 0
    };
    // One Otsu split of the block and the detection's box, without the
    // reach around them, where art dark enough to take the split from faint
    // lettering can lie. One tile covers the region, so the split is a
    // threshold, read back from it to divide the whole crop.
    let gray = imageops::grayscale(&crop);
    let [region_left, region_top, region_right, region_bottom] = boxed.map_or(
        [left, top, right, bottom],
        |[box_left, box_top, box_right, box_bottom]| {
            [
                box_left.min(block.x).max(left),
                box_top.min(block.y).max(top),
                box_right.max(block.x + block.width).min(right),
                box_bottom.max(block.y + block.height).min(bottom),
            ]
        },
    );
    let (region_width, region_height) = (region_right - region_left, region_bottom - region_top);
    let region = imageops::crop_imm(
        &gray,
        region_left - left,
        region_top - top,
        region_width,
        region_height,
    )
    .to_image();
    let sample = if region
        .as_raw()
        .first()
        .is_some_and(|v| region.as_raw().iter().all(|other| other == v))
    {
        &gray
    } else {
        &region
    };
    let mut histogram = [0.0; 256];
    for pixel in sample.pixels() {
        histogram[usize::from(pixel[0])] += 1.0;
    }
    let level = split_distribution(&histogram, 0.1);
    let mut threshold = sample
        .as_raw()
        .iter()
        .copied()
        .filter(|value| *value < level)
        .max()
        .map_or(0, |value| u16::from(value) + 1);
    let mut dark = GrayImage::from_fn(width, height, |x, y| {
        Luma([if u16::from(gray.get_pixel(x, y).0[0]) < threshold {
            255
        } else {
            0
        }])
    });
    let (mut around, mut dark_around, mut covered, mut dark_covered) = (0u32, 0u32, 0u32, 0u32);
    let ring_x0 = offset_x.saturating_sub(surround);
    let ring_y0 = offset_y.saturating_sub(surround);
    let ring_x1 = (offset_x + block.width + surround).min(width);
    let ring_y1 = (offset_y + block.height + surround).min(height);
    for (x, y, pixel) in dark.enumerate_pixels() {
        let is_dark = u32::from(pixel.0[0] != 0);
        if inside(x, y) {
            covered += 1;
            dark_covered += is_dark;
        } else if (ring_x0..ring_x1).contains(&x) && (ring_y0..ring_y1).contains(&y) {
            around += 1;
            dark_around += is_dark;
        }
    }
    // A solid rectangular box proposal holds both ink (minority) and background
    // (majority); its immediate exterior can be a frame border or dark art
    // outside a speech bubble.
    let solid_box = bbox.is_some()
        && !block.pixels.contains(&0)
        && dark_covered > 0
        && dark_covered < covered
        && (4 * dark_covered < covered || 4 * (covered - dark_covered) < covered);
    let mut light_ink = if solid_box {
        2 * dark_covered > covered
    } else if around > 0 {
        2 * dark_around > around
    } else {
        2 * dark_covered > covered
    };
    // On a translucent bubble over textured art, global Otsu often merges the
    // darkened bubble-tinted art with dark text ink into one bimodal class,
    // which can even flip `dark_around > around / 2`. Check both the initial
    // ink side and (when `light_ink` was flipped by mid-tone dark art) the
    // dark side; if refined, recompute `dark` and polarity.
    let refined = if !light_ink {
        refine_bimodal_ink_threshold(sample, threshold, false)
    } else {
        None
    }
    .or_else(|| {
        if light_ink && threshold > 110 {
            // Only try dark-side refinement when the light class also reaches
            // the region border (a light translucent bubble background, not
            // light lettering enclosed in a dark outline over grey art).
            let (sw, sh) = sample.dimensions();
            let (mut b_total, mut b_light) = (0u32, 0u32);
            for (x, y, p) in sample.enumerate_pixels() {
                if x < 2 || y < 2 || x + 2 >= sw || y + 2 >= sh {
                    b_total += 1;
                    if u16::from(p.0[0]) >= threshold {
                        b_light += 1;
                    }
                }
            }
            if b_total > 0 && (b_light as f32) >= 0.15 * b_total as f32 {
                refine_bimodal_ink_threshold(sample, threshold, false)
            } else {
                None
            }
        } else {
            None
        }
    });
    if let Some(refined) = refined {
        threshold = refined;
        dark = GrayImage::from_fn(width, height, |x, y| {
            Luma([if u16::from(gray.get_pixel(x, y).0[0]) < threshold {
                255
            } else {
                0
            }])
        });
        let (mut r_around, mut r_dark_around, mut r_covered, mut r_dark_covered) =
            (0u32, 0u32, 0u32, 0u32);
        for (x, y, pixel) in dark.enumerate_pixels() {
            let is_dark = u32::from(pixel.0[0] != 0);
            if inside(x, y) {
                r_covered += 1;
                r_dark_covered += is_dark;
            } else if (ring_x0..ring_x1).contains(&x) && (ring_y0..ring_y1).contains(&y) {
                r_around += 1;
                r_dark_around += is_dark;
            }
        }
        let r_solid_box = bbox.is_some()
            && !block.pixels.contains(&0)
            && r_dark_covered > 0
            && r_dark_covered < r_covered
            && (4 * r_dark_covered < r_covered || 4 * (r_covered - r_dark_covered) < r_covered);
        light_ink = if r_solid_box {
            2 * r_dark_covered > r_covered
        } else if r_around > 0 {
            2 * r_dark_around > r_around
        } else {
            2 * r_dark_covered > r_covered
        };
    }
    let is_ink = |x: u32, y: u32| (dark.get_pixel(x, y).0[0] != 0) != light_ink;

    if bbox.is_none() {
        let glyphs = GrayImage::from_fn(block.width, block.height, |x, y| {
            Luma([u8::from(
                block.pixels[(y * block.width + x) as usize] != 0
                    && is_ink(x + offset_x, y + offset_y),
            ) * 255])
        });
        return glyphs.as_raw().contains(&255).then(|| Ink {
            crop: imageops::crop_imm(&crop, offset_x, offset_y, block.width, block.height)
                .to_image(),
            glyphs,
            corner: [block.x, block.y],
            held: None,
            paper: None,
            borders: None,
        });
    }

    // Past the block, ink that joins the block's and ends within `reach` is a
    // glyph's end the mask cut off; ink that runs on is an outline, a border,
    // or art.
    let shape = GrayImage::from_fn(width, height, |x, y| {
        Luma([if inside(x, y) { 255 } else { 0 }])
    });
    let near = dilate(&shape, Norm::LInf, reach as u8);
    let ring = GrayImage::from_fn(width, height, |x, y| {
        Luma([
            if near.get_pixel(x, y).0[0] != 0 && !inside(x, y) && is_ink(x, y) {
                255
            } else {
                0
            },
        ])
    });
    let parts = connected_components(&ring, Connectivity::Eight, Luma([0]));
    // A flag for each part, by label: connected_components numbers them from
    // 1 up.
    let labels = |parts: &ImageBuffer<Luma<u32>, Vec<u32>>| {
        parts.pixels().map(|part| part.0[0]).max().unwrap_or(0) as usize + 1
    };
    let (mut joins, mut runs_on) = (vec![false; labels(&parts)], vec![false; labels(&parts)]);
    for (x, y, part) in parts.enumerate_pixels() {
        let part = part.0[0] as usize;
        if part == 0 {
            continue;
        }
        for (dx, dy) in (-1i64..=1).flat_map(|dx| (-1i64..=1).map(move |dy| (dx, dy))) {
            let (nx, ny) = (i64::from(x) + dx, i64::from(y) + dy);
            if !(0..i64::from(width)).contains(&nx) || !(0..i64::from(height)).contains(&ny) {
                runs_on[part] = true;
                continue;
            }
            let (nx, ny) = (nx as u32, ny as u32);
            if near.get_pixel(nx, ny).0[0] == 0 {
                runs_on[part] = true;
            } else if inside(nx, ny) && is_ink(nx, ny) {
                joins[part] = true;
            }
        }
    }
    // On plain paper every glyph-sized ink part wholly inside the box is
    // lettering: RF-DETR's mask can miss a thin glyph, such as 一, whole. So
    // is one the block holds inside the box, whole past it: RF-DETR's box can
    // end inside a line, and its mask with it, and the glyph's rest is one
    // line further at most, not the next. So is one in line with the box,
    // past its ends, when the paper there is plain too and the part stands
    // on the box's own paper: the box can miss it as well, but not past a
    // balloon's or panel's border, which parts the paper. The paper is plain
    // when nearly all of it away from the ink's edges is near its median
    // colour; on art no part counts.
    let all_ink = GrayImage::from_fn(width, height, |x, y| {
        Luma([if is_ink(x, y) { 255 } else { 0 }])
    });
    let pieces = connected_components(&all_ink, Connectivity::Eight, Luma([0]));
    let mut lettering = vec![false; labels(&pieces)];
    let mut drop_dot = vec![false; labels(&pieces)];
    // The extent of the glyphs the block holds inside the box, in the crop.
    let mut held_extent: Option<[u32; 4]> = None;
    let mut paper_color = None;
    let mut is_plain_bg = false;
    if let (Some(boxed), Some(lined)) = (boxed, lined) {
        let within = |[box_left, box_top, box_right, box_bottom]: [u32; 4], x: u32, y: u32| {
            (box_left..box_right).contains(&(x + left))
                && (box_top..box_bottom).contains(&(y + top))
        };
        let edges = dilate(&all_ink, Norm::LInf, 2);
        let plain = |region: [u32; 4]| {
            let paper = |x: u32, y: u32| within(region, x, y) && edges.get_pixel(x, y).0[0] == 0;
            let mut histograms = [[0usize; 256]; 3];
            let mut count = 0;
            for (x, y, pixel) in crop.enumerate_pixels() {
                if paper(x, y) {
                    count += 1;
                    for (histogram, value) in histograms.iter_mut().zip(pixel.0) {
                        histogram[usize::from(value)] += 1;
                    }
                }
            }
            if count < 16 {
                return None;
            }
            {
                // Each channel's value at the middle of its sorted values.
                let middle = count / 2;
                let medians = histograms.map(|histogram| {
                    let mut seen = 0;
                    (0..=u8::MAX)
                        .find(|&value| {
                            seen += histogram[usize::from(value)];
                            seen > middle
                        })
                        .expect("the middle value is counted")
                });
                let close = crop
                    .enumerate_pixels()
                    .filter(|&(x, y, pixel)| {
                        paper(x, y)
                            && (0..3).all(|channel| {
                                pixel.0[channel].abs_diff(medians[channel]) <= PAPER_TOLERANCE
                            })
                    })
                    .count();
                (close as f32 >= PLAIN_PAPER * count as f32).then_some(medians)
            }
        };
        // Screentone dot fields leave only uniform white gaps after dilating
        // `all_ink`, which would otherwise fool `plain()` into treating every
        // dot inside a loose box as lettering. Detect screentone by counting
        // tiny isolated ink specks when genuine glyph-sized components exist.
        let dot_side = scaled(5.0, page.width(), page.height()).max(5);
        let piece_count = labels(&pieces);
        let mut piece_area = vec![0u32; piece_count];
        let mut piece_extents: Vec<Option<[u32; 4]>> = vec![None; piece_count];
        let mut piece_in_box = vec![false; piece_count];
        for (x, y, p) in pieces.enumerate_pixels() {
            let l = p.0[0] as usize;
            if l == 0 {
                continue;
            }
            piece_area[l] += 1;
            let e = piece_extents[l].get_or_insert([x, y, x, y]);
            *e = [e[0].min(x), e[1].min(y), e[2].max(x), e[3].max(y)];
            if within(boxed, x, y) {
                piece_in_box[l] = true;
            }
        }
        let mut tiny_dots = 0u32;
        let mut has_large_glyph = false;
        let mut is_screentone_dot = vec![false; piece_count];
        for l in 1..piece_count {
            if let Some([x0, y0, x1, y1]) = piece_extents[l] {
                let side = (x1 - x0 + 1).max(y1 - y0 + 1);
                if side <= dot_side && piece_area[l] <= dot_side * dot_side {
                    tiny_dots += 1;
                    is_screentone_dot[l] = true;
                } else if piece_in_box[l] && side >= 12 && piece_area[l] >= 40 {
                    has_large_glyph = true;
                }
            }
        }
        let has_screentone = has_large_glyph && tiny_dots >= 12;
        if has_screentone {
            let large_img = GrayImage::from_fn(width, height, |x, y| {
                let l = pieces.get_pixel(x, y).0[0] as usize;
                Luma([u8::from(l != 0 && !is_screentone_dot[l]) * 255])
            });
            let near_large = dilate(&large_img, Norm::LInf, 5);
            let mut dot_near_glyph = vec![false; piece_count];
            for (x, y, p) in pieces.enumerate_pixels() {
                let l = p.0[0] as usize;
                if l != 0 && is_screentone_dot[l] && near_large.get_pixel(x, y).0[0] != 0 {
                    dot_near_glyph[l] = true;
                }
            }
            for l in 1..piece_count {
                if is_screentone_dot[l] && !dot_near_glyph[l] {
                    drop_dot[l] = true;
                }
            }
        }
        paper_color = if has_screentone { None } else { plain(boxed) };
        let (plain_box, plain_line) = (
            paper_color.is_some(),
            !has_screentone && plain(lined).is_some(),
        );
        is_plain_bg = plain_box || plain_line;
        if plain_box || plain_line {
            // The paper's parts, and the one covering most of the box.
            let paper = connected_components(&all_ink, Connectivity::Four, Luma([255]));
            let mut covering = vec![0usize; labels(&paper)];
            for (x, y, part) in paper.enumerate_pixels() {
                if within(boxed, x, y) {
                    covering[part.0[0] as usize] += 1;
                }
            }
            let own = (1..covering.len()).max_by_key(|&part| covering[part]);
            let mut extents: Vec<Option<[u32; 4]>> = vec![None; lettering.len()];
            let (mut leaves_box, mut held, mut leaves_line, mut on_own) = (
                vec![false; lettering.len()],
                vec![false; lettering.len()],
                vec![false; lettering.len()],
                vec![false; lettering.len()],
            );
            for (x, y, piece) in pieces.enumerate_pixels() {
                let piece = piece.0[0] as usize;
                if piece == 0 {
                    continue;
                }
                if !within(boxed, x, y) {
                    leaves_box[piece] = true;
                } else if inside(x, y) {
                    held[piece] = true;
                }
                if !within(lined, x, y) {
                    leaves_line[piece] = true;
                }
                let neighbours = [
                    x.checked_sub(1).map(|x| (x, y)),
                    (x + 1 < width).then_some((x + 1, y)),
                    y.checked_sub(1).map(|y| (x, y)),
                    (y + 1 < height).then_some((x, y + 1)),
                ];
                if neighbours
                    .into_iter()
                    .flatten()
                    .any(|(x, y)| Some(paper.get_pixel(x, y).0[0] as usize) == own)
                {
                    on_own[piece] = true;
                }
                let extent = extents[piece].get_or_insert([x, y, x, y]);
                *extent = [
                    extent[0].min(x),
                    extent[1].min(y),
                    extent[2].max(x),
                    extent[3].max(y),
                ];
            }
            for (piece, extent) in extents.into_iter().enumerate() {
                let Some([x0, y0, x1, y1]) = extent else {
                    continue;
                };
                // A glyph is no longer than its line is wide: a longer part,
                // such as a ruled line under the text, is art.
                let glyph = (x1 - x0 + 1).max(y1 - y0 + 1) <= line;
                let whole = glyph && plain_box && held[piece];
                lettering[piece] = whole
                    || glyph
                        && ((plain_box && !leaves_box[piece])
                            || (plain_line && !leaves_line[piece] && on_own[piece]));
                if whole {
                    let extent = held_extent.get_or_insert([x0, y0, x1, y1]);
                    *extent = [
                        extent[0].min(x0),
                        extent[1].min(y0),
                        extent[2].max(x1),
                        extent[3].max(y1),
                    ];
                }
            }
        }
    }
    let border_mask = if line >= 12 && is_plain_bg {
        crossing_border_mask(&all_ink, &pieces, &near, &inside, line)
    } else {
        None
    };
    let is_border = |x: u32, y: u32| {
        border_mask
            .as_ref()
            .is_some_and(|mask| mask.get_pixel(x, y).0[0] != 0)
    };
    let ink = GrayImage::from_fn(width, height, |x, y| {
        let part = parts.get_pixel(x, y).0[0] as usize;
        let piece = pieces.get_pixel(x, y).0[0] as usize;
        let cut_off = joins[part] && !runs_on[part];
        Luma([
            if !is_border(x, y)
                && !drop_dot[piece]
                && ((inside(x, y) && is_ink(x, y)) || cut_off || lettering[piece])
            {
                255
            } else {
                0
            },
        ])
    });
    let held =
        held_extent.map(|[x0, y0, x1, y1]| [x0 + left, y0 + top, x1 + left + 1, y1 + top + 1]);
    if !ink.pixels().any(|pixel| pixel[0] != 0) {
        return None;
    }
    Some(Ink {
        crop,
        glyphs: ink,
        corner: [left, top],
        held,
        paper: paper_color,
        borders: border_mask,
    })
}

/// Identifies pixels of panel bars, speedlines, grazing bubble walls, and
/// enclosing frame loops that cross or surround `block`. Intersections with
/// genuine glyph strokes are protected so glyph recall is preserved.
fn crossing_border_mask(
    all_ink: &GrayImage,
    pieces: &ImageBuffer<Luma<u32>, Vec<u32>>,
    near: &GrayImage,
    inside: &impl Fn(u32, u32) -> bool,
    line: u32,
) -> Option<GrayImage> {
    let (width, height) = all_ink.dimensions();
    let max_label = pieces.pixels().map(|p| p.0[0]).max().unwrap_or(0) as usize;
    if max_label == 0 {
        return None;
    }
    let mut leaves_near = vec![false; max_label + 1];
    let mut in_block = vec![false; max_label + 1];
    let mut comp_area = vec![0u64; max_label + 1];
    let mut extents: Vec<Option<[u32; 4]>> = vec![None; max_label + 1];
    for (x, y, p) in pieces.enumerate_pixels() {
        let l = p.0[0] as usize;
        if l == 0 {
            continue;
        }
        comp_area[l] += 1;
        if near.get_pixel(x, y).0[0] == 0 {
            leaves_near[l] = true;
        }
        if inside(x, y) {
            in_block[l] = true;
        }
        let e = extents[l].get_or_insert([x, y, x, y]);
        *e = [e[0].min(x), e[1].min(y), e[2].max(x), e[3].max(y)];
    }

    // 1. Detect enclosed frame loops: hollow components longer than `line` that
    //    enclose separate glyph-sized ink components inside their interior.
    let mut is_frame_loop = vec![false; max_label + 1];
    for l in 1..=max_label {
        if !in_block[l] {
            continue;
        }
        let Some([x0, y0, x1, y1]) = extents[l] else {
            continue;
        };
        let (ew, eh) = (u64::from(x1 - x0 + 1), u64::from(y1 - y0 + 1));
        if ew.max(eh) <= u64::from(line) || ew.min(eh) < 16 || comp_area[l] * 4 >= ew * eh {
            continue;
        }
        // Check if this component encloses both paper margin and separate
        // glyph-sized ink components (never triggers on an outline shell
        // tightly hugging its glyphs).
        let (cw, ch) = ((x1 - x0 + 3), (y1 - y0 + 3));
        let mut comp_img = GrayImage::new(cw, ch);
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                if pieces.get_pixel(xx, yy).0[0] as usize == l {
                    comp_img.put_pixel(xx - x0 + 1, yy - y0 + 1, Luma([255]));
                }
            }
        }
        let mut filled = comp_img.clone();
        fill_holes(&mut filled);
        let mut enclosed_other_ink = 0u32;
        let mut enclosed_paper = 0u32;
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                if filled.get_pixel(xx - x0 + 1, yy - y0 + 1).0[0] != 0
                    && comp_img.get_pixel(xx - x0 + 1, yy - y0 + 1).0[0] == 0
                {
                    let pl = pieces.get_pixel(xx, yy).0[0] as usize;
                    if pl == 0 {
                        enclosed_paper += 1;
                    } else if pl != l {
                        enclosed_other_ink += 1;
                    }
                }
            }
        }
        if enclosed_other_ink >= 16 && enclosed_paper > enclosed_other_ink {
            is_frame_loop[l] = true;
        }
    }

    // 2. Detect components exiting `near` (or enclosing the text as a frame loop)
    //    that are long and thin/sparse.
    let candidate: Vec<bool> = (0..=max_label)
        .map(|l| {
            in_block[l]
                && (is_frame_loop[l]
                    || (leaves_near[l]
                        && extents[l].is_some_and(|[x0, y0, x1, y1]| {
                            let (ew, eh) = (u64::from(x1 - x0 + 1), u64::from(y1 - y0 + 1));
                            let long = ew.max(eh);
                            let short = ew.min(eh);
                            let thin_bar = u64::from((line / 3).clamp(4, 16));
                            long > u64::from(line)
                                && (short <= thin_bar || comp_area[l] * 4 < ew * eh)
                        })))
        })
        .collect();

    if !candidate.iter().any(|&c| c) {
        return None;
    }

    // Precompute 36 fixed unit directions covering [0, pi) in 5-degree steps.
    let dirs: [(f32, f32); 36] = std::array::from_fn(|k| {
        let theta = (k as f32) * std::f32::consts::PI / 36.0;
        (theta.cos(), theta.sin())
    });
    let min_chord = (line as f32 * 0.85).max(20.0);
    let max_steps = ((width.max(height) / 2) as i32).min(200);

    let has_ink_3x3 = |fx: f32, fy: f32| -> Option<(u32, u32)> {
        let ix = fx.round() as i32;
        let iy = fy.round() as i32;
        if ix < 0 || iy < 0 || ix >= width as i32 || iy >= height as i32 {
            return None;
        }
        if all_ink.get_pixel(ix as u32, iy as u32).0[0] != 0 {
            return Some((ix as u32, iy as u32));
        }
        let (x0, y0) = ((ix - 1).max(0) as u32, (iy - 1).max(0) as u32);
        let (x1, y1) = (
            (ix + 1).min(width as i32 - 1) as u32,
            (iy + 1).min(height as i32 - 1) as u32,
        );
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                if all_ink.get_pixel(xx, yy).0[0] != 0 {
                    return Some((xx, yy));
                }
            }
        }
        None
    };

    let mut raw_border = GrayImage::new(width, height);
    let mut comp_in_total = vec![0u32; max_label + 1];
    let mut comp_in_border = vec![0u32; max_label + 1];

    for (x, y, p) in pieces.enumerate_pixels() {
        let l = p.0[0] as usize;
        if !candidate[l] || !inside(x, y) {
            continue;
        }
        comp_in_total[l] += 1;
        let require_exit = !is_frame_loop[l];
        let mut is_line = false;
        for &(dx, dy) in &dirs {
            let mut len = 1.0f32;
            let mut exits = !require_exit;
            for sign in [-1.0f32, 1.0f32] {
                let (sdx, sdy) = (dx * sign, dy * sign);
                let mut steps = 0;
                for s in 1..=max_steps {
                    let dist = 2.0 * s as f32;
                    let Some((sx, sy)) = has_ink_3x3(x as f32 + sdx * dist, y as f32 + sdy * dist)
                    else {
                        break;
                    };
                    steps = s;
                    if near.get_pixel(sx, sy).0[0] == 0 {
                        exits = true;
                        if len + 2.0 * steps as f32 >= min_chord {
                            break;
                        }
                    }
                }
                len += 2.0 * steps as f32;
                if exits && len >= min_chord {
                    break;
                }
            }
            if exits && len >= min_chord {
                is_line = true;
                break;
            }
        }
        if is_line {
            raw_border.put_pixel(x, y, Luma([255]));
            comp_in_border[l] += 1;
        }
    }

    // Protect glyph intersections: any non-border pixel of a crossing
    // component inside `block` is dilated by 4 px so glyph strokes crossed by
    // a panel bar, speedline, or bubble wall are never clipped. Pure borders
    // (>= 85% line/arc) and enclosed frame loops are dropped completely.
    let mut non_border = GrayImage::new(width, height);
    for (x, y, p) in pieces.enumerate_pixels() {
        let l = p.0[0] as usize;
        if !candidate[l] || !inside(x, y) {
            continue;
        }
        let pure_border =
            comp_in_total[l] > 0 && (comp_in_border[l] as f32) >= 0.85 * comp_in_total[l] as f32;
        if pure_border {
            raw_border.put_pixel(x, y, Luma([255]));
        } else if raw_border.get_pixel(x, y).0[0] == 0 {
            non_border.put_pixel(x, y, Luma([255]));
        }
    }
    let protected = dilate(&non_border, Norm::LInf, 4);
    let mut final_border = GrayImage::new(width, height);
    let mut any = false;
    for (x, y, p) in raw_border.enumerate_pixels() {
        if p.0[0] != 0 && protected.get_pixel(x, y).0[0] == 0 {
            final_border.put_pixel(x, y, Luma([255]));
            any = true;
        }
    }
    any.then_some(final_border)
}

/// Refines a single-pass Otsu threshold when the ink-side class is strongly
/// bimodal (such as dark text ink merged with darkened translucent-bubble art).
fn refine_bimodal_ink_threshold(
    sample: &GrayImage,
    threshold: u16,
    light_ink: bool,
) -> Option<u16> {
    let mut sub_hist = [0.0f32; 256];
    let mut sub_total = 0u32;
    for p in sample.pixels() {
        let v = u16::from(p.0[0]);
        let in_ink_side = if light_ink {
            v >= threshold
        } else {
            v < threshold
        };
        if in_ink_side {
            sub_hist[usize::from(p.0[0])] += 1.0;
            sub_total += 1;
        }
    }
    if sub_total < 64 {
        return None;
    }
    let sub_level = split_distribution(&sub_hist, 0.1);
    let sub_threshold = sample
        .as_raw()
        .iter()
        .copied()
        .filter(|&v| {
            let in_side = if light_ink {
                u16::from(v) >= threshold
            } else {
                u16::from(v) < threshold
            };
            in_side && v < sub_level
        })
        .max()
        .map_or(0, |v| u16::from(v) + 1);
    if sub_threshold == 0 || sub_threshold >= 256 {
        return None;
    }
    let (mut extreme_sum, mut extreme_cnt) = (0u64, 0u32);
    let (mut mid_sum, mut mid_cnt) = (0u64, 0u32);
    let (mut border_total, mut border_mid, mut border_extreme) = (0u32, 0u32, 0u32);
    let (w, h) = sample.dimensions();
    let mut extreme_img = GrayImage::new(w, h);
    for (x, y, p) in sample.enumerate_pixels() {
        let v = u16::from(p.0[0]);
        let in_side = if light_ink {
            v >= threshold
        } else {
            v < threshold
        };
        let on_border = x < 2 || y < 2 || x + 2 >= w || y + 2 >= h;
        if on_border {
            border_total += 1;
        }
        if !in_side {
            continue;
        }
        let is_extreme = if light_ink {
            v >= sub_threshold
        } else {
            v < sub_threshold
        };
        if is_extreme {
            extreme_sum += u64::from(v);
            extreme_cnt += 1;
            extreme_img.put_pixel(x, y, Luma([255]));
            if on_border {
                border_extreme += 1;
            }
        } else {
            mid_sum += u64::from(v);
            mid_cnt += 1;
            if on_border {
                border_mid += 1;
            }
        }
    }
    if extreme_cnt < 32 || mid_cnt < extreme_cnt {
        return None;
    }
    let mean_extreme = extreme_sum as f32 / extreme_cnt as f32;
    let mean_mid = mid_sum as f32 / mid_cnt as f32;
    let extreme_is_pure = if light_ink {
        mean_extreme >= 200.0
    } else {
        mean_extreme <= 60.0
    };
    if !extreme_is_pure || (mean_mid - mean_extreme).abs() < 60.0 {
        return None;
    }
    if border_total == 0
        || (border_mid as f32) < 0.12 * border_total as f32
        || border_mid <= border_extreme
    {
        return None;
    }
    let near_extreme = dilate(&extreme_img, Norm::LInf, 2);
    let mut far_mid = 0u32;
    for (x, y, p) in sample.enumerate_pixels() {
        let v = u16::from(p.0[0]);
        let in_side = if light_ink {
            v >= threshold
        } else {
            v < threshold
        };
        let is_extreme = if light_ink {
            v >= sub_threshold
        } else {
            v < sub_threshold
        };
        if in_side && !is_extreme && near_extreme.get_pixel(x, y).0[0] == 0 {
            far_mid += 1;
        }
    }
    if far_mid * 2 < mid_cnt {
        return None;
    }
    Some(sub_threshold)
}

/// Samples dominant non-ink background RGB colors around `bbox` (both outside
/// `mask` and inside `bbox` away from ink strokes), used to filter coarse
/// segmented blobs over translucent textured or gradient bubbles without
/// removing strokes or outlines.
fn sample_background_palette(
    page: &RgbImage,
    mask: &Mask,
    ink: &GrayImage,
    ink_corner: [u32; 2],
    bbox: [f32; 4],
    reach: u32,
) -> Vec<[u8; 3]> {
    let pad = reach.max(8);
    let bx0 = bbox[0].floor().max(0.0) as u32;
    let by0 = bbox[1].floor().max(0.0) as u32;
    let bx1 = (bbox[2].ceil().max(0.0) as u32).min(page.width());
    let by1 = (bbox[3].ceil().max(0.0) as u32).min(page.height());
    let x0 = bx0.saturating_sub(pad);
    let y0 = by0.saturating_sub(pad);
    let x1 = (bx1 + pad).min(page.width());
    let y1 = (by1 + pad).min(page.height());
    let ink_edges = dilate(ink, Norm::LInf, 2);
    let mut bins = HashMap::<[u8; 3], (u32, [u64; 3])>::new();
    let mut total = 0u32;
    for py in y0..y1 {
        for px in x0..x1 {
            if mask.contains(px, py) {
                continue;
            }
            if px >= ink_corner[0]
                && py >= ink_corner[1]
                && px - ink_corner[0] < ink_edges.width()
                && py - ink_corner[1] < ink_edges.height()
                && ink_edges
                    .get_pixel(px - ink_corner[0], py - ink_corner[1])
                    .0[0]
                    != 0
            {
                continue;
            }
            let rgb = page.get_pixel(px, py).0;
            let key = rgb.map(|c| c / 24);
            let (cnt, sum) = bins.entry(key).or_default();
            *cnt += 1;
            for (s, c) in sum.iter_mut().zip(rgb) {
                *s += u64::from(c);
            }
            total += 1;
        }
    }
    if total < 32 {
        return Vec::new();
    }
    // Never include the dominant ink colour in the background palette so we
    // never strip genuine glyph strokes even when art shares the ink tone.
    let mut ink_bins = HashMap::<[u8; 3], (u32, [u64; 3])>::new();
    for (x, y, p) in ink.enumerate_pixels() {
        if p.0[0] != 0 {
            let px = ink_corner[0] + x;
            let py = ink_corner[1] + y;
            if px < page.width() && py < page.height() {
                let rgb = page.get_pixel(px, py).0;
                let (cnt, sum) = ink_bins.entry(rgb.map(|c| c / 16)).or_default();
                *cnt += 1;
                for (s, c) in sum.iter_mut().zip(rgb) {
                    *s += u64::from(c);
                }
            }
        }
    }
    let dominant_ink = ink_bins
        .into_values()
        .max_by_key(|(cnt, _)| *cnt)
        .map(|(cnt, sum)| sum.map(|s| (s / u64::from(cnt)) as u8));

    let min_bin = (total / 40).max(8);
    bins.into_values()
        .filter(|(cnt, _)| *cnt >= min_bin)
        .map(|(cnt, sum)| sum.map(|s| (s / u64::from(cnt)) as u8))
        .filter(|bg| {
            let not_pure_dark = bg.iter().copied().max().unwrap_or(0) >= 70;
            not_pure_dark
                && dominant_ink.is_none_or(|ink_c| (0..3).any(|c| bg[c].abs_diff(ink_c[c]) > 48))
        })
        .collect()
}

/// `numaSplitDistribution`, plus one: the split with the best Otsu score,
/// moved to the histogram's minimum among the neighbouring splits scoring
/// within `score_fraction` of it.
fn split_distribution(histogram: &[f32; 256], score_fraction: f32) -> u8 {
    let n = histogram.len();
    let sum: f32 = histogram.iter().fold(0.0, |sum, &count| sum + count);
    let norm = 4.0f32 / ((n - 1) as f32 * (n - 1) as f32);
    let (mut moment, mut total) = (0.0f32, 0.0f32);
    for (index, &count) in histogram.iter().enumerate() {
        total += count;
        moment += index as f32 * count;
    }
    let (mut mean_below, mut mean_above) = (0.0f32, moment / total);
    let (mut below, mut above) = (0.0f32, sum);
    let mut scores = [0.0f32; 256];
    let (mut max_score, mut max_index) = (0.0f32, n / 2);
    for (index, &count) in histogram.iter().enumerate() {
        let next_below = below + count;
        let next_mean_below = if next_below == 0.0 {
            mean_below
        } else {
            (below * mean_below + index as f32 * count) / next_below
        };
        let next_above = above - count;
        let next_mean_above = if next_above == 0.0 {
            mean_above
        } else {
            (above * mean_above - index as f32 * count) / next_above
        };
        let fraction = next_below / sum;
        let score = norm
            * (fraction * (1.0 - fraction))
            * (next_mean_above - next_mean_below)
            * (next_mean_above - next_mean_below);
        scores[index] = score;
        if score > max_score {
            max_score = score;
            max_index = index;
        }
        (below, above, mean_below, mean_above) =
            (next_below, next_above, next_mean_below, next_mean_above);
    }

    let min_score = (1.0 - score_fraction) * max_score;
    let mut first = max_index;
    while first > 0 && scores[first - 1] >= min_score {
        first -= 1;
    }
    let mut last = max_index;
    while last + 1 < n && scores[last + 1] >= min_score {
        last += 1;
    }
    let mut split = first;
    for index in first + 1..=last {
        if histogram[index] < histogram[split] {
            split = index;
        }
    }
    (split + 1).min(255) as u8
}

/// Single-tile Leptonica stroke width. Padding puts the page beyond the
/// crop in the background for the eight-connected distance histogram.
fn find_stroke_width(ink: &GrayImage, threshold: f32) -> Option<f32> {
    let mut background = GrayImage::from_pixel(ink.width() + 2, ink.height() + 2, Luma([255]));
    for (x, y, pixel) in ink.enumerate_pixels() {
        background.put_pixel(x + 1, y + 1, Luma([u8::from(pixel[0] == 0) * 255]));
    }
    let distances = imageproc::distance_transform::distance_transform(&background, Norm::LInf);
    let mut histogram = [0usize; 256];
    for distance in distances.pixels() {
        histogram[usize::from(distance[0])] += 1;
    }
    let area: usize = histogram[1..].iter().sum();
    let boundary = histogram[1];
    if boundary < 2 {
        return None;
    }
    let deepest = (1..256).rfind(|&d| histogram[d] as f32 > threshold * boundary as f32)?;
    let last = histogram[deepest] + histogram.get(deepest + 1).copied().unwrap_or(0);
    let distance_width = 2.0 * (deepest as f32 - 1.0 + last as f32 / boundary as f32);
    Some((area as f32 / (boundary / 2) as f32 + distance_width) * 0.5)
}

/// Minimum relative projection-energy gain over upright (0°) and over the
/// mean across all candidate angles required before emitting a non-zero tilt.
const MIN_TILT_GAIN_OVER_UPRIGHT: f64 = 1.05;
const MIN_TILT_GAIN_OVER_MEAN: f64 = 1.15;

/// Cap on the block closing and stroke-extension radius in pixels so
/// ultra-high-resolution pages (e.g. 4961px) do not bridge inter-column
/// gutters.
const MAX_BLOCK_CLOSING_RADIUS: u32 = 12;

/// The tilt, in degrees, under which lettering reads as upright, as a missed
/// turn costs less than a false one: the 41 upright texts that the plain
/// average turned by up to 7° on 8 Manga109 spreads measure at most 1.3°.
const UPRIGHT: f32 = 3.0;

/// How far from the paper's median colour, in every channel, a pixel of
/// plain paper lies, and how much of it must.
const PAPER_TOLERANCE: u8 = 32;
const PLAIN_PAPER: f32 = 0.99;

/// `per_1024` pixels for every 1024 of the page's longer side.
pub(super) fn scaled(per_1024: f32, page_width: u32, page_height: u32) -> u32 {
    ((page_width.max(page_height) as f32 / 1024.0) * per_1024)
        .round()
        .clamp(1.0, 127.0) as u32
}

/// The covered part of `canvas`, placed at (`left`, `top`) on a page, cut to
/// the page.
fn crop(canvas: &GrayImage, left: i64, top: i64, page_width: u32, page_height: u32) -> Mask {
    let mut bounds: Option<[u32; 4]> = None;
    for (x, y, pixel) in canvas.enumerate_pixels() {
        if pixel[0] != 0
            && (0..i64::from(page_width)).contains(&(left + i64::from(x)))
            && (0..i64::from(page_height)).contains(&(top + i64::from(y)))
        {
            let [x0, y0, x1, y1] = bounds.get_or_insert([x, y, x + 1, y + 1]);
            *x0 = (*x0).min(x);
            *y0 = (*y0).min(y);
            *x1 = (*x1).max(x + 1);
            *y1 = (*y1).max(y + 1);
        }
    }
    let Some([x0, y0, x1, y1]) = bounds else {
        return Mask {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            pixels: Vec::new(),
        };
    };
    let (width, height) = (x1 - x0, y1 - y0);
    Mask {
        x: (left + i64::from(x0)) as u32,
        y: (top + i64::from(y0)) as u32,
        width,
        height,
        pixels: imageops::crop_imm(canvas, x0, y0, width, height)
            .to_image()
            .into_raw(),
    }
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    #[test]
    fn otsu_fits_both_polarities_and_fills_only_enclosed_counters() {
        for (ink, paper) in [(40, 210), (210, 40)] {
            let page = RgbImage::from_fn(120, 80, |x, y| {
                let ring = (28..42).contains(&x)
                    && (29..45).contains(&y)
                    && !((31..39).contains(&x) && (32..42).contains(&y));
                let stroke = (60..70).contains(&x) && (29..45).contains(&y);
                Rgb([if ring || stroke { ink } else { paper }; 3])
            });
            let bbox = [20.0, 20.0, 80.0, 55.0];
            // A coarse segmented blob with an open notch and an ink stroke
            // missed at the end. The space between the glyphs is open paper.
            let mask = Mask {
                x: 22,
                y: 25,
                width: 46,
                height: 24,
                pixels: (0..46 * 24)
                    .map(|i| if i % 46 < 2 && i / 46 < 2 { 0 } else { 255 })
                    .collect(),
            };
            let (fitted, _, _) = glyph_mask(&mask, bbox, &page, GlyphProposal::Segmented(bbox));
            assert!(fitted.contains(34, 38) && fitted.contains(69, 40));
            assert!(fitted.contains(70, 40)); // scale-dependent antialias margin
            assert!(!fitted.contains(51, 36) && !fitted.contains(22, 48));
        }
    }

    #[test]
    fn singleton_foreground_and_zero_area_are_safe() {
        for (w, h, pixels) in [(1, 1, vec![255]), (0, 1, vec![]), (1, 0, vec![])] {
            let page = RgbImage::from_pixel(w, h, Rgb([0; 3]));
            let mask = Mask {
                x: 0,
                y: 0,
                width: w,
                height: h,
                pixels,
            };
            let bbox = [0., 0., w as f32, h as f32];
            let (fitted, _, grown) = glyph_mask(&mask, bbox, &page, GlyphProposal::Segmented(bbox));
            assert_eq!(fitted.pixels, mask.pixels);
            assert_eq!(grown, bbox);
            assert!(fitted.typography(&page).is_none());
        }
    }

    #[test]
    fn conjoined_masks_keep_the_glyph_nearest_the_original_box() {
        let page = RgbImage::from_fn(80, 120, |x, y| {
            let glyph = (26..34).contains(&x) && ((44..52).contains(&y) || (58..66).contains(&y));
            Rgb([if glyph { 0 } else { 255 }; 3])
        });
        let text = |bbox: [f32; 4]| KoharuLayoutDetection {
            label_id: 0,
            label: "text".into(),
            score: 0.9,
            bbox,
            area: 20 * 53,
            mask: Arc::new(Mask {
                x: 20,
                y: 25,
                width: 20,
                height: 53,
                pixels: vec![255; 20 * 53],
            }),
        };
        let mut page_masks = [text([20., 10., 40., 50.]), text([20., 62., 40., 83.])];
        let proposals = page_masks.iter_mut().map(|d| {
            let original = d.bbox;
            // The same comic box enlarges both windows over both glyphs.
            d.bbox = [20., 10., 40., 83.];
            (GlyphProposal::Segmented(original), d)
        });
        fit_glyph_masks(proposals, &page);
        assert!(page_masks[0].mask.contains(30, 50));
        assert!(!page_masks[1].mask.contains(30, 50));
        assert!(page_masks[1].mask.contains(30, 60));
        assert!(!page_masks[0].mask.contains(30, 60));
    }

    #[test]
    fn a_glyph_past_the_box_counts_only_on_the_box_paper() {
        // A column of glyphs on white paper, a glyph the box cut off below
        // it, a border across the page, and a glyph past the border in line
        // with the column, as a copyright line under a panel lies.
        let mut page = RgbImage::from_pixel(200, 300, Rgb([255, 255, 255]));
        let mut square = |left: u32, top: u32, width: u32, height: u32| {
            for y in top..top + height {
                for x in left..left + width {
                    page.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        };
        let glyphs = [40, 64, 88, 112, 136, 160];
        for top in glyphs {
            square(92, top, 16, 16);
        }
        square(92, 184, 16, 16);
        square(0, 203, 200, 3);
        square(92, 209, 16, 10);
        let mut pixels = vec![0; 16 * 136];
        for top in glyphs {
            for y in top - 40..top - 40 + 16 {
                pixels[(y * 16) as usize..(y * 16 + 16) as usize].fill(255);
            }
        }
        let detected = Mask {
            x: 92,
            y: 40,
            width: 16,
            height: 136,
            pixels,
        };
        let bbox = [80.0, 36.0, 120.0, 180.0];
        let (fitted, _, _) = glyph_mask(&detected, bbox, &page, GlyphProposal::Segmented(bbox));
        assert!(fitted.contains(100, 192));
        assert!(!fitted.contains(100, 213));
    }

    #[test]
    fn a_mask_part_away_from_the_box_is_dropped() {
        // A column of glyphs under a border, and above the border a glyph of
        // another lettering that the detection's mask took in too.
        let mut page = RgbImage::from_pixel(200, 300, Rgb([255, 255, 255]));
        let mut square = |left: u32, top: u32, width: u32, height: u32| {
            for y in top..top + height {
                for x in left..left + width {
                    page.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        };
        for top in [40, 64, 88, 112, 136, 160] {
            square(92, top, 16, 16);
        }
        square(0, 30, 200, 3);
        square(92, 8, 16, 16);
        let pixels = (0..176)
            .flat_map(|y| (0..16).map(move |_| y))
            .map(
                |y| match (8..24).contains(&y) || (40..176).contains(&y) && (y - 40) % 24 < 16 {
                    true => 255,
                    false => 0,
                },
            )
            .collect();
        let detected = Mask {
            x: 92,
            y: 0,
            width: 16,
            height: 176,
            pixels,
        };
        let bbox = [80.0, 36.0, 120.0, 180.0];
        let (fitted, _, _) = glyph_mask(&detected, bbox, &page, GlyphProposal::Segmented(bbox));
        assert!(fitted.contains(100, 48));
        assert!(!fitted.contains(100, 16));
    }

    #[test]
    fn a_line_the_box_ends_inside_is_masked_whole() {
        // Three lines of glyphs on white paper, the box and its mask ending
        // halfway down the last, and another lettering's line under them.
        let mut page = RgbImage::from_pixel(300, 200, Rgb([255, 255, 255]));
        let lines = [60, 80, 100, 130];
        for top in lines {
            for left in (40..190).step_by(14) {
                for y in top..top + 14 {
                    for x in left..left + 10 {
                        page.put_pixel(x, y, Rgb([0, 0, 0]));
                    }
                }
            }
        }
        let bbox = [36.0, 56.0, 200.0, 107.0];
        let detected = Mask {
            x: 0,
            y: 0,
            width: 300,
            height: 200,
            pixels: (0..200)
                .flat_map(|y| (0..300).map(move |x| (x, y)))
                .map(|(x, y)| {
                    let glyph = page.get_pixel(x, y).0[0] == 0;
                    u8::from(glyph && y < 107) * u8::MAX
                })
                .collect(),
        };
        let (fitted, _, grown) = glyph_mask(&detected, bbox, &page, GlyphProposal::Segmented(bbox));
        assert!(fitted.contains(45, 113));
        assert!(grown[3] >= 114.0);
        assert!(!fitted.contains(45, 131) && grown[3] < 130.0);
    }

    #[test]
    fn a_ragged_column_stays_upright_and_a_turned_one_turns() {
        use imageproc::{drawing::draw_polygon_mut, point::Point};

        // Shapes around a column's top centre, turned clockwise about it, on
        // a 1024-pixel page, whose blocks close over gaps of up to 12 pixels.
        let tilt = |shapes: Vec<[(f32, f32); 4]>, turn: f32| {
            let mut page = RgbImage::from_pixel(1024, 1024, Rgb([255, 255, 255]));
            let (sin, cos) = turn.to_radians().sin_cos();
            for corners in shapes {
                let points = corners.map(|(x, y)| {
                    Point::new(
                        (500.0 + x * cos - y * sin).round() as i32,
                        (400.0 + x * sin + y * cos).round() as i32,
                    )
                });
                draw_polygon_mut(&mut page, &points, Rgb([0, 0, 0]));
            }
            let pixels = page
                .pixels()
                .map(|pixel| if pixel.0[0] < 128 { 255 } else { 0 })
                .collect();
            let lettering = Mask {
                x: 0,
                y: 0,
                width: 1024,
                height: 1024,
                pixels,
            };
            lettering
                .typography(&page)
                .expect("the column holds ink")
                .tilt
        };
        let bar = |left: f32, top: f32, width: f32, height: f32| {
            [
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
            ]
        };
        // Three 三, 24 pixels wide and 28 apart, ending in a hand-lettered
        // !!: two bars leaning 25°, narrower than the column, as ー, っ, and
        // !! make a column's sides ragged.
        let mut ragged: Vec<_> = (0..3)
            .flat_map(|row| {
                [0.0, 11.0, 22.0].map(|line| bar(-12.0, 28.0 * row as f32 + line, 24.0, 2.0))
            })
            .collect();
        ragged.extend([-4.0, 4.0].map(|left| {
            [
                (left, 82.0),
                (left + 3.0, 82.0),
                (left - 7.0, 104.0),
                (left - 10.0, 104.0),
            ]
        }));
        assert_eq!(tilt(ragged, 0.0), 0.0);
        let solid = (0..5)
            .map(|row| bar(-12.0, 28.0 * row as f32, 24.0, 24.0))
            .collect();
        let turned = tilt(solid, 12.0);
        assert!((turned - 12.0).abs() < 1.0, "turned {turned}°");
    }

    #[test]
    fn short_columns_read_down_and_short_rows_across() {
        // Hollow 24-pixel glyphs with 2-pixel strokes on a 1024-pixel page,
        // in `columns` by `rows`, `across` pixels apart side by side and
        // `down` pixels apart above and below, and with `ruby` a hollow
        // 10-pixel glyph with 1-pixel strokes a pixel right of each column's
        // first and centred on it, as ruby stands beside one word of a
        // column.
        let vertical = |columns: u32, rows: u32, across: u32, down: u32, ruby: bool| {
            let mut page = RgbImage::from_pixel(1024, 1024, Rgb([255, 255, 255]));
            let mut hollow = |left: u32, top: u32, size: u32, stroke: u32| {
                for (x, y) in
                    (left..left + size).flat_map(|x| (top..top + size).map(move |y| (x, y)))
                {
                    let edge = x < left + stroke
                        || x >= left + size - stroke
                        || y < top + stroke
                        || y >= top + size - stroke;
                    if edge {
                        page.put_pixel(x, y, Rgb([0, 0, 0]));
                    }
                }
            };
            for (column, row) in
                (0..columns).flat_map(|column| (0..rows).map(move |row| (column, row)))
            {
                let (left, top) = (400 + column * (24 + across), 400 + row * (24 + down));
                hollow(left, top, 24, 2);
                if ruby && row == 0 {
                    hollow(left + 25, top + 7, 10, 1);
                }
            }
            let pixels = page
                .pixels()
                .map(|pixel| if pixel.0[0] < 128 { 255 } else { 0 })
                .collect();
            let lettering = Mask {
                x: 0,
                y: 0,
                width: 1024,
                height: 1024,
                pixels,
            };
            lettering
                .typography(&page)
                .expect("the glyphs are ink")
                .vertical
        };
        // Four columns of three glyphs, wider than tall, the columns 12
        // pixels apart and their glyphs 4, with ruby and without.
        assert!(vertical(4, 3, 12, 4, false));
        assert!(vertical(4, 3, 12, 4, true));
        // Three rows of four glyphs, also wider than tall, the rows 12
        // pixels apart and their glyphs 4.
        assert!(!vertical(4, 3, 4, 12, false));
    }

    #[test]
    fn an_outline_is_a_band_unlike_the_ink_with_something_else_past_it() {
        // Two black glyphs in a column, in a white band 4 pixels wide.
        let page = |background: Rgb<u8>| {
            let mut page = RgbImage::from_pixel(200, 200, background);
            for y in 56..114 {
                for x in 56..84 {
                    page.put_pixel(x, y, Rgb([255, 255, 255]));
                }
            }
            for (x, y) in (60..80).flat_map(|x| (60..80).chain(90..110).map(move |y| (x, y))) {
                page.put_pixel(x, y, Rgb([0, 0, 0]));
            }
            page
        };
        let glyphs = GrayImage::from_fn(20, 50, |_, y| {
            Luma([if (20..30).contains(&y) { 0 } else { 255 }])
        });
        let over_art = outline(
            &page(Rgb([200, 40, 40])),
            &glyphs,
            [60, 60],
            20.0,
            Some([0; 3]),
        );
        assert_eq!(
            over_art,
            Some(Outline {
                color: [255; 3],
                width: 4.0,
            })
        );
        // Beside a white balloon, which takes the right half, the band runs
        // on into it there and ends over the art.
        let mut beside = page(Rgb([200, 40, 40]));
        for (x, y) in (84..200).flat_map(|x| (0..200).map(move |y| (x, y))) {
            beside.put_pixel(x, y, Rgb([255; 3]));
        }
        for (x, y) in (70..84).flat_map(|x| (0..56).chain(114..200).map(move |y| (x, y))) {
            beside.put_pixel(x, y, Rgb([255; 3]));
        }
        assert_eq!(
            outline(&beside, &glyphs, [60, 60], 20.0, Some([0; 3])),
            Some(Outline {
                color: [255; 3],
                width: 4.0,
            })
        );
        // On white paper the band never ends.
        let on_paper = outline(&page(Rgb([255; 3])), &glyphs, [60, 60], 20.0, Some([0; 3]));
        assert_eq!(on_paper, None);
    }

    #[test]
    fn steeply_tilted_column_recovers_tilt_without_collapsing_to_upright() {
        use imageproc::{drawing::draw_polygon_mut, point::Point};

        for expected in [-30.0f32, 35.0f32] {
            let mut page = RgbImage::from_pixel(560, 560, Rgb([255, 255, 255]));
            let (sin, cos) = expected.to_radians().sin_cos();
            for row in 0..5 {
                let top = -70.0 + 30.0 * row as f32;
                let corners = [
                    (-14.0, top),
                    (14.0, top),
                    (14.0, top + 22.0),
                    (-14.0, top + 22.0),
                ];
                let points = corners.map(|(x, y)| {
                    Point::new(
                        (280.0 + x * cos - y * sin).round() as i32,
                        (280.0 + x * sin + y * cos).round() as i32,
                    )
                });
                draw_polygon_mut(&mut page, &points, Rgb([0, 0, 0]));
            }
            let pixels = page
                .pixels()
                .map(|pixel| if pixel.0[0] < 128 { 255 } else { 0 })
                .collect();
            let lettering = Mask {
                x: 0,
                y: 0,
                width: 560,
                height: 560,
                pixels,
            };
            let typo = lettering
                .typography(&page)
                .expect("tilted column holds ink");
            assert!(
                (typo.tilt - expected).abs() < 2.0,
                "expected ~{expected}°, got {}°",
                typo.tilt
            );
        }
    }

    #[test]
    fn upright_circular_dot_column_stays_upright() {
        let mut page = RgbImage::from_pixel(440, 440, Rgb([255, 255, 255]));
        for cy in [160i32, 220, 280] {
            for y in cy - 5..=cy + 5 {
                for x in 215i32..=225 {
                    let dx = x - 220;
                    let dy = y - cy;
                    if dx * dx + dy * dy <= 16 {
                        page.put_pixel(x as u32, y as u32, Rgb([0, 0, 0]));
                    }
                }
            }
        }
        let pixels = page
            .pixels()
            .map(|pixel| if pixel.0[0] < 128 { 255 } else { 0 })
            .collect();
        let lettering = Mask {
            x: 0,
            y: 0,
            width: 440,
            height: 440,
            pixels,
        };
        let typo = lettering.typography(&page).expect("dot column holds ink");
        assert_eq!(
            typo.tilt, 0.0,
            "upright dots must not snap to diagonal tilt"
        );
    }

    #[test]
    fn tilted_adjacent_columns_preserve_target_ink_despite_overlapping_aabbs() {
        use imageproc::{drawing::draw_polygon_mut, point::Point};

        let mut page = RgbImage::from_pixel(300, 300, Rgb([255, 255, 255]));
        let (sin, cos) = (-12.0f32).to_radians().sin_cos();
        let mut draw_col = |cx: f32, rows: i32| -> (Mask, [f32; 4], Vec<(u32, u32)>) {
            let mut canvas = GrayImage::new(300, 300);
            let mut ink_pts = Vec::new();
            let (mut x0, mut y0, mut x1, mut y1) = (300u32, 300u32, 0u32, 0u32);
            for row in 0..rows {
                let top = -80.0 + 28.0 * row as f32;
                let corners = [
                    (-9.0, top),
                    (9.0, top),
                    (9.0, top + 20.0),
                    (-9.0, top + 20.0),
                ];
                let points = corners.map(|(x, y)| {
                    Point::new(
                        (cx + x * cos - y * sin).round() as i32,
                        (150.0 + x * sin + y * cos).round() as i32,
                    )
                });
                draw_polygon_mut(&mut page, &points, Rgb([0, 0, 0]));
                draw_polygon_mut(&mut canvas, &points, Luma([255]));
            }
            for (x, y, p) in canvas.enumerate_pixels() {
                if p.0[0] != 0 {
                    ink_pts.push((x, y));
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
            let mask = crop(&canvas, 0, 0, 300, 300);
            (mask, [x0 as f32, y0 as f32, x1 as f32, y1 as f32], ink_pts)
        };
        // Larger target column (6 glyphs) next to a smaller neighbor (4 glyphs)
        // 4px away so their axis-aligned bounding boxes overlap heavily.
        let (target_mask, target_bbox, target_ink) = draw_col(130.0, 6);
        let (neighbor_mask, neighbor_bbox, _) = draw_col(152.0, 4);
        assert!(
            target_bbox[2] > neighbor_bbox[0],
            "AABBs must overlap to exercise tie-breaking"
        );

        let make_det = |mask: Mask, bbox: [f32; 4]| KoharuLayoutDetection {
            label_id: 0,
            label: "text".into(),
            score: 0.9,
            bbox,
            area: mask.pixels.iter().filter(|&&v| v != 0).count() as u32,
            mask: Arc::new(mask),
        };
        let mut dets = [
            make_det(target_mask, target_bbox),
            make_det(neighbor_mask, neighbor_bbox),
        ];
        fit_glyph_masks(
            dets.iter_mut()
                .map(|d| (GlyphProposal::Segmented(d.bbox), d)),
            &page,
        );
        let kept = target_ink
            .iter()
            .filter(|&&(x, y)| dets[0].mask.contains(x, y))
            .count();
        assert_eq!(
            kept,
            target_ink.len(),
            "smaller tilted neighbor stole target ink pixels"
        );
    }

    #[test]
    fn thin_two_pixel_outline_detected_while_aa_fringe_is_rejected() {
        // Two black glyphs with a 2 px white outline over colored art.
        let mut outlined_page = RgbImage::from_pixel(160, 160, Rgb([180, 50, 50]));
        for y in 48..102 {
            for x in 58..82 {
                outlined_page.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
        for (x, y) in (60..80).flat_map(|x| (50..70).chain(80..100).map(move |y| (x, y))) {
            outlined_page.put_pixel(x, y, Rgb([0, 0, 0]));
        }
        let glyphs = GrayImage::from_fn(20, 50, |_, y| {
            Luma([if (20..30).contains(&y) { 0 } else { 255 }])
        });
        let found = outline(&outlined_page, &glyphs, [60, 50], 20.0, Some([0; 3]));
        assert_eq!(
            found,
            Some(Outline {
                color: [255; 3],
                width: 2.0,
            })
        );

        // Plain black glyphs on uniform white paper with a 1 px grey AA fringe
        // must not trigger a false outline.
        let mut aa_page = RgbImage::from_pixel(160, 160, Rgb([255, 255, 255]));
        for y in 49..101 {
            for x in 59..81 {
                aa_page.put_pixel(x, y, Rgb([128, 128, 128]));
            }
        }
        for (x, y) in (60..80).flat_map(|x| (50..70).chain(80..100).map(move |y| (x, y))) {
            aa_page.put_pixel(x, y, Rgb([0, 0, 0]));
        }
        assert_eq!(
            outline(&aa_page, &glyphs, [60, 50], 20.0, Some([0; 3])),
            None
        );
    }

    #[test]
    fn translucent_bubble_over_checker_art_keeps_glyphs_without_bleed() {
        let is_glyph = |x: u32, y: u32| {
            ((60..66).contains(&x) && (50..90).contains(&y))
                || ((82..88).contains(&x) && (50..90).contains(&y))
                || ((104..110).contains(&x) && (50..90).contains(&y))
        };
        let page = RgbImage::from_fn(200, 160, |x, y| {
            if is_glyph(x, y) {
                return Rgb([15, 15, 15]);
            }
            let art = if (x / 10 + y / 10).is_multiple_of(2) {
                20u16
            } else {
                245u16
            };
            // Translucent white bubble (alpha = 0.55) covering the central region.
            let in_bubble = (30..170).contains(&x) && (25..135).contains(&y);
            let v = if in_bubble {
                ((art * 45 + 255 * 55) / 100) as u8
            } else {
                art as u8
            };
            Rgb([v, v, v])
        });
        let bbox = [50.0, 40.0, 120.0, 100.0];
        let mut mask = Mask {
            x: 50,
            y: 40,
            width: 70,
            height: 60,
            pixels: vec![255; 70 * 60],
        };
        // Mark a non-solid segmented blob by clearing the 4 corners.
        mask.pixels[0] = 0;
        let (fitted, _, _) = glyph_mask(&mask, bbox, &page, GlyphProposal::Segmented(bbox));
        let mut glyph_total = 0u32;
        let mut glyph_hit = 0u32;
        let mut bg_bleed = 0u32;
        for y in 40..100 {
            for x in 50..120 {
                let in_fitted = fitted.contains(x, y);
                if is_glyph(x, y) {
                    glyph_total += 1;
                    glyph_hit += u32::from(in_fitted);
                } else if in_fitted {
                    bg_bleed += 1;
                }
            }
        }
        assert_eq!(glyph_hit, glyph_total, "must preserve all glyph pixels");
        assert!(
            (bg_bleed as f32) < 0.25 * (70 * 60 - glyph_total) as f32,
            "excessive translucent background bleed: {bg_bleed}"
        );
        let typo = fitted.typography(&page).expect("typography");
        let ink_c = typo.ink.expect("ink color");
        assert!(ink_c[0] < 40 && ink_c[1] < 40 && ink_c[2] < 40, "{ink_c:?}");
        assert_eq!(typo.outline, None);
    }

    #[test]
    fn crossing_panel_bar_is_excluded_without_clipping_glyph_strokes() {
        let is_glyph = |x: u32, y: u32| {
            ((70..76).contains(&x) && (55..95).contains(&y))
                || ((92..98).contains(&x) && (55..95).contains(&y))
        };
        let is_bar = |y: u32| (72..75).contains(&y);
        let page = RgbImage::from_fn(220, 160, |x, y| {
            Rgb(if is_glyph(x, y) || is_bar(y) {
                [10, 10, 10]
            } else {
                [250, 250, 250]
            })
        });
        let bbox = [55.0, 45.0, 125.0, 105.0];
        let mask = Mask {
            x: 55,
            y: 45,
            width: 70,
            height: 60,
            pixels: vec![255; 70 * 60],
        };
        let (fitted, _, _) = glyph_mask(&mask, bbox, &page, GlyphProposal::Box);
        // Every glyph pixel must remain covered.
        for y in 55..95 {
            for x in (70..76).chain(92..98) {
                assert!(fitted.contains(x, y), "clipped glyph pixel at ({x}, {y})");
            }
        }
        // Panel bar pixels well away from the glyphs inside the box should be excluded.
        assert!(
            !fitted.contains(58, 73),
            "panel bar left stub should be excluded"
        );
        assert!(
            !fitted.contains(120, 73),
            "panel bar right stub should be excluded"
        );
    }

    #[test]
    fn loose_bbox_filters_screentone_specks_but_preserves_ellipsis_dots() {
        // 1. Loose bbox containing two glyphs surrounded by periodic screentone dots.
        let is_glyph = |x: u32, y: u32| {
            ((80..96).contains(&x) && (60..76).contains(&y))
                || ((80..96).contains(&x) && (86..102).contains(&y))
        };
        let is_dot = |x: u32, y: u32| x % 12 < 2 && y % 12 < 2;
        let page = RgbImage::from_fn(220, 200, |x, y| {
            Rgb(if is_glyph(x, y) || is_dot(x, y) {
                [15, 15, 15]
            } else {
                [250, 250, 250]
            })
        });
        let bbox = [40.0, 30.0, 140.0, 140.0];
        let mask = Mask {
            x: 40,
            y: 30,
            width: 100,
            height: 110,
            pixels: vec![255; 100 * 110],
        };
        let (fitted, _, _) = glyph_mask(&mask, bbox, &page, GlyphProposal::Box);
        assert!(fitted.contains(88, 68) && fitted.contains(88, 94));
        // Far screentone dot inside the loose bbox must be dropped.
        assert!(
            !fitted.contains(48, 36),
            "stray screentone dot should be excluded"
        );

        // 2. Standalone 3-dot ellipsis block must keep all 3 dots.
        let ellipsis_page = RgbImage::from_fn(120, 120, |x, y| {
            let dot = (58..62).contains(&x)
                && ((36..40).contains(&y) || (52..56).contains(&y) || (68..72).contains(&y));
            Rgb(if dot { [15, 15, 15] } else { [250, 250, 250] })
        });
        let e_bbox = [50.0, 30.0, 70.0, 80.0];
        let e_mask = Mask {
            x: 50,
            y: 30,
            width: 20,
            height: 50,
            pixels: vec![255; 20 * 50],
        };
        let (e_fitted, _, _) = glyph_mask(&e_mask, e_bbox, &ellipsis_page, GlyphProposal::Box);
        assert!(
            e_fitted.contains(60, 38) && e_fitted.contains(60, 54) && e_fitted.contains(60, 70)
        );
    }
}
