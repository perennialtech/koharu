//! Page lettering adapts the room and size passes of upstream placement
//! (83cd4d96). Sibling sizes are matched before clearance so the final
//! collision checks use the lines the prepared frame will draw.

use super::*;
use crate::{HyphenationPolicy, TextLayout, fonts::Font};
use vello::kurbo::Point;

const MARGINS: [f32; 3] = [1.0, 0.875, 0.75];
const NEAR: f32 = 0.9;
const ROOMY: f32 = 1.15;
const ALIKE: f32 = 0.85;

/// Axial bounds reject distant obstacles before clipping their actual polygons.
pub(super) struct Polygon<P> {
    points: P,
    bounds: Rect,
}

type InkBox = Polygon<[Point; 4]>;
pub(super) type BalloonOutline = Polygon<Vec<Point>>;

impl<P: AsRef<[Point]>> Polygon<P> {
    fn new(points: P) -> Self {
        let vertices = points.as_ref();
        let bounds = vertices
            .iter()
            .fold(Rect::from_points(vertices[0], vertices[0]), |r, &p| {
                r.union_pt(p)
            });
        Self { points, bounds }
    }
}

impl BalloonOutline {
    pub(super) fn from_geometry(geometry: &Geometry) -> Self {
        Self::new(
            geometry
                .points
                .iter()
                .map(|p| Point::new(p.x, p.y))
                .collect(),
        )
    }
}

impl InkBox {
    fn from_rect(rect: Rect, onto: Affine) -> Self {
        Self::new(
            [
                Point::new(rect.x0, rect.y0),
                Point::new(rect.x1, rect.y0),
                Point::new(rect.x1, rect.y1),
                Point::new(rect.x0, rect.y1),
            ]
            .map(|p| onto * p),
        )
    }

    fn area(&self) -> f64 {
        let [a, b, _, d] = self.points;
        (b - a).cross(d - a)
    }

    /// Clip the obstacle against the four inward half-planes of the ink box.
    /// Signed area also counts disconnected pieces of a clipped concave wall:
    /// the connecting edges along the clip boundary cancel each other.
    fn overlap(&self, obstacle: &Polygon<impl AsRef<[Point]>>) -> f64 {
        if self.bounds.intersect(obstacle.bounds).is_zero_area() {
            return 0.0;
        }
        let mut polygon = obstacle.points.as_ref().to_vec();
        let mut clipped = Vec::with_capacity(polygon.len() + 4);
        for (a, b) in self.points.iter().zip(self.points.iter().cycle().skip(1)) {
            let Some(mut prev) = polygon.last().copied() else {
                return 0.0;
            };
            let side = |p: Point| (*b - *a).cross(p - *a);
            let mut before = side(prev);
            for &point in &polygon {
                let after = side(point);
                if (before < 0.0) != (after < 0.0) {
                    clipped.push(prev.lerp(point, before / (before - after)));
                }
                if after >= 0.0 {
                    clipped.push(point);
                }
                prev = point;
                before = after;
            }
            std::mem::swap(&mut polygon, &mut clipped);
            clipped.clear();
        }
        polygon
            .windows(2)
            .map(|edge| (edge[0] - polygon[0]).cross(edge[1] - polygon[0]))
            .sum::<f64>()
            .abs()
            * 0.5
    }
}

struct Measured {
    setting: TextSetting,
    size: f32,
    overflow: bool,
    empty: bool,
    ink: Vec<InkBox>,
}

struct Lettering {
    index: usize,
    fonts: Vec<Font>,
    measured: Measured,
    floor: Option<f32>,
}

fn measure(
    text: &TextNodeDescriptor,
    frame: GeometryFrame,
    fonts: &[Font],
    setting: TextSetting,
) -> Result<Measured> {
    let positioned = text.position(fonts, setting)?;
    let run = positioned.run;
    let onto = text_placement(frame)
        * Affine::translate((f64::from(positioned.x), f64::from(positioned.y)));
    let air = f64::from(run.font_size) / 8.0;
    let ink = run
        .lines
        .iter()
        .filter_map(|line| {
            let (x0, y0, x1, y1) =
                TextLayout::ink_bounds(run.font_size, std::slice::from_ref(line))?;
            Some(InkBox::from_rect(
                Rect::new(x0.into(), y0.into(), x1.into(), y1.into()).inflate(air, air),
                onto,
            ))
        })
        .collect();
    let empty = if text.writing_mode.is_vertical() {
        run.width < text.width * 0.5
    } else {
        run.height < text.height * 0.5
    };
    Ok(Measured {
        setting: TextSetting {
            size: Some(run.font_size),
            exact: true,
            breaking: run.breaking,
            wrap_width: run.wrap_width,
            ..setting
        },
        size: run.font_size,
        overflow: !run.fits()
            || (text.auto_fit
                && text
                    .source_area
                    .is_some_and(|area| run.width * run.height > area)),
        empty,
        ink,
    })
}

impl FlowShare {
    fn geometry(&self, reach: f32, origin: Origin) -> (Geometry, GeometryFrame) {
        let points = expanded_cell(&self.wall, &self.cell, reach);
        let geometry = page_polygon(self.frame, &points, origin);
        let frame = geometry_frame(&geometry, Some(self.frame.angle_degrees))
            .expect("a flow cell has positive extent");
        (geometry, frame)
    }
}

fn set_room(text: &mut TextNodeDescriptor, geometry: &Geometry, frame: GeometryFrame) {
    text.width = frame.bounds.width;
    text.height = frame.bounds.height;
    text.balloon_contour = Some(contour(geometry, frame));
}

/// Reuse the coordinated rooms as well as their glyphs for image-only edits.
/// Both sides matter: removing a text can change a size or clear an obstacle.
pub(super) fn reuse(
    page: &mut CompiledPage,
    previous: &Frame,
    affected: &AffectedDependencies,
) -> bool {
    let current: Vec<_> = page
        .layers
        .iter_mut()
        .filter(|l| matches!(l.descriptor, NodeDescriptor::Text(_)))
        .collect();
    let old: Vec<_> = previous
        .layers()
        .iter()
        .filter(|l| matches!(l.kind(), LayerKind::Text(_)))
        .collect();
    if previous.size() != (page.width, page.height)
        || current.len() != old.len()
        || current
            .first()
            .zip(old.first())
            .is_some_and(|(layer, old)| {
                // Moving a subtree can add or remove balloon outlines without
                // editing any component of the balloons themselves.
                layer.dependencies.as_ref() != old.dependencies()
                    || affected.intersects(&layer.dependencies)
            })
        || current.iter().zip(&old).any(|(layer, old)| {
            let (NodeDescriptor::Text(new), NodeDescriptor::Text(before)) =
                (&layer.descriptor, &old.0.node.descriptor)
            else {
                unreachable!()
            };
            layer.entity != old.entity() || new.font_families != before.font_families
        })
    {
        return false;
    }
    for (layer, old) in current.into_iter().zip(old) {
        let (_, intent) = inputs(layer);
        if !intent.authored_frame && (intent.share.is_some() || intent.original_room.is_some()) {
            layer.frame = geometry_frame(old.geometry(), Some(layer.frame.angle_degrees))
                .expect("a retained share has a frame");
            layer.geometry = old.geometry().clone();
        }
        layer.descriptor = old.0.node.descriptor.clone();
    }
    true
}

pub(super) fn coordinate(mut page: CompiledPage, fonts: &Fonts) -> Result<CompiledPage> {
    let mut jobs = page
        .layers
        .par_iter()
        .enumerate()
        .filter_map(|(index, layer)| {
            let (NodeDescriptor::Text(text), DraftMetadata::Text { lettering, .. }) =
                (&layer.descriptor, &layer.metadata)
            else {
                return None;
            };
            Some((|| {
                let fonts = text.resolve_fonts(fonts)?;
                let setting = TextSetting {
                    size: (text.auto_fit && !text.point_text && text.balloon_contour.is_none())
                        .then_some(text.width.min(text.height)),
                    room: if text.auto_fit {
                        None
                    } else {
                        lettering.original_room
                    },
                    breaking: HyphenationPolicy::Disabled,
                    ..Default::default()
                };
                // A persisted drag contains the already expanded share. Measure the
                // original split once, as for its siblings, to avoid feeding that
                // expansion back into the next page median.
                let measured = if let Some(share) = lettering
                    .share
                    .as_ref()
                    .filter(|_| lettering.authored_frame && text.auto_fit)
                {
                    let mut nominal = text.clone();
                    let (geometry, frame) = share.geometry(0.0, layer.geometry.origin.clone());
                    set_room(&mut nominal, &geometry, frame);
                    measure(&nominal, frame, &fonts, setting)?
                } else {
                    measure(text, layer.frame, &fonts, setting)?
                };
                Ok(Lettering {
                    index,
                    fonts,
                    measured,
                    floor: None,
                })
            })())
        })
        .collect::<Result<Vec<_>>>()?;

    let page_size = dialogue_size(&jobs, &page);

    for job in &jobs {
        let layer = &mut page.layers[job.index];
        let DraftMetadata::Text { lettering, .. } = &layer.metadata else {
            unreachable!()
        };
        if let (Some(share), Some(size)) = (&lettering.share, page_size)
            && !lettering.authored_frame
            && let NodeDescriptor::Text(text) = &mut layer.descriptor
        {
            let (geometry, frame) = share.geometry(size * 0.5, layer.geometry.origin.clone());
            set_room(text, &geometry, frame);
            layer.geometry = geometry;
            layer.frame = frame;
        }
    }
    jobs.par_iter_mut().try_for_each(|job| {
        let layer = &page.layers[job.index];
        let (text, intent) = inputs(layer);
        let at = |setting| measure(text, layer.frame, &job.fonts, setting);
        if !text.auto_fit || text.point_text {
            job.measured = at(TextSetting {
                room: intent.original_room,
                ..Default::default()
            })?;
        } else if text.balloon_contour.is_some() {
            let target = page_size.unwrap_or(job.measured.size);
            (job.measured, job.floor) = balloon(text, target, at)?;
        } else {
            job.measured = free(text, intent, page_size, at)?;
            // The fork's readable floor also applies during collision solving.
            job.floor =
                Some((job.measured.size * 0.5).max(text.minimum_font_size.min(job.measured.size)));
        }
        Ok::<_, Error>(())
    })?;
    let groups = together(&mut jobs, &page)?;
    clear(&mut jobs, &page, groups)?;
    for job in jobs {
        let layer = &mut page.layers[job.index];
        let NodeDescriptor::Text(text) = &mut layer.descriptor else {
            unreachable!()
        };
        text.setting = job.measured.setting;
        if let Some(room) = text.setting.room.take() {
            layer.geometry =
                page_polygon(layer.frame, &room.corners(), layer.geometry.origin.clone());
            layer.frame = geometry_frame(&layer.geometry, Some(layer.frame.angle_degrees))
                .ok_or_else(|| Error::invalid("free text room has no extent"))?;
            text.width = room.width;
            text.height = room.height;
        }
    }
    Ok(page)
}

fn visible(layer: &LayerDraft) -> bool {
    let (text, _) = inputs(layer);
    layer.presentation.visible
        && layer.presentation.opacity > 0.0
        && (text.foreground_color[3] > 0
            || text
                .stroke
                .is_some_and(|stroke| stroke.width_px > 0.0 && stroke.color[3] > 0))
}

fn dialogue_size(jobs: &[Lettering], page: &CompiledPage) -> Option<f32> {
    let candidates = jobs.iter().filter(|j| {
        let layer = &page.layers[j.index];
        let (text, intent) = inputs(layer);
        visible(layer)
            && !text.point_text
            && !intent.sound
            && !j.measured.ink.is_empty()
            && (text.auto_fit || !j.measured.overflow)
    });
    let balloons = candidates
        .clone()
        .any(|j| inputs(&page.layers[j.index]).1.balloon.is_some());
    let mut sizes: Vec<_> = candidates
        .filter(|j| !balloons || inputs(&page.layers[j.index]).1.balloon.is_some())
        .map(|j| j.measured.size)
        .collect();
    sizes.sort_by(f32::total_cmp);
    sizes.get(sizes.len() / 2).copied()
}

fn inputs(layer: &LayerDraft) -> (&TextNodeDescriptor, &LetteringIntent) {
    let (NodeDescriptor::Text(text), DraftMetadata::Text { lettering, .. }) =
        (&layer.descriptor, &layer.metadata)
    else {
        unreachable!()
    };
    (text, lettering)
}

fn balloon(
    text: &TextNodeDescriptor,
    page: f32,
    at: impl Fn(TextSetting) -> Result<Measured>,
) -> Result<(Measured, Option<f32>)> {
    let mut whole = None;
    for margin in MARGINS {
        let trial = at(TextSetting {
            margin,
            breaking: HyphenationPolicy::Disabled,
            ..Default::default()
        })?;
        let roomy = trial.size >= ROOMY * page && !trial.overflow;
        if whole
            .as_ref()
            .is_none_or(|best: &Measured| best.overflow || trial.size >= best.size)
        {
            whole = Some(trial);
        }
        if roomy {
            break;
        }
    }
    let mut best = whole.expect("there is a default margin");
    if best.size < NEAR * page || best.overflow {
        // Only once every whole-word room has been tested may a dictionary
        // break buy a tenth in size. Prefer the widest margin that reaches it.
        let words = best.size;
        for margin in MARGINS {
            let trial = at(TextSetting {
                size: Some(page),
                margin,
                ..Default::default()
            })?;
            if !trial.overflow
                && (best.overflow || (trial.size >= 1.1 * words && trial.size > best.size))
            {
                best = trial;
                if best.size >= 0.999 * page {
                    break;
                }
            }
        }
    }
    let held = best.size;
    let mut floor = None;
    if best.overflow || best.size < NEAR * page {
        best = at(TextSetting {
            size: Some(page),
            exact: true,
            margin: 0.75,
            ..Default::default()
        })?;
        floor = Some(held.min(page));
    }
    if let Some(source) = text.source_font_size.filter(|size| *size > best.size)
        && best.empty
    {
        for margin in MARGINS {
            let source_fit = at(TextSetting {
                size: Some(source),
                exact: true,
                margin,
                breaking: HyphenationPolicy::Disabled,
                ..Default::default()
            })?;
            if !source_fit.overflow {
                return Ok((source_fit, Some(best.size)));
            }
        }
        let mut low = best.size;
        let mut high = source;
        let setting = TextSetting {
            exact: true,
            margin: 0.75,
            ..Default::default()
        };
        let mut grown = at(TextSetting {
            size: Some(high),
            ..setting
        })?;
        if !grown.empty {
            for _ in 0..8 {
                let middle = (low + high) * 0.5;
                let trial = at(TextSetting {
                    size: Some(middle),
                    ..setting
                })?;
                if trial.empty {
                    low = middle;
                } else {
                    high = middle;
                    grown = trial;
                }
            }
        }
        floor = Some(floor.unwrap_or(best.size));
        best = grown;
    }
    Ok((best, floor))
}

fn free(
    text: &TextNodeDescriptor,
    intent: &LetteringIntent,
    page: Option<f32>,
    at: impl Fn(TextSetting) -> Result<Measured>,
) -> Result<Measured> {
    let source = text.source_font_size;
    let size = if intent.sound {
        page.into_iter().chain(source).reduce(f32::min)
    } else {
        page.into_iter().chain(source).reduce(f32::max)
    };
    let setting = TextSetting {
        size,
        ..Default::default()
    };
    let Some(own) = intent.original_room else {
        return at(setting);
    };
    if text.writing_mode.is_vertical() {
        return at(TextSetting {
            room: Some(own),
            ..setting
        });
    }
    let widest = LayoutBox {
        x: 0.0,
        y: 0.0,
        width: text.width,
        height: text.height,
    };
    let mut rooms = Vec::new();
    // Widen along the lines first. Across-line growth is reserved for a
    // short source frame that cannot carry the translation in one row.
    for across in 0..=4 {
        for along in 0..=16 {
            let x = along as f32 / 16.0;
            let y = across as f32 / 4.0;
            let room = LayoutBox {
                x: own.x * (1.0 - x),
                y: own.y * (1.0 - y),
                width: own.width + (widest.width - own.width) * x,
                height: own.height + (widest.height - own.height) * y,
            };
            if rooms.last() != Some(&room) && (along == 0 || own.width != widest.width) {
                rooms.push(room);
            }
        }
        if own.height == widest.height {
            break;
        }
    }
    let target = size.unwrap_or(own.width.min(own.height));
    let clean = TextSetting {
        breaking: HyphenationPolicy::Disabled,
        ..setting
    };
    for &room in &rooms {
        let trial = at(TextSetting {
            size: Some(target),
            exact: true,
            room: Some(room),
            ..clean
        })?;
        if !trial.overflow {
            return Ok(trial);
        }
    }
    let mut held = at(TextSetting {
        room: Some(widest),
        ..clean
    })?;
    if held.size < NEAR * target || held.overflow {
        // The widest frame contains every smaller candidate. Compare its
        // best unbroken setting before considering any dictionary breaks.
        let broken = at(TextSetting {
            room: Some(widest),
            ..setting
        })?;
        if !broken.overflow && (held.overflow || broken.size >= 1.1 * held.size) {
            held = broken;
        }
    }
    // Search all allowed frames at the winning size. Each trial only shapes
    // that em; it need not repeat the full size search for every frame.
    for room in rooms {
        let trial = at(TextSetting {
            size: Some(held.size),
            exact: true,
            breaking: held.setting.breaking,
            room: Some(room),
            ..setting
        })?;
        if !trial.overflow {
            return Ok(trial);
        }
    }
    Ok(held)
}

fn coverage<'a>(
    mut trials: impl Iterator<Item = &'a Measured> + Clone,
    sheet: &InkBox,
    obstacles: &[&InkBox],
    walls: &[&BalloonOutline],
) -> f64 {
    let mut score = 0.0;
    while let Some(trial) = trials.next() {
        for line in &trial.ink {
            if !sheet.bounds.contains_rect(line.bounds) {
                score += (line.area() - line.overlap(sheet)).max(0.0);
            }
            let siblings = trials.clone().flat_map(|next| &next.ink);
            score += obstacles
                .iter()
                .copied()
                .chain(siblings)
                .map(|other| line.overlap(other))
                .sum::<f64>();
            score += walls.iter().map(|wall| line.overlap(wall)).sum::<f64>();
        }
    }
    score
}

fn clear(jobs: &mut [Lettering], page: &CompiledPage, mut groups: Vec<Vec<usize>>) -> Result<()> {
    let sheet = InkBox::from_rect(
        Rect::new(0.0, 0.0, page.width.into(), page.height.into()),
        Affine::IDENTITY,
    );
    let rank = |j: &Lettering| {
        let (_, intent) = inputs(&page.layers[j.index]);
        if intent.sound {
            0
        } else if intent.balloon.is_none() {
            1
        } else {
            2
        }
    };
    groups.sort_by(|a, b| {
        rank(&jobs[a[0]]).cmp(&rank(&jobs[b[0]])).then_with(|| {
            let area = |group: &[usize]| {
                group
                    .iter()
                    .flat_map(|&i| &jobs[i].measured.ink)
                    .map(InkBox::area)
                    .sum::<f64>()
            };
            area(b).total_cmp(&area(a))
        })
    });
    for group in groups {
        let first = &jobs[group[0]];
        let (text, intent) = inputs(&page.layers[first.index]);
        let obstacles: Vec<_> = jobs
            .iter()
            .enumerate()
            .filter(|(i, j)| !group.contains(i) && visible(&page.layers[j.index]))
            .flat_map(|(_, j)| &j.measured.ink)
            .collect();
        let walls: Vec<_> = page
            .balloons
            .iter()
            .filter(|(id, _)| Some(*id) != intent.balloon)
            .map(|(_, wall)| wall)
            .collect();
        let mut score = coverage(
            group.iter().map(|&i| &jobs[i].measured),
            &sheet,
            &obstacles,
            &walls,
        );
        if score <= 0.000001 {
            continue;
        }
        let high = first.measured.size;
        let low = group
            .iter()
            .filter_map(|&i| jobs[i].floor)
            .reduce(f32::max)
            .unwrap_or(text.minimum_font_size)
            .min(high);
        let at = |em| -> Result<Vec<Measured>> {
            group
                .iter()
                .map(|&i| {
                    let job = &jobs[i];
                    let layer = &page.layers[job.index];
                    let (text, _) = inputs(layer);
                    measure(
                        text,
                        layer.frame,
                        &job.fonts,
                        TextSetting {
                            size: Some(em),
                            exact: text.balloon_contour.is_some(),
                            wrap_width: None,
                            ..job.measured.setting
                        },
                    )
                })
                .collect()
        };
        let cost = |trial: &[Measured]| coverage(trial.iter(), &sheet, &obstacles, &walls);
        let mut best = None;
        // Reflow can make a smaller run wider. Search down in eighths before
        // refining the first clear interval, retaining least coverage if none clears.
        for step in 1..=8 {
            let em = high - (high - low) * step as f32 / 8.0;
            let trial = at(em)?;
            let covered = cost(&trial);
            if covered < score {
                score = covered;
                best = Some(trial);
                if covered <= 0.000001 {
                    let (mut l, mut h) = (em, em + (high - low) / 8.0);
                    for _ in 0..8 {
                        let middle = (l + h) * 0.5;
                        let trial = at(middle)?;
                        if cost(&trial) <= 0.000001 {
                            l = middle;
                            best = Some(trial);
                        } else {
                            h = middle;
                        }
                    }
                    break;
                }
            }
        }
        if let Some(best) = best {
            for (i, measured) in group.into_iter().zip(best) {
                jobs[i].measured = measured;
            }
        }
    }
    Ok(())
}

/// Match visible siblings first. A size change can reflow several lines into
/// one wider line. Keep matched siblings together in all clearance trials.
fn together(jobs: &mut [Lettering], page: &CompiledPage) -> Result<Vec<Vec<usize>>> {
    let mut movable = Vec::new();
    let mut balloons = BTreeMap::<EntityId, Vec<(usize, f32)>>::new();
    for (i, job) in jobs.iter().enumerate() {
        let layer = &page.layers[job.index];
        if !visible(layer) {
            continue;
        }
        let (text, intent) = inputs(layer);
        if let Some(id) = intent.balloon
            && text.auto_fit
            && !intent.sound
        {
            balloons
                .entry(id)
                .or_default()
                .push((i, text.source_font_size.unwrap_or(text.minimum_font_size)));
        } else if job.floor.is_some() {
            movable.push(vec![i]);
        }
    }
    for members in balloons.values_mut() {
        members.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut rest = members.as_slice();
        while let Some(&(_, largest)) = rest.first() {
            let end = rest.partition_point(|&(_, source)| source >= largest * ALIKE);
            let (group, next) = rest.split_at(end);
            rest = next;
            let least = group
                .iter()
                .map(|&(i, _)| jobs[i].measured.size)
                .fold(f32::INFINITY, f32::min);
            if group
                .iter()
                .any(|&(i, _)| jobs[i].floor.is_some() || jobs[i].measured.size > least)
            {
                movable.push(group.iter().map(|&(i, _)| i).collect());
            }
            for &(i, _) in group {
                let job = &mut jobs[i];
                if job.measured.size > least {
                    let layer = &page.layers[job.index];
                    let (text, _) = inputs(layer);
                    // Matching below a lift's floor supersedes it for this sibling.
                    job.floor = job.floor.filter(|&floor| floor <= least);
                    job.measured = measure(
                        text,
                        layer.frame,
                        &job.fonts,
                        TextSetting {
                            size: Some(least),
                            exact: true,
                            wrap_width: None,
                            ..job.measured.setting
                        },
                    )?;
                }
            }
        }
    }
    Ok(movable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use koharu_scene::{At, Authored, Edit, PageDraft, Session};

    fn text_layer(
        edit: &mut Edit,
        page: EntityId,
        words: &str,
        em: f32,
        auto_fit: bool,
        geometry: Geometry,
        balloon: bool,
    ) -> koharu_scene::Result<(EntityId, EntityId)> {
        let content = edit.add_text_content(page, At::End)?;
        edit.set(
            content,
            &SourceText {
                text: Authored::user("声".into()),
                language: None,
            },
        )?;
        edit.set(
            content,
            &Translation {
                text: Authored::user(words.to_owned()),
                language: None,
            },
        )?;
        let layer = edit.add_text_layer(
            page,
            At::End,
            content,
            &SceneTextLayout {
                origin: Origin::User,
                kind: TextLayoutKind::Paragraph,
                angle_degrees: None,
            },
        )?;
        edit.set(
            layer,
            &Typography {
                origin: Origin::User,
                preferred_font: None,
                font_weight: None,
                font_style: None,
                size: Some(em),
                auto_fit,
                color: Some([160, 164, 158, 255]),
                stroke_color: Some([90, 92, 89, 255]),
                stroke_width: Some(20.0),
                alignment: None,
                writing_mode: None,
                extensions: BTreeMap::new(),
            },
        )?;
        if balloon {
            let bubble =
                edit.add_analysis_region::<BubbleRegion>(page, At::End, &geometry, None)?;
            edit.relate::<FlowsIn>(layer, bubble)?;
        } else {
            edit.set(layer, &geometry)?;
        }
        Ok((layer, content))
    }

    fn renderer() -> Renderer {
        let font = crate::fonts::FontSystem::new().first_font().unwrap();
        Renderer::with_typesetting(TypesettingConfig {
            font_families: vec![font.family_name().to_owned()],
        })
    }

    fn metadata(frame: &Frame, layer: EntityId) -> &TextMetadata {
        let LayerKind::Text(text) = frame.layer(layer).unwrap().kind() else {
            unreachable!()
        };
        text
    }

    fn committed_ink(renderer: &Renderer, frame: &Frame, id: EntityId) -> Result<Vec<InkBox>> {
        let layer = frame.layer(id).unwrap();
        let NodeDescriptor::Text(text) = &layer.0.node.descriptor else {
            unreachable!()
        };
        let placement =
            geometry_frame(layer.geometry(), Some(metadata(frame, id).angle_degrees)).unwrap();
        let fonts = text.resolve_fonts(&renderer.inner.fonts)?;
        Ok(measure(text, placement, &fonts, text.setting)?.ink)
    }

    #[tokio::test]
    async fn parallel_tilted_captions_clear_one_another_at_their_original_size()
    -> anyhow::Result<()> {
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("sheet", 1000.0, 800.0), At::End)?;
        text_layer(
            &mut edit,
            page,
            "Yes",
            32.0,
            false,
            Geometry::rectangle(50.0, 30.0, 180.0, 120.0),
            true,
        )?;
        let tilt = Affine::translate((350.0, 200.0)) * Affine::rotate(30_f64.to_radians());
        let mut add = |offset, auto_fit| {
            let mut geometry = Geometry::rectangle(0.0, offset, 300.0, 90.0);
            for corner in &mut geometry.points {
                let p = tilt * Point::new(corner.x, corner.y);
                (corner.x, corner.y) = (p.x, p.y);
            }
            text_layer(
                &mut edit,
                page,
                "WAVE WAVE",
                32.0,
                auto_fit,
                geometry,
                false,
            )
        };
        let (free, _) = add(0.0, true)?;
        let (fixed, _) = add(48.0, false)?;
        let snapshot = session.commit(edit.finish()?).await?.snapshot;
        let renderer = renderer();
        let frame = renderer.render(&snapshot, page).await?;
        assert_eq!(metadata(&frame, free).font_size, 32.0);
        let free = committed_ink(&renderer, &frame, free)?;
        let fixed = committed_ink(&renderer, &frame, fixed)?;
        assert_eq!((free.len(), fixed.len()), (1, 1));
        assert!(free[0].bounds.intersect(fixed[0].bounds).area() > 0.0);
        assert_eq!(free[0].overlap(&fixed[0]), 0.0);
        Ok(())
    }

    #[test]
    fn clipped_area_preserves_tilt_and_concave_balloon_margins() {
        let onto = Affine::translate((600.0, 700.0)) * Affine::rotate(30_f64.to_radians());
        let line = InkBox::from_rect(Rect::new(0.0, 0.0, 10.0, 4.0), onto);
        let neighbour = InkBox::from_rect(Rect::new(5.0, 1.0, 15.0, 5.0), onto);
        assert!((line.overlap(&neighbour) - 15.0).abs() < 1e-9);
        // The notch cuts out the centre of the line and leaves two disjoint pieces.
        let wall = BalloonOutline::new(
            [
                (2.0, -1.0),
                (2.0, 5.0),
                (4.0, 5.0),
                (4.0, -0.5),
                (6.0, -0.5),
                (6.0, 5.0),
                (8.0, 5.0),
                (8.0, -1.0),
            ]
            .map(|(x, y)| onto * Point::new(x, y))
            .to_vec(),
        );
        assert!((line.overlap(&wall) - 16.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn matching_siblings_clears_a_caption_beside_the_reflowed_line() -> anyhow::Result<()> {
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("sheet", 1200.0, 800.0), At::End)?;
        let bubble = edit.add_analysis_region::<BubbleRegion>(
            page,
            At::End,
            &Geometry::rectangle(120.0, 180.0, 680.0, 250.0),
            None,
        )?;
        let (first, _) = text_layer(
            &mut edit,
            page,
            "WAVE WAVE",
            4.0,
            true,
            Geometry::rectangle(160.0, 200.0, 240.0, 190.0),
            false,
        )?;
        let (second, _) = text_layer(
            &mut edit,
            page,
            "WAVE",
            12.0,
            true,
            Geometry::rectangle(540.0, 250.0, 125.0, 80.0),
            false,
        )?;
        edit.relate::<FlowsIn>(first, bubble)?;
        edit.relate::<FlowsIn>(second, bubble)?;
        for x in [40.0, 340.0, 640.0] {
            text_layer(
                &mut edit,
                page,
                "Hi",
                16.0,
                false,
                Geometry::rectangle(x, 520.0, 220.0, 120.0),
                true,
            )?;
        }
        let snapshot = session.commit(edit.finish()?).await?.snapshot;
        let renderer = renderer();
        let separate = renderer.render(&snapshot, page).await?;
        let two = committed_ink(&renderer, &separate, first)?;
        assert_eq!(two.len(), 2);
        let mut style = snapshot.component::<Typography>(second)?.unwrap();
        style.size = Some(4.0);
        let matched = session
            .commit(snapshot.patch(|edit| edit.set(second, &style))?)
            .await?;
        let frame = renderer
            .update(&separate, &matched.snapshot, &matched.changes)
            .await?;
        let one = committed_ink(&renderer, &frame, first)?;
        assert_eq!(one.len(), 1);
        assert!(two.iter().all(|line| line.bounds.x1 < one[0].bounds.x1));
        let mut edit = matched.snapshot.edit();
        let (caption, _) = text_layer(
            &mut edit,
            page,
            "I",
            16.0,
            false,
            Geometry::rectangle(
                one[0].bounds.x1 - 12.0,
                one[0].bounds.center().y - 20.0,
                20.0,
                40.0,
            ),
            false,
        )?;
        let change = session.commit(edit.finish()?).await?;
        let cleared = renderer
            .update(&frame, &change.snapshot, &change.changes)
            .await?;
        let obstacle = committed_ink(&renderer, &cleared, caption)?;
        assert!(two.iter().all(|line| line.overlap(&obstacle[0]) == 0.0));
        assert!(one[0].overlap(&obstacle[0]) > 0.0);
        let final_lines = committed_ink(&renderer, &cleared, first)?;
        assert!(
            final_lines
                .iter()
                .all(|line| line.overlap(&obstacle[0]) == 0.0)
        );
        assert!(metadata(&cleared, first).font_size < metadata(&frame, first).font_size);
        assert_eq!(
            metadata(&cleared, first).font_size,
            metadata(&cleared, second).font_size
        );
        Ok(())
    }

    async fn clear_siblings_against_caption(
        siblings: [(f32, Option<f32>); 2],
        clear_size: f32,
    ) -> anyhow::Result<(f32, f64)> {
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("sheet", 900.0, 600.0), At::End)?;
        let bubble = edit.add_analysis_region::<BubbleRegion>(
            page,
            At::End,
            &Geometry::rectangle(40.0, 40.0, 800.0, 320.0),
            None,
        )?;
        for (words, x, sibling) in [
            ("WAVE", 80.0, true),
            ("Hi", 480.0, true),
            ("I", 80.0, false),
        ] {
            let (id, _) = text_layer(
                &mut edit,
                page,
                words,
                48.0,
                sibling,
                Geometry::rectangle(x, 100.0, 260.0, 160.0),
                false,
            )?;
            if sibling {
                edit.relate::<FlowsIn>(id, bubble)?;
            }
        }
        let snapshot = session.commit(edit.finish()?).await?.snapshot;
        let renderer = renderer();
        let mut compiled = renderer.compile(&snapshot, page)?;
        renderer
            .inner
            .fonts
            .prepare(&font_requests(&compiled.layers))
            .await?;
        let mut jobs = Vec::new();
        for (index, (em, floor)) in siblings.into_iter().chain([(16.0, None)]).enumerate() {
            let layer = &compiled.layers[index];
            let (text, _) = inputs(layer);
            let fonts = text.resolve_fonts(&renderer.inner.fonts)?;
            let measured = measure(
                text,
                layer.frame,
                &fonts,
                TextSetting {
                    size: Some(em),
                    exact: true,
                    ..Default::default()
                },
            )?;
            jobs.push(Lettering {
                index,
                fonts,
                measured,
                floor,
            });
        }
        let groups = together(&mut jobs, &compiled)?;
        assert_eq!(jobs[0].measured.size, siblings[0].0.min(siblings[1].0));
        assert_eq!(jobs[0].measured.size, jobs[1].measured.size);
        let first = &compiled.layers[0];
        let fit = measure(
            inputs(first).0,
            first.frame,
            &jobs[0].fonts,
            TextSetting {
                size: Some(clear_size),
                ..jobs[0].measured.setting
            },
        )?;
        // Place the caption just beyond the line's right edge at the clear size.
        let edge = fit.ink[0].bounds;
        let caption = &mut compiled.layers[2];
        let bounds = jobs[2].measured.ink[0].bounds;
        caption.frame.bounds.x += (edge.x1 + 0.001 - bounds.x0) as f32;
        caption.frame.bounds.y += (edge.center().y - bounds.center().y) as f32;
        jobs[2].measured = measure(
            inputs(caption).0,
            caption.frame,
            &jobs[2].fonts,
            jobs[2].measured.setting,
        )?;
        assert!(jobs[0].measured.ink[0].overlap(&jobs[2].measured.ink[0]) > 0.0);
        assert_eq!(fit.ink[0].overlap(&jobs[2].measured.ink[0]), 0.0);

        clear(&mut jobs, &compiled, groups)?;
        assert_eq!(jobs[0].measured.size, jobs[1].measured.size);
        Ok((
            jobs[0].measured.size,
            jobs[0].measured.ink[0].overlap(&jobs[2].measured.ink[0]),
        ))
    }

    #[tokio::test]
    async fn matched_siblings_can_clear_a_caption_below_a_pre_match_floor() -> anyhow::Result<()> {
        let (size, overlap) =
            clear_siblings_against_caption([(48.0, Some(30.0)), (24.0, None)], 23.0).await?;
        assert!((size - 23.0).abs() < 0.1);
        assert_eq!(overlap, 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn matched_siblings_stop_at_the_highest_active_lift_floor() -> anyhow::Result<()> {
        let (size, overlap) =
            clear_siblings_against_caption([(48.0, Some(36.0)), (48.0, Some(18.0))], 35.0).await?;
        assert_eq!(size, 36.0);
        assert!(overlap > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn matching_below_one_lift_floor_preserves_the_other() -> anyhow::Result<()> {
        let (size, overlap) =
            clear_siblings_against_caption([(48.0, Some(36.0)), (30.0, Some(18.0))], 17.0).await?;
        assert_eq!(size, 18.0);
        assert!(overlap > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn matching_exactly_at_a_lift_floor_preserves_it() -> anyhow::Result<()> {
        let (size, overlap) =
            clear_siblings_against_caption([(48.0, Some(36.0)), (36.0, None)], 35.0).await?;
        assert_eq!(size, 36.0);
        assert!(overlap > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn a_smaller_margin_keeps_whole_words_at_the_page_size() {
        let mut session = Session::memory().await.unwrap();
        let mut page = None;
        let create = session
            .snapshot()
            .patch(|edit| {
                let p = edit.add_page(PageDraft::new("sheet", 600.0, 600.0), At::End)?;
                text_layer(
                    edit,
                    p,
                    "extraordinary",
                    4.0,
                    true,
                    Geometry::rectangle(20.0, 20.0, 200.0, 400.0),
                    true,
                )?;
                page = Some(p);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(create).await.unwrap().snapshot;
        let renderer = renderer();
        let compiled = renderer.compile(&snapshot, page.unwrap()).unwrap();
        renderer
            .inner
            .fonts
            .prepare(&font_requests(&compiled.layers))
            .await
            .unwrap();
        let layer = &compiled.layers[0];
        let (text, _) = inputs(layer);
        let fonts = text.resolve_fonts(&renderer.inner.fonts).unwrap();
        let at = |setting| measure(text, layer.frame, &fonts, setting);
        let capacities = MARGINS.map(|margin| {
            at(TextSetting {
                margin,
                breaking: HyphenationPolicy::Disabled,
                ..Default::default()
            })
            .unwrap()
            .size
        });
        assert!(capacities[0] < capacities[1] && capacities[1] < capacities[2]);
        for (page, margin) in [
            (capacities[0] / ROOMY * 0.99, 1.0),
            ((capacities[0] + capacities[1]) / (2.0 * ROOMY), 0.875),
            ((capacities[1] + capacities[2]) * 0.5, 0.75),
        ] {
            let (fit, floor) = balloon(text, page, at).unwrap();
            assert_eq!(fit.setting.margin, margin);
            assert_eq!(fit.setting.breaking, HyphenationPolicy::Disabled);
            assert!(fit.size >= page && floor.is_none());
        }
        let target = capacities[2] * 1.25;
        let (fit, _) = balloon(text, target, at).unwrap();
        assert!(fit.size > capacities[2] * 1.1 && !fit.overflow);
        assert_ne!(fit.setting.breaking, HyphenationPolicy::Disabled);
        let mut measured_source = text.clone();
        measured_source.source_font_size = Some(capacities[2] * 2.5);
        let at = |setting| measure(&measured_source, layer.frame, &fonts, setting);
        let (grown, floor) = balloon(&measured_source, capacities[2], at).unwrap();
        assert!(grown.size > capacities[2] * 1.2 && grown.size <= capacities[2] * 2.5);
        assert!(floor.is_some());
    }

    fn sound_layer(
        edit: &mut Edit,
        page: EntityId,
        em: f32,
        source: Geometry,
    ) -> koharu_scene::Result<EntityId> {
        let (layer, content) = text_layer(edit, page, "Bam", em, true, source.clone(), false)?;
        edit.remove::<Geometry>(layer)?;
        let region = edit.add_analysis_region::<TextRegion>(
            page,
            At::End,
            &source,
            Some("onomatopoeia".into()),
        )?;
        edit.relate::<RecognizedFrom>(content, region)?;
        Ok(layer)
    }

    #[tokio::test]
    async fn source_punctuation_uses_its_run_extent_or_the_page_minimum() -> anyhow::Result<()> {
        use TextDirection::{Horizontal, Vertical};
        use koharu_scene::{Generation, ProducerId};

        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("sheet", 3240.0, 3240.0), At::End)?;
        let snapshot = session.commit(edit.finish()?).await?.snapshot;
        let generation = Generation::new(ProducerId::new("dev.koharu.test")?);
        let mut edit = snapshot.edit_as(generation);
        let mut expected = Vec::new();
        let mut user_sized_layer = None;
        for (index, (source, direction, width, height, measured, em, user_size)) in [
            ("…", Vertical, 3.0, 26.0, 3.0, 26.0, false),
            ("……", Vertical, 4.0, 48.0, 4.0, 24.0, false),
            ("・・・", Vertical, 5.0, 66.0, 5.0, 22.0, false),
            ("・", Vertical, 5.0, 5.0, 5.0, 18.0, false),
            ("...", Horizontal, 18.0, 3.0, 3.0, 24.0, false),
            ("…！", Vertical, 8.0, 60.0, 8.0, 30.0, false),
            ("！", Vertical, 65.0, 130.0, 65.0, 65.0, false),
            ("!", Vertical, 65.0, 130.0, 65.0, 65.0, false),
            ("!!\n!!", Horizontal, 60.0, 100.0, 50.0, 50.0, false),
            ("待って…", Vertical, 24.0, 100.0, 22.0, 22.0, false),
            ("Hey!", Horizontal, 92.0, 34.0, 28.0, 28.0, false),
            ("小声", Vertical, 8.0, 16.0, 6.0, 6.0, false),
            ("……", Vertical, 4.0, 48.0, 4.0, 4.0, true),
        ]
        .into_iter()
        .enumerate()
        {
            let x = 12.0 + 150.0 * index as f64;
            let (layer, content) = text_layer(
                &mut edit,
                page,
                "Ah",
                measured,
                false,
                Geometry::rectangle(x, 12.0, 130.0, 110.0),
                true,
            )?;
            edit.set(
                content,
                &SourceText {
                    text: Authored::user(source.into()),
                    language: None,
                },
            )?;
            let region = edit.add_analysis_region::<TextRegion>(
                page,
                At::End,
                &Geometry::rectangle(x + 15.0, 27.0, width, height),
                None,
            )?;
            edit.set(
                region,
                &OcrAnalysis {
                    origin: Origin::User,
                    direction,
                    confidence: None,
                    line_boundaries: Vec::new(),
                },
            )?;
            edit.relate::<RecognizedFrom>(content, region)?;
            if user_size {
                user_sized_layer = Some(layer);
            }
            expected.push((layer, source, em));
        }
        let snapshot = session.commit(edit.finish()?).await?.snapshot;
        let user_sized_layer = user_sized_layer.unwrap();
        let mut edit = snapshot.edit();
        edit.set(
            user_sized_layer,
            &snapshot.component::<Typography>(user_sized_layer)?.unwrap(),
        )?;
        let snapshot = session.commit(edit.finish()?).await?.snapshot;
        let frame = renderer().render(&snapshot, page).await?;
        for (layer, source, em) in expected {
            let prepared = frame.layer(layer).unwrap();
            let NodeDescriptor::Text(text) = &prepared.0.node.descriptor else {
                panic!("expected text layer");
            };
            assert_eq!(text.source_font_size, Some(em), "{source}");
            assert!(
                (metadata(&frame, layer).font_size - em).abs() < 0.001,
                "{source}"
            );
        }

        Ok(())
    }

    #[test]
    fn padded_vertical_ellipsis_recovers_its_source_em() {
        let bounds = LayoutBox {
            x: 0.0,
            y: 0.0,
            width: 17.0,
            height: 33.0,
        };
        let page_minimum = readable_font_size(1800, 3240);

        assert_eq!(
            punctuation_source_em(
                "…",
                bounds,
                TextDirection::Vertical,
                page_minimum,
                Some(5.0)
            ),
            Some(33.0)
        );
    }

    #[test]
    fn punctuation_advance_uses_the_longest_line_in_each_direction() {
        for (source, direction, width, height, em) in [
            ("!?\n!", TextDirection::Vertical, 65.0, 130.0, 65.0),
            ("!!\n!!", TextDirection::Horizontal, 60.0, 100.0, 60.0),
        ] {
            let bounds = LayoutBox {
                x: 0.0,
                y: 0.0,
                width,
                height,
            };
            assert_eq!(
                punctuation_source_em(source, bounds, direction, 18.0, None),
                Some(em)
            );
        }
    }

    #[tokio::test]
    async fn a_free_sound_uses_only_the_extra_room_its_measured_source_em_needs()
    -> anyhow::Result<()> {
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("sheet", 600.0, 600.0), At::End)?;
        let layer = sound_layer(
            &mut edit,
            page,
            28.0,
            Geometry::rectangle(240.0, 200.0, 36.0, 100.0),
        )?;
        let snapshot = session.commit(edit.finish()?).await?.snapshot;
        let frame = renderer().render(&snapshot, page).await?;
        let placed = geometry_bounds(frame.layer(layer).unwrap().geometry()).unwrap();
        assert!(placed.width > 36.0 && placed.width < 90.0);
        assert_eq!(metadata(&frame, layer).font_size, 28.0);
        Ok(())
    }

    #[tokio::test]
    async fn a_tall_free_source_offers_at_most_two_and_a_half_times_its_width() -> anyhow::Result<()>
    {
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("sheet", 600.0, 600.0), At::End)?;
        let layer = sound_layer(
            &mut edit,
            page,
            28.0,
            Geometry::rectangle(240.0, 200.0, 20.0, 200.0),
        )?;
        let snapshot = session.commit(edit.finish()?).await?.snapshot;
        let compiled = renderer().compile(&snapshot, page)?;
        let (text, intent) = inputs(compiled.layers.iter().find(|l| l.entity == layer).unwrap());
        assert_eq!(intent.original_room.unwrap().width, 20.0);
        assert_eq!((text.width, text.height), (50.0, 200.0));
        Ok(())
    }

    #[tokio::test]
    async fn a_sound_that_fits_at_dialogue_size_keeps_its_source_frame() -> anyhow::Result<()> {
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("sheet", 900.0, 600.0), At::End)?;
        text_layer(
            &mut edit,
            page,
            "Yes",
            20.0,
            false,
            Geometry::rectangle(40.0, 40.0, 150.0, 100.0),
            true,
        )?;
        let source = Geometry::rectangle(350.0, 250.0, 65.0, 160.0);
        let layer = sound_layer(&mut edit, page, 48.0, source.clone())?;
        let snapshot = session.commit(edit.finish()?).await?.snapshot;
        let frame = renderer().render(&snapshot, page).await?;
        assert_eq!(metadata(&frame, layer).font_size, 20.0);
        assert_eq!(
            geometry_bounds(frame.layer(layer).unwrap().geometry()),
            geometry_bounds(&source)
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_sound_inside_its_own_balloon_keeps_its_free_text_size() -> anyhow::Result<()> {
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("sheet", 600.0, 600.0), At::End)?;
        let source = Geometry::rectangle(200.0, 240.0, 180.0, 100.0);
        let sound = sound_layer(&mut edit, page, 28.0, source.clone())?;
        let bubble = edit.add_analysis_region::<BubbleRegion>(
            page,
            At::End,
            &Geometry::rectangle(100.0, 120.0, 400.0, 360.0),
            None,
        )?;
        let flow = edit.relate::<FlowsIn>(sound, bubble)?;
        let snapshot = session.commit(edit.finish()?).await?.snapshot;
        let renderer = renderer();
        let frame = renderer.render(&snapshot, page).await?;
        assert_eq!(metadata(&frame, sound).font_size, 28.0);
        assert_eq!(committed_ink(&renderer, &frame, sound)?.len(), 1);
        assert_eq!(
            geometry_bounds(frame.layer(sound).unwrap().geometry()),
            geometry_bounds(&source)
        );

        let change = session
            .commit(snapshot.patch(|edit| edit.remove_relation(flow))?)
            .await?;
        let foreign = renderer
            .update(&frame, &change.snapshot, &change.changes)
            .await?;
        assert!(metadata(&foreign, sound).font_size < 28.0);
        Ok(())
    }

    #[tokio::test]
    async fn transparent_dialogue_only_affects_lettering_with_a_visible_stroke()
    -> anyhow::Result<()> {
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("sheet", 1200.0, 800.0), At::End)?;
        text_layer(
            &mut edit,
            page,
            "Yes",
            16.0,
            false,
            Geometry::rectangle(20.0, 20.0, 160.0, 100.0),
            true,
        )?;
        let (large, _) = text_layer(
            &mut edit,
            page,
            "Hey",
            72.0,
            false,
            Geometry::rectangle(600.0, 20.0, 500.0, 300.0),
            true,
        )?;
        // Move the dialogue away from its balloon so only its letters can obstruct the caption.
        let room = Geometry::rectangle(320.0, 400.0, 500.0, 260.0);
        edit.set(large, &room)?;
        let (caption, _) = text_layer(&mut edit, page, "Yes", 4.0, true, room, false)?;
        let (distant, _) = text_layer(
            &mut edit,
            page,
            "Yes",
            4.0,
            true,
            Geometry::rectangle(850.0, 400.0, 300.0, 260.0),
            false,
        )?;
        let mut snapshot = session.commit(edit.finish()?).await?.snapshot;
        let renderer = renderer();
        let mut frame = renderer.render(&snapshot, page).await?;
        let mut style = snapshot.component::<Typography>(large)?.unwrap();
        style.color = Some([0, 0, 0, 0]);
        for (width, alpha) in [
            (None, 255),
            (Some(0.0), 255),
            (Some(2.0), 0),
            (Some(2.0), 255),
        ] {
            style.stroke_width = width;
            style.stroke_color = Some([255, 255, 255, alpha]);
            let change = session
                .commit(snapshot.patch(|edit| edit.set(large, &style))?)
                .await?;
            snapshot = change.snapshot;
            frame = renderer.update(&frame, &snapshot, &change.changes).await?;
            if width == Some(2.0) && alpha > 0 {
                assert_eq!(metadata(&frame, distant).font_size, 72.0);
                assert!(metadata(&frame, caption).font_size < 72.0);
            } else {
                assert_eq!(metadata(&frame, distant).font_size, 16.0);
                assert_eq!(metadata(&frame, caption).font_size, 16.0);
                let hidden = committed_ink(&renderer, &frame, large)?;
                let letters = committed_ink(&renderer, &frame, caption)?;
                assert!(letters[0].overlap(&hidden[0]) > 0.0);
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn hidden_and_overflowing_fixed_dialogue_do_not_set_the_page_size() -> anyhow::Result<()>
    {
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("sheet", 1200.0, 800.0), At::End)?;
        text_layer(
            &mut edit,
            page,
            "Yes",
            16.0,
            false,
            Geometry::rectangle(50.0, 50.0, 140.0, 100.0),
            true,
        )?;
        let (large, _) = text_layer(
            &mut edit,
            page,
            "Hey",
            72.0,
            false,
            Geometry::rectangle(500.0, 40.0, 500.0, 300.0),
            true,
        )?;
        let (caption, _) = text_layer(
            &mut edit,
            page,
            "Yes",
            4.0,
            true,
            Geometry::rectangle(50.0, 480.0, 450.0, 180.0),
            false,
        )?;
        edit.set(
            large,
            &Visibility {
                origin: Origin::User,
                visible: false,
                opacity: 1.0,
            },
        )?;
        let hidden = session.commit(edit.finish()?).await?.snapshot;
        let renderer = renderer();
        let frame = renderer.render(&hidden, page).await?;
        assert_eq!(metadata(&frame, caption).font_size, 16.0);

        let shown = session
            .commit(hidden.patch(|edit| edit.remove::<Visibility>(large))?)
            .await?;
        let frame = renderer
            .update(&frame, &shown.snapshot, &shown.changes)
            .await?;
        assert_eq!(metadata(&frame, caption).font_size, 72.0);

        let bubble = shown
            .snapshot
            .relation_from::<FlowsIn>(large)?
            .unwrap()
            .value()
            .target;
        let tight =
            session
                .commit(shown.snapshot.patch(|edit| {
                    edit.set(bubble, &Geometry::rectangle(500.0, 40.0, 40.0, 30.0))
                })?)
                .await?;
        let frame = renderer
            .update(&frame, &tight.snapshot, &tight.changes)
            .await?;
        assert_eq!(metadata(&frame, large).font_size, 72.0);
        assert_eq!(metadata(&frame, caption).font_size, 16.0);
        Ok(())
    }

    #[tokio::test]
    async fn overflow_clears_an_untranslated_balloon_after_its_outline_moves() -> anyhow::Result<()>
    {
        let mut session = Session::memory().await?;
        let mut edit = session.snapshot().edit();
        let page = edit.add_page(PageDraft::new("sheet", 1200.0, 1000.0), At::End)?;
        let (tight, _) = text_layer(
            &mut edit,
            page,
            "Run with me to the end of the road",
            4.0,
            true,
            Geometry::rectangle(240.0, 450.0, 125.0, 70.0),
            true,
        )?;
        text_layer(
            &mut edit,
            page,
            "Hi",
            36.0,
            false,
            Geometry::rectangle(700.0, 400.0, 180.0, 160.0),
            true,
        )?;
        let panel = edit.add_analysis_region::<PanelRegion>(
            page,
            At::End,
            &Geometry::rectangle(40.0, 20.0, 1100.0, 880.0),
            None,
        )?;
        let other_page = edit.add_page(PageDraft::new("next", 1200.0, 1000.0), At::End)?;
        let neighbour = edit.add_analysis_region::<BubbleRegion>(
            panel,
            At::End,
            &Geometry::rectangle(650.0, 30.0, 160.0, 100.0),
            None,
        )?;
        let base = session.commit(edit.finish()?).await?.snapshot;
        let renderer = renderer();
        let before = renderer.render(&base, page).await?;
        let original = metadata(&before, tight);
        assert!(original.rendered_bounds.y < 450.0);
        let bottom = f64::from(original.rendered_bounds.y) + 6.0;
        let moved = session
            .commit(base.patch(|edit| {
                edit.set(
                    neighbour,
                    &Geometry::rectangle(200.0, bottom - 90.0, 210.0, 90.0),
                )
            })?)
            .await?;
        let after = renderer
            .update(&before, &moved.snapshot, &moved.changes)
            .await?;
        let fit = metadata(&after, tight);
        assert!(fit.font_size < original.font_size);
        assert!(f64::from(fit.rendered_bounds.y) >= bottom);
        let cold = self::renderer().render(&moved.snapshot, page).await?;
        assert_eq!(
            after.prepared().manifest()?.encode()?,
            cold.prepared().manifest()?.encode()?
        );

        let transferred = session
            .commit(
                moved
                    .snapshot
                    .patch(|edit| edit.move_entity(panel, Some(other_page), At::End))?,
            )
            .await?;
        let clear_page = renderer
            .update(&after, &transferred.snapshot, &transferred.changes)
            .await?;
        assert_eq!(metadata(&clear_page, tight).font_size, original.font_size);
        let returned = session.undo(transferred.revision).await?;
        let after = renderer
            .update(&clear_page, &returned.snapshot, &returned.changes)
            .await?;
        assert_eq!(metadata(&after, tight).font_size, fit.font_size);

        let mut region = returned.snapshot.component::<Region>(neighbour)?.unwrap();
        region.kind = PanelRegion::kind();
        let changed = session
            .commit(
                returned
                    .snapshot
                    .patch(|edit| edit.set(neighbour, &region))?,
            )
            .await?;
        let released = renderer
            .update(&after, &changed.snapshot, &changed.changes)
            .await?;
        assert_eq!(metadata(&released, tight).font_size, original.font_size);
        Ok(())
    }

    #[tokio::test]
    async fn editing_one_balloon_resizes_dialogue_and_matches_a_cold_frame() {
        let mut session = Session::memory().await.unwrap();
        let mut ids = None;
        let create = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("sheet", 1400.0, 1200.0), At::End)?;
                let (tight, _) = text_layer(
                    edit,
                    page,
                    "Run with me to the end of the road",
                    4.0,
                    true,
                    Geometry::rectangle(240.0, 550.0, 125.0, 70.0),
                    true,
                )?;
                let (_, content) = text_layer(
                    edit,
                    page,
                    "Wait for me",
                    4.0,
                    true,
                    Geometry::rectangle(620.0, 550.0, 240.0, 150.0),
                    true,
                )?;
                let (large, _) = text_layer(
                    edit,
                    page,
                    "Go",
                    4.0,
                    true,
                    Geometry::rectangle(970.0, 550.0, 310.0, 210.0),
                    true,
                )?;
                for x in [240.0, 600.0, 960.0] {
                    text_layer(
                        edit,
                        page,
                        "A",
                        90.0,
                        false,
                        Geometry::rectangle(x, 30.0, 170.0, 135.0),
                        false,
                    )?;
                }
                ids = Some((page, tight, content, large));
                Ok(())
            })
            .unwrap();
        let base = session.commit(create).await.unwrap().snapshot;
        let (page, tight, content, large) = ids.unwrap();
        let renderer = renderer();
        let before = renderer.render(&base, page).await.unwrap();
        let old = metadata(&before, tight);
        assert!(old.font_size > 4.0 && old.font_size < 90.0);
        assert!(old.rendered_bounds.height > 70.0, "{old:?}");
        assert!(old.rendered_bounds.width < 125.0, "{old:?}");
        assert_eq!(old.color, [0, 0, 0, 255]);
        let commands = &before.layer(tight).unwrap().0.node.scene.commands;
        let border = commands
            .iter()
            .filter_map(|command| match command {
                koharu_rasterizer::PreparedSceneCommand::GlyphRun(run) if !run.hint => Some(run),
                _ => None,
            })
            .next()
            .unwrap();
        let fill = commands
            .iter()
            .filter_map(|command| match command {
                koharu_rasterizer::PreparedSceneCommand::GlyphRun(run) if run.hint => Some(run),
                _ => None,
            })
            .next()
            .unwrap();
        assert_eq!(border.color, [255; 4]);
        assert!((border.embolden[0] - fill.embolden[0] - old.font_size / 8.0).abs() < 0.001);
        let longer =
            "We have a very long way to go and there is still a long way back to the start";
        let mut obstacle = None;
        let add = base
            .patch(|edit| {
                let bounds = old.rendered_bounds;
                let (fixed, _) = text_layer(
                    edit,
                    page,
                    "M",
                    20.0,
                    false,
                    Geometry::rectangle(
                        f64::from(bounds.x + bounds.width * 0.5) - 20.0,
                        f64::from(bounds.y) - 16.0,
                        40.0,
                        40.0,
                    ),
                    false,
                )?;
                obstacle = Some(fixed);
                Ok(())
            })
            .unwrap();
        let obstructed = session.commit(add).await.unwrap();
        let cleared = renderer
            .update(&before, &obstructed.snapshot, &obstructed.changes)
            .await
            .unwrap();
        let fitted = metadata(&cleared, tight);
        let neighbour = metadata(&cleared, obstacle.unwrap());
        assert!(fitted.font_size < old.font_size);
        assert!(
            fitted.rendered_bounds.y
                > neighbour.rendered_bounds.y + neighbour.rendered_bounds.height
        );

        let undo = session.undo(obstructed.revision).await.unwrap();
        let edited = undo
            .snapshot
            .patch(|edit| {
                edit.set(
                    content,
                    &Translation {
                        text: Authored::user(longer.into()),
                        language: None,
                    },
                )
            })
            .unwrap();
        let changed = session.commit(edited).await.unwrap();
        let restored = renderer
            .update(&cleared, &undo.snapshot, &undo.changes)
            .await
            .unwrap();
        let patched = renderer
            .update(&restored, &changed.snapshot, &changed.changes)
            .await
            .unwrap();
        let cold = self::renderer()
            .render(&changed.snapshot, page)
            .await
            .unwrap();
        assert!(metadata(&patched, tight).font_size < old.font_size * 0.9);
        assert!(patched.stats().rebuilt_layers >= 2);
        assert!(Arc::ptr_eq(
            &before.layer(large).unwrap().0.node,
            &patched.layer(large).unwrap().0.node
        ));
        assert_eq!(patched.diagnostics(), cold.diagnostics());
        assert_eq!(
            patched.prepared().manifest().unwrap().encode().unwrap(),
            cold.prepared().manifest().unwrap().encode().unwrap()
        );
    }

    #[tokio::test]
    async fn like_source_ems_share_a_size_and_a_smaller_remark_keeps_its_own() {
        let mut session = Session::memory().await.unwrap();
        let mut ids = None;
        let create = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("sheet", 1400.0, 800.0), At::End)?;
                let bubble = edit.add_analysis_region::<BubbleRegion>(
                    page,
                    At::End,
                    &Geometry::rectangle(200.0, 250.0, 500.0, 280.0),
                    None,
                )?;
                let mut layers = Vec::new();
                for (words, em) in [
                    ("Ah", 20.0),
                    ("What are you doing all the way out here", 21.0),
                ] {
                    let (layer, _) = text_layer(
                        edit,
                        page,
                        words,
                        em,
                        true,
                        Geometry::rectangle(0.0, 0.0, 1.0, 1.0),
                        false,
                    )?;
                    edit.remove::<Geometry>(layer)?;
                    edit.relate::<FlowsIn>(layer, bubble)?;
                    layers.push(layer);
                }
                text_layer(
                    edit,
                    page,
                    "I know a way back",
                    10.0,
                    true,
                    Geometry::rectangle(900.0, 300.0, 130.0, 110.0),
                    true,
                )?;
                ids = Some((page, layers));
                Ok(())
            })
            .unwrap();
        let base = session.commit(create).await.unwrap().snapshot;
        let (page, layers) = ids.unwrap();
        let renderer = renderer();
        let frame = renderer.render(&base, page).await.unwrap();
        let shared = metadata(&frame, layers[1]).font_size;
        assert_eq!(metadata(&frame, layers[0]).font_size, shared);
        let mut style = base.component::<Typography>(layers[1]).unwrap().unwrap();
        style.size = Some(10.0);
        let change = base.patch(|edit| edit.set(layers[1], &style)).unwrap();
        let changed = session.commit(change).await.unwrap();
        let after = renderer
            .update(&frame, &changed.snapshot, &changed.changes)
            .await
            .unwrap();
        assert!(metadata(&after, layers[0]).font_size > shared * 1.1);
        assert_eq!(metadata(&after, layers[1]).font_size, shared);
    }

    #[tokio::test]
    async fn free_text_shrinks_against_the_letters_of_a_fixed_neighbour() {
        let mut session = Session::memory().await.unwrap();
        let mut ids = None;
        let create = session
            .snapshot()
            .patch(|edit| {
                let page = edit.add_page(PageDraft::new("sheet", 900.0, 600.0), At::End)?;
                let (free, _) = text_layer(
                    edit,
                    page,
                    "WAVE",
                    48.0,
                    true,
                    Geometry::rectangle(200.0, 200.0, 240.0, 100.0),
                    false,
                )?;
                let (fixed, _) = text_layer(
                    edit,
                    page,
                    "I",
                    48.0,
                    false,
                    Geometry::rectangle(650.0, 200.0, 80.0, 100.0),
                    false,
                )?;
                ids = Some((page, free, fixed));
                Ok(())
            })
            .unwrap();
        let base = session.commit(create).await.unwrap().snapshot;
        let (page, free, fixed) = ids.unwrap();
        let renderer = renderer();
        let before = renderer.render(&base, page).await.unwrap();
        let old = metadata(&before, free);
        let fixed_ink = metadata(&before, fixed).rendered_bounds;
        let left = old.rendered_bounds.x + old.rendered_bounds.width - 5.0;
        let shift = f64::from(left - fixed_ink.x);
        let change = base
            .patch(|edit| {
                edit.set(
                    fixed,
                    &Geometry::rectangle(650.0 + shift, 200.0, 80.0, 100.0),
                )
            })
            .unwrap();
        let changed = session.commit(change).await.unwrap();
        let after = renderer
            .update(&before, &changed.snapshot, &changed.changes)
            .await
            .unwrap();
        let new = metadata(&after, free);
        assert!(new.font_size < old.font_size);
        assert!(new.font_size >= old.font_size * 0.5);
        assert!(new.rendered_bounds.x + new.rendered_bounds.width < left);
        assert_eq!(metadata(&after, fixed).font_size, 48.0);
    }
}
