//! Unicode-aware text shaping, line breaking, and layout.

use std::{collections::HashMap, ops::Range};
use unicode_bidi::BidiInfo;

use anyhow::Result;
use harfrust::{Feature, Tag};
use hypher::Lang;
use icu_normalizer::ComposingNormalizerBorrowed;
use icu_properties::{
    CodePointMapData,
    props::{GeneralCategory, GeneralCategoryGroup},
};
use skrifa::{MetadataProvider, instance::Size};

use crate::{
    fonts::{Font, font_key},
    script::shaping_direction_for_text,
    segment::{HyphenationOptions, LineBreakSuffix, LineBreaker, hyphenation_lang_from_tag},
    shape::{PositionedGlyph, ShapedRun, ShapingOptions, TextShaper, shape_script_runs},
    types::TextAlign,
};

const HYPHENATION_MIN_WORD_LEN: usize = 5;
const CENTRAL_HYPHEN_MIN_WORD_LEN: usize = 6;
const WHOLE_WORD_TARGET_SHARE: f32 = 0.9;
const MINIMUM_HYPHENATED_GAIN: f32 = 1.1;
const COMPACT_HYPHENATION_FRAGMENT_LEN: usize = 2;
const LINE_BREAK_HYPHEN_PENALTY: f32 = 2_000.0;
const LINE_BREAK_OVERFLOW_MULTIPLIER: f32 = 10_000.0;
const COMIC_LINE_OVERFLOW_PENALTY: f32 = 1_000_000.0;
const COMIC_MAX_LINES: usize = 64;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HyphenationPolicy {
    /// Do not introduce discretionary hyphenation opportunities.
    Disabled,
    /// Use a discretionary hyphen only when the unhyphenated text overflows.
    LastResort,
    /// Try one central word break to recover font size.
    Central,
    /// Consider discretionary hyphens during normal line optimization.
    #[default]
    Normal,
}

/// Writing mode for text layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WritingMode {
    /// Horizontal text, left-to-right, lines flow top-to-bottom.
    #[default]
    Horizontal,
    /// Vertical text, right-to-left columns (traditional CJK).
    VerticalRl,
}

impl WritingMode {
    /// Returns true if the writing mode is vertical.
    pub const fn is_vertical(self) -> bool {
        matches!(self, WritingMode::VerticalRl)
    }
}

/// Glyphs for one line alongside metadata required by the renderer.
#[derive(Debug, Clone, Default)]
pub struct LayoutLine<'a> {
    /// Positioned glyphs in this line.
    pub glyphs: Vec<PositionedGlyph<'a>>,
    /// Range in the original text that this line covers.
    pub range: Range<usize>,
    /// Total advance (width for horizontal, height for vertical) of this line.
    pub advance: f32,
    hyphenated: bool,
    /// Baseline position for this line (x, y).
    pub baseline: (f32, f32),
}

/// A collection of laid out lines.
#[derive(Debug, Clone)]
pub struct LayoutRun<'a> {
    /// Lines in this layout run.
    pub lines: Vec<LayoutLine<'a>>,
    /// Total width of the layout.
    pub width: f32,
    /// Total height of the layout.
    pub height: f32,
    /// Font size used to generate this layout.
    pub font_size: f32,
    overflowed: bool,
    placement_offset_x: f32,
    placement_offset_y: f32,
}

impl LayoutRun<'_> {
    #[must_use]
    pub const fn overflowed(&self) -> bool {
        self.overflowed
    }

    pub(crate) const fn placement_offset_x(&self) -> f32 {
        self.placement_offset_x
    }

    pub(crate) const fn placement_offset_y(&self) -> f32 {
        self.placement_offset_y
    }
}

#[derive(Clone)]
struct LineRun<'a> {
    shaped: ShapedRun<'a>,
    level: unicode_bidi::Level,
}

#[derive(Clone)]
struct ShapedBreakSuffix<'a> {
    runs: Vec<LineRun<'a>>,
    advance: f32,
}

#[derive(Clone)]
struct ShapedSegment<'a> {
    range: Range<usize>,
    visible_end: usize,
    next_offset: usize,
    is_mandatory: bool,
    runs: Vec<LineRun<'a>>,
    advance: f32,
    trailing_advance: f32,
    break_penalty: f32,
    break_suffix: Option<ShapedBreakSuffix<'a>>,
}

#[derive(Clone, Copy, Debug)]
struct LineBreakMeasure {
    advance: f32,
    trailing_advance: f32,
    break_suffix_advance: f32,
    break_penalty: f32,
    is_mandatory: bool,
}

#[derive(Clone, Copy, Debug)]
struct LineProfile {
    /// Available extent on the inline axis: X for horizontal text, Y for vertical text.
    width: f32,
    /// Start of the usable inline room in balloon coordinates.
    start: Option<f32>,
    center_offset: f32,
    block_baseline: f32,
}

#[derive(Clone, Copy, Debug)]
struct InkBand {
    before: f32,
    after: f32,
}

impl InkBand {
    fn thickness(self) -> f32 {
        self.before + self.after
    }
}

#[derive(Debug)]
struct LineBreakResult {
    breaks: Vec<usize>,
    profiles: Vec<LineProfile>,
    overflowed: bool,
    cost: f32,
}

#[derive(Clone, Debug)]
struct ComicBalloon {
    width: f32,
    height: f32,
    contour: Vec<(f32, f32)>,
    minimum_air: f32,
}

struct BalloonProfiles {
    bands: Vec<Vec<(f32, f32)>>,
    baselines: Vec<f32>,
    inline_center: f32,
    alignment: TextAlign,
    axes: Vec<f32>,
}

#[derive(Clone)]
pub struct TextLayout<'a> {
    writing_mode: WritingMode,
    center_vertical_punctuation: bool,
    hyphenation_lang: Option<Lang>,
    hyphenation_policy: HyphenationPolicy,
    comic_balloon: Option<ComicBalloon>,
    font: &'a Font,
    fallback_fonts: &'a [Font],
    font_size: Option<f32>,
    min_font_size: Option<f32>,
    max_font_size: Option<f32>,
    max_width: Option<f32>,
    max_height: Option<f32>,
    alignment: Option<TextAlign>,
    line_height: Option<f32>,
    letter_spacing: f32,
    word_spacing: f32,
    cjk_punctuation_layout: bool,
}

pub(crate) fn largest_fitting_font_size<T>(
    minimum: f32,
    maximum: f32,
    mut layout_at: impl FnMut(f32) -> Result<T>,
    fits: impl Fn(&T) -> bool,
) -> Result<Option<T>> {
    const PROBES: usize = 8;
    if maximum - minimum <= f32::EPSILON {
        let candidate = layout_at(maximum)?;
        return Ok(fits(&candidate).then_some(candidate));
    }
    let step = (maximum - minimum) / PROBES as f32;
    let mut larger_non_fit = None;
    for probe in 0..=PROBES {
        let size = if probe == PROBES {
            minimum
        } else {
            maximum - step * probe as f32
        };
        let candidate = layout_at(size)?;
        if fits(&candidate) {
            let Some(mut high) = larger_non_fit else {
                return Ok(Some(candidate));
            };
            let mut low = size;
            let mut best = candidate;
            let mut iterations = 0u32;
            while high - low > 0.01 && iterations < 10 {
                iterations += 1;
                let midpoint = (low + high) * 0.5;
                let candidate = layout_at(midpoint)?;
                if fits(&candidate) {
                    best = candidate;
                    low = midpoint;
                } else {
                    high = midpoint;
                }
            }
            return Ok(Some(best));
        }

        larger_non_fit = Some(size);
    }
    Ok(None)
}

impl<'a> TextLayout<'a> {
    #[must_use]
    pub fn new(font: &'a Font) -> Self {
        Self {
            writing_mode: WritingMode::Horizontal,
            center_vertical_punctuation: true,
            hyphenation_lang: Some(Lang::English),
            hyphenation_policy: HyphenationPolicy::Normal,
            comic_balloon: None,
            font,
            fallback_fonts: &[],
            font_size: None,
            min_font_size: None,
            max_font_size: None,
            max_width: None,
            max_height: None,
            alignment: None,
            line_height: None,
            letter_spacing: 0.0,
            word_spacing: 0.0,
            cjk_punctuation_layout: false,
        }
    }

    #[must_use]
    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = Some(size);
        self.min_font_size = None;
        self.max_font_size = None;
        self
    }

    /// Automatically fit the text to its bounds without growing beyond `size`.
    #[must_use]
    pub fn with_max_font_size(mut self, size: f32) -> Self {
        self.font_size = None;
        self.max_font_size = Some(size);
        self
    }

    /// Prevent automatically fitted text from becoming unreadably small.
    #[must_use]
    pub fn with_min_font_size(mut self, size: f32) -> Self {
        self.font_size = None;
        self.min_font_size = Some(size);
        self
    }

    #[must_use]
    pub fn with_writing_mode(mut self, mode: WritingMode) -> Self {
        self.writing_mode = mode;
        self
    }

    #[must_use]
    pub fn with_hyphenation_language_tag(mut self, tag: &str) -> Self {
        self.hyphenation_lang = hyphenation_lang_from_tag(tag);
        self
    }

    #[must_use]
    pub fn with_hyphenation_policy(mut self, policy: HyphenationPolicy) -> Self {
        self.hyphenation_policy = policy;
        self
    }

    pub(crate) fn with_comic_balloon(
        mut self,
        width: f32,
        height: f32,
        contour: Vec<(f32, f32)>,
        minimum_air: f32,
    ) -> Self {
        self.comic_balloon = Some(ComicBalloon {
            width,
            height,
            contour,
            minimum_air: minimum_air.max(0.0),
        });
        self
    }

    #[must_use]
    pub fn with_fallback_fonts(mut self, fonts: &'a [Font]) -> Self {
        self.fallback_fonts = fonts;
        self
    }

    #[must_use]
    pub fn with_max_width(mut self, width: f32) -> Self {
        self.max_width = Some(width);
        self
    }

    #[must_use]
    pub fn with_max_height(mut self, height: f32) -> Self {
        self.max_height = Some(height);
        self
    }

    #[must_use]
    pub fn with_alignment(mut self, alignment: TextAlign) -> Self {
        self.alignment = Some(alignment);
        self
    }

    #[must_use]
    pub fn with_line_height(mut self, ratio: f32) -> Self {
        self.line_height = Some(ratio);
        self
    }

    #[must_use]
    pub fn with_spacing(mut self, letter: f32, word: f32) -> Self {
        self.letter_spacing = letter;
        self.word_spacing = word;
        self
    }

    pub(crate) fn with_cjk_punctuation_layout(mut self, enabled: bool) -> Self {
        self.cjk_punctuation_layout = enabled;
        self
    }

    pub fn run(&self, text: &str) -> Result<LayoutRun<'a>> {
        if let Some(font_size) = self.font_size {
            // Even an unused discretionary break divides a shaping run and
            // can change kerning or ligatures across that boundary.
            if self.hyphenation_policy == HyphenationPolicy::LastResort {
                let whole = self
                    .clone()
                    .with_hyphenation_policy(HyphenationPolicy::Disabled)
                    .run_with_size(text, font_size)?;
                if !whole.overflowed() {
                    return Ok(whole);
                }
            }
            return self.run_with_size(text, font_size);
        }

        self.run_auto(text)
    }

    fn run_auto(&self, text: &str) -> Result<LayoutRun<'a>> {
        let maximum = self.max_font_size.unwrap_or(300.0).max(0.5);
        let minimum = self
            .min_font_size
            .unwrap_or(maximum.min(1.0))
            .max(0.5)
            .min(maximum);
        if let Some(best) =
            self.fit_with_word_breaks(text, minimum, maximum, |layout, low, high| {
                layout.largest_fit(text, low, high)
            })?
        {
            return Ok(best);
        }
        self.run_with_size(text, minimum)
    }

    /// Try every room with whole words before spending one hyphen on a size
    /// gain. All legal breaks remain available if no whole-word setting fits.
    pub(crate) fn fit_with_word_breaks(
        &self,
        text: &str,
        minimum: f32,
        maximum: f32,
        mut fit: impl FnMut(&Self, f32, f32) -> Result<Option<LayoutRun<'a>>>,
    ) -> Result<Option<LayoutRun<'a>>> {
        if self.hyphenation_policy == HyphenationPolicy::Disabled {
            return fit(self, minimum, maximum);
        }
        let words = self
            .clone()
            .with_hyphenation_policy(HyphenationPolicy::Disabled);
        let clean = fit(&words, minimum, maximum)?;
        let threshold = WHOLE_WORD_TARGET_SHARE * maximum;
        if clean.as_ref().is_some_and(|run| run.font_size >= threshold) {
            return Ok(clean);
        }
        let least = clean
            .as_ref()
            .map_or(minimum, |run| run.font_size * MINIMUM_HYPHENATED_GAIN);
        let central = self
            .clone()
            .with_hyphenation_policy(HyphenationPolicy::Central);
        if central
            .line_breaker()
            .line_segments(text)
            .iter()
            .any(|s| s.break_suffix.is_some())
            && let Some(broken) = fit(&central, least, maximum)?
        {
            // A search returns the fitting side of its bracket. At the cutoff,
            // test the exact size before trading a whole word for a hyphen.
            if clean.is_some()
                && let Some(at_threshold) = fit(&words, threshold, threshold)?
            {
                return Ok(Some(at_threshold));
            }
            return Ok(Some(broken));
        }
        if clean.is_some() {
            return Ok(clean);
        }
        fit(self, minimum, maximum)
    }

    pub(crate) fn accepts(&self, run: &LayoutRun<'_>) -> bool {
        !run.overflowed()
            && (self.hyphenation_policy != HyphenationPolicy::Central
                || run.lines.iter().filter(|line| line.hyphenated).count() <= 1)
    }

    /// The largest size in `minimum..=maximum` whose layout fits, if any does.
    fn largest_fit(&self, text: &str, minimum: f32, maximum: f32) -> Result<Option<LayoutRun<'a>>> {
        let fits = |run: &LayoutRun<'_>| self.accepts(run);

        if self.comic_balloon.is_some() {
            // A balloon's usable width changes when the text reflows to a different
            // number of lines, so a smaller font can fail even though a larger one
            // fits. Search from largest to smallest instead of assuming monotonicity.
            return largest_fitting_font_size(
                minimum,
                maximum,
                |size| self.run_with_size(text, size),
                fits,
            );
        }

        let maximum_layout = self.run_with_size(text, maximum)?;
        if fits(&maximum_layout) {
            return Ok(Some(maximum_layout));
        }
        let mut best = self.run_with_size(text, minimum)?;
        if !fits(&best) {
            return Ok(None);
        }
        let mut low = minimum;
        let mut high = maximum;
        let mut iterations = 0u32;
        while high - low > 0.01 && iterations < 16 {
            iterations += 1;
            let size = (low + high) * 0.5;
            let layout = self.run_with_size(text, size)?;
            if fits(&layout) {
                best = layout;
                low = size;
            } else {
                high = size;
            }
        }
        Ok(Some(best))
    }

    fn line_breaker(&self) -> LineBreaker {
        let breaker = LineBreaker::new();
        if self.hyphenation_policy == HyphenationPolicy::Disabled {
            return breaker;
        }
        let central = self.hyphenation_policy == HyphenationPolicy::Central;
        let minimum_length = if central {
            CENTRAL_HYPHEN_MIN_WORD_LEN
        } else {
            HYPHENATION_MIN_WORD_LEN
        };
        let mut options = HyphenationOptions::new(self.hyphenation_lang, minimum_length);
        if self.comic_balloon.is_some() {
            options = options.with_fragment_bounds(
                options
                    .minimum_prefix_length
                    .min(COMPACT_HYPHENATION_FRAGMENT_LEN),
                options
                    .minimum_suffix_length
                    .min(COMPACT_HYPHENATION_FRAGMENT_LEN),
            );
        }
        options.central_only = central;
        breaker.with_hyphenation(options)
    }

    fn run_with_size(&self, text: &str, font_size: f32) -> Result<LayoutRun<'a>> {
        let shaper = TextShaper::new();
        let line_breaker = self.line_breaker();
        // Use real font metrics for consistent line sizing across modes.
        let font_ref = self.font.skrifa_ref()?;
        let metrics = font_ref.metrics(Size::new(font_size), self.font.location());
        let ascent = metrics.ascent;
        let descent = -metrics.descent;
        let line_height = self.line_height.map_or_else(
            || (ascent + descent + metrics.leading).max(font_size),
            |ratio| font_size * ratio,
        );

        let bidi_info = BidiInfo::new(text, None);

        let (direction, script) = shaping_direction_for_text(text, self.writing_mode);
        let options = ShapingOptions {
            direction,
            script,
            font_size,
            features: if self.writing_mode.is_vertical() {
                &[
                    Feature::new(Tag::new(b"vert"), 1, ..),
                    Feature::new(Tag::new(b"vrt2"), 1, ..),
                ]
            } else {
                &[]
            },
        };
        let max_extent = if self.writing_mode.is_vertical() {
            self.max_height
        } else {
            self.max_width
        }
        .unwrap_or(f32::INFINITY);

        let mut fonts: Vec<&Font> = Vec::with_capacity(1 + self.fallback_fonts.len());
        fonts.push(self.font);
        fonts.extend(self.fallback_fonts.iter());

        let shape_break_suffix = |suffix: LineBreakSuffix,
                                  level: unicode_bidi::Level,
                                  cluster: usize|
         -> Result<ShapedBreakSuffix<'a>> {
            let mut suffix_options = options.clone();
            suffix_options.direction = if self.writing_mode.is_vertical() {
                harfrust::Direction::TopToBottom
            } else if level.is_rtl() {
                harfrust::Direction::RightToLeft
            } else {
                harfrust::Direction::LeftToRight
            };

            let mut runs = Vec::new();
            let mut advance = 0.0f32;
            for mut shaped in shape_script_runs(&shaper, suffix.as_str(), &fonts, &suffix_options)?
            {
                for glyph in &mut shaped.glyphs {
                    glyph.cluster += cluster as u32;
                }
                advance += if self.writing_mode.is_vertical() {
                    shaped.y_advance.abs()
                } else {
                    shaped.x_advance.abs()
                };
                runs.push(LineRun { shaped, level });
            }

            Ok(ShapedBreakSuffix { runs, advance })
        };

        let mut shaped_segments = Vec::new();
        for segment in line_breaker.line_segments(text) {
            let segment_text = &text[segment.range.clone()];
            let visible_end = segment.range.start + segment_text.trim_end().len();

            let mut segment_runs = Vec::new();
            let mut segment_advance = 0.0f32;

            if !segment_text.is_empty() {
                // Subdivide segment into constant BiDi level runs.
                let mut char_indices = segment_text
                    .char_indices()
                    .map(|(id, _)| segment.range.start + id)
                    .peekable();

                while let Some(run_start) = char_indices.next() {
                    let level = bidi_info.levels[run_start];
                    let mut run_end = segment.range.end;

                    while let Some(&next_char_start) = char_indices.peek() {
                        if bidi_info.levels[next_char_start] != level {
                            run_end = next_char_start;
                            break;
                        }
                        char_indices.next();
                    }

                    let run_text = &text[run_start..run_end];
                    let mut run_options = options.clone();
                    run_options.direction = if self.writing_mode.is_vertical() {
                        harfrust::Direction::TopToBottom
                    } else if level.is_rtl() {
                        harfrust::Direction::RightToLeft
                    } else {
                        harfrust::Direction::LeftToRight
                    };

                    let normalized_punctuation = self
                        .cjk_punctuation_layout
                        .then(|| normalize_cjk_emphasis_punctuation(run_text))
                        .flatten();
                    let shaping_text = normalized_punctuation
                        .as_ref()
                        .map_or(run_text, |(text, _)| text.as_str());
                    let script_runs =
                        shape_script_runs(&shaper, shaping_text, &fonts, &run_options)?;
                    for mut shaped in script_runs {
                        self.apply_spacing(shaping_text, &mut shaped);
                        if self.writing_mode.is_vertical() && self.center_vertical_punctuation {
                            self.center_vertical_punctuation(
                                font_size,
                                shaping_text,
                                &mut shaped.glyphs,
                            );
                        }
                        if self.cjk_punctuation_layout {
                            self.layout_cjk_emphasis_runs(font_size, shaping_text, &mut shaped);
                        }
                        if let Some((_, cluster_map)) = &normalized_punctuation {
                            for glyph in &mut shaped.glyphs {
                                if let Some(&source_cluster) =
                                    cluster_map.get(glyph.cluster as usize)
                                {
                                    glyph.cluster = source_cluster;
                                }
                            }
                        }

                        for glyph in &mut shaped.glyphs {
                            glyph.cluster += run_start as u32;
                        }

                        segment_advance += if self.writing_mode.is_vertical() {
                            shaped.y_advance.abs()
                        } else {
                            shaped.x_advance.abs()
                        };

                        segment_runs.push(LineRun { shaped, level });
                    }
                }
            }
            let segment_break_suffix = if let (Some(suffix), Some(level)) =
                (segment.break_suffix, segment_runs.last().map(|r| r.level))
            {
                Some(shape_break_suffix(suffix, level, segment.range.end)?)
            } else {
                None
            };

            let trailing_advance = segment_runs
                .iter()
                .flat_map(|run| &run.shaped.glyphs)
                .filter(|glyph| glyph.cluster as usize >= visible_end)
                .map(|glyph| {
                    if self.writing_mode.is_vertical() {
                        glyph.y_advance.abs()
                    } else {
                        glyph.x_advance.abs()
                    }
                })
                .sum();
            shaped_segments.push(ShapedSegment {
                range: segment.range,
                visible_end,
                next_offset: segment.next_offset,
                is_mandatory: segment.is_mandatory,
                runs: segment_runs,
                advance: segment_advance,
                trailing_advance,
                break_penalty: if self.comic_balloon.is_some() {
                    comic_break_penalty(text, segment.next_offset)
                } else {
                    0.0
                },
                break_suffix: segment_break_suffix,
            });
        }

        let fallback_ink = if self.writing_mode.is_vertical() {
            InkBand {
                before: line_height * 0.5,
                after: line_height * 0.5,
            }
        } else {
            InkBand {
                before: ascent,
                after: descent,
            }
        };
        let line_ink = if self.comic_balloon.is_some() {
            self.shaped_block_extents(font_size, &shaped_segments)
                .unwrap_or(fallback_ink)
        } else {
            fallback_ink
        };
        let balloon_air = self
            .comic_balloon
            .as_ref()
            .map_or(0.0, |balloon| balloon.air(line_ink.thickness()));

        let mut lines: Vec<LayoutLine<'a>> = Vec::new();
        let mut line_profiles = Vec::new();
        let mut contour_overflowed = false;
        let mut line_offset = 0usize;
        if self.comic_balloon.is_some() {
            contour_overflowed = self.append_balanced_segment_lines(
                &shaped_segments,
                &mut line_offset,
                text.len(),
                false,
                max_extent,
                line_height,
                line_ink,
                balloon_air,
                balloon_air,
                &bidi_info,
                &mut lines,
                &mut line_profiles,
            );
        } else {
            let mut paragraph_start = 0usize;
            for (index, segment) in shaped_segments.iter().enumerate() {
                if !segment.is_mandatory {
                    continue;
                }
                contour_overflowed |= self.append_balanced_segment_lines(
                    &shaped_segments[paragraph_start..=index],
                    &mut line_offset,
                    segment.next_offset,
                    true,
                    max_extent,
                    line_height,
                    line_ink,
                    balloon_air,
                    balloon_air,
                    &bidi_info,
                    &mut lines,
                    &mut line_profiles,
                );
                paragraph_start = index + 1;
            }
            if paragraph_start < shaped_segments.len() {
                contour_overflowed |= self.append_balanced_segment_lines(
                    &shaped_segments[paragraph_start..],
                    &mut line_offset,
                    text.len(),
                    false,
                    max_extent,
                    line_height,
                    line_ink,
                    balloon_air,
                    balloon_air,
                    &bidi_info,
                    &mut lines,
                    &mut line_profiles,
                );
            }
        }

        // Baselines depend only on line index and metrics. For vertical text we compute absolute X
        // positions within the layout bounds (0..width) so the renderer can draw from the left.
        let line_count = lines.len();
        let effective_alignment = self.alignment.unwrap_or(TextAlign::Left);

        for (i, line) in lines.iter_mut().enumerate() {
            line.baseline = match self.writing_mode {
                WritingMode::VerticalRl => (
                    self.comic_balloon
                        .as_ref()
                        .and_then(|_| line_profiles.get(i))
                        .map_or(
                            (line_count.saturating_sub(1) as f32 - i as f32) * line_height
                                + line_height * 0.5,
                            |profile| profile.block_baseline,
                        ),
                    ascent,
                ),
                WritingMode::Horizontal => {
                    let y = self
                        .comic_balloon
                        .as_ref()
                        .and_then(|_| line_profiles.get(i))
                        .map_or(ascent + i as f32 * line_height, |profile| {
                            profile.block_baseline
                        });
                    (0.0, y)
                }
            };
        }

        if effective_alignment == TextAlign::Justify && !self.writing_mode.is_vertical() {
            justify_lines(text, &mut lines, max_extent, &line_profiles);
        }

        // Alignment is an inline-axis operation inside one intrinsic text block. Balloon
        // contours constrain line breaking and block placement, but do not bend the text axis.
        const PAD: f32 = 1.0;
        let inline_extent = lines
            .iter()
            .filter_map(|line| self.ink_bounds(font_size, std::slice::from_ref(line)))
            .map(|(min_x, min_y, max_x, max_y)| {
                if self.writing_mode.is_vertical() {
                    max_y - min_y
                } else {
                    max_x - min_x
                }
            })
            .fold(0.0_f32, f32::max);
        let inline_box = (inline_extent > 0.0).then(|| {
            let extent = inline_extent + PAD * 2.0;
            (extent, PAD, extent * 0.5, extent - PAD)
        });
        let mut earliest = f32::NEG_INFINITY;
        let mut latest = f32::INFINITY;
        if let Some((_, inline_start, inline_center, inline_end)) = inline_box {
            for (line, profile) in lines.iter_mut().zip(&line_profiles) {
                if let Some((min_x, min_y, max_x, max_y)) =
                    self.ink_bounds(font_size, std::slice::from_ref(line))
                {
                    let (minimum, center, maximum) = if self.writing_mode.is_vertical() {
                        (min_y, (min_y + max_y) * 0.5, max_y)
                    } else {
                        (min_x, (min_x + max_x) * 0.5, max_x)
                    };
                    let offset = match effective_alignment {
                        TextAlign::Left | TextAlign::Justify => inline_start - minimum,
                        TextAlign::Center => inline_center - center,
                        TextAlign::Right => inline_end - maximum,
                    };
                    if let Some(start) = profile.start {
                        earliest = earliest.max(start - (minimum + offset));
                        latest = latest.min(start + profile.width - (maximum + offset));
                    }
                    if self.writing_mode.is_vertical() {
                        line.baseline.1 += offset;
                    } else {
                        line.baseline.0 += offset;
                    }
                }
            }
        }

        // Compute a tight ink bounding box using per-glyph bounds from the font tables (via skrifa),
        // then normalize only the block axis. The inline axis remains in the fixed layout box.
        let (mut width, mut height) = (0.0, 0.0);
        let mut placement_offset_x = 0.0;
        let mut placement_offset_y = 0.0;
        if let Some((mut min_x, mut min_y, mut max_x, mut max_y)) =
            self.ink_bounds(font_size, &lines)
        {
            // Keep a tiny safety pad for hinting/AA differences.
            min_x -= PAD;
            min_y -= PAD;
            max_x += PAD;
            max_y += PAD;

            for line in &mut lines {
                if inline_box.is_none() || self.writing_mode.is_vertical() {
                    line.baseline.0 -= min_x;
                }
                if inline_box.is_none() || !self.writing_mode.is_vertical() {
                    line.baseline.1 -= min_y;
                }
            }
            let actual_width = (max_x - min_x).max(0.0);
            let actual_height = (max_y - min_y).max(0.0);
            if self.writing_mode.is_vertical() {
                width = actual_width;
                if let Some((inline_extent, _, _, _)) = inline_box {
                    height = inline_extent;
                } else {
                    height = actual_height;
                }
            } else {
                height = actual_height;
                if let Some((inline_extent, _, _, _)) = inline_box {
                    width = inline_extent;
                } else {
                    width = actual_width;
                }
            }

            if let Some(balloon) = &self.comic_balloon {
                let axis_offset = line_profiles
                    .first()
                    .map_or(0.0, |profile| profile.center_offset);
                // Centre an edge-aligned block as far as the individual
                // lines' ink can travel together inside their rooms.
                let default_start = if self.writing_mode.is_vertical() {
                    (balloon.height - height) * 0.5
                } else {
                    (balloon.width - width) * 0.5
                };
                let inline_offset = match effective_alignment {
                    TextAlign::Left | TextAlign::Justify => {
                        (default_start + axis_offset).min(latest).max(earliest) - default_start
                    }
                    TextAlign::Right => {
                        (default_start + axis_offset).max(earliest).min(latest) - default_start
                    }
                    TextAlign::Center => axis_offset,
                };
                if self.writing_mode.is_vertical() {
                    let default_left = (balloon.width - width) * 0.5;
                    placement_offset_x = min_x - default_left;
                    placement_offset_y = inline_offset;
                } else {
                    let default_top = (balloon.height - height) * 0.5;
                    placement_offset_x = inline_offset;
                    placement_offset_y = min_y - default_top;
                }
            }
        }

        let overflowed = contour_overflowed
            || self
                .max_width
                .is_some_and(|maximum| width > maximum + f32::EPSILON)
            || self
                .max_height
                .is_some_and(|maximum| height > maximum + f32::EPSILON);

        Ok(LayoutRun {
            lines,
            width,
            height,
            font_size,
            overflowed,
            placement_offset_x,
            placement_offset_y,
        })
    }

    fn shaped_block_extents(
        &self,
        font_size: f32,
        segments: &[ShapedSegment<'a>],
    ) -> Option<InkBand> {
        let mut metrics_cache = HashMap::new();
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;

        for segment in segments {
            let suffix_runs = segment
                .break_suffix
                .iter()
                .flat_map(|suffix| suffix.runs.iter());
            for run in segment.runs.iter().chain(suffix_runs) {
                for glyph in &run.shaped.glyphs {
                    let key = font_key(glyph.font);
                    let glyph_metrics = match metrics_cache.entry(key) {
                        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            let Ok(font_ref) = glyph.font.skrifa_ref() else {
                                continue;
                            };
                            entry.insert(
                                font_ref.glyph_metrics(Size::new(font_size), glyph.font.location()),
                            )
                        }
                    };

                    let glyph_id = skrifa::GlyphId::new(glyph.glyph_id);
                    if let Some(bounds) = glyph_metrics.bounds(glyph_id) {
                        if self.writing_mode.is_vertical() {
                            let synthetic_pad = glyph
                                .font
                                .synthetic_skew()
                                .map_or(0.0, |_| font_size * 0.25)
                                + if glyph.font.synthetic_bold() {
                                    font_size * 0.05
                                } else {
                                    0.0
                                };
                            minimum = minimum.min(glyph.x_offset + bounds.x_min - synthetic_pad);
                            maximum = maximum.max(glyph.x_offset + bounds.x_max + synthetic_pad);
                        } else {
                            minimum = minimum.min(-glyph.y_offset - bounds.y_max);
                            maximum = maximum.max(-glyph.y_offset - bounds.y_min);
                        }
                    }
                }
            }
        }

        minimum.is_finite().then(|| InkBand {
            before: (-minimum).max(0.0),
            after: maximum.max(0.0),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn append_balanced_segment_lines(
        &self,
        segments: &[ShapedSegment<'a>],
        line_offset: &mut usize,
        final_next_offset: usize,
        force_final_line: bool,
        max_extent: f32,
        line_height: f32,
        line_ink: InkBand,
        balloon_air_x: f32,
        balloon_air_y: f32,
        bidi_info: &BidiInfo<'_>,
        lines: &mut Vec<LayoutLine<'a>>,
        line_profiles: &mut Vec<LineProfile>,
    ) -> bool {
        if segments.is_empty() {
            if force_final_line {
                *line_offset = self.push_layout_line(
                    Vec::new(),
                    *line_offset,
                    *line_offset,
                    final_next_offset,
                    None,
                    true,
                    bidi_info,
                    lines,
                );
                line_profiles.push(LineProfile {
                    width: max_extent,
                    start: None,
                    center_offset: 0.0,
                    block_baseline: 0.0,
                });
            }
            return false;
        }

        let measures = segments
            .iter()
            .map(|segment| LineBreakMeasure {
                advance: segment.advance,
                trailing_advance: segment.trailing_advance,
                break_suffix_advance: segment
                    .break_suffix
                    .as_ref()
                    .map_or(0.0, |suffix| suffix.advance),
                break_penalty: segment.break_penalty,
                is_mandatory: segment.is_mandatory,
            })
            .collect::<Vec<_>>();
        let result = if let Some(balloon) = self.comic_balloon.as_ref() {
            comic_line_breaks(
                &measures,
                balloon,
                (self.writing_mode, self.alignment.unwrap_or(TextAlign::Left)),
                line_height,
                line_ink,
                (balloon_air_x, balloon_air_y),
                self.hyphenation_policy,
            )
        } else if max_extent.is_finite() && max_extent > 0.0 {
            line_breaks_with_policy(&measures, max_extent, self.hyphenation_policy)
        } else {
            LineBreakResult {
                breaks: vec![segments.len()],
                profiles: vec![LineProfile {
                    width: measures.iter().map(|measure| measure.advance).sum(),
                    start: None,
                    center_offset: 0.0,
                    block_baseline: 0.0,
                }],
                overflowed: false,
                cost: 0.0,
            }
        };

        let mut start = 0usize;
        for (line_index, end) in result.breaks.iter().copied().enumerate() {
            if end <= start || end > segments.len() {
                continue;
            }
            let final_line = end == segments.len();
            let mandatory_line = segments[end - 1].is_mandatory;
            let visible_end = segments[end - 1].visible_end;
            let next_offset = if mandatory_line {
                segments[end - 1].next_offset
            } else if final_line {
                final_next_offset
            } else {
                segments[end].range.start
            };
            let break_suffix = if final_line || mandatory_line {
                None
            } else {
                segments[end - 1].break_suffix.clone()
            };
            let mut runs = segments[start..end - 1]
                .iter()
                .flat_map(|segment| segment.runs.iter().cloned())
                .collect::<Vec<_>>();
            let mut final_runs = segments[end - 1].runs.clone();
            for run in &mut final_runs {
                run.shaped
                    .glyphs
                    .retain(|glyph| (glyph.cluster as usize) < visible_end);
            }
            runs.extend(final_runs);
            *line_offset = self.push_layout_line(
                runs,
                *line_offset,
                visible_end,
                next_offset,
                break_suffix,
                mandatory_line || (force_final_line && final_line),
                bidi_info,
                lines,
            );
            line_profiles.push(
                result
                    .profiles
                    .get(line_index)
                    .copied()
                    .unwrap_or(LineProfile {
                        width: max_extent,
                        start: None,
                        center_offset: 0.0,
                        block_baseline: 0.0,
                    }),
            );
            start = end;
        }
        result.overflowed
    }

    #[allow(clippy::too_many_arguments)]
    fn push_layout_line(
        &self,
        mut runs: Vec<LineRun<'a>>,
        offset: usize,
        visible_end: usize,
        next_offset: usize,
        break_suffix: Option<ShapedBreakSuffix<'a>>,
        force_push: bool,
        _bidi_info: &BidiInfo<'_>,
        lines: &mut Vec<LayoutLine<'a>>,
    ) -> usize {
        if runs.is_empty() && !force_push {
            return next_offset;
        }

        let hyphenated = break_suffix.is_some();
        if let Some(mut suffix) = break_suffix {
            runs.append(&mut suffix.runs);
        }

        let levels: Vec<unicode_bidi::Level> = runs.iter().map(|r| r.level).collect();
        let visual_indices = reorder_visual(&levels);

        let mut line = LayoutLine {
            range: offset..visible_end,
            hyphenated,
            ..Default::default()
        };

        let mut pen_x = 0.0f32;
        let mut pen_y = 0.0f32;

        for idx in visual_indices {
            let run = &mut runs[idx];
            for glyph in std::mem::take(&mut run.shaped.glyphs) {
                line.glyphs.push(glyph);
            }
            if self.writing_mode.is_vertical() {
                pen_y -= run.shaped.y_advance;
            } else {
                pen_x += run.shaped.x_advance;
            }
        }

        line.advance = if self.writing_mode.is_vertical() {
            pen_y.abs()
        } else {
            pen_x
        };

        lines.push(line);
        next_offset
    }

    fn ink_bounds(&self, font_size: f32, lines: &[LayoutLine<'a>]) -> Option<(f32, f32, f32, f32)> {
        let mut metrics_cache = HashMap::new();

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for line in lines {
            let (mut x, mut y) = line.baseline;
            for g in &line.glyphs {
                let key = font_key(g.font);
                let glyph_metrics = match metrics_cache.entry(key) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let Ok(font_ref) = g.font.skrifa_ref() else {
                            x += g.x_advance;
                            y -= g.y_advance;
                            continue;
                        };
                        entry
                            .insert(font_ref.glyph_metrics(Size::new(font_size), g.font.location()))
                    }
                };

                let gid = skrifa::GlyphId::new(g.glyph_id);
                if let Some(b) = glyph_metrics.bounds(gid) {
                    let x0 = x + g.x_offset + b.x_min;
                    let x1 = x + g.x_offset + b.x_max;
                    let synthetic_pad = g.font.synthetic_skew().map_or(0.0, |_| font_size * 0.25)
                        + if g.font.synthetic_bold() {
                            font_size * 0.05
                        } else {
                            0.0
                        };

                    // `b` is in a Y-up font coordinate system. Our layout coordinates are Y-down
                    // while screen-space Y grows downward, so we flip by subtracting.
                    let y0 = (y - g.y_offset) - b.y_max;
                    let y1 = (y - g.y_offset) - b.y_min;

                    min_x = min_x.min(x0 - synthetic_pad).min(x1 - synthetic_pad);
                    max_x = max_x.max(x0 + synthetic_pad).max(x1 + synthetic_pad);
                    min_y = min_y.min(y0).min(y1);
                    max_y = max_y.max(y0).max(y1);
                }

                x += g.x_advance;
                y -= g.y_advance;
            }
        }

        if min_x.is_finite() {
            Some((min_x, min_y, max_x, max_y))
        } else {
            None
        }
    }

    fn center_vertical_punctuation(
        &self,
        font_size: f32,
        segment: &str,
        glyphs: &mut [PositionedGlyph<'a>],
    ) {
        if segment.is_empty() || glyphs.is_empty() {
            return;
        }

        let mut metrics_cache = HashMap::new();
        let categories = CodePointMapData::<GeneralCategory>::new();
        for glyph in glyphs {
            let cluster = glyph.cluster as usize;
            let Some(ch) = segment.get(cluster..).and_then(|tail| tail.chars().next()) else {
                continue;
            };
            if !GeneralCategoryGroup::Punctuation.contains(categories.get(ch)) {
                continue;
            }

            let key = font_key(glyph.font);
            let glyph_metrics = match metrics_cache.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let Ok(font_ref) = glyph.font.skrifa_ref() else {
                        continue;
                    };
                    entry
                        .insert(font_ref.glyph_metrics(Size::new(font_size), glyph.font.location()))
                }
            };

            let gid = skrifa::GlyphId::new(glyph.glyph_id);
            let Some(bounds) = glyph_metrics.bounds(gid) else {
                continue;
            };
            glyph.x_offset = centered_x_offset(bounds.x_min, bounds.x_max);
            glyph.y_offset = glyph.y_advance * 0.5 - (bounds.y_min + bounds.y_max) * 0.5;
        }
    }

    fn layout_cjk_emphasis_runs(&self, font_size: f32, text: &str, shaped: &mut ShapedRun<'a>) {
        const GAP_EM: f32 = 0.04;

        let mut metrics_cache = HashMap::new();
        let mut start = 0;
        while start < shaped.glyphs.len() {
            let is_emphasis = |glyph: &PositionedGlyph<'_>| {
                text.get(glyph.cluster as usize..)
                    .and_then(|tail| tail.chars().next())
                    .and_then(cjk_emphasis_mark)
                    .is_some()
            };
            if !is_emphasis(&shaped.glyphs[start]) {
                start += 1;
                continue;
            }

            let mut end = start + 1;
            while end < shaped.glyphs.len() && end - start < 3 && is_emphasis(&shaped.glyphs[end]) {
                end += 1;
            }
            if end - start < 2 {
                start = end;
                continue;
            }

            let bounds = shaped.glyphs[start..end]
                .iter()
                .map(|glyph| {
                    let key = font_key(glyph.font);
                    let glyph_metrics = match metrics_cache.entry(key) {
                        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            let font_ref = glyph.font.skrifa_ref().ok()?;
                            entry.insert(
                                font_ref.glyph_metrics(Size::new(font_size), glyph.font.location()),
                            )
                        }
                    };
                    let bounds = glyph_metrics.bounds(skrifa::GlyphId::new(glyph.glyph_id))?;
                    Some((bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max))
                })
                .collect::<Option<Vec<_>>>();
            let Some(bounds) = bounds else {
                start = end;
                continue;
            };

            let gap = font_size * GAP_EM;
            if self.writing_mode.is_vertical() {
                let cell_advance = -shaped.glyphs[start..end]
                    .iter()
                    .map(|glyph| glyph.y_advance.abs())
                    .fold(font_size, f32::max);
                let group_width = bounds
                    .iter()
                    .map(|(x_min, _, x_max, _)| x_max - x_min)
                    .sum::<f32>()
                    + gap * (bounds.len() - 1) as f32;
                let mut cursor = -group_width * 0.5;
                let last = end - start - 1;
                for (index, (glyph, (x_min, y_min, x_max, y_max))) in
                    shaped.glyphs[start..end].iter_mut().zip(bounds).enumerate()
                {
                    glyph.x_offset = cursor - x_min;
                    glyph.y_offset = cell_advance * 0.5 - (y_min + y_max) * 0.5;
                    glyph.x_advance = 0.0;
                    glyph.y_advance = if index == last { cell_advance } else { 0.0 };
                    cursor += x_max - x_min + gap;
                }
            } else {
                let last = end - start - 1;
                for (index, (glyph, (x_min, _, x_max, _))) in
                    shaped.glyphs[start..end].iter_mut().zip(bounds).enumerate()
                {
                    glyph.x_offset = -x_min;
                    glyph.x_advance = if index == last {
                        extend_advance(x_max - x_min, self.letter_spacing)
                    } else {
                        x_max - x_min + gap
                    };
                }
            }

            start = end;
        }

        shaped.x_advance = shaped.glyphs.iter().map(|glyph| glyph.x_advance).sum();
        shaped.y_advance = shaped.glyphs.iter().map(|glyph| glyph.y_advance).sum();
    }

    fn apply_spacing(&self, text: &str, shaped: &mut ShapedRun<'a>) {
        if self.letter_spacing == 0.0 && self.word_spacing == 0.0 {
            return;
        }
        for glyph in &mut shaped.glyphs {
            let character = text
                .get(glyph.cluster as usize..)
                .and_then(|tail| tail.chars().next());
            let extra = self.letter_spacing
                + character
                    .filter(|character| character.is_whitespace())
                    .map_or(0.0, |_| self.word_spacing);
            if self.writing_mode.is_vertical() {
                glyph.y_advance = extend_advance(glyph.y_advance, extra);
            } else {
                glyph.x_advance = extend_advance(glyph.x_advance, extra);
            }
        }
        shaped.x_advance = shaped.glyphs.iter().map(|glyph| glyph.x_advance).sum();
        shaped.y_advance = shaped.glyphs.iter().map(|glyph| glyph.y_advance).sum();
    }
}

fn extend_advance(advance: f32, extra: f32) -> f32 {
    if advance < 0.0 {
        advance - extra
    } else {
        advance + extra
    }
}

fn justify_lines(
    text: &str,
    lines: &mut [LayoutLine<'_>],
    max_width: f32,
    profiles: &[LineProfile],
) {
    if profiles.is_empty() && (!max_width.is_finite() || max_width <= 0.0) {
        return;
    }
    let last = lines.len().saturating_sub(1);
    for (index, line) in lines.iter_mut().enumerate() {
        if index == last
            || text
                .get(line.range.end..)
                .and_then(|tail| tail.chars().next())
                .is_some_and(|character| {
                    matches!(
                        character,
                        '\n' | '\r' | '\u{0085}' | '\u{2028}' | '\u{2029}'
                    )
                })
        {
            continue;
        }
        let target_width = profiles
            .get(index)
            .map_or(max_width, |profile| profile.width);
        if !target_width.is_finite() || target_width <= 0.0 {
            continue;
        }
        let is_space = |glyph: &PositionedGlyph<'_>| {
            text.get(glyph.cluster as usize..)
                .and_then(|tail| tail.chars().next())
                .is_some_and(char::is_whitespace)
        };
        let spaces = line.glyphs.iter().filter(|glyph| is_space(glyph)).count();
        if spaces == 0 || line.advance >= target_width {
            continue;
        }
        let extra = (target_width - line.advance) / spaces as f32;
        for glyph in &mut line.glyphs {
            if is_space(glyph) {
                glyph.x_advance = extend_advance(glyph.x_advance, extra);
            }
        }
        line.advance = target_width;
    }
}

fn line_breaks_with_policy(
    segments: &[LineBreakMeasure],
    max_extent: f32,
    policy: HyphenationPolicy,
) -> LineBreakResult {
    if policy == HyphenationPolicy::LastResort {
        let without_hyphens = optimal_uniform_line_breaks(segments, max_extent, false);
        if !without_hyphens.overflowed {
            return without_hyphens;
        }
    }
    optimal_uniform_line_breaks(segments, max_extent, policy != HyphenationPolicy::Disabled)
}

#[cfg(test)]
fn optimal_line_breaks(segments: &[LineBreakMeasure], max_extent: f32) -> Vec<usize> {
    optimal_uniform_line_breaks(segments, max_extent, true).breaks
}

fn optimal_uniform_line_breaks(
    segments: &[LineBreakMeasure],
    max_extent: f32,
    allow_hyphenation: bool,
) -> LineBreakResult {
    let len = segments.len();
    if len == 0 {
        return LineBreakResult {
            breaks: Vec::new(),
            profiles: Vec::new(),
            overflowed: false,
            cost: 0.0,
        };
    }
    if !max_extent.is_finite() || max_extent <= 0.0 {
        return LineBreakResult {
            breaks: vec![len],
            profiles: vec![LineProfile {
                width: max_extent,
                start: None,
                center_offset: 0.0,
                block_baseline: 0.0,
            }],
            overflowed: false,
            cost: 0.0,
        };
    }

    let mut dp = vec![f32::INFINITY; len + 1];
    let mut prev = vec![None; len + 1];
    dp[0] = 0.0;

    for start in 0..len {
        if !dp[start].is_finite() {
            continue;
        }
        let mut advance = 0.0f32;
        for end in start + 1..=len {
            advance += segments[end - 1].advance;
            let suffix_advance = if end < len {
                segments[end - 1].break_suffix_advance
            } else {
                0.0
            };
            let line_advance = advance - segments[end - 1].trailing_advance + suffix_advance;
            let hyphenated_break = end < len && suffix_advance > 0.0;
            if hyphenated_break && !allow_hyphenation {
                continue;
            }
            let mut cost = dp[start] + line_break_badness(line_advance, max_extent);
            if hyphenated_break {
                cost += LINE_BREAK_HYPHEN_PENALTY;
            }
            if end < len {
                cost += segments[end - 1].break_penalty;
            }

            if cost < dp[end] {
                dp[end] = cost;
                prev[end] = Some(start);
            }
            if segments[end - 1].is_mandatory || advance > max_extent {
                break;
            }
        }
    }

    if !dp[len].is_finite() {
        return LineBreakResult {
            breaks: vec![len],
            profiles: vec![LineProfile {
                width: max_extent,
                start: None,
                center_offset: 0.0,
                block_baseline: 0.0,
            }],
            overflowed: segments.iter().map(|segment| segment.advance).sum::<f32>()
                - segments
                    .last()
                    .map_or(0.0, |segment| segment.trailing_advance)
                > max_extent,
            cost: f32::INFINITY,
        };
    }

    let mut breaks = Vec::new();
    let mut index = len;
    while index > 0 {
        breaks.push(index);
        let Some(previous) = prev[index] else {
            return LineBreakResult {
                breaks: vec![len],
                profiles: vec![LineProfile {
                    width: max_extent,
                    start: None,
                    center_offset: 0.0,
                    block_baseline: 0.0,
                }],
                overflowed: true,
                cost: f32::INFINITY,
            };
        };
        index = previous;
    }
    breaks.reverse();
    let overflowed = breaks_overflow(segments, &breaks, &[max_extent; 1]);
    let profiles = vec![
        LineProfile {
            width: max_extent,
            start: None,
            center_offset: 0.0,
            block_baseline: 0.0,
        };
        breaks.len()
    ];
    LineBreakResult {
        breaks,
        profiles,
        overflowed,
        cost: dp[len],
    }
}

fn comic_line_breaks(
    segments: &[LineBreakMeasure],
    balloon: &ComicBalloon,
    (writing_mode, alignment): (WritingMode, TextAlign),
    line_height: f32,
    line_ink: InkBand,
    air: (f32, f32),
    policy: HyphenationPolicy,
) -> LineBreakResult {
    let (air_x, air_y) = air;
    let block_extent = if writing_mode.is_vertical() {
        balloon.width
    } else {
        balloon.height
    };
    let block_air = if writing_mode.is_vertical() {
        air_x
    } else {
        air_y
    };
    let available_block_extent = (block_extent - block_air * 2.0).max(0.0);
    let maximum_lines = if line_ink.thickness() <= available_block_extent {
        1 + ((available_block_extent - line_ink.thickness()) / line_height).floor() as usize
    } else {
        0
    }
    .min(COMIC_MAX_LINES)
    .min(segments.len());
    if maximum_lines == 0 {
        let mut fallback = line_breaks_with_policy(
            segments,
            balloon.inline_extent(writing_mode, air_x, air_y),
            policy,
        );
        fallback.overflowed = true;
        return fallback;
    }

    // A mandatory break on the final segment already ends the last line.
    let minimum_lines = 1 + segments[..segments.len() - 1]
        .iter()
        .filter(|segment| segment.is_mandatory)
        .count();
    let select = |allow_hyphenation: bool| {
        let mut fallback: Option<LineBreakResult> = None;
        for line_count in minimum_lines..=maximum_lines {
            let geometry = balloon.line_profiles(
                writing_mode,
                alignment,
                line_count,
                line_height,
                line_ink,
                air,
            );
            for profiles in geometry.ranked() {
                let Some(result) =
                    exact_profiled_line_breaks(segments, profiles, allow_hyphenation, true)
                else {
                    continue;
                };
                if !result.overflowed {
                    return Some(result);
                }
                // A cheaper break can overflow even when another break fits on this axis.
                if alignment == TextAlign::Center
                    && let Some(fit) = exact_profiled_line_breaks(
                        segments,
                        result.profiles.clone(),
                        allow_hyphenation,
                        false,
                    )
                {
                    return Some(fit);
                }
                if fallback
                    .as_ref()
                    .is_none_or(|best| line_count == best.profiles.len() && result.cost < best.cost)
                {
                    fallback = Some(result);
                }
            }
            if let Some(axis) = geometry.feasible_center(segments, allow_hyphenation)
                && let Some(fit) = exact_profiled_line_breaks(
                    segments,
                    geometry.at_axis(axis),
                    allow_hyphenation,
                    false,
                )
            {
                return Some(fit);
            }
        }
        fallback
    };

    if policy == HyphenationPolicy::LastResort
        && let Some(without_hyphens) = select(false)
        && !without_hyphens.overflowed
    {
        return without_hyphens;
    }

    select(policy != HyphenationPolicy::Disabled).unwrap_or_else(|| {
        let mut fallback = line_breaks_with_policy(
            segments,
            balloon.inline_extent(writing_mode, air_x, air_y),
            policy,
        );
        fallback.overflowed = true;
        fallback
    })
}

fn line_choices(
    segments: &[LineBreakMeasure],
    start: usize,
    last_end: usize,
    max_width: f32,
    allow_hyphenation: bool,
) -> impl Iterator<Item = (usize, f32, bool)> {
    let mut ends = start + 1..=last_end;
    let mut advance = 0.0;
    let mut finished = false;
    std::iter::from_fn(move || {
        if finished {
            return None;
        }
        for end in ends.by_ref() {
            let segment = segments[end - 1];
            advance += segment.advance;
            let suffix = if end < segments.len() {
                segment.break_suffix_advance
            } else {
                0.0
            };
            if suffix > 0.0 && !allow_hyphenation {
                continue;
            }
            finished = segment.is_mandatory || advance > max_width;
            return Some((
                end,
                advance - segment.trailing_advance + suffix,
                suffix > 0.0,
            ));
        }
        None
    })
}

fn exact_profiled_line_breaks(
    segments: &[LineBreakMeasure],
    profiles: Vec<LineProfile>,
    allow_hyphenation: bool,
    allow_overflow: bool,
) -> Option<LineBreakResult> {
    let len = segments.len();
    let line_count = profiles.len();
    if line_count == 0 || line_count > len {
        return None;
    }
    let mut dp = vec![vec![f32::INFINITY; len + 1]; line_count + 1];
    let mut previous = vec![vec![None; len + 1]; line_count + 1];
    dp[0][0] = 0.0;

    for line in 0..line_count {
        let remaining_lines = line_count - line - 1;
        for start in line..len {
            if !dp[line][start].is_finite() {
                continue;
            }
            let last_end = len - remaining_lines;
            let width = profiles[line].width.max(1.0);
            for (end, line_advance, hyphenated_break) in
                line_choices(segments, start, last_end, width, allow_hyphenation)
            {
                if !allow_overflow && line_advance > profiles[line].width + f32::EPSILON {
                    continue;
                }
                let overflow = (line_advance - width).max(0.0) / width;
                let slack = (width - line_advance).max(0.0) / width;
                let mut cost = dp[line][start]
                    + slack * slack * 1_000.0
                    + overflow * overflow * COMIC_LINE_OVERFLOW_PENALTY;
                if hyphenated_break {
                    cost += LINE_BREAK_HYPHEN_PENALTY;
                }
                if end < len {
                    cost += segments[end - 1].break_penalty;
                }
                if cost < dp[line + 1][end] {
                    dp[line + 1][end] = cost;
                    previous[line + 1][end] = Some(start);
                }
            }
        }
    }

    let mut cost = dp[line_count][len];
    if !cost.is_finite() {
        return None;
    }
    cost = cost / line_count as f32 + line_count as f32 * 8.0;
    let mut breaks = Vec::with_capacity(line_count);
    let mut end = len;
    for line in (1..=line_count).rev() {
        breaks.push(end);
        end = previous[line][end]?;
    }
    if end != 0 {
        return None;
    }
    breaks.reverse();
    let widths = profiles
        .iter()
        .map(|profile| profile.width)
        .collect::<Vec<_>>();
    let overflowed = breaks_overflow(segments, &breaks, &widths);
    Some(LineBreakResult {
        breaks,
        profiles,
        overflowed,
        cost,
    })
}

fn breaks_overflow(segments: &[LineBreakMeasure], breaks: &[usize], widths: &[f32]) -> bool {
    let mut start = 0usize;
    for (line, end) in breaks.iter().copied().enumerate() {
        let mut advance = segments[start..end]
            .iter()
            .map(|segment| segment.advance)
            .sum::<f32>();
        advance -= segments[end - 1].trailing_advance;
        if end < segments.len() {
            advance += segments[end - 1].break_suffix_advance;
        }
        let Some(width) = widths
            .get(line)
            .copied()
            .or_else(|| widths.first().copied())
        else {
            return true;
        };
        if advance > width + f32::EPSILON {
            return true;
        }
        start = end;
    }
    false
}

impl ComicBalloon {
    fn air(&self, ink_height: f32) -> f32 {
        self.minimum_air.max(ink_height)
    }

    fn inline_extent(&self, writing_mode: WritingMode, air_x: f32, air_y: f32) -> f32 {
        if writing_mode.is_vertical() {
            (self.height - air_y * 2.0).max(1.0)
        } else {
            (self.width - air_x * 2.0).max(1.0)
        }
    }

    fn line_profiles(
        &self,
        writing_mode: WritingMode,
        alignment: TextAlign,
        line_count: usize,
        line_height: f32,
        line_ink: InkBand,
        (air_x, air_y): (f32, f32),
    ) -> BalloonProfiles {
        let (block_extent, inline_extent, block_air, inline_air) = if writing_mode.is_vertical() {
            (self.width, self.height, air_x, air_y)
        } else {
            (self.height, self.width, air_y, air_x)
        };
        let inline_center = inline_extent * 0.5;
        let inline_radius = inline_center - inline_air;
        let block_size = line_ink.thickness() + line_count.saturating_sub(1) as f32 * line_height;
        let mut bands = Vec::with_capacity(line_count);
        let mut baselines = Vec::with_capacity(line_count);
        if inline_radius > 0.0
            && let Some(origin) =
                self.centered_block_origin(writing_mode, block_extent, block_air, block_size)
        {
            for line in 0..line_count {
                let index = if writing_mode.is_vertical() {
                    line_count - 1 - line
                } else {
                    line
                };
                let start = origin + index as f32 * line_height;
                let end = start + line_ink.thickness();
                baselines.push(start + line_ink.before);
                let spans = if self.contour.len() >= 3 {
                    self.band_spans(writing_mode, start, end, inline_air)
                } else {
                    let radius = block_extent * 0.5 - block_air;
                    let distance = (block_extent * 0.5 - start).max(end - block_extent * 0.5);
                    let fraction = (distance / radius).clamp(0.0, 1.0);
                    let half = inline_radius * (1.0 - fraction * fraction).sqrt();
                    vec![(inline_center - half, inline_center + half)]
                };
                bands.push(spans);
            }
        }

        let mut axes = balloon_axes(&bands);
        if alignment != TextAlign::Center {
            axes.truncate(1);
        }
        BalloonProfiles {
            bands,
            baselines,
            inline_center,
            alignment,
            axes,
        }
    }

    fn centered_block_origin(
        &self,
        writing_mode: WritingMode,
        block_extent: f32,
        block_air: f32,
        block_size: f32,
    ) -> Option<f32> {
        let mut first = block_air;
        let mut last = block_extent - block_air;
        if self.contour.len() >= 3 {
            let (minimum, maximum) = self.contour.iter().fold(
                (f32::INFINITY, f32::NEG_INFINITY),
                |(minimum, maximum), &(x, y)| {
                    let block = if writing_mode.is_vertical() { x } else { y };
                    (minimum.min(block), maximum.max(block))
                },
            );
            first = first.max(minimum + block_air);
            last = last.min(maximum - block_air);
        }
        ((last - first) + f32::EPSILON >= block_size)
            .then_some(first + (last - first - block_size) * 0.5)
    }

    /// Intersect every component across the ink band before choosing a room. A
    /// concave contour can change which component is widest halfway through.
    fn band_spans(&self, mode: WritingMode, start: f32, end: f32, air: f32) -> Vec<(f32, f32)> {
        let first = start.next_up().min(end);
        let last = end.next_down().max(start);
        let mut samples: Vec<_> = (0..5)
            .map(|i| start + (end - start) * (i as f32 + 0.5) / 5.0)
            // Sample each boundary from inside the band, including at horizontal contour edges.
            .chain([first, last])
            .chain(
                self.contour
                    .iter()
                    .map(|&(x, y)| if mode.is_vertical() { x } else { y })
                    .filter(|&v| start < v && v < end)
                    // A horizontal edge can have different spans on either side.
                    .flat_map(|v| [v.next_down().max(first), v, v.next_up().min(last)]),
            )
            .collect();
        samples.sort_unstable_by(f32::total_cmp);
        samples.dedup();
        let mut spans = polygon_inline_spans(&self.contour, mode, samples[0]);
        for v in samples.into_iter().skip(1) {
            if spans.is_empty() {
                break;
            }
            spans = intersect_spans(&spans, &polygon_inline_spans(&self.contour, mode, v));
        }
        spans
            .into_iter()
            .map(|(left, right)| (left + air, right - air))
            .filter(|(left, right)| right > left)
            .collect()
    }
}

impl BalloonProfiles {
    fn ranked(&self) -> impl Iterator<Item = Vec<LineProfile>> {
        self.axes.iter().map(|&axis| self.at_axis(axis))
    }

    fn at_axis(&self, axis: f32) -> Vec<LineProfile> {
        let spans: Vec<_> = self
            .bands
            .iter()
            .map(|band| {
                band.iter()
                    .copied()
                    .find(|&(left, right)| left < axis && axis < right)
                    .unwrap_or((axis, axis))
            })
            .collect();
        let shared_start = spans
            .iter()
            .map(|span| span.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let shared_end = spans
            .iter()
            .map(|span| span.1)
            .fold(f32::INFINITY, f32::min);
        spans
            .into_iter()
            .zip(&self.baselines)
            .map(|((left, right), &block_baseline)| {
                let (start, width) = match self.alignment {
                    TextAlign::Center => {
                        let reach = (axis - left).min(right - axis);
                        (axis - reach, 2.0 * reach)
                    }
                    TextAlign::Left | TextAlign::Justify => (shared_start, right - shared_start),
                    TextAlign::Right => (left, shared_end - left),
                };
                LineProfile {
                    width,
                    start: Some(start),
                    center_offset: axis - self.inline_center,
                    block_baseline,
                }
            })
            .collect()
    }

    fn feasible_center(
        &self,
        segments: &[LineBreakMeasure],
        allow_hyphenation: bool,
    ) -> Option<f32> {
        if self.alignment != TextAlign::Center || self.axes.len() < 2 {
            return None;
        }
        // For each possible line break, retain the union of axes that fit all
        // lines up to it. A closed interval allows an exact-width fit at one axis.
        let len = segments.len();
        let mut feasible = vec![Vec::new(); len + 1];
        feasible[0].push((f32::NEG_INFINITY, f32::INFINITY));
        for (line, band) in self.bands.iter().enumerate() {
            let max_width = band.iter().map(|(l, r)| r - l).reduce(f32::max)?;
            let last_end = len - (self.bands.len() - line - 1);
            let mut next = vec![Vec::new(); len + 1];
            for (start, prior) in feasible[..last_end].iter().enumerate() {
                if prior.is_empty() {
                    continue;
                }
                for (end, width, _) in
                    line_choices(segments, start, last_end, max_width, allow_hyphenation)
                {
                    for &(left, right) in band {
                        for &(lo, hi) in prior {
                            let lo = lo.max(left + width * 0.5);
                            let hi = hi.min(right - width * 0.5);
                            if lo <= hi {
                                next[end].push((lo, hi));
                            }
                        }
                    }
                }
            }
            for intervals in &mut next {
                intervals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                intervals.dedup_by(|later, earlier| {
                    if later.0 > earlier.1 {
                        return false;
                    }
                    earlier.1 = earlier.1.max(later.1);
                    true
                });
            }
            feasible = next;
        }
        let &(lo, hi) = feasible[len].first()?;
        Some(lo.midpoint(hi))
    }
}

fn intersect_spans(a: &[(f32, f32)], b: &[(f32, f32)]) -> Vec<(f32, f32)> {
    a.iter()
        .flat_map(|&(left, right)| {
            b.iter().filter_map(move |&(start, end)| {
                let overlap = (left.max(start), right.min(end));
                (overlap.0 < overlap.1).then_some(overlap)
            })
        })
        .collect()
}

fn balloon_axes(bands: &[Vec<(f32, f32)>]) -> Vec<f32> {
    // Include the common span's middle: two overlapping bands need not
    // contain each other's midpoint.
    let Some((first, rest)) = bands.split_first() else {
        return Vec::new();
    };
    let common = rest
        .iter()
        .fold(first.clone(), |common, band| intersect_spans(&common, band));
    let mut candidates: Vec<_> = bands
        .iter()
        .flatten()
        .chain(&common)
        .map(|(left, right)| (left + right) * 0.5)
        .collect();
    candidates.sort_unstable_by(f32::total_cmp);
    candidates.dedup();
    let mut ranked: Vec<_> = candidates
        .into_iter()
        .map(|axis| {
            let mut reached = 0;
            let mut total = 0.0_f32;
            for band in bands {
                if let Some(&(left, right)) = band
                    .iter()
                    .find(|&&(left, right)| left < axis && axis < right)
                {
                    reached += 1;
                    total += (axis - left).min(right - axis);
                }
            }
            (reached, total, axis)
        })
        .collect();
    ranked.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
    ranked.into_iter().rev().map(|(_, _, axis)| axis).collect()
}

fn polygon_inline_spans(
    contour: &[(f32, f32)],
    writing_mode: WritingMode,
    block: f32,
) -> Vec<(f32, f32)> {
    if contour.len() < 3 {
        return Vec::new();
    }
    let mut intersections = Vec::new();
    for index in 0..contour.len() {
        let first = contour[index];
        let second = contour[(index + 1) % contour.len()];
        let (first_block, first_inline, second_block, second_inline) = if writing_mode.is_vertical()
        {
            (first.0, first.1, second.0, second.1)
        } else {
            (first.1, first.0, second.1, second.0)
        };
        if (first_block <= block && second_block > block)
            || (second_block <= block && first_block > block)
        {
            let fraction = (block - first_block) / (second_block - first_block);
            intersections.push(first_inline + (second_inline - first_inline) * fraction);
        }
    }
    intersections.sort_by(f32::total_cmp);
    intersections
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

fn comic_break_penalty(text: &str, boundary: usize) -> f32 {
    let boundary = boundary.min(text.len());
    let before = text[..boundary].trim_end();
    let after = text[boundary..].trim_start();
    match before.chars().next_back() {
        Some('.' | '!' | '?' | '…' | '‼' | '⁇' | '⁈' | '⁉') => return 0.0,
        Some(',' | ';' | ':' | '—' | '–') => return 20.0,
        _ => {}
    }
    let next_word = after
        .split(|character: char| !character.is_ascii_alphabetic())
        .next()
        .unwrap_or_default();
    if matches!(
        next_word.to_ascii_lowercase().as_str(),
        "and" | "but" | "or" | "so" | "because" | "although" | "while" | "then"
    ) {
        return 40.0;
    }
    let previous_word = before
        .rsplit(|character: char| !character.is_ascii_alphabetic())
        .next()
        .unwrap_or_default();
    if matches!(
        previous_word.to_ascii_lowercase().as_str(),
        "a" | "an" | "the" | "to" | "of" | "for" | "in" | "on" | "at" | "with" | "from"
    ) {
        300.0
    } else {
        100.0
    }
}

fn line_break_badness(line_advance: f32, max_extent: f32) -> f32 {
    if line_advance <= max_extent {
        (max_extent - line_advance).powi(3)
    } else {
        (line_advance - max_extent).powi(3) * LINE_BREAK_OVERFLOW_MULTIPLIER
    }
}

fn centered_x_offset(x_min: f32, x_max: f32) -> f32 {
    -((x_min + x_max) * 0.5)
}

fn cjk_emphasis_mark(character: char) -> Option<char> {
    let mut normalized =
        ComposingNormalizerBorrowed::new_nfkc().normalize_iter(std::iter::once(character));
    let character = normalized.next()?;
    if normalized.next().is_some() {
        return None;
    }
    match character {
        '!' => Some('！'),
        '?' => Some('？'),
        _ => None,
    }
}

fn normalize_cjk_emphasis_punctuation(text: &str) -> Option<(String, Vec<u32>)> {
    if !text
        .chars()
        .any(|character| cjk_emphasis_mark(character).is_some_and(|mark| mark != character))
    {
        return None;
    }

    let mut output = String::with_capacity(text.len());
    let mut cluster_map = Vec::with_capacity(text.len());
    for (source_offset, character) in text.char_indices() {
        let output_character = cjk_emphasis_mark(character).unwrap_or(character);
        output.push(output_character);
        cluster_map.resize(output.len(), source_offset as u32);
    }

    Some((output, cluster_map))
}

fn reorder_visual(levels: &[unicode_bidi::Level]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..levels.len()).collect();
    if levels.is_empty() {
        return indices;
    }

    let max_level = levels.iter().map(|l| l.number()).max().unwrap();
    let min_odd_level = levels
        .iter()
        .map(|l| l.number())
        .filter(|&n| n % 2 != 0)
        .min()
        .unwrap_or(u8::MAX);

    if min_odd_level == u8::MAX {
        return indices;
    }

    for level in (min_odd_level..=max_level).rev() {
        let mut i = 0;
        while i < levels.len() {
            if levels[i].number() >= level {
                let mut j = i;
                while j < levels.len() && levels[j].number() >= level {
                    j += 1;
                }
                indices[i..j].reverse();
                i = j;
            } else {
                i += 1;
            }
        }
    }
    indices
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::fonts::{Font, FontSystem};
    use skrifa::{MetadataProvider, instance::Size};

    fn any_system_font() -> Font {
        let mut fonts = FontSystem::new();

        // Prefer fonts that are commonly available depending on OS/environment.
        // This is only used to construct a `TextLayout` for calling `compute_bounds`.
        let preferred = [
            "Yu Gothic",
            "MS Gothic",
            "Noto Sans CJK JP",
            "Noto Sans",
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
        ];

        for name in preferred {
            if let Ok(font) = fonts.query_family(name) {
                return font;
            }
        }
        fonts
            .first_font()
            .expect("no system font available for tests")
    }

    fn assert_approx_eq(actual: f32, expected: f32) {
        if actual.is_infinite()
            && expected.is_infinite()
            && actual.is_sign_positive() == expected.is_sign_positive()
        {
            return;
        }
        let eps = 1e-4;
        assert!(
            (actual - expected).abs() <= eps,
            "expected {expected}, got {actual}"
        );
    }

    fn comic_balloon(
        width: f32,
        height: f32,
        contour: Vec<(f32, f32)>,
        minimum_air: f32,
    ) -> ComicBalloon {
        ComicBalloon {
            width,
            height,
            contour,
            minimum_air,
        }
    }

    #[test]
    fn capped_auto_size_shrinks_text_to_fit_the_available_height() -> anyhow::Result<()> {
        let font = any_system_font();
        let preferred_size = 24.0;
        let fixed = TextLayout::new(&font)
            .with_font_size(preferred_size)
            .with_max_width(1_000.0)
            .run("First\nSecond\nThird")?;
        let max_height = fixed.height * 0.65;

        let fitted = TextLayout::new(&font)
            .with_max_font_size(preferred_size)
            .with_max_width(1_000.0)
            .with_max_height(max_height)
            .run("First\nSecond\nThird")?;

        assert!(fitted.font_size < preferred_size);
        assert!(fitted.width <= 1_000.0);
        assert!(fitted.height <= max_height + 0.01);
        Ok(())
    }

    #[test]
    fn capped_auto_size_does_not_enlarge_text_that_already_fits() -> anyhow::Result<()> {
        let font = any_system_font();
        let preferred_size = 18.0;

        let fitted = TextLayout::new(&font)
            .with_max_font_size(preferred_size)
            .with_max_width(1_000.0)
            .with_max_height(1_000.0)
            .run("Fits")?;

        assert_eq!(fitted.font_size, preferred_size);
        Ok(())
    }

    #[test]
    fn capped_auto_size_respects_the_readable_floor() -> anyhow::Result<()> {
        let font = any_system_font();

        let fitted = TextLayout::new(&font)
            .with_max_font_size(16.0)
            .with_min_font_size(8.0)
            .with_max_width(12.0)
            .with_max_height(12.0)
            .run("This translation cannot fit")?;

        assert_eq!(fitted.font_size, 8.0);
        Ok(())
    }

    #[test]
    fn capped_auto_size_keeps_the_configured_line_height() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "First\nSecond\nThird\nFourth";
        let expected = TextLayout::new(&font)
            .with_font_size(16.0)
            .with_line_height(1.2)
            .run(text)?;
        let max_height = expected.height - 1.0;

        let fitted = TextLayout::new(&font)
            .with_max_font_size(16.0)
            .with_min_font_size(16.0)
            .with_line_height(1.2)
            .with_max_width(1_000.0)
            .with_max_height(max_height)
            .run(text)?;

        assert_eq!(fitted.font_size, 16.0);
        assert!(fitted.height > max_height);
        assert_approx_eq(fitted.height, expected.height);
        let leading = fitted.lines[1].baseline.1 - fitted.lines[0].baseline.1;
        assert_approx_eq(leading, 16.0 * 1.2);
        Ok(())
    }

    #[test]
    fn font_size_search_handles_non_monotonic_balloon_fit() -> anyhow::Result<()> {
        let calls = Cell::new(0);
        let fitted = largest_fitting_font_size(
            9.0,
            24.0,
            |size| {
                calls.set(calls.get() + 1);
                Ok(size)
            },
            |size| (10.0..=12.0).contains(size),
        )?
        .expect("the larger fitting range should be found");

        assert!((11.99..=12.0).contains(&fitted));
        assert!(calls.get() <= 18);
        Ok(())
    }

    #[test]
    fn comic_auto_size_uses_tall_balloon_capacity() -> anyhow::Result<()> {
        let font = any_system_font();
        let width = 130.0;
        let height = 250.0;
        let fitted = TextLayout::new(&font)
            .with_max_font_size(width)
            .with_min_font_size(9.0)
            .with_line_height(1.2)
            .with_hyphenation_policy(HyphenationPolicy::LastResort)
            .with_max_width(width)
            .with_max_height(height)
            .with_comic_balloon(
                width,
                height,
                vec![(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)],
                4.0,
            )
            .run(
                "The Justice Realization Committee is a volunteer organization dedicated to the cause of justice.",
            )?;

        assert!(
            !fitted.overflowed(),
            "auto-fit overflowed at size {} with {} lines and height {}",
            fitted.font_size,
            fitted.lines.len(),
            fitted.height
        );
        assert!(
            fitted.height >= height * 0.5,
            "auto-fit left excessive vertical space: font size {}, height {}",
            fitted.font_size,
            fitted.height
        );
        Ok(())
    }

    #[test]
    fn optimal_line_breaks_balance_ragged_lines() {
        let segments = vec![
            LineBreakMeasure {
                advance: 30.0,
                trailing_advance: 0.0,
                break_suffix_advance: 0.0,
                break_penalty: 0.0,
                is_mandatory: false,
            };
            7
        ];

        assert_eq!(optimal_line_breaks(&segments, 100.0), vec![2, 4, 7]);
    }

    #[test]
    fn line_fit_excludes_invisible_trailing_break_space() {
        let segments = [
            LineBreakMeasure {
                advance: 55.0,
                trailing_advance: 10.0,
                break_suffix_advance: 0.0,
                break_penalty: 0.0,
                is_mandatory: false,
            },
            LineBreakMeasure {
                advance: 40.0,
                trailing_advance: 0.0,
                break_suffix_advance: 0.0,
                break_penalty: 0.0,
                is_mandatory: false,
            },
        ];

        let result = line_breaks_with_policy(&segments, 50.0, HyphenationPolicy::Disabled);

        assert_eq!(result.breaks, [1, 2]);
        assert!(!result.overflowed);
    }

    #[test]
    fn comic_profiles_are_widest_in_the_middle() {
        let balloon = comic_balloon(200.0, 120.0, Vec::new(), 0.0);
        let profiles = balloon
            .line_profiles(
                WritingMode::Horizontal,
                TextAlign::Center,
                5,
                16.0,
                InkBand {
                    before: 8.0,
                    after: 8.0,
                },
                (10.0, 10.0),
            )
            .ranked()
            .next()
            .unwrap();

        assert!(profiles[0].width < profiles[1].width);
        assert!(profiles[1].width < profiles[2].width);
        assert!((profiles[0].width - profiles[4].width).abs() < 0.001);
        assert!((profiles[1].width - profiles[3].width).abs() < 0.001);
    }

    #[test]
    fn vertical_comic_profiles_are_tallest_in_the_middle() {
        let balloon = comic_balloon(120.0, 200.0, Vec::new(), 0.0);
        let profiles = balloon
            .line_profiles(
                WritingMode::VerticalRl,
                TextAlign::Center,
                5,
                16.0,
                InkBand {
                    before: 8.0,
                    after: 8.0,
                },
                (10.0, 10.0),
            )
            .ranked()
            .next()
            .unwrap();

        assert!(profiles[0].width < profiles[1].width);
        assert!(profiles[1].width < profiles[2].width);
        assert!((profiles[0].width - profiles[4].width).abs() < 0.001);
        assert!((profiles[1].width - profiles[3].width).abs() < 0.001);
    }

    #[test]
    fn comic_profiles_use_the_detected_contour_without_an_extra_ellipse() {
        let balloon = comic_balloon(
            200.0,
            120.0,
            vec![(0.0, 0.0), (200.0, 0.0), (200.0, 120.0), (0.0, 120.0)],
            0.0,
        );
        let profiles = balloon
            .line_profiles(
                WritingMode::Horizontal,
                TextAlign::Center,
                5,
                16.0,
                InkBand {
                    before: 8.0,
                    after: 8.0,
                },
                (10.0, 10.0),
            )
            .ranked()
            .next()
            .unwrap();

        for profile in profiles {
            assert_approx_eq(profile.width, 180.0);
        }
    }

    #[test]
    fn comic_profiles_measure_the_glyph_ink_band_instead_of_leading() {
        let balloon = comic_balloon(
            200.0,
            120.0,
            vec![(100.0, 0.0), (200.0, 60.0), (100.0, 120.0), (0.0, 60.0)],
            0.0,
        );
        let result = comic_line_breaks(
            &[LineBreakMeasure {
                advance: 140.0,
                trailing_advance: 0.0,
                break_suffix_advance: 0.0,
                break_penalty: 0.0,
                is_mandatory: false,
            }],
            &balloon,
            (WritingMode::Horizontal, TextAlign::Center),
            80.0,
            InkBand {
                before: 20.0,
                after: 20.0,
            },
            (0.0, 10.0),
            HyphenationPolicy::Disabled,
        );

        assert_approx_eq(result.profiles[0].width, 200.0 * 40.0 / 60.0);
        assert!(result.overflowed);
    }

    #[test]
    fn centered_forced_lines_fit_the_common_span_at_full_size() -> anyhow::Result<()> {
        let font = any_system_font();
        let size = 16.0;
        let air = 20.0;
        let width = 149.0 + air * 2.0;
        let height = 144.0;
        // Relative to the air margin, the first two spans are [0, 100] and the third is [49, 149].
        let layout = TextLayout::new(&font)
            .with_line_height(2.0)
            .with_alignment(TextAlign::Center)
            .with_max_width(width)
            .with_max_height(height)
            .with_comic_balloon(
                width,
                height,
                vec![
                    (0.0, 0.0),
                    (100.0 + air * 2.0, 0.0),
                    (100.0 + air * 2.0, 88.0),
                    (width, 88.0),
                    (width, height),
                    (49.0, height),
                    (49.0, 88.0),
                    (0.0, 88.0),
                ],
                air,
            );

        for builder in [
            layout.clone().with_font_size(size),
            layout.with_min_font_size(8.0).with_max_font_size(size),
        ] {
            let run = builder.run("WW\nWW\nWW")?;
            assert!(!run.overflowed());
            assert_eq!(run.lines.len(), 3);
            assert_eq!(run.font_size, size);
            assert_approx_eq(width * 0.5 + run.placement_offset_x() - air, 74.5);
        }
        Ok(())
    }

    #[test]
    fn centered_balloon_prefers_a_fitting_break_over_a_natural_pause() {
        let text = "Go to bed. Now!";
        let segments: Vec<_> = LineBreaker::new()
            .line_segments(text)
            .into_iter()
            .map(|segment| {
                let word = &text[segment.range];
                LineBreakMeasure {
                    advance: word.len() as f32 * 10.0,
                    trailing_advance: (word.len() - word.trim_end().len()) as f32 * 10.0,
                    break_suffix_advance: 0.0,
                    break_penalty: comic_break_penalty(text, segment.next_offset),
                    is_mandatory: segment.is_mandatory,
                }
            })
            .collect();
        let balloon = comic_balloon(
            99.0,
            40.0,
            vec![(0.0, 0.0), (99.0, 0.0), (99.0, 40.0), (0.0, 40.0)],
            0.0,
        );
        let result = comic_line_breaks(
            &segments,
            &balloon,
            (WritingMode::Horizontal, TextAlign::Center),
            20.0,
            InkBand {
                before: 5.0,
                after: 5.0,
            },
            (0.0, 0.0),
            HyphenationPolicy::Disabled,
        );

        assert!(!result.overflowed, "chosen breaks: {:?}", result.breaks);
        assert_eq!(result.breaks, [2, 4]);
    }

    #[test]
    fn unequal_centered_lines_fit_between_the_span_midpoints() {
        let balloon = comic_balloon(
            140.0,
            40.0,
            vec![
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 20.0),
                (140.0, 20.0),
                (140.0, 40.0),
                (40.0, 40.0),
                (40.0, 20.0),
                (0.0, 20.0),
            ],
            0.0,
        );
        let measure = |advance, is_mandatory| LineBreakMeasure {
            advance,
            trailing_advance: 0.0,
            break_suffix_advance: 0.0,
            break_penalty: 0.0,
            is_mandatory,
        };
        for (segments, breaks, axis) in [
            (vec![measure(80.0, true), measure(36.0, true)], [1, 2], 59.0),
            (
                vec![measure(80.0, true), measure(40.0, false)],
                [1, 2],
                60.0,
            ),
            // The optional break after the first segment leaves the second line too wide.
            (
                vec![
                    measure(12.0, false),
                    measure(68.0, false),
                    measure(36.0, false),
                ],
                [2, 3],
                59.0,
            ),
        ] {
            let result = comic_line_breaks(
                &segments,
                &balloon,
                (WritingMode::Horizontal, TextAlign::Center),
                20.0,
                InkBand {
                    before: 5.0,
                    after: 5.0,
                },
                (0.0, 0.0),
                HyphenationPolicy::Disabled,
            );
            assert!(!result.overflowed);
            assert_eq!(result.breaks, breaks);
            for profile in result.profiles {
                assert_approx_eq(70.0 + profile.center_offset, axis);
            }
        }
    }

    #[test]
    fn comic_profiles_respect_asymmetric_contours() {
        let balloon = comic_balloon(
            200.0,
            120.0,
            vec![(0.0, 0.0), (140.0, 0.0), (140.0, 120.0), (0.0, 120.0)],
            0.0,
        );
        let profile = balloon
            .line_profiles(
                WritingMode::Horizontal,
                TextAlign::Center,
                1,
                16.0,
                InkBand {
                    before: 8.0,
                    after: 8.0,
                },
                (10.0, 10.0),
            )
            .ranked()
            .next()
            .unwrap()[0];

        assert!(profile.center_offset < -20.0);
        assert!(profile.width < 130.0);
    }

    #[test]
    fn comic_profiles_keep_one_axis_across_an_irregular_contour() {
        let balloon = comic_balloon(
            200.0,
            160.0,
            vec![(30.0, 0.0), (200.0, 0.0), (165.0, 160.0), (0.0, 160.0)],
            0.0,
        );
        let horizontal = balloon
            .line_profiles(
                WritingMode::Horizontal,
                TextAlign::Center,
                4,
                24.0,
                InkBand {
                    before: 10.0,
                    after: 10.0,
                },
                (8.0, 8.0),
            )
            .ranked()
            .next()
            .unwrap();
        let vertical = balloon
            .line_profiles(
                WritingMode::VerticalRl,
                TextAlign::Center,
                4,
                24.0,
                InkBand {
                    before: 10.0,
                    after: 10.0,
                },
                (8.0, 8.0),
            )
            .ranked()
            .next()
            .unwrap();

        for profiles in [&horizontal, &vertical] {
            let axis = profiles[0].center_offset;
            assert!(
                profiles
                    .iter()
                    .all(|profile| (profile.center_offset - axis).abs() < 0.001)
            );
        }
    }

    #[test]
    fn comic_profiles_stay_centered_when_wider_space_is_elsewhere() {
        let balloon = comic_balloon(
            100.0,
            200.0,
            vec![
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 200.0),
                (50.0, 200.0),
                (50.0, 80.0),
                (0.0, 80.0),
            ],
            0.0,
        );

        let profiles = balloon
            .line_profiles(
                WritingMode::Horizontal,
                TextAlign::Center,
                3,
                20.0,
                InkBand {
                    before: 10.0,
                    after: 10.0,
                },
                (0.0, 0.0),
            )
            .ranked()
            .next()
            .unwrap();

        assert!((profiles[0].block_baseline - 80.0).abs() < 0.01);
        assert!((profiles[2].block_baseline - 120.0).abs() < 0.01);
        assert!(profiles.iter().all(|profile| profile.width <= 50.01));
    }

    #[test]
    fn comic_profiles_center_on_the_divided_lobe_bounds() {
        let balloon = ComicBalloon {
            width: 100.0,
            height: 200.0,
            contour: vec![(0.0, 40.0), (100.0, 40.0), (100.0, 160.0), (0.0, 160.0)],
            minimum_air: 0.0,
        };

        let profiles = balloon
            .line_profiles(
                WritingMode::Horizontal,
                TextAlign::Center,
                3,
                20.0,
                InkBand {
                    before: 10.0,
                    after: 10.0,
                },
                (0.0, 0.0),
            )
            .ranked()
            .next()
            .unwrap();

        assert!((profiles[0].block_baseline - 80.0).abs() < 0.01);
        assert!((profiles[2].block_baseline - 120.0).abs() < 0.01);
    }

    #[test]
    fn comic_profiles_never_offer_an_off_center_origin() {
        let balloon = ComicBalloon {
            width: 100.0,
            height: 160.0,
            contour: vec![(0.0, 0.0), (100.0, 0.0), (100.0, 160.0), (48.0, 160.0)],
            minimum_air: 0.0,
        };
        let line_ink = InkBand {
            before: 5.0,
            after: 5.0,
        };
        let profiles = balloon
            .line_profiles(
                WritingMode::Horizontal,
                TextAlign::Center,
                5,
                20.0,
                line_ink,
                (0.0, 0.0),
            )
            .ranked()
            .next()
            .unwrap();

        assert!((profiles[0].block_baseline - 40.0).abs() < 0.01);
    }

    #[test]
    fn comic_contour_preserves_disjoint_rooms() {
        let balloon = ComicBalloon {
            width: 100.0,
            height: 100.0,
            contour: vec![
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 100.0),
                (70.0, 100.0),
                (70.0, 20.0),
                (40.0, 20.0),
                (40.0, 100.0),
                (0.0, 100.0),
            ],
            minimum_air: 0.0,
        };

        assert_eq!(
            balloon.band_spans(WritingMode::Horizontal, 40.0, 60.0, 0.0),
            [(0.0, 40.0), (70.0, 100.0)]
        );
    }

    #[test]
    fn comic_band_spans_include_the_narrower_contour_edge_at_either_end() {
        for (narrow, wide) in [(0.0, 60.0), (60.0, 0.0)] {
            let balloon = comic_balloon(
                200.0,
                60.0,
                vec![(50.0, narrow), (150.0, narrow), (200.0, wide), (0.0, wide)],
                0.0,
            );
            let band = balloon.band_spans(WritingMode::Horizontal, 0.0, 60.0, 0.0);
            assert_eq!(band.len(), 1);
            assert_approx_eq(band[0].0, 50.0);
            assert_approx_eq(band[0].1, 150.0);
        }
    }

    #[test]
    fn ink_band_includes_the_tip_of_a_notch_ending_in_a_horizontal_edge() {
        let balloon = comic_balloon(
            100.0,
            40.0,
            vec![
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 16.0),
                (60.0, 18.0),
                (100.0, 18.0),
                (100.0, 40.0),
                (0.0, 40.0),
            ],
            0.0,
        );

        let band = balloon.band_spans(WritingMode::Horizontal, 0.0, 40.0, 0.0);
        assert_eq!(band.len(), 1);
        assert_approx_eq(band[0].0, 0.0);
        assert_approx_eq(band[0].1, 60.0);
    }

    #[test]
    fn a_concave_band_keeps_continuous_rooms_and_samples_interior_vertices() {
        // The wider arm switches at y=50. A spike from the right at y=48
        // narrows the right arm between the five regular samples.
        let balloon = comic_balloon(
            100.0,
            100.0,
            vec![
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 47.0),
                (85.0, 48.0),
                (100.0, 49.0),
                (100.0, 100.0),
                (60.0, 100.0),
                (60.0, 50.0),
                (80.0, 50.0),
                (80.0, 20.0),
                (30.0, 20.0),
                (30.0, 100.0),
                (0.0, 100.0),
            ],
            0.0,
        );
        let band = balloon.band_spans(WritingMode::Horizontal, 40.0, 60.0, 0.0);
        assert_eq!(band, [(0.0, 30.0), (80.0, 85.0)]);
        assert_eq!(balloon_axes(&[band]).first(), Some(&15.0));
    }

    #[test]
    fn edge_aligned_ink_uses_the_long_side_of_a_lopsided_balloon() -> anyhow::Result<()> {
        let font = any_system_font();
        let size = 20.0;
        let text = "WWWWWWWW\nWW";
        let base = TextLayout::new(&font)
            .with_font_size(size)
            .with_line_height(2.0);
        let measured = base.run(text)?;
        let wide = measured.lines[0].advance + 3.0 * size;
        let narrow = measured.lines[1].advance + 3.0 * size;
        let height = 150.0;
        for align in [TextAlign::Left, TextAlign::Right] {
            let mut outline = vec![
                (0.0, 0.0),
                (wide, 0.0),
                (wide, height / 2.0),
                (narrow, height / 2.0),
                (narrow, height),
                (0.0, height),
            ];
            if align == TextAlign::Right {
                for (x, _) in &mut outline {
                    *x = wide - *x;
                }
            }
            let builder = base
                .clone()
                .with_max_width(wide)
                .with_max_height(height)
                .with_alignment(align)
                .with_comic_balloon(wide, height, outline, 0.0);
            let run = builder.run(text)?;
            assert!(!run.overflowed());
            assert_eq!(run.lines.len(), 2);
            let origin = (wide - run.width) * 0.5 + run.placement_offset_x();
            let edges = |line: &LayoutLine<'_>| {
                let (x0, _, x1, _) = builder
                    .ink_bounds(size, std::slice::from_ref(line))
                    .unwrap();
                if align == TextAlign::Left {
                    (origin + x0, origin + x1)
                } else {
                    (wide - origin - x1, wide - origin - x0)
                }
            };
            let (start, far) = edges(&run.lines[0]);
            let (short_start, short_far) = edges(&run.lines[1]);
            assert_approx_eq(start, short_start);
            assert!(start >= 0.0 && far > narrow && far <= wide);
            assert!(short_far <= narrow);
        }
        Ok(())
    }

    #[test]
    fn comic_contour_keeps_air_at_both_edges() {
        let balloon = ComicBalloon {
            width: 100.0,
            height: 100.0,
            contour: vec![(50.0, 0.0), (100.0, 0.0), (100.0, 100.0), (50.0, 100.0)],
            minimum_air: 0.0,
        };

        assert_eq!(
            balloon.band_spans(WritingMode::Horizontal, 40.0, 60.0, 10.0),
            [(60.0, 90.0)]
        );
    }

    #[test]
    fn comic_layout_preserves_the_contour_center_for_both_writing_modes() -> anyhow::Result<()> {
        let font = any_system_font();
        let horizontal = TextLayout::new(&font)
            .with_font_size(16.0)
            .with_alignment(TextAlign::Center)
            .with_max_width(200.0)
            .with_max_height(120.0)
            .with_comic_balloon(
                200.0,
                120.0,
                vec![(0.0, 0.0), (140.0, 0.0), (140.0, 120.0), (0.0, 120.0)],
                0.0,
            )
            .run("Hello")?;
        let vertical = TextLayout::new(&font)
            .with_font_size(16.0)
            .with_writing_mode(WritingMode::VerticalRl)
            .with_alignment(TextAlign::Left)
            .with_max_width(120.0)
            .with_max_height(200.0)
            .with_comic_balloon(
                120.0,
                200.0,
                vec![(0.0, 0.0), (120.0, 0.0), (120.0, 140.0), (0.0, 140.0)],
                0.0,
            )
            .run("Hello")?;

        assert!(horizontal.placement_offset_x() < -10.0);
        assert!(vertical.placement_offset_y() < -10.0);
        Ok(())
    }

    #[test]
    fn gain_based_hyphenation_uses_the_words_measured_size() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "Incomprehensibilities";
        let target = 32.0;
        let layout = TextLayout::new(&font)
            .with_line_height(1.2)
            .with_hyphenation_language_tag("en")
            .with_hyphenation_policy(HyphenationPolicy::LastResort);
        let measured = layout.clone().with_font_size(target).run(text)?;
        let extent = measured.lines[0].advance.max(measured.width);
        let room = |ratio| {
            layout
                .clone()
                .with_min_font_size(8.0)
                .with_max_font_size(target)
                .with_max_width(ratio * extent)
                .with_max_height(4.0 * target)
        };
        let near = room(0.94).run(text)?;
        assert!(near.font_size >= 0.9 * target && near.font_size < target);
        assert_eq!(near.lines.len(), 1);

        let cutoff = layout.clone().with_font_size(target * 0.9).run(text)?;
        let exact = room(1.0)
            .with_max_width(cutoff.width.max(cutoff.lines[0].advance))
            .run(text)?;
        assert_eq!(exact.lines.len(), 1);
        assert!(exact.font_size >= target * 0.9);

        let narrow = room(0.70);
        let clean = narrow
            .clone()
            .with_hyphenation_policy(HyphenationPolicy::Disabled)
            .run(text)?;
        let larger = narrow.run(text)?;
        assert!(clean.font_size < 0.9 * target);
        assert!(larger.font_size >= 1.1 * clean.font_size && !larger.overflowed());
        assert_eq!(larger.lines.len(), 2);
        assert!(larger.lines[0].hyphenated);

        // Two lines are possible here, but recover only 5% of the clean size.
        let limited = narrow
            .clone()
            .with_font_size(clean.font_size * 1.05)
            .run(text)?;
        assert!(limited.lines[0].hyphenated && !limited.overflowed());
        let modest = narrow.with_max_height(limited.height).run(text)?;
        assert_eq!(modest.lines.len(), 1);
        assert_approx_eq(modest.font_size, clean.font_size);
        Ok(())
    }

    #[test]
    fn a_drawn_out_sound_breaks_inside_the_run_to_fit() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "Gwaaaaaaaaah";
        let base = TextLayout::new(&font)
            .with_font_size(24.0)
            .with_hyphenation_policy(HyphenationPolicy::LastResort);
        let width = base.run(text)?.lines[0].advance;
        for lang in ["en", "und"] {
            let layout = base.clone().with_hyphenation_language_tag(lang);
            let whole = layout.clone().with_max_width(width * 1.1).run(text)?;
            assert_eq!(whole.lines.len(), 1);
            let split = layout.with_max_width(width * 0.7).run(text)?;
            assert!(!split.overflowed());
            assert_eq!(split.lines.len(), 2);
            assert!(split.lines[0].hyphenated);
            assert_eq!(&text[split.lines[0].range.clone()], "Gwaaaa");
            assert_eq!(&text[split.lines[1].range.clone()], "aaaaah");
        }
        Ok(())
    }

    #[test]
    fn comic_auto_size_does_not_fake_capacity_with_extra_leading() -> anyhow::Result<()> {
        let font = any_system_font();
        let font_size = 16.0;
        let layout = TextLayout::new(&font)
            .with_max_font_size(font_size)
            .with_min_font_size(font_size)
            .with_line_height(1.2)
            .with_alignment(TextAlign::Center)
            .with_max_width(200.0)
            .with_max_height(150.0)
            .with_comic_balloon(
                200.0,
                150.0,
                vec![(0.0, 0.0), (200.0, 0.0), (200.0, 150.0), (0.0, 150.0)],
                10.0,
            )
            .run("First\nSecond\nThird")?;

        let leading = layout.lines[1].baseline.1 - layout.lines[0].baseline.1;
        assert_approx_eq(leading, font_size * 1.2);
        Ok(())
    }

    #[test]
    fn comic_auto_size_preserves_air_for_an_already_filled_layout() -> anyhow::Result<()> {
        let font = any_system_font();
        let font_size = 16.0;
        let layout = TextLayout::new(&font)
            .with_max_font_size(font_size)
            .with_min_font_size(font_size)
            .with_line_height(1.2)
            .with_alignment(TextAlign::Center)
            .with_max_width(200.0)
            .with_max_height(150.0)
            .with_comic_balloon(
                200.0,
                150.0,
                vec![(0.0, 0.0), (200.0, 0.0), (200.0, 150.0), (0.0, 150.0)],
                10.0,
            )
            .run("First\nSecond\nThird\nFourth\nFifth\nSixth")?;

        let leading = layout.lines[1].baseline.1 - layout.lines[0].baseline.1;
        assert_approx_eq(leading, font_size * 1.2);
        Ok(())
    }

    #[test]
    fn comic_breaks_prefer_natural_pauses() {
        let mut segments = vec![
            LineBreakMeasure {
                advance: 30.0,
                trailing_advance: 0.0,
                break_suffix_advance: 0.0,
                break_penalty: 0.0,
                is_mandatory: false,
            };
            5
        ];
        segments[2].break_penalty = 500.0;

        assert_eq!(optimal_line_breaks(&segments, 90.0), vec![2, 5]);
        assert!(comic_break_penalty("Stop! Now", 6) < comic_break_penalty("go and", 3));
        assert!(comic_break_penalty("go and", 3) < comic_break_penalty("hello world", 6));
        assert!(comic_break_penalty("hello world", 6) < comic_break_penalty("the word", 4));
    }

    #[test]
    fn comic_breaks_use_a_coherent_block_in_both_writing_directions() {
        let segments = [LineBreakMeasure {
            advance: 40.0,
            trailing_advance: 0.0,
            break_suffix_advance: 0.0,
            break_penalty: 0.0,
            is_mandatory: false,
        }; 4];
        let balloon = comic_balloon(
            100.0,
            200.0,
            vec![(0.0, 0.0), (100.0, 0.0), (100.0, 200.0), (0.0, 200.0)],
            0.0,
        );

        let horizontal = comic_line_breaks(
            &segments,
            &balloon,
            (WritingMode::Horizontal, TextAlign::Center),
            24.0,
            InkBand {
                before: 10.0,
                after: 10.0,
            },
            (0.0, 0.0),
            HyphenationPolicy::Disabled,
        );
        let vertical_balloon = comic_balloon(
            200.0,
            100.0,
            vec![(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)],
            0.0,
        );
        let vertical = comic_line_breaks(
            &segments,
            &vertical_balloon,
            (WritingMode::VerticalRl, TextAlign::Center),
            24.0,
            InkBand {
                before: 10.0,
                after: 10.0,
            },
            (0.0, 0.0),
            HyphenationPolicy::Disabled,
        );

        assert_eq!(horizontal.breaks, [2, 4]);
        assert_eq!(vertical.breaks, [2, 4]);
    }

    #[test]
    fn last_resort_hyphenation_is_used_only_to_avoid_overflow() {
        let fits_without_hyphen = [
            LineBreakMeasure {
                advance: 30.0,
                trailing_advance: 0.0,
                break_suffix_advance: 5.0,
                break_penalty: 0.0,
                is_mandatory: false,
            },
            LineBreakMeasure {
                advance: 30.0,
                trailing_advance: 0.0,
                break_suffix_advance: 0.0,
                break_penalty: 0.0,
                is_mandatory: false,
            },
        ];
        let unbroken =
            line_breaks_with_policy(&fits_without_hyphen, 70.0, HyphenationPolicy::LastResort);
        assert_eq!(unbroken.breaks, [2]);
        assert!(!unbroken.overflowed);

        let needs_hyphen = [
            LineBreakMeasure {
                advance: 55.0,
                trailing_advance: 0.0,
                break_suffix_advance: 10.0,
                break_penalty: 0.0,
                is_mandatory: false,
            },
            LineBreakMeasure {
                advance: 55.0,
                trailing_advance: 0.0,
                break_suffix_advance: 0.0,
                break_penalty: 0.0,
                is_mandatory: false,
            },
        ];
        let hyphenated =
            line_breaks_with_policy(&needs_hyphen, 70.0, HyphenationPolicy::LastResort);
        assert_eq!(hyphenated.breaks, [1, 2]);
        assert!(!hyphenated.overflowed);
    }

    #[test]
    fn comic_auto_fit_uses_hyphenation_to_preserve_readable_size() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "A healthy mind dwells in a healthy body. Exercise is important!";
        let layout = |policy| {
            TextLayout::new(&font)
                .with_max_font_size(36.0)
                .with_min_font_size(8.0)
                .with_line_height(1.2)
                .with_hyphenation_language_tag("en")
                .with_hyphenation_policy(policy)
                .with_alignment(TextAlign::Center)
                .with_max_width(125.0)
                .with_max_height(220.0)
                .with_comic_balloon(
                    125.0,
                    220.0,
                    vec![(0.0, 0.0), (125.0, 0.0), (125.0, 220.0), (0.0, 220.0)],
                    4.0,
                )
                .run(text)
        };
        let unhyphenated = layout(HyphenationPolicy::Disabled)?;
        let hyphenated = layout(HyphenationPolicy::LastResort)?;

        assert!(!hyphenated.overflowed());
        assert!(hyphenated.font_size + 0.01 >= unhyphenated.font_size);
        assert!(hyphenated.lines.windows(2).any(|lines| {
            let before = text[..lines[0].range.end].chars().next_back();
            let after = text[lines[1].range.start..].chars().next();
            matches!((before, after), (Some(left), Some(right)) if left.is_alphabetic() && right.is_alphabetic())
        }));
        Ok(())
    }

    #[test]
    fn comic_auto_fit_hyphenates_at_the_readability_floor() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "For the villagers, it's a common occurrence.";
        let layout = |policy| {
            TextLayout::new(&font)
                .with_max_font_size(80.0)
                .with_min_font_size(9.0)
                .with_line_height(1.2)
                .with_hyphenation_language_tag("en")
                .with_hyphenation_policy(policy)
                .with_alignment(TextAlign::Center)
                .with_max_width(68.0)
                .with_max_height(218.0)
                .with_comic_balloon(
                    68.0,
                    218.0,
                    vec![(0.0, 0.0), (68.0, 0.0), (68.0, 218.0), (0.0, 218.0)],
                    4.0,
                )
                .run(text)
        };
        let clean = layout(HyphenationPolicy::Disabled)?;
        let last_resort = layout(HyphenationPolicy::LastResort)?;

        assert_eq!(clean.font_size.floor(), 9.0);
        assert!(last_resort.font_size.floor() > clean.font_size.floor());
        assert!(last_resort.lines.windows(2).any(|lines| {
            let before = text[..lines[0].range.end].chars().next_back();
            let after = text[lines[1].range.start..].chars().next();
            matches!((before, after), (Some(left), Some(right)) if left.is_alphabetic() && right.is_alphabetic())
        }));
        Ok(())
    }

    #[test]
    fn short_comic_text_keeps_clean_words_when_an_unhyphenated_setting_fits() -> anyhow::Result<()>
    {
        let font = any_system_font();
        let text = "A cat tower?";
        let layout = |policy| {
            TextLayout::new(&font)
                .with_max_font_size(80.0)
                .with_min_font_size(8.0)
                .with_hyphenation_language_tag("en")
                .with_hyphenation_policy(policy)
                .with_alignment(TextAlign::Center)
                .with_max_width(80.0)
                .with_max_height(160.0)
                .with_comic_balloon(
                    80.0,
                    160.0,
                    vec![(0.0, 0.0), (80.0, 0.0), (80.0, 160.0), (0.0, 160.0)],
                    4.0,
                )
                .run(text)
        };
        let unhyphenated = layout(HyphenationPolicy::Disabled)?;
        let hyphenated = layout(HyphenationPolicy::LastResort)?;

        assert!(!hyphenated.overflowed());
        assert!((hyphenated.font_size - unhyphenated.font_size).abs() < 0.01);
        let tower = text.find("tower").unwrap();
        assert!(
            !hyphenated
                .lines
                .iter()
                .any(|line| (tower + 1..tower + "tower".len()).contains(&line.range.end))
        );
        Ok(())
    }

    #[test]
    fn mandatory_breaks_are_respected_by_the_global_balloon_profile() {
        let segments = [
            LineBreakMeasure {
                advance: 20.0,
                trailing_advance: 0.0,
                break_suffix_advance: 0.0,
                break_penalty: 0.0,
                is_mandatory: true,
            },
            LineBreakMeasure {
                advance: 20.0,
                trailing_advance: 0.0,
                break_suffix_advance: 0.0,
                break_penalty: 0.0,
                is_mandatory: false,
            },
            LineBreakMeasure {
                advance: 5.0,
                trailing_advance: 0.0,
                break_suffix_advance: 0.0,
                break_penalty: 0.0,
                is_mandatory: false,
            },
        ];
        let profiles = vec![
            LineProfile {
                width: 50.0,
                start: None,
                center_offset: 0.0,
                block_baseline: 0.0,
            },
            LineProfile {
                width: 25.0,
                start: None,
                center_offset: 0.0,
                block_baseline: 0.0,
            },
        ];

        let result = exact_profiled_line_breaks(&segments, profiles, true, true).unwrap();

        assert_eq!(result.breaks, [1, 3]);
    }

    #[test]
    fn a_discretionary_suffix_does_not_hide_a_later_unbroken_fit() {
        let segments = [
            LineBreakMeasure {
                advance: 30.0,
                trailing_advance: 0.0,
                break_suffix_advance: 5.0,
                break_penalty: 0.0,
                is_mandatory: false,
            },
            LineBreakMeasure {
                advance: 25.0,
                trailing_advance: 0.0,
                break_suffix_advance: 20.0,
                break_penalty: 0.0,
                is_mandatory: false,
            },
            LineBreakMeasure {
                advance: 5.0,
                trailing_advance: 0.0,
                break_suffix_advance: 0.0,
                break_penalty: 0.0,
                is_mandatory: false,
            },
        ];

        let result = line_breaks_with_policy(&segments, 65.0, HyphenationPolicy::Normal);

        assert_eq!(result.breaks, [3]);
        assert!(!result.overflowed);
    }

    #[test]
    fn comic_balloon_reports_when_relative_air_cannot_fit() -> anyhow::Result<()> {
        let font = any_system_font();
        let layout = TextLayout::new(&font)
            .with_font_size(16.0)
            .with_max_width(100.0)
            .with_max_height(10.0)
            .with_comic_balloon(
                100.0,
                10.0,
                vec![(0.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0)],
                4.0,
            )
            .run("Hi")?;

        assert!(layout.overflowed());
        Ok(())
    }

    #[test]
    fn comic_balloon_air_uses_the_visible_ink_height_and_explicit_floor() {
        let balloon = comic_balloon(100.0, 100.0, Vec::new(), 4.0);

        assert_approx_eq(balloon.air(12.0), 12.0);
        assert_approx_eq(balloon.air(24.0), 24.0);
        assert_approx_eq(balloon.air(3.0), 4.0);
    }

    #[test]
    #[ignore = "skipping for now"]
    fn layout_baselines_horizontal_follow_font_metrics() -> anyhow::Result<()> {
        let font = any_system_font();
        let font_size = 16.0;
        let layout = TextLayout::new(&font)
            .with_font_size(font_size)
            .with_writing_mode(WritingMode::Horizontal)
            .run("A\nB\nC")?;

        assert!(layout.lines.len() >= 2);

        let metrics = font
            .skrifa_ref()?
            .metrics(Size::new(font_size), font.location());
        let ascent = metrics.ascent;
        let descent = -metrics.descent;
        let line_height = (ascent + descent + metrics.leading).max(font_size);

        let base_x = layout.lines[0].baseline.0;
        for line in &layout.lines {
            assert_approx_eq(line.baseline.0, base_x);
        }
        for i in 1..layout.lines.len() {
            let dy = layout.lines[i].baseline.1 - layout.lines[i - 1].baseline.1;
            assert_approx_eq(dy, line_height);
        }

        Ok(())
    }

    #[test]
    fn mandatory_newlines_are_not_shaped_as_glyphs() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "A\nB\nC";
        let layout = TextLayout::new(&font)
            .with_font_size(16.0)
            .with_writing_mode(WritingMode::Horizontal)
            .run(text)?;

        assert_eq!(layout.lines.len(), 3);
        for (line, expected) in layout.lines.iter().zip(["A", "B", "C"]) {
            assert_eq!(&text[line.range.clone()], expected);
            assert_eq!(line.glyphs.len(), 1);
        }

        Ok(())
    }

    #[test]
    #[ignore = "skipping for now"]
    fn layout_baselines_vertical_follow_font_metrics() -> anyhow::Result<()> {
        let font = any_system_font();
        let font_size = 16.0;
        let layout = TextLayout::new(&font)
            .with_font_size(font_size)
            .with_writing_mode(WritingMode::VerticalRl)
            .run("A\nB\nC")?;

        assert!(layout.lines.len() >= 2);

        let metrics = font
            .skrifa_ref()?
            .metrics(Size::new(font_size), font.location());
        let ascent = metrics.ascent;
        let descent = -metrics.descent;
        let line_height = (ascent + descent + metrics.leading).max(font_size);
        let base_y = layout.lines[0].baseline.1;
        for line in &layout.lines {
            assert_approx_eq(line.baseline.1, base_y);
        }

        for i in 1..layout.lines.len() {
            let dx = layout.lines[i - 1].baseline.0 - layout.lines[i].baseline.0;
            assert_approx_eq(dx, line_height);
        }

        Ok(())
    }

    #[test]
    fn vertical_layout_preserves_original_source_offsets() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "！！A";
        let layout = TextLayout::new(&font)
            .with_font_size(16.0)
            .with_writing_mode(WritingMode::VerticalRl)
            .run(text)?;

        assert_eq!(layout.lines[0].range, 0..text.len());
        assert!(layout.lines[0].glyphs.iter().all(|glyph| {
            let cluster = glyph.cluster as usize;
            cluster <= text.len() && text.is_char_boundary(cluster)
        }));
        Ok(())
    }

    #[test]
    fn vertical_alignment_keeps_one_layout_box() -> anyhow::Result<()> {
        let font = any_system_font();
        let max_height = 100.0;
        let layout = |alignment| {
            TextLayout::new(&font)
                .with_font_size(16.0)
                .with_writing_mode(WritingMode::VerticalRl)
                .with_max_width(60.0)
                .with_max_height(max_height)
                .with_alignment(alignment)
                .run("AAAA\ng")
        };
        let start = layout(TextAlign::Left)?;
        let center = layout(TextAlign::Center)?;
        let end = layout(TextAlign::Right)?;

        assert!(start.height < max_height);
        assert_approx_eq(start.height, center.height);
        assert_approx_eq(center.height, end.height);
        assert_approx_eq(start.width, end.width);
        assert_approx_eq(start.placement_offset_y(), center.placement_offset_y());
        assert_approx_eq(center.placement_offset_y(), end.placement_offset_y());
        let metrics = TextLayout::new(&font);
        let line_top = |layout: &LayoutRun<'_>| {
            metrics
                .ink_bounds(16.0, std::slice::from_ref(&layout.lines[1]))
                .unwrap()
                .1
        };
        let start_top = line_top(&start);
        let center_top = line_top(&center);
        let end_top = line_top(&end);
        assert!(start_top < center_top);
        assert!(center_top < end_top);

        Ok(())
    }

    #[test]
    fn vertical_start_alignment_shares_one_visual_top() -> anyhow::Result<()> {
        let font = any_system_font();
        let layout = TextLayout::new(&font)
            .with_font_size(16.0)
            .with_writing_mode(WritingMode::VerticalRl)
            .with_alignment(TextAlign::Left)
            .with_max_width(80.0)
            .with_max_height(160.0)
            .with_comic_balloon(
                80.0,
                160.0,
                vec![(0.0, 0.0), (80.0, 0.0), (80.0, 160.0), (0.0, 160.0)],
                0.0,
            )
            .run("A\ng\nE")?;
        let metrics = TextLayout::new(&font);
        let tops = layout
            .lines
            .iter()
            .map(|line| {
                metrics
                    .ink_bounds(16.0, std::slice::from_ref(line))
                    .unwrap()
                    .1
            })
            .collect::<Vec<_>>();

        assert_eq!(tops.len(), 3);
        for top in &tops[1..] {
            assert_approx_eq(*top, tops[0]);
        }
        let (_, ink_top, _, ink_bottom) = metrics.ink_bounds(16.0, &layout.lines).unwrap();
        assert_approx_eq(ink_top, layout.height - ink_bottom);
        assert!(layout.height < 160.0);
        assert_approx_eq(layout.placement_offset_y(), 0.0);

        Ok(())
    }

    #[test]
    fn horizontal_alignment_keeps_one_layout_box() -> anyhow::Result<()> {
        let font = any_system_font();
        let max_width = 100.0;
        let layout = |alignment| {
            TextLayout::new(&font)
                .with_font_size(16.0)
                .with_max_width(max_width)
                .with_alignment(alignment)
                .run("AAAA\ng")
        };
        let start = layout(TextAlign::Left)?;
        let center = layout(TextAlign::Center)?;
        let end = layout(TextAlign::Right)?;
        let metrics = TextLayout::new(&font);
        let ink_left = |layout: &LayoutRun<'_>| {
            metrics
                .ink_bounds(16.0, std::slice::from_ref(&layout.lines[1]))
                .unwrap()
                .0
        };

        assert!(start.width < max_width);
        assert_approx_eq(start.width, center.width);
        assert_approx_eq(center.width, end.width);
        assert_approx_eq(start.placement_offset_x(), center.placement_offset_x());
        assert_approx_eq(center.placement_offset_x(), end.placement_offset_x());
        assert!(ink_left(&start) < ink_left(&center));
        assert!(ink_left(&center) < ink_left(&end));

        Ok(())
    }

    #[test]
    fn horizontal_center_alignment_centres_short_lines() -> anyhow::Result<()> {
        // Two lines of clearly different widths — a wide "HELLOWORLD" and
        // a narrow "HI". In a max_width wider than the long line, the
        // narrow line should be offset so its centre matches the long
        // line's centre (and the sprite centre).
        let font = any_system_font();
        let max_width = 400.0;
        let layout = TextLayout::new(&font)
            .with_font_size(20.0)
            .with_max_width(max_width)
            .with_alignment(TextAlign::Center)
            .run("HELLOWORLD\nHI")?;

        assert_eq!(layout.lines.len(), 2);
        let w0 = layout.lines[0].advance;
        let w1 = layout.lines[1].advance;
        let c0 = layout.lines[0].baseline.0 + w0 * 0.5;
        let c1 = layout.lines[1].baseline.0 + w1 * 0.5;
        // Line centres must coincide (within rounding / float slack).
        assert!(
            (c0 - c1).abs() < 1.0,
            "expected line centres to match, got c0={c0} c1={c1}",
        );
        Ok(())
    }

    #[test]
    fn horizontal_layout_hyphenates_long_words() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "antidisestablishmentarianism";
        let font_size = 24.0;
        let unwrapped = TextLayout::new(&font).with_font_size(font_size).run(text)?;
        let max_width = (unwrapped.lines[0].advance * 0.45).max(font_size * 4.0);

        let layout = TextLayout::new(&font)
            .with_font_size(font_size)
            .with_max_width(max_width)
            .run(text)?;

        assert!(
            layout.lines.len() > 1,
            "expected hyphenation to wrap long word, got {layout:?}"
        );
        for line in layout.lines.iter().take(layout.lines.len() - 1) {
            assert!(
                line.advance <= max_width + 1.0,
                "hyphenated line should fit max width {max_width}, got {}",
                line.advance
            );
        }
        assert!(
            layout
                .lines
                .iter()
                .take(layout.lines.len() - 1)
                .any(|line| line
                    .glyphs
                    .iter()
                    .any(|glyph| glyph.cluster as usize == line.range.end)),
            "expected a synthetic hyphen glyph at a discretionary break"
        );

        Ok(())
    }

    #[test]
    fn horizontal_layout_wraps_chinese_on_jieba_word_boundaries() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "\u{5357}\u{4eac}\u{5e02}\u{957f}\u{6c5f}\u{5927}\u{6865}";
        let font_size = 24.0;
        let unwrapped = TextLayout::new(&font).with_font_size(font_size).run(text)?;
        let layout = TextLayout::new(&font)
            .with_font_size(font_size)
            .with_max_width(unwrapped.lines[0].advance * 0.5)
            .run(text)?;

        assert!(
            layout.lines.len() > 1,
            "expected Chinese text to wrap, got {layout:?}"
        );
        assert_eq!(
            &text[layout.lines[0].range.clone()],
            "\u{5357}\u{4eac}\u{5e02}"
        );

        Ok(())
    }

    #[test]
    fn cjk_emphasis_normalization_only_expands_ascii_marks() {
        let source = "A!?！？⁉⁈‼⁇";
        let (normalized, cluster_map) = normalize_cjk_emphasis_punctuation(source).unwrap();

        assert_eq!(normalized, "A！？！？⁉⁈‼⁇");
        assert_eq!(
            normalized
                .char_indices()
                .map(|(offset, _)| cluster_map[offset])
                .collect::<Vec<_>>(),
            [0, 1, 2, 3, 6, 9, 12, 15, 18]
        );
        assert!(normalize_cjk_emphasis_punctuation("！？⁉⁈‼⁇").is_none());
    }

    #[test]
    fn cjk_emphasis_layout_keeps_glyphs_and_groups_at_most_three_per_vertical_row()
    -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "中!?A!!!B????C⁉⁈‼⁇";
        let layout = TextLayout::new(&font)
            .with_font_size(32.0)
            .with_writing_mode(WritingMode::VerticalRl)
            .with_cjk_punctuation_layout(true)
            .run(text)?;
        let glyphs = layout
            .lines
            .iter()
            .flat_map(|line| &line.glyphs)
            .collect::<Vec<_>>();

        for (offset, _) in text.char_indices() {
            assert!(
                glyphs.iter().any(|glyph| glyph.cluster == offset as u32),
                "missing source glyph at byte offset {offset}"
            );
        }
        let y_advance = |cluster| {
            glyphs
                .iter()
                .find(|glyph| glyph.cluster == cluster)
                .unwrap()
                .y_advance
        };
        assert_approx_eq(y_advance(3), 0.0);
        assert!(y_advance(4) < 0.0);
        assert_approx_eq(y_advance(6), 0.0);
        assert_approx_eq(y_advance(7), 0.0);
        assert!(y_advance(8) < 0.0);
        assert_approx_eq(y_advance(10), 0.0);
        assert_approx_eq(y_advance(11), 0.0);
        assert!(y_advance(12) < 0.0);
        assert!(y_advance(13) < 0.0);
        Ok(())
    }

    #[test]
    fn cjk_emphasis_layout_applies_to_horizontal_text() -> anyhow::Result<()> {
        let font = any_system_font();
        let font_size = 32.0;
        let text = "中!?A!!!";
        let layout = TextLayout::new(&font)
            .with_font_size(font_size)
            .with_cjk_punctuation_layout(true)
            .run(text)?;
        let glyphs = &layout.lines[0].glyphs;
        let punctuation = [3, 4, 6, 7, 8].map(|cluster| {
            glyphs
                .iter()
                .find(|glyph| glyph.cluster == cluster)
                .unwrap()
        });

        for glyph in punctuation {
            assert!(glyph.x_advance > 0.0);
        }
        let first = glyphs.iter().find(|glyph| glyph.cluster == 3).unwrap();
        let metrics = first
            .font
            .skrifa_ref()?
            .glyph_metrics(Size::new(font_size), first.font.location());
        let bounds = metrics
            .bounds(skrifa::GlyphId::new(first.glyph_id))
            .unwrap();
        assert_approx_eq(
            first.x_advance - (bounds.x_max - bounds.x_min),
            font_size * 0.04,
        );
        Ok(())
    }

    #[test]
    fn vertical_punctuation_centering_enabled_by_default() {
        let font = any_system_font();
        let layout = TextLayout::new(&font).with_font_size(16.0);
        assert!(layout.center_vertical_punctuation);
    }

    #[test]
    fn centered_x_offset_uses_absolute_center() {
        assert_approx_eq(centered_x_offset(2.0, 6.0), -4.0);
        assert_approx_eq(centered_x_offset(-3.0, 1.0), 1.0);
    }

    #[test]
    fn all_vertical_punctuation_is_centered_in_its_advance_cell() -> anyhow::Result<()> {
        let font = any_system_font();
        let font_size = 32.0;
        let text = "A,A.A;A:A!A?A-A_A(A)A[A]A{A}A，A。A！A？A—A“A”A";
        let layout = TextLayout::new(&font)
            .with_font_size(font_size)
            .with_writing_mode(WritingMode::VerticalRl)
            .run(text)?;
        let categories = CodePointMapData::<GeneralCategory>::new();
        let punctuation_offsets = text
            .char_indices()
            .filter_map(|(offset, character)| {
                GeneralCategoryGroup::Punctuation
                    .contains(categories.get(character))
                    .then_some(offset as u32)
            })
            .collect::<Vec<_>>();
        assert_eq!(punctuation_offsets.len(), 21);

        for cluster in punctuation_offsets {
            let punctuation = layout.lines[0]
                .glyphs
                .iter()
                .find(|glyph| glyph.cluster == cluster)
                .unwrap_or_else(|| panic!("missing punctuation glyph at byte offset {cluster}"));
            let font_ref = punctuation.font.skrifa_ref()?;
            let metrics = font_ref.glyph_metrics(Size::new(font_size), punctuation.font.location());
            let bounds = metrics
                .bounds(skrifa::GlyphId::new(punctuation.glyph_id))
                .expect("punctuation glyph has no bounds");

            let ink_center_x = punctuation.x_offset + (bounds.x_min + bounds.x_max) * 0.5;
            let ink_center_y = -punctuation.y_offset - (bounds.y_min + bounds.y_max) * 0.5;
            assert_approx_eq(ink_center_x, 0.0);
            assert_approx_eq(ink_center_y, -punctuation.y_advance * 0.5);
        }
        Ok(())
    }

    #[test]
    fn horizontal_center_alignment_with_overflow_is_aligned_relative_to_widest()
    -> anyhow::Result<()> {
        let font = any_system_font();
        // A very narrow container.
        let max_width = 20.0;
        // A very long word that is guaranteed to overflow 20px in any font.
        let text = "LONGWORDTHATWILLOVERFLOW,\nHI";
        let layout = TextLayout::new(&font)
            .with_font_size(20.0)
            .with_max_width(max_width)
            .with_alignment(TextAlign::Center)
            .run(text)?;

        let w0 = layout.lines[0].advance;
        let w1 = layout.lines[1].advance;

        // Ensure we are actually testing the overflow case.
        assert!(
            w0 > max_width,
            "Test error: widest line {w0} did not overflow max_width {max_width}"
        );

        let c0 = layout.lines[0].baseline.0 + w0 * 0.5;
        let c1 = layout.lines[1].baseline.0 + w1 * 0.5;

        // In a fixed system, the center of the short line should match the center
        // of the overflowing line, NOT the center of the original max_width constraint.
        assert!(
            (c0 - c1).abs() < 1.0,
            "expected line centres to match even with overflow, got c0={c0} c1={c1} (max_width={max_width})",
        );
        Ok(())
    }
}
