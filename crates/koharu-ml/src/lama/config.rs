//! IOPaint inference orchestration settings.
//!
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/schema.py#L206-L214

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HDStrategy {
    Original,
    Resize,
    #[default]
    Crop,
}

#[derive(Debug, Clone)]
pub struct InpaintRequest {
    pub hd_strategy: HDStrategy,
    pub hd_strategy_crop_trigger_size: u32,
    pub hd_strategy_crop_margin: u32,
    pub hd_strategy_resize_limit: u32,
    pub sd_keep_unmasked_area: bool,
}

impl Default for InpaintRequest {
    fn default() -> Self {
        Self {
            hd_strategy: HDStrategy::Crop,
            hd_strategy_crop_trigger_size: 800,
            hd_strategy_crop_margin: 128,
            hd_strategy_resize_limit: 1280,
            sd_keep_unmasked_area: true,
        }
    }
}
