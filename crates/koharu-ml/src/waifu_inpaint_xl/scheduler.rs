//! Euler sampling for Waifu-Inpaint-XL's scheduler configuration.
//!
//! Port of diffusers' `EulerDiscreteScheduler` restricted to the checkpoint's
//! `scheduler_config.json`: scaled-linear betas from 0.00085 to 0.012 over
//! 1000 steps, rescaled to zero terminal SNR, v-prediction, trailing timestep
//! spacing, linear sigma interpolation, a final sigma of zero and no
//! stochastic churn.
//! https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/schedulers/scheduling_euler_discrete.py
//!
//! The schedule is computed in f64 where diffusers uses f32; sigmas agree to
//! f32 rounding.

const TRAIN_STEPS: usize = 1000;
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;

pub(super) struct EulerSchedule {
    /// Discrete timesteps fed to the UNet, from noisiest.
    pub timesteps: Vec<f32>,
    /// One more than `timesteps`; the last is zero.
    sigmas: Vec<f32>,
}

impl EulerSchedule {
    /// `set_timesteps(steps)` with trailing spacing.
    pub fn new(steps: usize) -> Self {
        let train_sigmas = train_sigmas();
        let stride = TRAIN_STEPS as f64 / steps as f64;
        let timesteps = (0..steps)
            .map(|step| (TRAIN_STEPS as f64 - step as f64 * stride).round() - 1.0)
            .collect::<Vec<_>>();
        // Trailing timesteps are integers, so linear interpolation reads the
        // training sigma at each one.
        let mut sigmas = timesteps
            .iter()
            .map(|&timestep| train_sigmas[timestep as usize] as f32)
            .collect::<Vec<_>>();
        sigmas.push(0.0);
        Self {
            timesteps: timesteps.into_iter().map(|t| t as f32).collect(),
            sigmas,
        }
    }

    /// `init_noise_sigma`: with trailing spacing, the largest sigma.
    pub fn init_noise_sigma(&self) -> f32 {
        self.sigmas[0]
    }

    /// The first step run at `strength`, as the pipeline's `get_timesteps`:
    /// `strength` 1 runs every step from pure noise.
    pub fn first_step(&self, strength: f32) -> usize {
        let steps = self.timesteps.len();
        let run = ((steps as f32 * strength) as usize).min(steps);
        steps - run
    }

    /// `add_noise` at step `index`: the sigma noise is scaled by.
    pub fn sigma(&self, index: usize) -> f32 {
        self.sigmas[index]
    }

    /// `scale_model_input` at step `index`.
    pub fn input_scale(&self, index: usize) -> f32 {
        1.0 / (self.sigmas[index].powi(2) + 1.0).sqrt()
    }

    /// One Euler step from step `index` with the UNet's v-prediction.
    pub fn step(&self, index: usize, prediction: &[f32], sample: &mut [f32]) {
        let sigma = self.sigmas[index];
        let variance = sigma * sigma + 1.0;
        let original_from_prediction = -sigma / variance.sqrt();
        let dt = self.sigmas[index + 1] - sigma;
        for (sample, &prediction) in sample.iter_mut().zip(prediction) {
            let original = prediction * original_from_prediction + *sample / variance;
            let derivative = (*sample - original) / sigma;
            *sample += derivative * dt;
        }
    }
}

/// sqrt((1 - alphas_cumprod) / alphas_cumprod) over the training timesteps.
fn train_sigmas() -> Vec<f64> {
    let betas = (0..TRAIN_STEPS).map(|step| {
        let t = step as f64 / (TRAIN_STEPS - 1) as f64;
        (BETA_START.sqrt() + t * (BETA_END.sqrt() - BETA_START.sqrt())).powi(2)
    });
    let mut cumprod = 1.0;
    let mut alphas_bar_sqrt = betas
        .map(|beta| {
            cumprod *= 1.0 - beta;
            f64::sqrt(cumprod)
        })
        .collect::<Vec<_>>();
    // rescale_zero_terminal_snr: shift and scale sqrt(alpha_bar) so the last
    // step has zero signal while the first is unchanged. Rebuilding betas from
    // the rescaled product and taking its cumulative product again gives
    // alpha_bar back, so it is used directly.
    let first = alphas_bar_sqrt[0];
    let last = alphas_bar_sqrt[TRAIN_STEPS - 1];
    for value in &mut alphas_bar_sqrt {
        *value = (*value - last) * first / (first - last);
    }
    alphas_bar_sqrt
        .into_iter()
        .enumerate()
        .map(|(step, value)| {
            // EulerDiscreteScheduler keeps the last sigma finite.
            let alpha_bar = if step == TRAIN_STEPS - 1 {
                2f64.powi(-24)
            } else {
                value * value
            };
            ((1.0 - alpha_bar) / alpha_bar).sqrt()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // From diffusers 0.36.0: EulerDiscreteScheduler.from_pretrained(
    // "ShinoharaHare/Waifu-Inpaint-XL", subfolder="scheduler") then
    // set_timesteps(20).
    #[test]
    fn matches_diffusers_schedule() {
        let schedule = EulerSchedule::new(20);
        assert_eq!(schedule.timesteps[..4], [999.0, 949.0, 899.0, 849.0]);
        assert_eq!(schedule.timesteps[19], 49.0);
        assert_eq!(schedule.first_step(1.0), 0);
        assert_eq!(schedule.first_step(0.8), 4);
        let expected = EXPECTED_SIGMAS;
        for (actual, expected) in schedule.sigmas.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= expected * 1e-4,
                "{actual} != {expected}"
            );
        }
    }

    const EXPECTED_SIGMAS: [f32; 21] = [
        4096.0,
        40.392345,
        18.130056,
        10.9017,
        7.4107075,
        5.398_741,
        4.1141853,
        3.2360516,
        2.6048563,
        2.1328518,
        1.768081,
        1.4780177,
        1.2413754,
        1.0436839,
        0.8747436,
        0.7270561,
        0.594_751_8,
        0.47266248,
        0.35498944,
        0.23120026,
        0.0,
    ];
}
