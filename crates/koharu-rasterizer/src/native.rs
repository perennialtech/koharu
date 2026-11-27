//! Reusable native WGPU readback and export supersampling.

use std::{
    num::NonZeroUsize,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};

use anyhow::{Context, Result as AnyResult, anyhow, bail};
use fast_image_resize::{
    FilterType, IntoImageView as _, IntoImageViewMut as _, ResizeAlg, ResizeOptions, Resizer,
    pixels::U8x4,
};
use image::RgbaImage;
use parking_lot::Mutex;
use vello::{
    AaConfig, AaSupport, RenderParams, RendererOptions, Scene,
    kurbo::Affine,
    peniko::Color,
    wgpu::{
        self, Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d,
        TexelCopyBufferInfo, Texture, TextureDescriptor, TextureFormat, TextureUsages, TextureView,
    },
};

use crate::{CompositionCommand, Error, Frame, GpuCompositor, RasterDraw, Result};

const MAX_SUPERSAMPLING_FACTOR: u32 = 4;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DownsampleFilter {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    #[default]
    Lanczos3,
}

impl From<DownsampleFilter> for ResizeAlg {
    fn from(value: DownsampleFilter) -> Self {
        match value {
            DownsampleFilter::Nearest => ResizeAlg::Nearest,
            DownsampleFilter::Triangle => ResizeAlg::Convolution(FilterType::Bilinear),
            DownsampleFilter::CatmullRom => ResizeAlg::Convolution(FilterType::CatmullRom),
            DownsampleFilter::Gaussian => ResizeAlg::Convolution(FilterType::Gaussian),
            DownsampleFilter::Lanczos3 => ResizeAlg::Convolution(FilterType::Lanczos3),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RasterOptions {
    /// Render passes at this multiple of the frame size and downsample to it.
    pub supersampling_factor: u32,
    pub downsample_filter: DownsampleFilter,
}

impl RasterOptions {
    #[must_use]
    pub fn supersampled(factor: u32) -> Self {
        Self {
            supersampling_factor: factor,
            ..Default::default()
        }
    }

    fn surface(self, width: u32, height: u32) -> AnyResult<Surface> {
        checked_surface(width, height)?;
        let scale = self.supersampling_factor.clamp(1, MAX_SUPERSAMPLING_FACTOR);
        let size = |length: u32| {
            length
                .checked_mul(scale)
                .context("render surface size overflow")
        };
        Ok(Surface {
            scale,
            width,
            height,
            raster_width: size(width)?,
            raster_height: size(height)?,
        })
    }
}

/// Output size and the render surface that is downsampled to it.
#[derive(Clone, Copy)]
struct Surface {
    scale: u32,
    width: u32,
    height: u32,
    raster_width: u32,
    raster_height: u32,
}

impl Default for RasterOptions {
    fn default() -> Self {
        Self {
            supersampling_factor: 1,
            downsample_filter: DownsampleFilter::Lanczos3,
        }
    }
}

#[derive(Debug)]
pub struct Raster {
    pub image: RgbaImage,
    pub left: i32,
    pub top: i32,
}

impl Raster {
    /// Encodes the image as PNG. The generic encoder is instantiated in this
    /// optimized crate rather than in callers built without optimization.
    pub fn encode_png(&self) -> Result<Vec<u8>> {
        let mut png = std::io::Cursor::new(Vec::new());
        self.image
            .write_to(&mut png, image::ImageFormat::Png)
            .map_err(Error::Png)?;
        Ok(png.into_inner())
    }
}

#[derive(Clone)]
struct GpuDevice {
    raw: wgpu::Device,
    retired: Arc<AtomicBool>,
}

impl GpuDevice {
    fn check_live(&self) -> AnyResult<()> {
        if self.retired.load(Ordering::Acquire) {
            bail!("WGPU device lost or retired");
        }
        Ok(())
    }

    fn checked<T>(&self, work: impl FnOnce() -> AnyResult<T>) -> AnyResult<T> {
        self.check_live()?;
        // wgpu can panic in poll on a driver reset after firing the loss callback.
        let result = pollster::block_on(crate::compositor::checked_gpu(&self.raw, || {
            catch_unwind(AssertUnwindSafe(work))
        }));
        let result = match result {
            Err(error) => Err(error.into()),
            Ok(Ok(result)) => result,
            Ok(Err(panic)) => {
                self.check_live()?;
                resume_unwind(panic)
            }
        };
        if let Err(error) = &result {
            // Vello and the compositor retain buffers and bind groups. A failed
            // GPU operation can leave those resources invalid on a live device.
            if !matches!(error.downcast_ref::<Error>(), Some(Error::Invalid(_))) {
                self.retired.store(true, Ordering::Release);
            }
        } else {
            self.check_live()?;
        }
        result
    }
}

struct GpuState {
    device: GpuDevice,
    limits: wgpu::Limits,
    queue: wgpu::Queue,
    renderer: vello::Renderer,
    compositor: GpuCompositor,
    targets: Vec<RenderTarget>,
}

struct RenderTarget {
    width: u32,
    height: u32,
    padded_width: u32,
    texture: Texture,
    view: TextureView,
    readback: Buffer,
}

impl GpuState {
    fn new() -> AnyResult<Self> {
        let (raw, queue) = pollster::block_on(request_device())?;
        let retired = Arc::new(AtomicBool::new(false));
        raw.set_device_lost_callback({
            let retired = Arc::clone(&retired);
            move |_, _| retired.store(true, Ordering::Release)
        });
        let device = GpuDevice { raw, retired };
        let (renderer, compositor) = device.checked(|| {
            let renderer = vello::Renderer::new(
                &device.raw,
                RendererOptions {
                    antialiasing_support: AaSupport::area_only(),
                    // Shader creation must use the thread that owns the error scopes.
                    num_init_threads: Some(NonZeroUsize::MIN),
                    ..Default::default()
                },
            )
            .context("failed to create Vello renderer")?;
            Ok((renderer, GpuCompositor::new(&device.raw)))
        })?;
        Ok(Self {
            limits: device.raw.limits(),
            device,
            queue,
            renderer,
            compositor,
            targets: Vec::new(),
        })
    }

    fn take_target(&mut self, width: u32, height: u32) -> AnyResult<RenderTarget> {
        if let Some(index) = self
            .targets
            .iter()
            .position(|target| target.width == width && target.height == height)
        {
            return Ok(self.targets.swap_remove(index));
        }
        let limit = self.limits.max_texture_dimension_2d;
        if width > limit || height > limit {
            bail!("render surface {width}x{height} exceeds the device limit {limit}");
        }
        let padded_width = width
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT))
            .context("render target row size overflow")?;
        let buffer_size = u64::from(padded_width)
            .checked_mul(u64::from(height))
            .context("render target buffer size overflow")?;
        let limit = self.limits.max_buffer_size;
        if buffer_size > limit {
            bail!(
                "readback buffer requires {buffer_size} bytes, exceeding the device limit {limit}"
            );
        }
        self.device.checked(|| {
            Ok(RenderTarget::new(
                &self.device.raw,
                width,
                height,
                padded_width,
                buffer_size,
            ))
        })
    }
}

struct Worker {
    generation: u64,
    state: Option<GpuState>,
}

impl Worker {
    fn current(&mut self) -> AnyResult<(u64, &mut GpuState)> {
        if self
            .state
            .as_ref()
            .is_none_or(|gpu| gpu.device.retired.load(Ordering::Acquire))
        {
            // Release the old renderer, tile cache, and pool before allocating replacements.
            self.state = None;
            self.state = Some(GpuState::new()?);
            self.generation += 1;
        }
        Ok((
            self.generation,
            self.state.as_mut().expect("device initialized"),
        ))
    }

    fn return_target(&mut self, generation: u64, target: RenderTarget) {
        if self.generation == generation
            && let Some(gpu) = &mut self.state
            && !gpu.device.retired.load(Ordering::Acquire)
            && gpu.targets.len() < 4
        {
            gpu.targets.push(target);
        }
    }
}

/// Reusable headless Vello renderer with a bounded readback-target pool.
pub struct Rasterizer {
    gpu: Mutex<Worker>,
}

impl Rasterizer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            gpu: Mutex::new(Worker {
                generation: 1,
                state: Some(GpuState::new().map_err(Error::backend)?),
            }),
        })
    }

    pub fn rasterize(&self, frame: &Frame, options: RasterOptions) -> Result<Raster> {
        let (width, height) = frame.size();
        let (left, top) = frame.origin();
        let image = self
            .rasterize_frame_inner(frame, width, height, options)
            .map_err(Error::backend)?;
        Ok(Raster { image, left, top })
    }

    fn rasterize_frame_inner(
        &self,
        frame: &Frame,
        width: u32,
        height: u32,
        options: RasterOptions,
    ) -> AnyResult<RgbaImage> {
        let surface = options.surface(width, height)?;
        let scale = surface.scale;
        let commands = frame.composition_commands(scale);
        let pixels =
            self.readback_commands(&commands, surface.raster_width, surface.raster_height)?;
        finish_raster(pixels, surface, options.downsample_filter)
    }

    pub fn rasterize_commands(
        &self,
        commands: &[CompositionCommand],
        width: u32,
        height: u32,
        options: RasterOptions,
    ) -> Result<RgbaImage> {
        self.rasterize_commands_inner(commands, width, height, options)
            .map_err(Error::backend)
    }

    fn rasterize_commands_inner(
        &self,
        commands: &[CompositionCommand],
        width: u32,
        height: u32,
        options: RasterOptions,
    ) -> AnyResult<RgbaImage> {
        let surface = options.surface(width, height)?;
        let scale = surface.scale;
        let scaled;
        let commands = if scale == 1 {
            commands
        } else {
            let transform = Affine::scale(f64::from(scale));
            scaled = commands
                .iter()
                .map(|command| match command {
                    CompositionCommand::Raster(draw) => CompositionCommand::Raster(RasterDraw {
                        image: draw.image.clone(),
                        transform: transform * draw.transform,
                        opacity: draw.opacity,
                        erase: draw.erase,
                    }),
                    CompositionCommand::Vector(scene) => {
                        let mut scaled = Scene::new();
                        scaled.append(scene, Some(transform));
                        CompositionCommand::Vector(Box::new(scaled))
                    }
                })
                .collect::<Vec<_>>();
            &scaled
        };
        let pixels =
            self.readback_commands(commands, surface.raster_width, surface.raster_height)?;
        finish_raster(pixels, surface, options.downsample_filter)
    }

    pub fn rasterize_scene(
        &self,
        scene: &Scene,
        width: u32,
        height: u32,
        background: [u8; 4],
        options: RasterOptions,
    ) -> Result<RgbaImage> {
        self.rasterize_scene_inner(scene, width, height, background, options)
            .map_err(Error::backend)
    }

    fn rasterize_scene_inner(
        &self,
        scene: &Scene,
        width: u32,
        height: u32,
        background: [u8; 4],
        options: RasterOptions,
    ) -> AnyResult<RgbaImage> {
        let surface = options.surface(width, height)?;
        let scale = surface.scale;
        let scaled;
        let scene = if scale == 1 {
            scene
        } else {
            scaled = {
                let mut scaled = Scene::new();
                scaled.append(scene, Some(Affine::scale(f64::from(scale))));
                scaled
            };
            &scaled
        };
        let pixels = self.readback_scene(
            scene,
            surface.raster_width,
            surface.raster_height,
            background,
        )?;
        finish_raster(pixels, surface, options.downsample_filter)
    }

    fn readback_commands(
        &self,
        commands: &[CompositionCommand],
        width: u32,
        height: u32,
    ) -> AnyResult<Vec<u8>> {
        // Compositor targets are premultiplied; native images expose straight RGBA.
        self.readback(width, height, true, |gpu, view| {
            Ok(gpu.compositor.render(
                &gpu.device.raw,
                &gpu.queue,
                &mut gpu.renderer,
                view,
                (width, height),
                commands,
                None,
                [0; 4],
                [0, 0, width, height],
            )?)
        })
    }

    fn readback_scene(
        &self,
        scene: &Scene,
        width: u32,
        height: u32,
        background: [u8; 4],
    ) -> AnyResult<Vec<u8>> {
        self.readback(width, height, false, |gpu, view| {
            gpu.renderer
                .render_to_texture(
                    &gpu.device.raw,
                    &gpu.queue,
                    scene,
                    view,
                    &RenderParams {
                        base_color: rgba(background),
                        width,
                        height,
                        antialiasing_method: AaConfig::Area,
                    },
                )
                .context("Vello rendering failed")
        })
    }

    fn readback(
        &self,
        width: u32,
        height: u32,
        premultiplied: bool,
        render: impl FnOnce(&mut GpuState, &TextureView) -> AnyResult<()>,
    ) -> AnyResult<Vec<u8>> {
        let (generation, device, target, submission) = {
            let mut worker = self.gpu.lock();
            let (generation, gpu) = worker.current()?;
            let device = gpu.device.clone();
            let target = gpu.take_target(width, height)?;
            device.checked(|| render(gpu, &target.view))?;
            let submission = device.checked(|| {
                let mut encoder = device
                    .raw
                    .create_command_encoder(&CommandEncoderDescriptor {
                        label: Some("koharu readback encoder"),
                    });
                encoder.copy_texture_to_buffer(
                    target.texture.as_image_copy(),
                    TexelCopyBufferInfo {
                        buffer: &target.readback,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(target.padded_width),
                            rows_per_image: None,
                        },
                    },
                    target.texture.size(),
                );
                Ok(gpu.queue.submit([encoder.finish()]))
            })?;
            (generation, device, target, submission)
        };
        let pixels = device.checked(|| target.read_pixels(&device, submission, premultiplied))?;
        self.gpu.lock().return_target(generation, target);
        Ok(pixels)
    }
}

impl RenderTarget {
    fn read_pixels(
        &self,
        device: &GpuDevice,
        submission: wgpu::SubmissionIndex,
        premultiplied: bool,
    ) -> AnyResult<Vec<u8>> {
        let width = self.width;
        let height = self.height;
        let slice = self.readback.slice(..);
        let (sender, receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .raw
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| anyhow!("WGPU device polling failed: {error:?}"))?;
        device.check_live()?;
        receiver
            .recv()
            .context("WGPU closed the readback channel")?
            .context("failed to map WGPU readback buffer")?;
        let mapped = slice
            .get_mapped_range()
            .context("failed to read mapped WGPU readback buffer")?;
        let row_len = (width * 4) as usize;
        let mut pixels = Vec::with_capacity(row_len * height as usize);
        for row in mapped
            .chunks_exact(self.padded_width as usize)
            .take(height as usize)
        {
            let start = pixels.len();
            pixels.extend_from_slice(&row[..row_len]);
            if premultiplied {
                for pixel in pixels[start..].as_chunks_mut::<4>().0 {
                    match pixel[3] {
                        0 => pixel.fill(0),
                        255 => {}
                        alpha => {
                            let alpha = u32::from(alpha);
                            for channel in &mut pixel[..3] {
                                *channel = ((u32::from(*channel) * 255 + alpha / 2) / alpha)
                                    .min(255) as u8;
                            }
                        }
                    }
                }
            }
        }
        drop(mapped);
        self.readback.unmap();
        Ok(pixels)
    }
}

/// Creates the device as Vello's `RenderContext::device` does, but with the
/// adapter's texture and buffer limits. Vello requests `Limits::default()`,
/// which caps every surface at 8192 px and its readback at 256 MiB whatever
/// the GPU supports.
async fn request_device() -> AnyResult<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        display: None,
        backends: wgpu::Backends::from_env().unwrap_or_default(),
        flags: wgpu::InstanceFlags::from_build_config().with_env(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::from_env_or_default(),
    });
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
        .await
        .context("no WGPU adapter is available")?;
    let supported = adapter.limits();
    let required_limits = wgpu::Limits {
        max_texture_dimension_2d: supported.max_texture_dimension_2d,
        max_buffer_size: supported.max_buffer_size,
        ..wgpu::Limits::default()
    };
    let optional_features = wgpu::Features::CLEAR_TEXTURE | wgpu::Features::PIPELINE_CACHE;
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("koharu rasterizer device"),
            required_features: adapter.features() & optional_features,
            required_limits,
            ..Default::default()
        })
        .await
        .context("no WGPU adapter supports Vello's required features")
}

fn checked_surface(width: u32, height: u32) -> AnyResult<()> {
    if width == 0 || height == 0 {
        bail!("invalid render surface {width}x{height}");
    }
    Ok(())
}

impl RenderTarget {
    fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        padded_width: u32,
        buffer_size: u64,
    ) -> Self {
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("koharu rasterizer target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("koharu rasterizer readback"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            width,
            height,
            padded_width,
            texture,
            view,
            readback,
        }
    }
}

fn finish_raster(
    pixels: Vec<u8>,
    surface: Surface,
    filter: DownsampleFilter,
) -> AnyResult<RgbaImage> {
    let image = RgbaImage::from_raw(surface.raster_width, surface.raster_height, pixels)
        .context("WGPU returned an invalid RGBA buffer")?;
    if surface.raster_width == surface.width {
        return Ok(image);
    }
    let mut downsampled = RgbaImage::new(surface.width, surface.height);
    let resize_options = ResizeOptions::new()
        .resize_alg(filter.into())
        .use_alpha(true);
    // `resize_typed` instantiates the resampler for one pixel type; the untyped
    // `resize` monomorphizes all of them into this crate at opt-level 3.
    {
        let (Some(source), Some(mut destination)) = (
            image.image_view::<U8x4>(),
            downsampled.image_view_mut::<U8x4>(),
        ) else {
            bail!("WGPU render is not 8-bit RGBA");
        };
        Resizer::new()
            .resize_typed(&source, &mut destination, &resize_options)
            .context("failed to downsample WGPU render")?;
    }
    Ok(downsampled)
}

fn rgba([r, g, b, a]: [u8; 4]) -> Color {
    Color::from_rgba8(r, g, b, a)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::sync::LazyLock;

    static RASTERIZER: LazyLock<Rasterizer> = LazyLock::new(|| Rasterizer::new().unwrap());

    pub(crate) fn with_device(work: impl FnOnce(&wgpu::Device, &wgpu::Queue)) {
        let mut worker = RASTERIZER.gpu.lock();
        let (_, gpu) = worker.current().unwrap();
        work(&gpu.device.raw, &gpu.queue);
    }

    #[test]
    fn an_oversized_padded_readback_preserves_the_device_generation() {
        static LIMITED: LazyLock<Rasterizer> = LazyLock::new(|| Rasterizer::new().unwrap());
        let (generation, device) = {
            let mut worker = LIMITED.gpu.lock();
            let (generation, gpu) = worker.current().unwrap();
            gpu.limits.max_buffer_size = 512;
            (generation, gpu.device.clone())
        };

        let render = |height| {
            LIMITED.rasterize_scene(
                &Scene::new(),
                1,
                height,
                [12, 34, 56, 255],
                RasterOptions::default(),
            )
        };
        let error = render(3).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("readback buffer requires 768 bytes")
        );
        device.check_live().unwrap();
        assert_eq!(render(2).unwrap().get_pixel(0, 1).0, [12, 34, 56, 255]);
        assert_eq!(LIMITED.gpu.lock().generation, generation);
    }

    #[test]
    fn a_lost_device_is_replaced_before_render_and_rejects_old_targets() {
        let (generation, old_target) = {
            let mut worker = RASTERIZER.gpu.lock();
            let (generation, gpu) = worker.current().unwrap();
            let target = gpu.take_target(5, 3).unwrap();
            gpu.device.raw.destroy();
            gpu.device
                .raw
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            assert!(gpu.device.retired.load(Ordering::Acquire));
            (generation, target)
        };

        let image = RASTERIZER
            .rasterize_scene(
                &Scene::new(),
                2,
                2,
                [12, 34, 56, 255],
                RasterOptions::default(),
            )
            .unwrap();
        assert_eq!(image.get_pixel(1, 1).0, [12, 34, 56, 255]);
        let mut worker = RASTERIZER.gpu.lock();
        assert_eq!(worker.generation, generation + 1);
        worker.return_target(generation, old_target);
        let targets = &worker.state.as_ref().unwrap().targets;
        assert_eq!(targets.len(), 1);
        assert_eq!((targets[0].width, targets[0].height), (2, 2));
    }
}
