//! One model on one GPU, with its engine and compiled kernels cached on disk.

use std::{
    collections::HashSet,
    ffi::CString,
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard, mpsc},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, anyhow, bail, ensure};

use crate::{
    DeviceBuffer, ShapeRange, Stream, TensorInfo, cuda,
    engine::{self, Engine, ExecutionContext},
};

/// A buffer bound to one engine tensor for a run. Inputs carry their shape;
/// outputs take the shape TensorRT infers from the inputs.
pub struct Binding<'a> {
    name: &'a str,
    buffer: &'a DeviceBuffer,
    shape: Option<&'a [i64]>,
}

impl<'a> Binding<'a> {
    pub fn input(name: &'a str, buffer: &'a DeviceBuffer, shape: &'a [i64]) -> Self {
        Self {
            name,
            buffer,
            shape: Some(shape),
        }
    }

    pub fn output(name: &'a str, buffer: &'a DeviceBuffer) -> Self {
        Self {
            name,
            buffer,
            shape: None,
        }
    }
}

// TensorRT-RTX compiles kernels for a new dynamic shape in the background
// while runs use fallback kernels, so nothing marks when they are ready. After
// a new shape a saver thread writes the runtime cache this often, for this
// long, whether or not more runs come.
const CACHE_SAVE_INTERVAL: Duration = Duration::from_secs(5);
const CACHE_SAVE_WINDOW: Duration = Duration::from_secs(30);

/// `Drop` stops the saver, so the context is no longer shared, and field
/// order then drops the context before the engine.
pub struct Session {
    saver: Option<Saver>,
    live: Arc<Mutex<Live>>,
    engine: Engine,
    names: Vec<CString>,
    device: u32,
    /// Input shapes run so far; `None` for an engine without dynamic inputs,
    /// whose kernels are all compiled when the context is created.
    shapes: Option<HashSet<Vec<Vec<i64>>>>,
}

/// The context and its runtime cache's bookkeeping, shared with the saver.
struct Live {
    context: ExecutionContext,
    cache_path: PathBuf,
    /// The runtime cache as last written.
    saved: Vec<u8>,
    last_new_shape: Option<Instant>,
    /// Whether a saver thread is running; it clears this as it exits.
    saving: bool,
}

struct Saver {
    stop: mpsc::Sender<()>,
    thread: thread::JoinHandle<()>,
}

impl Session {
    /// Opens `onnx` on `device`, keeping the serialized engine and the
    /// runtime cache of compiled kernels in `cache_dir`.
    ///
    /// The engine file is keyed by the ONNX file's contents and the
    /// TensorRT-RTX version, and the runtime cache additionally by the GPU's
    /// compute capability and the driver version, since compiled kernels are
    /// specific to both. With `cuda_graph`, runs replay a captured CUDA graph.
    /// `ranges` bounds each dynamic input and is part of the engine's key.
    #[tracing::instrument(level = "info", skip_all, fields(model = %onnx.display()))]
    pub fn open(
        onnx: &Path,
        cache_dir: &Path,
        device: u32,
        cuda_graph: bool,
        ranges: &[ShapeRange<'_>],
    ) -> Result<Self> {
        crate::cudart()?;
        cuda::set_device(device)?;
        fs::create_dir_all(cache_dir)
            .with_context(|| format!("failed to create {}", cache_dir.display()))?;
        let key = format!(
            "{}-{}-rtx{}",
            onnx.file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("model"),
            content_hash(onnx, ranges)?,
            crate::version()?
        );
        let engine = load_engine(onnx, ranges, &cache_dir.join(format!("{key}.engine")))?;

        let (major, minor) = cuda::compute_capability(device)?;
        let cache_path = cache_dir.join(format!(
            "{key}-sm{major}{minor}-driver{}.jit",
            cuda::driver_version()?
        ));
        let cached = match fs::read(&cache_path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error).context("failed to read the runtime cache"),
        };
        let started = Instant::now();
        let context = ExecutionContext::new(&engine, &cached, cuda_graph)?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            cached = !cached.is_empty(),
            "created TensorRT-RTX execution context"
        );
        // Kernels for fixed shapes are compiled with the context; save them
        // for the next process if anything new was compiled.
        let compiled = context.runtime_cache()?;
        if compiled != cached {
            write_atomic(&cache_path, &compiled)?;
        }
        let live = Live {
            context,
            cache_path,
            saved: compiled,
            last_new_shape: None,
            saving: false,
        };

        let names = engine
            .tensors()
            .iter()
            .map(|tensor| CString::new(tensor.name.as_str()))
            .collect::<Result<_, _>>()?;
        let dynamic = engine
            .tensors()
            .iter()
            .any(|tensor| tensor.input && tensor.shape.iter().any(|extent| *extent < 0));
        Ok(Self {
            saver: None,
            live: Arc::new(Mutex::new(live)),
            engine,
            names,
            device,
            shapes: dynamic.then(HashSet::new),
        })
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        self.engine.tensors()
    }

    /// Activation memory the engine needs per context, in bytes.
    pub fn device_memory(&self) -> i64 {
        self.engine.device_memory()
    }

    /// Runs the model on `stream` and returns once the outputs are written.
    /// Every engine tensor must be bound exactly once.
    pub fn run(&mut self, bindings: &[Binding<'_>], stream: &Stream) -> Result<()> {
        cuda::set_device(self.device)?;
        let mut live = lock(&self.live)?;
        let context = &mut live.context;
        let mut bound = vec![None; self.names.len()];
        for binding in bindings {
            let index = self
                .engine
                .tensors()
                .iter()
                .position(|tensor| tensor.name == binding.name)
                .with_context(|| format!("the engine has no tensor {}", binding.name))?;
            ensure!(
                bound[index].is_none(),
                "tensor {} is bound twice",
                binding.name
            );
            bound[index] = Some(binding);
        }

        let mut input_shapes = Vec::new();
        for (index, tensor) in self.engine.tensors().iter().enumerate() {
            let binding =
                bound[index].with_context(|| format!("tensor {} is not bound", tensor.name))?;
            if !tensor.input {
                continue;
            }
            let shape = binding
                .shape
                .with_context(|| format!("input {} needs a shape", tensor.name))?;
            ensure!(
                shape.len() == tensor.shape.len()
                    && shape
                        .iter()
                        .zip(&tensor.shape)
                        .all(|(given, fixed)| *fixed < 0 || given == fixed),
                "input {} has shape {shape:?}, the engine expects {:?}",
                tensor.name,
                tensor.shape
            );
            context.set_input_shape(&self.names[index], shape)?;
            input_shapes.push(shape.to_vec());
        }

        for (index, tensor) in self.engine.tensors().iter().enumerate() {
            let binding = bound[index].expect("every tensor is bound");
            let shape = context.shape(&self.names[index])?;
            if shape.iter().any(|extent| *extent < 0) {
                bail!("tensor {} has unresolved shape {shape:?}", tensor.name);
            }
            let bytes = shape.iter().product::<i64>() as usize * tensor.data_type.size();
            ensure!(
                binding.buffer.len() >= bytes,
                "tensor {} needs {bytes} bytes, its buffer has {}",
                tensor.name,
                binding.buffer.len()
            );
            // SAFETY: the buffer is large enough, and it stays borrowed until
            // the stream is synchronized below.
            unsafe {
                context.set_tensor_address(&self.names[index], binding.buffer.raw())?;
            }
        }

        // SAFETY: every tensor is bound to a large enough live buffer and the
        // stream is live; the borrows outlast the synchronization.
        unsafe { context.enqueue(stream.raw())? };
        stream.synchronize()?;

        if let Some(shapes) = &mut self.shapes
            && shapes.insert(input_shapes)
        {
            live.last_new_shape = Some(Instant::now());
            if !live.saving {
                live.saving = true;
                drop(live);
                self.start_saver()?;
            }
        }
        Ok(())
    }

    /// Starts a thread that saves the runtime cache until the save window
    /// after the latest new shape closes.
    fn start_saver(&mut self) -> Result<()> {
        if let Some(saver) = self.saver.take() {
            // It has cleared `saving`, so it is exiting.
            let _ = saver.thread.join();
        }
        let (stop, stopped) = mpsc::channel();
        let live = Arc::clone(&self.live);
        let device = self.device;
        let spawned = thread::Builder::new()
            .name("tensorrt-cache".into())
            .spawn(move || {
                if let Err(error) = cuda::set_device(device) {
                    tracing::warn!(%error, "TensorRT-RTX cache saver cannot select the GPU");
                }
                while let Err(mpsc::RecvTimeoutError::Timeout) =
                    stopped.recv_timeout(CACHE_SAVE_INTERVAL)
                {
                    let Ok(mut live) = live.lock() else { return };
                    live.save();
                    if live
                        .last_new_shape
                        .is_none_or(|shape| shape.elapsed() >= CACHE_SAVE_WINDOW)
                    {
                        live.last_new_shape = None;
                        live.saving = false;
                        return;
                    }
                }
            });
        match spawned {
            Ok(thread) => {
                self.saver = Some(Saver { stop, thread });
                Ok(())
            }
            Err(error) => {
                // Only the next process's warm start depends on the saver;
                // `Drop` still saves.
                tracing::warn!(%error, "failed to start the TensorRT-RTX cache saver");
                lock(&self.live)?.saving = false;
                Ok(())
            }
        }
    }
}

impl Live {
    /// Writes kernels compiled since the last save. A failed save only costs
    /// the next process a recompile, so it is logged rather than returned.
    /// The lock keeps it apart from runs, while TensorRT-RTX's background
    /// compilation may still be adding to the cache.
    fn save(&mut self) {
        let result = self.context.runtime_cache().and_then(|bytes| {
            if bytes != self.saved {
                write_atomic(&self.cache_path, &bytes)?;
                self.saved = bytes;
            }
            Ok(())
        });
        if let Err(error) = result {
            tracing::warn!(%error, "failed to save the TensorRT-RTX runtime cache");
        }
    }
}

fn lock(live: &Mutex<Live>) -> Result<MutexGuard<'_, Live>> {
    live.lock()
        .map_err(|_| anyhow!("TensorRT-RTX session lock is poisoned"))
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(Saver { stop, thread }) = self.saver.take() {
            drop(stop);
            let _ = thread.join();
        }
        if self.shapes.is_some()
            && let Ok(mut live) = self.live.lock()
        {
            live.save();
        }
    }
}

fn load_engine(onnx: &Path, ranges: &[ShapeRange<'_>], path: &Path) -> Result<Engine> {
    if let Ok(bytes) = fs::read(path) {
        match Engine::deserialize(&bytes) {
            Ok(engine) => return Ok(engine),
            Err(error) => tracing::warn!(%error, "rebuilding unreadable cached engine"),
        }
    }
    let started = Instant::now();
    let bytes = engine::build(onnx, ranges)?;
    tracing::info!(
        elapsed_ms = started.elapsed().as_millis(),
        bytes = bytes.len(),
        "built TensorRT-RTX engine"
    );
    write_atomic(path, &bytes)?;
    Engine::deserialize(&bytes)
}

fn content_hash(path: &Path, ranges: &[ShapeRange<'_>]) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    io::copy(&mut file, &mut hasher)?;
    for range in ranges {
        hasher.update(range.name.as_bytes());
        for extents in [range.min, range.opt, range.max] {
            hasher.update(&(extents.len() as u64).to_le_bytes());
            for extent in extents {
                hasher.update(&extent.to_le_bytes());
            }
        }
    }
    Ok(hasher.finalize().to_hex()[..16].to_owned())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let directory = path
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_owned);
    let mut file = tempfile::NamedTempFile::new_in(&directory)?;
    file.write_all(bytes)?;
    file.persist(path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}
