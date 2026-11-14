//! One model on one GPU, with its engine and compiled kernels cached on disk.

use std::{
    ffi::CString,
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context as _, Result, bail, ensure};

use crate::{
    DeviceBuffer, Stream, TensorInfo, cuda,
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

/// Field order drops the context before the engine.
pub struct Session {
    context: ExecutionContext,
    engine: Engine,
    names: Vec<CString>,
    device: u32,
}

impl Session {
    /// Opens `onnx` on `device`, keeping the serialized engine and the
    /// runtime cache of compiled kernels in `cache_dir`.
    ///
    /// The engine file is keyed by the ONNX file's contents and the
    /// TensorRT-RTX version, and the runtime cache additionally by the GPU's
    /// compute capability and the driver version, since compiled kernels are
    /// specific to both. With `cuda_graph`, runs replay a captured CUDA graph.
    #[tracing::instrument(level = "info", skip_all, fields(model = %onnx.display()))]
    pub fn open(onnx: &Path, cache_dir: &Path, device: u32, cuda_graph: bool) -> Result<Self> {
        crate::cudart()?;
        cuda::set_device(device)?;
        fs::create_dir_all(cache_dir)
            .with_context(|| format!("failed to create {}", cache_dir.display()))?;
        let key = format!(
            "{}-{}-rtx{}",
            onnx.file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("model"),
            content_hash(onnx)?,
            crate::version()?
        );
        let engine = load_engine(onnx, &cache_dir.join(format!("{key}.engine")))?;

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

        let names = engine
            .tensors()
            .iter()
            .map(|tensor| CString::new(tensor.name.as_str()))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            context,
            engine,
            names,
            device,
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
            self.context.set_input_shape(&self.names[index], shape)?;
        }

        for (index, tensor) in self.engine.tensors().iter().enumerate() {
            let binding = bound[index].expect("every tensor is bound");
            let shape = self.context.shape(&self.names[index])?;
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
                self.context
                    .set_tensor_address(&self.names[index], binding.buffer.raw())?;
            }
        }

        // SAFETY: every tensor is bound to a large enough live buffer and the
        // stream is live; the borrows outlast the synchronization.
        unsafe { self.context.enqueue(stream.raw())? };
        stream.synchronize()
    }
}

fn load_engine(onnx: &Path, path: &Path) -> Result<Engine> {
    if let Ok(bytes) = fs::read(path) {
        match Engine::deserialize(&bytes) {
            Ok(engine) => return Ok(engine),
            Err(error) => tracing::warn!(%error, "rebuilding unreadable cached engine"),
        }
    }
    let started = Instant::now();
    let bytes = engine::build(onnx)?;
    tracing::info!(
        elapsed_ms = started.elapsed().as_millis(),
        bytes = bytes.len(),
        "built TensorRT-RTX engine"
    );
    write_atomic(path, &bytes)?;
    Engine::deserialize(&bytes)
}

fn content_hash(path: &Path) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    io::copy(&mut file, &mut hasher)?;
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
