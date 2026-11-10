//! Safe wrapper around `gguf_context` for reading GGUF file metadata and
//! writing tensor subsets of GGUF files.
//!
//! Provides metadata-only access to GGUF files without loading tensor data.
//! Useful for inspecting model architecture parameters before loading a model.

use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::ptr::NonNull;

/// A safe wrapper around `gguf_context`.
///
/// Opens a GGUF file and parses only the metadata header; tensor weights are
/// never loaded into memory (`no_alloc = true`).
#[derive(Debug)]
pub struct GgufContext {
    ctx: NonNull<koharu_llama_sys::gguf_context>,
}

impl GgufContext {
    /// Open a GGUF file and parse its metadata header.
    ///
    /// Returns `None` if the path contains a null byte, the file does not
    /// exist, or the file is not a valid GGUF file.
    pub fn from_file(path: &Path) -> Option<Self> {
        let c_path = CString::new(path.to_str()?).ok()?;
        let params = koharu_llama_sys::gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut(),
        };
        let ptr = unsafe { koharu_llama_sys::gguf_init_from_file(c_path.as_ptr(), params) };
        Some(Self {
            ctx: NonNull::new(ptr)?,
        })
    }

    /// Total number of key-value pairs in the metadata.
    pub fn n_kv(&self) -> i64 {
        unsafe { koharu_llama_sys::gguf_get_n_kv(self.ctx.as_ptr()) }
    }

    /// Find the index of a key by name. Returns `-1` if not found.
    pub fn find_key(&self, key: &str) -> i64 {
        let Ok(c_key) = CString::new(key) else {
            return -1;
        };
        unsafe { koharu_llama_sys::gguf_find_key(self.ctx.as_ptr(), c_key.as_ptr()) }
    }

    /// Return the key name at the given index, or `None` if out of range.
    pub fn key_at(&self, idx: i64) -> Option<&str> {
        let ptr = unsafe { koharu_llama_sys::gguf_get_key(self.ctx.as_ptr(), idx) };
        if ptr.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }

    /// Return the value type of the KV pair at `idx`.
    pub fn kv_type(&self, idx: i64) -> koharu_llama_sys::gguf_type {
        unsafe { koharu_llama_sys::gguf_get_kv_type(self.ctx.as_ptr(), idx) }
    }

    /// Read a `uint32` value. Panics (inside llama.cpp) if the stored type is
    /// not `GGUF_TYPE_UINT32` â€” check `kv_type` first if unsure.
    pub fn val_u32(&self, idx: i64) -> u32 {
        unsafe { koharu_llama_sys::gguf_get_val_u32(self.ctx.as_ptr(), idx) }
    }

    /// Read an `int32` value.
    pub fn val_i32(&self, idx: i64) -> i32 {
        unsafe { koharu_llama_sys::gguf_get_val_i32(self.ctx.as_ptr(), idx) }
    }

    /// Read a `bool` value.
    pub fn val_bool(&self, idx: i64) -> bool {
        unsafe { koharu_llama_sys::gguf_get_val_bool(self.ctx.as_ptr(), idx) }
    }

    /// Read a `uint64` value.
    pub fn val_u64(&self, idx: i64) -> u64 {
        unsafe { koharu_llama_sys::gguf_get_val_u64(self.ctx.as_ptr(), idx) }
    }

    /// Read a string value. Returns `None` if the pointer is null or not
    /// valid UTF-8.
    pub fn val_str(&self, idx: i64) -> Option<&str> {
        let ptr = unsafe { koharu_llama_sys::gguf_get_val_str(self.ctx.as_ptr(), idx) };
        if ptr.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }

    /// Total number of tensors described in the file.
    pub fn n_tensors(&self) -> i64 {
        unsafe { koharu_llama_sys::gguf_get_n_tensors(self.ctx.as_ptr()) }
    }
}

impl Drop for GgufContext {
    fn drop(&mut self) {
        unsafe { koharu_llama_sys::gguf_free(self.ctx.as_ptr()) }
    }
}

/// Write a copy of the GGUF file at `source` that keeps only the tensors whose
/// names satisfy `keep`, with each `flags` key set to its boolean value.
///
/// Tensor data is streamed from `source`; only metadata is held in memory.
pub fn write_subset(
    source: &Path,
    destination: &Path,
    keep: impl Fn(&str) -> bool,
    flags: &[(&str, bool)],
) -> io::Result<()> {
    struct Owned {
        source: *mut koharu_llama_sys::gguf_context,
        tensors: *mut koharu_llama_sys::ggml_context,
        output: *mut koharu_llama_sys::gguf_context,
    }
    impl Drop for Owned {
        fn drop(&mut self) {
            unsafe {
                if !self.output.is_null() {
                    koharu_llama_sys::gguf_free(self.output);
                }
                if !self.tensors.is_null() {
                    koharu_llama_sys::ggml_free(self.tensors);
                }
                if !self.source.is_null() {
                    koharu_llama_sys::gguf_free(self.source);
                }
            }
        }
    }

    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message.to_owned());
    let path = source
        .to_str()
        .and_then(|path| CString::new(path).ok())
        .ok_or_else(|| invalid("GGUF path is not a valid C string"))?;
    let mut owned = Owned {
        source: std::ptr::null_mut(),
        tensors: std::ptr::null_mut(),
        output: std::ptr::null_mut(),
    };
    let params = koharu_llama_sys::gguf_init_params {
        no_alloc: true,
        ctx: &raw mut owned.tensors,
    };
    owned.source = unsafe { koharu_llama_sys::gguf_init_from_file(path.as_ptr(), params) };
    if owned.source.is_null() || owned.tensors.is_null() {
        return Err(invalid("failed to read GGUF metadata"));
    }
    owned.output = unsafe { koharu_llama_sys::gguf_init_empty() };
    if owned.output.is_null() {
        return Err(invalid("failed to create GGUF metadata"));
    }
    unsafe { koharu_llama_sys::gguf_set_kv(owned.output, owned.source) };
    for (key, value) in flags {
        let key = CString::new(*key).map_err(|_| invalid("GGUF key contains NUL"))?;
        unsafe { koharu_llama_sys::gguf_set_val_bool(owned.output, key.as_ptr(), *value) };
    }

    let data_offset = unsafe { koharu_llama_sys::gguf_get_data_offset(owned.source) } as u64;
    let mut ranges = Vec::new();
    for index in 0..unsafe { koharu_llama_sys::gguf_get_n_tensors(owned.source) } {
        let name = unsafe { koharu_llama_sys::gguf_get_tensor_name(owned.source, index) };
        let name_str = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| invalid("GGUF tensor name is not UTF-8"))?;
        if !keep(name_str) {
            continue;
        }
        let tensor = unsafe { koharu_llama_sys::ggml_get_tensor(owned.tensors, name) };
        if tensor.is_null() {
            return Err(invalid("GGUF tensor metadata is missing"));
        }
        unsafe { koharu_llama_sys::gguf_add_tensor(owned.output, tensor) };
        ranges.push((
            data_offset
                + unsafe { koharu_llama_sys::gguf_get_tensor_offset(owned.source, index) } as u64,
            unsafe { koharu_llama_sys::gguf_get_tensor_size(owned.source, index) } as u64,
        ));
    }

    let mut metadata = vec![0_u8; unsafe { koharu_llama_sys::gguf_get_meta_size(owned.output) }];
    unsafe { koharu_llama_sys::gguf_get_meta_data(owned.output, metadata.as_mut_ptr().cast()) };
    let mut input = File::open(source)?;
    let mut output = io::BufWriter::new(File::create(destination)?);
    output.write_all(&metadata)?;
    let mut written = metadata.len() as u64;
    for (index, (offset, size)) in ranges.into_iter().enumerate() {
        let target = metadata.len() as u64
            + unsafe { koharu_llama_sys::gguf_get_tensor_offset(owned.output, index as i64) }
                as u64;
        io::copy(&mut io::repeat(0).take(target - written), &mut output)?;
        input.seek(SeekFrom::Start(offset))?;
        if io::copy(&mut (&mut input).take(size), &mut output)? != size {
            return Err(invalid("GGUF tensor data is truncated"));
        }
        written = target + size;
    }
    output
        .into_inner()
        .map_err(io::IntoInnerError::into_error)?
        .sync_all()
}
