use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use koharu_llama_sys as sys;

/// Shape of a layer's KV tensors in an interleaved attention model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufKvLayer {
    pub sliding: bool,
    pub heads: u32,
    pub key_length: u32,
    pub value_length: u32,
}

/// Model metadata and storage requirements from the GGUF headers.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct GgufInfo {
    pub architecture: Option<String>,
    pub block_count: Option<u32>,
    /// Per-layer KV geometry when the model describes an interleaved sliding window.
    pub kv_layers: Option<Vec<GgufKvLayer>>,
    pub expert_count: u32,
    /// Maximum experts selected by one token, including per-layer routing counts.
    pub expert_used_count: u32,
    pub total_model_bytes: u64,
    expert_layouts: Vec<ExpertLayout>,
}

#[derive(Debug, PartialEq, Eq)]
struct ExpertLayout {
    layers: Vec<u32>,
    bytes_per_layer: u64,
    upload_bytes: u64,
    cpu_share_bytes: u64,
}

impl GgufInfo {
    /// Minimum single-device CUDA budgets for upload and CPU miss processing.
    /// `Some(n)` places layers below `n` on the host, and `None` places all layers there.
    #[must_use]
    pub fn min_moe_cache_bytes_for_host_layers(&self, host_layers: Option<u32>) -> (u64, u64) {
        let host_layouts = self.expert_layouts.iter().filter_map(|layout| {
            let n = host_layers.map_or_else(
                || layout.layers.len(),
                |end| layout.layers.partition_point(|&layer| layer < end),
            );
            (n > 0).then_some((layout, n as u64 * layout.bytes_per_layer))
        });
        let total_bytes: u64 = host_layouts.clone().map(|(_, bytes)| bytes).sum();
        host_layouts.fold((0, 0), |(upload, cpu_share), (layout, bytes)| {
            // Invert the native cache's truncated proportional share for this layout.
            let required = |bank_bytes: u64| {
                let budget =
                    (u128::from(bank_bytes) * u128::from(total_bytes)).div_ceil(u128::from(bytes));
                u64::try_from(budget).unwrap_or(u64::MAX)
            };
            (
                upload.max(required(layout.upload_bytes)),
                cpu_share.max(required(layout.cpu_share_bytes)),
            )
        })
    }
}

struct Header(NonNull<sys::gguf_context>);

impl Drop for Header {
    fn drop(&mut self) {
        unsafe { sys::gguf_free(self.0.as_ptr()) };
    }
}

impl Header {
    fn open(path: &Path) -> Option<Self> {
        let path = CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
        let params = sys::gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut(),
        };
        NonNull::new(unsafe { sys::gguf_init_from_file(path.as_ptr(), params) }).map(Self)
    }

    fn key(&self, name: &CStr) -> Option<i64> {
        let idx = unsafe { sys::gguf_find_key(self.0.as_ptr(), name.as_ptr()) };
        (idx >= 0).then_some(idx)
    }

    fn split_value(&self, name: &CStr, default: u16) -> Option<u16> {
        let Some(idx) = self.key(name) else {
            return Some(default);
        };
        let ctx = self.0.as_ptr();
        (unsafe { sys::gguf_get_kv_type(ctx, idx) } == sys::GGUF_TYPE_UINT16)
            .then(|| unsafe { sys::gguf_get_val_u16(ctx, idx) })
    }

    fn u32_at(&self, idx: i64) -> Option<u32> {
        let ctx = self.0.as_ptr();
        (unsafe { sys::gguf_get_kv_type(ctx, idx) } == sys::GGUF_TYPE_UINT32)
            .then(|| unsafe { sys::gguf_get_val_u32(ctx, idx) })
    }

    fn u32_array(&self, idx: i64) -> Option<impl ExactSizeIterator<Item = u32> + '_> {
        let ctx = self.0.as_ptr();
        if unsafe { sys::gguf_get_kv_type(ctx, idx) } != sys::GGUF_TYPE_ARRAY {
            return None;
        }
        let is_bool = match unsafe { sys::gguf_get_arr_type(ctx, idx) } {
            sys::GGUF_TYPE_UINT32 | sys::GGUF_TYPE_INT32 => false,
            sys::GGUF_TYPE_BOOL => true,
            _ => return None,
        };
        let n = unsafe { sys::gguf_get_arr_n(ctx, idx) };
        let data = unsafe { sys::gguf_get_arr_data(ctx, idx) };
        // The header owns n elements of the checked type for this borrow.
        Some((0..n).map(move |i| unsafe {
            if is_bool {
                u32::from(*data.cast::<u8>().add(i) != 0)
            } else {
                data.cast::<u32>().add(i).read_unaligned()
            }
        }))
    }

    fn kv_layers(
        &self,
        key: impl Fn(&str) -> Option<i64>,
        blocks: u32,
    ) -> Option<Vec<GgufKvLayer>> {
        let sliding = self.u32_array(key("attention.sliding_window_pattern")?)?;
        if sliding.len() == 0 || sliding.len() != blocks as usize {
            return None;
        }
        let layer_counts = |idx| {
            if let Some(value) = self.u32_at(idx) {
                Some(vec![value; sliding.len()])
            } else {
                self.u32_array(idx)
                    .filter(|a| a.len() == sliding.len())
                    .map(Iterator::collect)
            }
        };
        let heads = layer_counts(key("attention.head_count")?)?;
        let default_width = self
            .u32_at(key("embedding_length")?)?
            .checked_div(heads[0])?;
        let kv_heads = match key("attention.head_count_kv") {
            Some(idx) => layer_counts(idx)?,
            None => heads,
        };
        let full_k = key("attention.key_length")
            .and_then(|idx| self.u32_at(idx))
            .unwrap_or(default_width);
        let full_v = key("attention.value_length")
            .and_then(|idx| self.u32_at(idx))
            .unwrap_or(default_width);
        let swa_k = key("attention.key_length_swa")
            .and_then(|idx| self.u32_at(idx))
            .unwrap_or(full_k);
        let swa_v = key("attention.value_length_swa")
            .and_then(|idx| self.u32_at(idx))
            .unwrap_or(full_v);
        Some(
            sliding
                .zip(kv_heads)
                .map(|(swa, heads)| GgufKvLayer {
                    sliding: swa != 0,
                    heads,
                    key_length: if swa != 0 { swa_k } else { full_k },
                    value_length: if swa != 0 { swa_v } else { full_v },
                })
                .collect(),
        )
    }

    fn model_info(&self) -> Option<GgufInfo> {
        let ctx = self.0.as_ptr();
        let architecture = self.key(c"general.architecture").and_then(|idx| {
            (unsafe { sys::gguf_get_kv_type(ctx, idx) } == sys::GGUF_TYPE_STRING).then(|| {
                unsafe { CStr::from_ptr(sys::gguf_get_val_str(ctx, idx)) }
                    .to_string_lossy()
                    .into_owned()
            })
        });
        let arch_key = |suffix: &str| {
            self.key(&CString::new(format!("{}.{suffix}", architecture.as_ref()?)).ok()?)
        };
        let block_count = arch_key("block_count")
            .and_then(|idx| self.u32_at(idx))
            .or_else(|| {
                (0..unsafe { sys::gguf_get_n_kv(ctx) }).find_map(|idx| {
                    let key = unsafe { CStr::from_ptr(sys::gguf_get_key(ctx, idx)) };
                    key.to_bytes()
                        .ends_with(b".block_count")
                        .then(|| self.u32_at(idx))
                        .flatten()
                })
            });
        let expert_count = match arch_key("expert_count") {
            Some(idx) => self.u32_at(idx)?,
            None => 0,
        };
        let expert_used_count = match arch_key("expert_used_count") {
            None => 0,
            Some(idx) if unsafe { sys::gguf_get_kv_type(ctx, idx) } == sys::GGUF_TYPE_ARRAY => {
                self.u32_array(idx)?.max().unwrap_or(0)
            }
            Some(idx) => self.u32_at(idx)?,
        };
        let kv_layers = block_count.and_then(|blocks| self.kv_layers(arch_key, blocks));
        Some(GgufInfo {
            architecture,
            kv_layers,
            block_count,
            expert_count,
            expert_used_count,
            ..Default::default()
        })
    }

    fn collect_expert_banks(&self, layers: &mut ExpertLayers) -> Option<()> {
        let ctx = self.0.as_ptr();
        for idx in 0..unsafe { sys::gguf_get_n_tensors(ctx) } {
            let name = unsafe { CStr::from_ptr(sys::gguf_get_tensor_name(ctx, idx)) };
            let Some((layer, bank)) = name.to_str().ok().and_then(expert_bank) else {
                continue;
            };
            // gguf_get_tensor_ne exposes all four dimensions, including trailing ones.
            let shape = unsafe { *sys::gguf_get_tensor_ne(ctx, idx).cast::<[i64; 4]>() };
            let bytes = u64::try_from(unsafe { sys::gguf_get_tensor_size(ctx, idx) }).ok()?;
            let experts = u64::try_from(shape[2]).ok()?;
            if experts == 0 || shape[3] != 1 || !bytes.is_multiple_of(experts) {
                return None;
            }
            let layout = Bank {
                shape,
                dtype: unsafe { sys::gguf_get_tensor_type(ctx, idx) },
                bytes_per_expert: bytes / experts,
            };
            if layers.entry(layer).or_default()[bank]
                .replace(layout)
                .is_some()
            {
                return None;
            }
        }
        Some(())
    }
}

/// The native loader reads model metadata from the first shard. Numeric GGUF
/// arrays, including the sliding-window pattern, are absent from a loaded model.
pub(crate) fn model_kv_layers(
    path: &Path,
    architecture: &str,
    blocks: u32,
) -> Option<Vec<GgufKvLayer>> {
    let header = Header::open(path)?;
    let key = |suffix: &str| header.key(&CString::new(format!("{architecture}.{suffix}")).ok()?);
    header.kv_layers(key, blocks)
}

fn shard_paths(path: &Path, index: u16, count: u16) -> Option<Vec<PathBuf>> {
    if count <= 1 {
        return (index == 0).then(|| vec![path.to_owned()]);
    }
    if index >= count {
        return None;
    }
    let suffix = format!("-{:05}-of-{count:05}.gguf", index + 1);
    let prefix = path.file_name()?.to_str()?.strip_suffix(&suffix)?;
    if prefix.is_empty() {
        return None;
    }
    Some(
        (1..=count)
            .map(|n| path.with_file_name(format!("{prefix}-{n:05}-of-{count:05}.gguf")))
            .collect(),
    )
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Bank {
    shape: [i64; 4],
    dtype: sys::ggml_type,
    bytes_per_expert: u64,
}

type ExpertLayers = BTreeMap<u32, [Option<Bank>; 4]>;

fn expert_bank(name: &str) -> Option<(u32, usize)> {
    let (layer, tensor) = name.strip_prefix("blk.")?.split_once('.')?;
    let bank = match tensor {
        "ffn_gate_up_exps.weight" => 0,
        "ffn_gate_exps.weight" => 1,
        "ffn_up_exps.weight" => 2,
        "ffn_down_exps.weight" => 3,
        _ => return None,
    };
    Some((layer.parse().ok()?, bank))
}

fn expert_layouts(layers: ExpertLayers, expert_used_count: u32) -> Option<Vec<ExpertLayout>> {
    let mut groups = BTreeMap::<Vec<Bank>, Vec<u32>>::new();
    for (layer, banks) in layers {
        // llama.cpp compares the sequence of present tensors in bank order.
        groups
            .entry(banks.into_iter().flatten().collect())
            .or_default()
            .push(layer);
    }

    let used = u64::from(expert_used_count);
    let mut total_bytes = 0u64;
    groups
        .into_iter()
        .map(|(banks, layers)| {
            let bytes_per_layer = banks.iter().try_fold(0u64, |sum, bank| {
                sum.checked_add(bank.bytes_per_expert.checked_mul(bank.shape[2] as u64)?)
            })?;
            if bytes_per_layer == 0 {
                return None;
            }
            // Validate the sum once so every host subset fits in u64.
            total_bytes =
                total_bytes.checked_add(bytes_per_layer.checked_mul(layers.len() as u64)?)?;
            // CUDA pads each bank and reserves a trailing slot for MMQ reads. With
            // CPU misses the device also reads up to 64 distinct zero slots per token.
            let bank_bytes = |slots| {
                banks.iter().try_fold(0u64, |sum, bank| {
                    let size = bank.bytes_per_expert.checked_mul(slots)?;
                    sum.checked_add(size.checked_next_multiple_of(128)?)
                })
            };
            Some(ExpertLayout {
                layers,
                bytes_per_layer,
                upload_bytes: bank_bytes(used + 1)?,
                cpu_share_bytes: bank_bytes(used + used.min(64) + 1)?,
            })
        })
        .collect()
}

/// Inspects one GGUF model without loading tensor data. A numbered shard resolves
/// to the complete set in its directory. Missing or inconsistent shards fail inspection.
#[must_use]
pub fn inspect_gguf(path: &Path) -> Option<GgufInfo> {
    let header = Header::open(path)?;
    let count = header.split_value(c"split.count", 1)?;
    let index = header.split_value(c"split.no", 0)?;
    let paths = shard_paths(path, index, count)?;
    let mut supplied = Some(header);
    let mut layers = ExpertLayers::new();
    let mut info = GgufInfo::default();
    for (idx, shard) in (0u16..).zip(paths) {
        let header = if idx == index {
            supplied.take()?
        } else {
            Header::open(&shard)?
        };
        if count > 1
            && (header.split_value(c"split.no", u16::MAX)? != idx
                || header.split_value(c"split.count", 0)? != count)
        {
            return None;
        }
        if idx == 0 {
            info = header.model_info()?;
        }
        info.total_model_bytes = info
            .total_model_bytes
            .checked_add(shard.metadata().ok()?.len())?;
        if info.expert_count > 0 {
            header.collect_expert_banks(&mut layers)?;
        }
    }
    info.expert_layouts = expert_layouts(layers, info.expert_used_count)?;
    Some(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_paths_use_the_numbered_suffix_of_the_supplied_shard() {
        let first = Path::new("models/q4-00001-of-00002.gguf");
        let second = Path::new("models/q4-00002-of-00002.gguf");
        assert_eq!(shard_paths(second, 1, 2).unwrap(), [first, second]);
        assert!(shard_paths(second, 0, 2).is_none());
        assert!(shard_paths(second, 1, 3).is_none());
        let single = Path::new("models/dense.gguf");
        assert_eq!(shard_paths(single, 0, 1).unwrap(), [single]);
    }

    fn moe_info(layers: ExpertLayers, used: u32) -> GgufInfo {
        GgufInfo {
            expert_layouts: expert_layouts(layers, used).unwrap(),
            ..Default::default()
        }
    }

    #[test]
    fn cache_shares_identical_layouts_and_pads_each_expert_bank() {
        let mut layers = ExpertLayers::new();
        // The gate and down tensors of a layer can be in separate shards.
        for (name, stride) in [
            ("blk.0.ffn_gate_up_exps.weight", 192),
            ("blk.1.ffn_gate_up_exps.weight", 192),
            ("blk.0.ffn_down_exps.weight", 96),
            ("blk.1.ffn_down_exps.weight", 96),
        ] {
            let (layer, index) = expert_bank(name).unwrap();
            layers.entry(layer).or_default()[index] = Some(Bank {
                shape: [stride, 1, 8, 1],
                dtype: sys::GGML_TYPE_I8,
                bytes_per_expert: stride as u64,
            });
        }
        // Three slots per bank for two routed experts. Layers 0 and 1 share banks.
        assert_eq!(
            moe_info(layers, 2).min_moe_cache_bytes_for_host_layers(None),
            (640 + 384, 1024 + 512)
        );
        assert_eq!(
            GgufInfo::default().min_moe_cache_bytes_for_host_layers(None),
            (0, 0)
        );
        assert!(expert_bank("blk.0.ffn_gate_inp.weight").is_none());
        assert!(expert_bank("blk.0.ffn_down_exps.scale").is_none());
    }

    #[test]
    fn cache_budget_accounts_for_layer_multiplicity_and_cpu_zero_slots() {
        let bank = |stride: u64| Bank {
            shape: [stride as i64, 1, 128, 1],
            dtype: sys::GGML_TYPE_I8,
            bytes_per_expert: stride,
        };
        let mut layers = ExpertLayers::new();
        for layer in 0..9 {
            layers.entry(layer).or_default()[3] = Some(bank(128));
        }
        layers.entry(9).or_default()[3] = Some(bank(320));
        // The rare layout gets 5/23 of the budget. Its three upload slots
        // require 1024 padded bytes, or 1664 bytes for five slots with CPU misses.
        assert_eq!(
            moe_info(layers, 2).min_moe_cache_bytes_for_host_layers(None),
            (4711, 7655)
        );

        let single = BTreeMap::from([(0, [None, None, None, Some(bank(128))])]);
        assert_eq!(
            moe_info(single, 65).min_moe_cache_bytes_for_host_layers(None),
            (66 * 128, 130 * 128)
        );
    }

    #[test]
    fn cache_floors_exclude_gpu_resident_expert_layouts() {
        let mut layers = ExpertLayers::new();
        for layer in 0..30 {
            let stride = if layer == 29 { 256 } else { 128 };
            layers.entry(layer).or_default()[3] = Some(Bank {
                shape: [stride, 1, 128, 1],
                dtype: sys::GGML_TYPE_I8,
                bytes_per_expert: stride as u64,
            });
        }
        let info = moe_info(layers, 2);
        assert_eq!(info.min_moe_cache_bytes_for_host_layers(Some(0)), (0, 0));
        for n in [23, 29] {
            assert_eq!(
                info.min_moe_cache_bytes_for_host_layers(Some(n)),
                (384, 640)
            );
        }
        // With all layers on the host the last layout gets 2/31 of the budget.
        for host_layers in [Some(30), None] {
            assert_eq!(
                info.min_moe_cache_bytes_for_host_layers(host_layers),
                (11_904, 19_840)
            );
        }
    }
}
