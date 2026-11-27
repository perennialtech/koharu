//! Inference-only Hayai OCR network.
//!
//! Hayai architecture and generation implementation:
//! https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.7-supernova/blob/b00317ed69fccee5f787187d913fcf8a96f37f65/modeling_hayai.py
//! Upstream generates through `_FastEngine`, an inference copy of the
//! modules' math with fused weights; this port follows the modules and
//! `_reference_generate`, which compute the same result. Upstream skips the
//! last decoder layer's vision queries, whose outputs nothing reads; this
//! port runs them, which leaves the logits the same.
//! SigLIP2 NaFlex vision encoder instantiated by the upstream `HayaiModel`:
//! https://github.com/huggingface/transformers/blob/598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1/src/transformers/models/siglip2/modeling_siglip2.py

use std::path::Path;

use anyhow::{Result, ensure};
use koharu_torch::{Device, Kind, Tensor, nn};

use super::{
    config::{HayaiConfig, decoder, siglip2},
    processor::{ImageInput, Tokenizer},
};

/// Upstream `generate`'s default; it decodes greedily and ignores its
/// `num_beams` and `repetition_penalty` arguments.
const MAX_NEW_TOKENS: i64 = 128;
const RMS_NORM_EPS: f64 = 1e-6;

#[derive(Debug)]
pub(super) struct Model {
    vs: nn::VarStore,
    vision_encoder: Siglip2VisionModel,
    decoder: Decoder,
    text_rope_cos: Tensor,
    text_rope_sin: Tensor,
    bos_token_id: i64,
    eos_token_id: i64,
}

impl Model {
    pub(super) fn new(
        config: &HayaiConfig,
        tokenizer: &Tokenizer,
        device: Device,
        kind: Kind,
    ) -> Result<Self> {
        let mut vs = nn::VarStore::new(device);
        vs.set_kind(kind);
        let root = vs.root();
        let vision_encoder = Siglip2VisionModel::new(&(&root / "vision_encoder" / "vision_model"))?;
        let decoder = Decoder::new(&(&root / "decoder"), config)?;

        // Text tokens use plain 1D RoPE from `TEXT_ROPE_OFFSET`; positions
        // 0..=MAX_NEW_TOKENS cover every decoding step. Visual tokens receive
        // their own 2D frequencies per spatial shape, computed during
        // generation.
        let inv_freq = inverse_frequencies(decoder::HEAD_DIM / 2, decoder::ROPE_THETA, device)?;
        let positions = Tensor::f_arange(MAX_NEW_TOKENS + 1, (Kind::Float, device))?
            .f_add_scalar(decoder::TEXT_ROPE_OFFSET)?
            .f_unsqueeze(1)?
            .f_matmul(&inv_freq.f_unsqueeze(0)?)?;
        let angles = Tensor::f_cat(&[positions.f_alias()?, positions], -1)?;
        let text_rope_cos = angles.f_cos()?;
        let text_rope_sin = angles.f_sin()?;

        Ok(Self {
            vs,
            vision_encoder,
            decoder,
            text_rope_cos,
            text_rope_sin,
            bos_token_id: tokenizer.bos_token_id(),
            eos_token_id: tokenizer.eos_token_id(),
        })
    }

    pub(super) fn load(&mut self, path: impl AsRef<Path>) -> Result<()> {
        // The checkpoint must match every declared name and shape. Copy into the
        // model's chosen precision (VarStore::load adopts the checkpoint dtype).
        let mut parameters = self.vs.variables();
        for (name, source) in Tensor::read_safetensors(path)? {
            let mut target = parameters
                .remove(&name)
                .ok_or_else(|| anyhow::anyhow!("unexpected Hayai weight {name}"))?;
            ensure!(
                source.size() == target.size(),
                "Hayai weight {name} has shape {:?}, expected {:?}",
                source.size(),
                target.size()
            );
            target.f_copy_(&source)?;
        }
        ensure!(
            parameters.is_empty(),
            "missing Hayai weights: {:?}",
            parameters.keys()
        );
        Ok(())
    }

    /// Greedy decoding for each crop, the crops decoded together. Every
    /// crop is padded to the batch's longest, so their prefixes are the
    /// same length.
    pub(super) fn generate(
        &self,
        inputs: &[ImageInput],
        tokenizer: &Tokenizer,
        device: Device,
    ) -> Result<Vec<Vec<i64>>> {
        let kind = self.vs.kind();
        let rows = inputs.len();
        ensure!(rows > 0, "Hayai OCR needs at least one crop");
        let spatial_shapes = inputs
            .iter()
            .map(|input| input.spatial_shape)
            .collect::<Vec<_>>();
        let lengths = spatial_shapes
            .iter()
            .map(|&[height, width]| height * width)
            .collect::<Vec<_>>();
        let patches = lengths.iter().copied().max().unwrap_or(1);
        let pixel_values = Tensor::f_stack(
            &inputs
                .iter()
                .zip(&lengths)
                .map(|(input, &length)| {
                    input
                        .pixel_values
                        .f_constant_pad_nd([0, 0, 0, patches - length])
                })
                .collect::<koharu_torch::Result<Vec<_>>>()?,
            0,
        )?
        .f_to_kind(kind)?;

        // Padded patches are masked as keys; their own rows are never read,
        // since the projector gathers inside each crop's grid.
        let key_mask = if lengths.iter().any(|&length| length < patches) {
            let valid = lengths
                .iter()
                .flat_map(|&length| (0..patches).map(move |patch| i64::from(patch < length)))
                .collect::<Vec<_>>();
            Some(additive_mask(
                &Tensor::f_from_slice(&valid)?
                    .f_view([rows as i64, patches])?
                    .f_to_device(device)?,
                kind,
            )?)
        } else {
            None
        };
        let (encoder_states, visual_features) =
            self.vision_encoder
                .forward(&pixel_values, &spatial_shapes, key_mask.as_ref())?;
        let (vision_embeddings, merged_shapes, blocks) = self
            .decoder
            .projector
            .forward(&visual_features, &spatial_shapes)?;
        // One residual onto the visual tokens for each decoder layer.
        let depth_residuals = self.decoder.depth_fusion.forward(
            &encoder_states,
            &blocks,
            vision_embeddings.size()[1],
        )?;

        let bos = Tensor::f_full([rows as i64, 1], self.bos_token_id, (Kind::Int64, device))?;
        let mut hidden = Tensor::f_cat(
            &[
                vision_embeddings,
                Tensor::f_embedding(&self.decoder.token_embeddings, &bos, -1, false, false)?,
            ],
            1,
        )?;

        let m_vision = hidden.size()[1] - 1;
        let total_length = m_vision + 1;
        // Block-causal prefill mask: bidirectional across visual tokens, causal
        // onto the text token (upstream `generate_block_causal_mask`).
        let text_future = Tensor::f_cat(
            &[
                Tensor::f_full([m_vision, 1], f64::NEG_INFINITY, (kind, device))?,
                Tensor::f_zeros([1, 1], (kind, device))?,
            ],
            0,
        )?;
        let prefill_mask = Tensor::f_cat(
            &[
                Tensor::f_zeros([total_length, m_vision], (kind, device))?,
                text_future,
            ],
            1,
        )?
        .f_unsqueeze(0)?
        .f_unsqueeze(0)?;

        // Tokens past a crop's own merged grid only pad the batch; upstream
        // masks them as keys in prefill and in every decoding step.
        let vision_keys = merged_shapes
            .iter()
            .flat_map(|&[height, width]| {
                (0..m_vision).map(move |token| i64::from(token < height * width))
            })
            .collect::<Vec<_>>();
        let vision_key_mask = if vision_keys.contains(&0) {
            Some(additive_mask(
                &Tensor::f_from_slice(&vision_keys)?
                    .f_view([rows as i64, m_vision])?
                    .f_to_device(device)?,
                kind,
            )?)
        } else {
            None
        };
        let key_mask = |text_keys: i64| -> koharu_torch::Result<Option<Tensor>> {
            vision_key_mask
                .as_ref()
                .map(|vision| {
                    Tensor::f_cat(
                        &[
                            vision.f_alias()?,
                            Tensor::f_zeros([rows as i64, 1, 1, text_keys], (kind, device))?,
                        ],
                        3,
                    )
                })
                .transpose()
        };
        let prefill_mask = match key_mask(1)? {
            Some(keys) => prefill_mask.f_add(&keys)?,
            None => prefill_mask,
        };

        // Each crop's angles: its merged grid's, then its padding tokens',
        // which upstream leaves at cos=1 / sin=0 (no rotation), then the
        // first text position's.
        let mut cos = Vec::with_capacity(rows);
        let mut sin = Vec::with_capacity(rows);
        for &[patch_height, patch_width] in &merged_shapes {
            let (visual_cos, visual_sin) = self.visual_rope(patch_height, patch_width)?;
            let padding_rows = m_vision - patch_height * patch_width;
            let rope_device = self.text_rope_cos.device();
            let padding_cos = Tensor::f_ones(
                [padding_rows, decoder::HEAD_DIM / 2],
                (Kind::Float, rope_device),
            )?;
            let padding_sin = Tensor::f_zeros(
                [padding_rows, decoder::HEAD_DIM / 2],
                (Kind::Float, rope_device),
            )?;
            cos.push(Tensor::f_cat(
                &[
                    visual_cos,
                    padding_cos,
                    self.text_rope_cos.f_narrow(0, 0, 1)?,
                ],
                0,
            )?);
            sin.push(Tensor::f_cat(
                &[
                    visual_sin,
                    padding_sin,
                    self.text_rope_sin.f_narrow(0, 0, 1)?,
                ],
                0,
            )?);
        }
        let cos = Tensor::f_stack(&cos, 0)?;
        let sin = Tensor::f_stack(&sin, 0)?;

        let mut caches: Vec<LayerCache> = Vec::with_capacity(self.decoder.layers.len());
        for (index, layer) in self.decoder.layers.iter().enumerate() {
            // Upstream `_reference_generate` adds each layer's depth
            // residual to the visual tokens before the layer runs.
            hidden = Tensor::f_cat(
                &[
                    hidden
                        .f_narrow(1, 0, m_vision)?
                        .f_add(&depth_residuals.f_select(0, index as i64)?)?,
                    hidden.f_narrow(1, m_vision, 1)?,
                ],
                1,
            )?;
            let (next_hidden_states, present) = layer.forward(
                &hidden,
                Some(&prefill_mask),
                (&cos, &sin),
                caches.get(index),
            )?;
            hidden = next_hidden_states;
            caches.push(present);
        }
        let logits = self.decoder.logits(&hidden)?;

        let stops = |token: i64| token == self.eos_token_id || token == tokenizer.pad_token_id();
        let mut next_ids = logits.f_argmax(-1, false)?;
        let first_tokens = Vec::<i64>::try_from(&next_ids)?;
        let mut finished = first_tokens
            .iter()
            .map(|&token| stops(token))
            .collect::<Vec<_>>();
        let mut tokens = first_tokens
            .iter()
            .zip(&finished)
            .map(|(&token, &finished)| if finished { Vec::new() } else { vec![token] })
            .collect::<Vec<_>>();

        for step in 1..MAX_NEW_TOKENS {
            if finished.iter().all(|finished| *finished) {
                break;
            }
            // Argmax IDs stay on the embedding device. Rows decode independently;
            // later predictions for a stopped row are discarded.
            let embedding = Tensor::f_embedding(
                &self.decoder.token_embeddings,
                &next_ids.f_unsqueeze(1)?,
                -1,
                false,
                false,
            )?;
            let cos_step = self.text_rope_cos.f_narrow(0, step, 1)?.f_unsqueeze(0)?;
            let sin_step = self.text_rope_sin.f_narrow(0, step, 1)?.f_unsqueeze(0)?;
            hidden = embedding;
            let step_mask = key_mask(step + 1)?;
            for (index, layer) in self.decoder.layers.iter().enumerate() {
                let (next_hidden_states, present) = layer.forward(
                    &hidden,
                    step_mask.as_ref(),
                    (&cos_step, &sin_step),
                    caches.get(index),
                )?;
                caches[index] = present;
                hidden = next_hidden_states;
            }
            let step_logits = self.decoder.logits(&hidden)?;
            next_ids = step_logits.f_argmax(-1, false)?;
            let next_tokens = Vec::<i64>::try_from(&next_ids)?;
            for (row, &next_token) in next_tokens.iter().enumerate() {
                if finished[row] {
                    continue;
                }
                if stops(next_token) {
                    finished[row] = true;
                } else {
                    tokens[row].push(next_token);
                }
            }
        }
        Ok(tokens)
    }

    /// Interleaved 2D rotary angles over this image's patch grid
    /// (upstream `_get_2d_visual_freqs`), returning `(cos, sin)` of shape
    /// `(height * width, head_dim / 2)`.
    fn visual_rope(&self, height: i64, width: i64) -> koharu_torch::Result<(Tensor, Tensor)> {
        let device = self.text_rope_cos.device();
        let inv_freq = inverse_frequencies(decoder::HEAD_DIM / 2, decoder::ROPE_THETA, device)?;
        let grid_y = Tensor::f_arange(height, (Kind::Float, device))?
            .f_unsqueeze(1)?
            .f_matmul(&inv_freq.f_unsqueeze(0)?)?;
        let grid_x = Tensor::f_arange(width, (Kind::Float, device))?
            .f_unsqueeze(1)?
            .f_matmul(&inv_freq.f_unsqueeze(0)?)?;
        let grid_y = grid_y
            .f_unsqueeze(1)?
            .f_expand([height, width, -1], false)?;
        let grid_x = grid_x
            .f_unsqueeze(0)?
            .f_expand([height, width, -1], false)?;
        let angles = Tensor::f_cat(&[grid_y, grid_x], -1)?.f_flatten(0, 1)?;
        Ok((angles.f_cos()?, angles.f_sin()?))
    }
}

fn inverse_frequencies(d_axis: i64, theta: f64, device: Device) -> koharu_torch::Result<Tensor> {
    let dimensions = Tensor::f_arange_start_step(0, d_axis, 2, (Kind::Float, device))?;
    (dimensions.f_div_scalar(d_axis as f64)?)
        .f_mul_scalar(theta.ln())?
        .f_exp()?
        .f_reciprocal()
}

/// Applies an interleaved (even/odd pairs) rotary embedding over the last
/// dimension (upstream `apply_rotary_emb_2d`). Angles have shape
/// `(batch, sequence, head_dim / 2)` and broadcast across heads.
fn apply_rotary_emb(
    x: &Tensor,
    angles_cos: &Tensor,
    angles_sin: &Tensor,
) -> koharu_torch::Result<Tensor> {
    let original_kind = x.kind();
    let original_size = x.size();
    let mut pairs_shape = original_size.clone();
    let d_head = original_size[original_size.len() - 1];
    let d_axis = d_head / 2;
    let last = pairs_shape.len() - 1;
    pairs_shape[last] = d_axis;
    pairs_shape.push(2);
    let cos = angles_cos.f_unsqueeze(2)?;
    let sin = angles_sin.f_unsqueeze(2)?;
    let x = x.f_to_kind(Kind::Float)?;
    let pairs = x.f_reshape(&pairs_shape)?;
    let even = pairs.f_select(-1, 0)?;
    let odd = pairs.f_select(-1, 1)?;
    let rotated_even = (even.f_mul(&cos)?).f_sub(&odd.f_mul(&sin)?)?;
    let rotated_odd = (even.f_mul(&sin)?).f_add(&odd.f_mul(&cos)?)?;
    Tensor::f_stack(&[rotated_even, rotated_odd], -1)?
        .f_reshape(&original_size)?
        .f_to_kind(original_kind)
}

/// Converts a `[batch, keys]` padding mask into an additive SDPA bias,
/// `[batch, 1, 1, keys]`, where `0` entries become forbidden keys.
fn additive_mask(mask: &Tensor, kind: Kind) -> koharu_torch::Result<Tensor> {
    mask.f_lt(0.5)?
        .f_to_kind(Kind::Float)?
        .f_mul_scalar(f32::MIN as f64)?
        .f_to_kind(kind)?
        .f_unsqueeze(1)?
        .f_unsqueeze(1)
}

#[derive(Debug)]
struct Siglip2VisionModel {
    embeddings: VisionEmbeddings,
    layers: Vec<VisionEncoderLayer>,
    post_layernorm: LayerNorm,
    _head: PoolingHead,
}

impl Siglip2VisionModel {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            embeddings: VisionEmbeddings::new(&(path / "embeddings"))?,
            layers: (0..siglip2::NUM_HIDDEN_LAYERS)
                .map(|index| VisionEncoderLayer::new(&(path / "encoder" / "layers" / index)))
                .collect::<Result<_>>()?,
            post_layernorm: LayerNorm::new(
                path / "post_layernorm",
                siglip2::HIDDEN_SIZE,
                siglip2::LAYER_NORM_EPS,
            )?,
            _head: PoolingHead::new(&(path / "head"))?,
        })
    }

    /// Each layer's input, the embeddings first, as the depths Hayai fuses,
    /// and the last hidden state.
    fn forward(
        &self,
        pixel_values: &Tensor,
        spatial_shapes: &[[i64; 2]],
        attention_mask: Option<&Tensor>,
    ) -> koharu_torch::Result<(Vec<Tensor>, Tensor)> {
        let mut states = Vec::with_capacity(self.layers.len());
        let mut hidden_states = self.embeddings.forward(pixel_values, spatial_shapes)?;
        for layer in &self.layers {
            let next = layer.forward(&hidden_states, attention_mask)?;
            states.push(hidden_states);
            hidden_states = next;
        }
        Ok((states, self.post_layernorm.forward(&hidden_states)?))
    }
}

/// Upstream's `Siglip2MultiheadAttentionPoolingHead`, which the encoder
/// builds as `vision_use_head` is on. Hayai reads the last hidden state, so
/// the head's weights load but never run.
/// https://github.com/huggingface/transformers/blob/598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1/src/transformers/models/siglip2/modeling_siglip2.py#L664-L676
#[derive(Debug)]
struct PoolingHead {
    _probe: Tensor,
    /// `torch.nn.MultiheadAttention`'s packed query, key, and value
    /// projection, then its output projection.
    _in_proj_weight: Tensor,
    _in_proj_bias: Tensor,
    _out_proj: Linear,
    _layernorm: LayerNorm,
    _mlp: VisionMlp,
}

impl PoolingHead {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        let hidden = siglip2::HIDDEN_SIZE;
        let attention = path / "attention";
        Ok(Self {
            _probe: weight(path, "probe", &[1, 1, hidden])?,
            _in_proj_weight: weight(&attention, "in_proj_weight", &[3 * hidden, hidden])?,
            _in_proj_bias: weight(&attention, "in_proj_bias", &[3 * hidden])?,
            _out_proj: Linear::new(&attention / "out_proj", hidden, hidden, true)?,
            _layernorm: LayerNorm::new(path / "layernorm", hidden, siglip2::LAYER_NORM_EPS)?,
            _mlp: VisionMlp::new(&(path / "mlp"))?,
        })
    }
}

#[derive(Debug)]
struct VisionEmbeddings {
    /// NaFlex patchifies on the host, so the upstream convolution collapses to
    /// a linear layer over flattened patches.
    patch_embedding: Linear,
    position_embedding: Tensor,
}

impl VisionEmbeddings {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            patch_embedding: Linear::new(
                path / "patch_embedding",
                siglip2::NUM_CHANNELS * siglip2::PATCH_SIZE * siglip2::PATCH_SIZE,
                siglip2::HIDDEN_SIZE,
                true,
            )?,
            position_embedding: weight(
                &(path / "position_embedding"),
                "weight",
                &[
                    siglip2::POSITION_EMBEDDING_SIZE * siglip2::POSITION_EMBEDDING_SIZE,
                    siglip2::HIDDEN_SIZE,
                ],
            )?,
        })
    }

    /// Patch embeddings plus each image's position embeddings, resized to its
    /// patch grid (upstream `resize_positional_embeddings`, which loops over
    /// the batch).
    fn forward(
        &self,
        pixel_values: &Tensor,
        spatial_shapes: &[[i64; 2]],
    ) -> koharu_torch::Result<Tensor> {
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;
        let patches = pixel_values.size()[1];
        let position_embeddings = spatial_shapes
            .iter()
            .map(|&[height, width]| {
                // Padded patches get none, as upstream adds them only to a
                // crop's own.
                self.position_embeddings(height, width)?.f_constant_pad_nd([
                    0,
                    0,
                    0,
                    patches - height * width,
                ])
            })
            .collect::<koharu_torch::Result<Vec<_>>>()?;
        patch_embeds.f_add(&Tensor::f_stack(&position_embeddings, 0)?)
    }

    fn position_embeddings(&self, height: i64, width: i64) -> koharu_torch::Result<Tensor> {
        // Antialiased bilinear interpolation of the square base grid to this
        // image's patch shape, as upstream: a grid side under 16 shrinks.
        let size = siglip2::POSITION_EMBEDDING_SIZE;
        self.position_embedding
            .f_reshape([size, size, siglip2::HIDDEN_SIZE])?
            .f_permute([2, 0, 1])?
            .f_unsqueeze(0)?
            .f_to_kind(Kind::Float)?
            .f_internal_upsample_bilinear2d_aa([height, width], false, None::<f64>, None::<f64>)?
            .f_to_kind(self.position_embedding.kind())?
            .f_reshape([siglip2::HIDDEN_SIZE, height * width])?
            .f_transpose(0, 1)
    }
}

#[derive(Debug)]
struct VisionEncoderLayer {
    layer_norm1: LayerNorm,
    attention: VisionAttention,
    layer_norm2: LayerNorm,
    mlp: VisionMlp,
}

impl VisionEncoderLayer {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            layer_norm1: LayerNorm::new(
                path / "layer_norm1",
                siglip2::HIDDEN_SIZE,
                siglip2::LAYER_NORM_EPS,
            )?,
            attention: VisionAttention::new(&(path / "self_attn"))?,
            layer_norm2: LayerNorm::new(
                path / "layer_norm2",
                siglip2::HIDDEN_SIZE,
                siglip2::LAYER_NORM_EPS,
            )?,
            mlp: VisionMlp::new(&(path / "mlp"))?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> koharu_torch::Result<Tensor> {
        let attended = self
            .attention
            .forward(&self.layer_norm1.forward(hidden_states)?, attention_mask)?;
        let hidden_states = hidden_states.f_add(&attended)?;
        hidden_states.f_add(
            &self
                .mlp
                .forward(&self.layer_norm2.forward(&hidden_states)?)?,
        )
    }
}

#[derive(Debug)]
struct VisionAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: i64,
    head_dim: i64,
}

impl VisionAttention {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            query: Linear::new(
                path / "q_proj",
                siglip2::HIDDEN_SIZE,
                siglip2::HIDDEN_SIZE,
                true,
            )?,
            key: Linear::new(
                path / "k_proj",
                siglip2::HIDDEN_SIZE,
                siglip2::HIDDEN_SIZE,
                true,
            )?,
            value: Linear::new(
                path / "v_proj",
                siglip2::HIDDEN_SIZE,
                siglip2::HIDDEN_SIZE,
                true,
            )?,
            output: Linear::new(
                path / "out_proj",
                siglip2::HIDDEN_SIZE,
                siglip2::HIDDEN_SIZE,
                true,
            )?,
            num_heads: siglip2::NUM_ATTENTION_HEADS,
            head_dim: siglip2::HEAD_DIM,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> koharu_torch::Result<Tensor> {
        let size = hidden_states.size();
        let [batch_size, sequence_length] = [size[0], size[1]];
        let shape = [batch_size, sequence_length, self.num_heads, self.head_dim];
        let query = self
            .query
            .forward(hidden_states)?
            .f_view(shape)?
            .f_transpose(1, 2)?;
        let key = self
            .key
            .forward(hidden_states)?
            .f_view(shape)?
            .f_transpose(1, 2)?;
        let value = self
            .value
            .forward(hidden_states)?
            .f_view(shape)?
            .f_transpose(1, 2)?;
        let attended = Tensor::f_scaled_dot_product_attention(
            &query,
            &key,
            &value,
            attention_mask,
            0.0,
            false,
            (self.head_dim as f64).powf(-0.5),
            false,
        )?
        .f_transpose(1, 2)?
        .f_contiguous()?
        .f_reshape([batch_size, sequence_length, self.num_heads * self.head_dim])?;
        self.output.forward(&attended)
    }
}

#[derive(Debug)]
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
}

impl VisionMlp {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            fc1: Linear::new(
                path / "fc1",
                siglip2::HIDDEN_SIZE,
                siglip2::INTERMEDIATE_SIZE,
                true,
            )?,
            fc2: Linear::new(
                path / "fc2",
                siglip2::INTERMEDIATE_SIZE,
                siglip2::HIDDEN_SIZE,
                true,
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> koharu_torch::Result<Tensor> {
        // Upstream activation is `gelu_pytorch_tanh`.
        self.fc2
            .forward(&self.fc1.forward(hidden_states)?.f_gelu("tanh")?)
    }
}

#[derive(Debug)]
struct Decoder {
    projector: Projector,
    depth_fusion: DepthFusion,
    token_embeddings: Tensor,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    output_head: Linear,
    /// The IDS classification head, trained alongside the decoder and shipped
    /// in the checkpoint; the model card says it never runs at inference.
    _ids_norm: RmsNorm,
    _ids_head: Linear,
}

impl Decoder {
    fn new(path: &nn::Path<'_>, config: &HayaiConfig) -> Result<Self> {
        Ok(Self {
            _ids_norm: RmsNorm::new(&(path / "ids_norm"), config.d_model)?,
            _ids_head: Linear::new(
                path / "ids_head",
                config.d_model,
                config.num_ids_classes,
                false,
            )?,
            projector: Projector::new(&(path / "projector"), config.d_vision, config.d_model)?,
            depth_fusion: DepthFusion::new(&(path / "depth_fusion"), config)?,
            token_embeddings: weight(
                &(path / "token_embeddings"),
                "weight",
                &[config.vocab_size, config.d_model],
            )?,
            layers: (0..config.n_layers)
                .map(|index| {
                    DecoderLayer::new(&(path / "layers" / index), config.d_model, config.d_ffn)
                })
                .collect::<Result<_>>()?,
            final_norm: RmsNorm::new(&(path / "final_norm"), config.d_model)?,
            output_head: Linear::new(
                path / "output_head",
                config.d_model,
                config.vocab_size,
                false,
            )?,
        })
    }

    /// Logits of the final sequence position; generation consumes nothing else.
    fn logits(&self, hidden_states: &Tensor) -> koharu_torch::Result<Tensor> {
        let last_hidden_state = hidden_states.f_select(1, hidden_states.size()[1] - 1)?;
        self.output_head
            .forward(&self.final_norm.forward(&last_hidden_state)?)
    }
}

/// Upstream `DSCProjector`: each 2x2 block of patches becomes one token,
/// mixed with its 3x3 neighbours, then read in both orders by
/// `VisualCausalFlow`.
#[derive(Debug)]
struct Projector {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    dw_weight: Tensor,
    dw_bias: Tensor,
    causal_flow: CausalFlow,
    out_norm: RmsNorm,
}

impl Projector {
    fn new(path: &nn::Path<'_>, d_vision: i64, d_model: i64) -> Result<Self> {
        let unshuffled = d_vision * decoder::DOWNSCALE * decoder::DOWNSCALE;
        Ok(Self {
            norm: LayerNorm::new(path / "norm", unshuffled, 1e-5)?,
            linear1: Linear::new(path / "mlp" / 0, unshuffled, d_model, false)?,
            linear2: Linear::new(path / "mlp" / 2, d_model, d_model, false)?,
            dw_weight: weight(path, "dw_weight", &[9, d_model])?,
            dw_bias: weight(path, "dw_bias", &[d_model])?,
            causal_flow: CausalFlow::new(&(path / "causal_flow"), d_model)?,
            out_norm: RmsNorm::new(&(path / "out_norm"), d_model)?,
        })
    }

    /// The tokens of `(batch, patches, d_vision)` patch features laid out
    /// row-major on `spatial_shapes`, padded to the batch's longest merged
    /// grid, each crop's merged grid, and the flat patch index of each
    /// token's 2x2 block, which depth fusion pools by.
    fn forward(
        &self,
        visual_features: &Tensor,
        spatial_shapes: &[[i64; 2]],
    ) -> koharu_torch::Result<(Tensor, Vec<[i64; 2]>, Tensor)> {
        let r = decoder::DOWNSCALE;
        let size = visual_features.size();
        let [rows, patches, d_vision] = [size[0], size[1], size[2]];
        let merged = spatial_shapes
            .iter()
            .map(|&[height, width]| [(height + r - 1) / r, (width + r - 1) / r])
            .collect::<Vec<_>>();
        let length = merged
            .iter()
            .map(|&[height, width]| height * width)
            .max()
            .unwrap_or(0)
            .max(1);
        // Upstream `gather_patches`: the r x r neighbours of every merged
        // token, clamped into the grid, which replicate-pads its bottom and
        // right edges. Tokens past a crop's own grid gather clamped garbage
        // that the decoder masks.
        let mut index = Vec::with_capacity((rows * length * r * r) as usize);
        for (row, (&[height, width], &[_, merged_width])) in
            spatial_shapes.iter().zip(&merged).enumerate()
        {
            let merged_width = merged_width.max(1);
            for token in 0..length {
                let (y, x) = (token / merged_width, token % merged_width);
                for dy in 0..r {
                    for dx in 0..r {
                        let source_y = (y * r + dy).min(height - 1).max(0);
                        let source_x = (x * r + dx).min(width - 1).max(0);
                        index.push(row as i64 * patches + source_y * width + source_x);
                    }
                }
            }
        }
        let index = Tensor::f_from_slice(&index)?.f_to_device(visual_features.device())?;
        let tokens = visual_features
            .f_reshape([rows * patches, d_vision])?
            .f_index_select(0, &index)?
            .f_view([rows, length, r * r, d_vision])?
            // Channel c * r * r + k, as `F.pixel_unshuffle` orders them.
            .f_transpose(2, 3)?
            .f_reshape([rows, length, d_vision * r * r])?;
        let tokens = self.linear2.forward(
            &self
                .linear1
                .forward(&self.norm.forward(&tokens)?)?
                .f_gelu("none")?,
        )?;
        let tokens = self.mix(&tokens, &merged)?;
        let tokens = self.causal_flow.forward(&tokens, &merged)?;
        Ok((self.out_norm.forward(&tokens)?, merged, index))
    }

    /// Upstream `_dw_gather_mix_compiled`: each token plus a depthwise 3x3
    /// filter over its neighbours on the merged grid, clamped into it, so
    /// the grid's edges replicate.
    fn mix(&self, tokens: &Tensor, merged: &[[i64; 2]]) -> koharu_torch::Result<Tensor> {
        let size = tokens.size();
        let [rows, length, d_model] = [size[0], size[1], size[2]];
        let mut index = Vec::with_capacity((rows * length * 9) as usize);
        for (row, &[height, width]) in merged.iter().enumerate() {
            let (last_y, last_x) = ((height - 1).max(0), (width - 1).max(0));
            for token in 0..length {
                let (y, x) = (token / width.max(1), token % width.max(1));
                let ys = [(y - 1).max(0), y.min(last_y), (y + 1).min(last_y)];
                let xs = [(x - 1).max(0), x.min(last_x), (x + 1).min(last_x)];
                for source_y in ys {
                    for source_x in xs {
                        index.push(row as i64 * length + source_y * width + source_x);
                    }
                }
            }
        }
        let index = Tensor::f_from_slice(&index)?.f_to_device(tokens.device())?;
        let neighbours = tokens
            .f_reshape([rows * length, d_model])?
            .f_index_select(0, &index)?
            .f_view([rows, length, 9, d_model])?;
        let filtered = neighbours
            .f_mul(&self.dw_weight)?
            .f_sum_dim_intlist(&[2i64][..], false, None::<Kind>)?
            .f_add(&self.dw_bias)?;
        neighbours.f_select(2, 4)?.f_add(&filtered)
    }
}

/// Upstream `VisualCausalFlow`: gated attention over the merged grid, half
/// its heads reading rows left to right, half columns right to left, each
/// token seeing only what comes before it.
#[derive(Debug)]
struct CausalFlow {
    norm: RmsNorm,
    wqkv: Linear,
    wo: Linear,
    flow_gate: Tensor,
}

impl CausalFlow {
    fn new(path: &nn::Path<'_>, d_model: i64) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::new(&(path / "norm"), d_model)?,
            wqkv: Linear::new(path / "wqkv", d_model, 3 * d_model, false)?,
            wo: Linear::new(path / "wo", d_model, d_model, false)?,
            flow_gate: weight(path, "flow_gate", &[d_model])?,
        })
    }

    fn forward(&self, tokens: &Tensor, merged: &[[i64; 2]]) -> koharu_torch::Result<Tensor> {
        let size = tokens.size();
        let [rows, length, d_model] = [size[0], size[1], size[2]];
        let heads = decoder::FLOW_HEADS;
        let head_dim = d_model / heads;
        let qkv = self
            .wqkv
            .forward(&self.norm.forward(tokens)?)?
            .f_view([rows, length, 3, heads, head_dim])?
            .f_permute([2, 0, 3, 1, 4])?;
        let mask = dual_causal_mask(merged, length, tokens.device())?.f_to_kind(tokens.kind())?;
        let attended = Tensor::f_scaled_dot_product_attention(
            &qkv.f_select(0, 0)?,
            &qkv.f_select(0, 1)?,
            &qkv.f_select(0, 2)?,
            Some(&mask),
            0.0,
            false,
            (head_dim as f64).powf(-0.5),
            false,
        )?
        .f_transpose(1, 2)?
        .f_contiguous()?
        .f_reshape([rows, length, d_model])?;
        tokens.f_add(&self.wo.forward(&attended)?.f_mul(&self.flow_gate)?)
    }
}

/// Upstream `VisualCausalFlow.build_dual_causal_mask`, `(batch, FLOW_HEADS,
/// tokens, tokens)`: the first half of the heads lets a token see the grid's
/// tokens up to it row by row, the second half column by column from the
/// right. A token past its crop's grid sees the first, so no row is empty.
fn dual_causal_mask(
    merged: &[[i64; 2]],
    length: i64,
    device: Device,
) -> koharu_torch::Result<Tensor> {
    let mut mask = Vec::with_capacity(merged.len() * 2 * (length * length) as usize);
    for &[height, width] in merged {
        let (height, width) = (height.max(1), width.max(1));
        let valid = height * width;
        let rank = |token: i64| (width - 1 - token % width) * height + token / width;
        for vertical in [false, true] {
            for query in 0..length {
                for key in 0..length {
                    let before = match vertical {
                        false => query >= key,
                        true => rank(query) >= rank(key),
                    };
                    let seen = (before && key < valid) || (query >= valid && key == 0);
                    mask.push(if seen { 0.0f32 } else { f32::NEG_INFINITY });
                }
            }
        }
    }
    let rows = merged.len() as i64;
    let half = decoder::FLOW_HEADS / 2;
    Tensor::f_from_slice(&mask)?
        .f_view([rows, 2, 1, length, length])?
        .f_expand([rows, 2, half, length, length], false)?
        .f_reshape([rows, 2 * half, length, length])?
        .f_to_device(device)
}

/// Upstream `StaticDepthFusion`: each encoder depth, pooled over the
/// projector's 2x2 blocks, normalized, and projected, mixed by a softmax
/// of each decoder layer's weights and scaled per channel. `(layers, batch,
/// tokens, d_model)`.
#[derive(Debug)]
struct DepthFusion {
    norms: Vec<RmsNorm>,
    wo: Linear,
    gamma: Tensor,
    depth_logits: Tensor,
}

impl DepthFusion {
    fn new(path: &nn::Path<'_>, config: &HayaiConfig) -> Result<Self> {
        Ok(Self {
            norms: (0..config.n_vision_layers)
                .map(|index| RmsNorm::new(&(path / "norms" / index), config.d_vision))
                .collect::<Result<_>>()?,
            wo: Linear::new(path / "wo", config.d_vision, config.d_model, false)?,
            gamma: weight(path, "gamma", &[config.n_layers as i64, config.d_model])?,
            depth_logits: weight(
                path,
                "depth_logits",
                &[config.n_layers as i64, config.n_vision_layers as i64],
            )?,
        })
    }

    fn forward(
        &self,
        states: &[Tensor],
        blocks: &Tensor,
        length: i64,
    ) -> koharu_torch::Result<Tensor> {
        let r = decoder::DOWNSCALE;
        let projected = self
            .norms
            .iter()
            .zip(states)
            .map(|(norm, state)| {
                let size = state.size();
                let [rows, d_vision] = [size[0], size[2]];
                // Upstream `_pool_2x2_enc_state`.
                let pooled = state
                    .f_reshape([rows * size[1], d_vision])?
                    .f_index_select(0, blocks)?
                    .f_view([rows, length, r * r, d_vision])?
                    .f_sum_dim_intlist(&[2i64][..], false, None::<Kind>)?
                    .f_mul_scalar(0.25)?;
                self.wo.forward(&norm.forward(&pooled)?)
            })
            .collect::<koharu_torch::Result<Vec<_>>>()?;
        let projected = Tensor::f_stack(&projected, 0)?;
        let alpha = self
            .depth_logits
            .f_softmax(-1, Kind::Float)?
            .f_to_kind(projected.kind())?;
        Tensor::f_einsum("jl,lbnd->jbnd", &[alpha, projected], None::<&[i64]>)?
            .f_mul(&self.gamma.f_unsqueeze(1)?.f_unsqueeze(1)?)
    }
}

#[derive(Debug)]
struct DecoderLayer {
    attn_norm: RmsNorm,
    attn: GroupedQueryAttention,
    ffn_norm: RmsNorm,
    ffn: SwiGlu,
    attn_res_scale: Tensor,
    ffn_res_scale: Tensor,
}

impl DecoderLayer {
    fn new(path: &nn::Path<'_>, d_model: i64, d_ffn: i64) -> Result<Self> {
        Ok(Self {
            attn_norm: RmsNorm::new(&(path / "attn_norm"), d_model)?,
            attn: GroupedQueryAttention::new(&(path / "attn"), d_model)?,
            ffn_norm: RmsNorm::new(&(path / "ffn_norm"), d_model)?,
            ffn: SwiGlu::new(&(path / "ffn"), d_model, d_ffn)?,
            attn_res_scale: weight(path, "attn_res_scale", &[d_model])?,
            ffn_res_scale: weight(path, "ffn_res_scale", &[d_model])?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        rope: (&Tensor, &Tensor),
        past_key_value: Option<&LayerCache>,
    ) -> koharu_torch::Result<(Tensor, LayerCache)> {
        let normalized = self.attn_norm.forward(hidden_states)?;
        let (attention, present) =
            self.attn
                .forward(&normalized, attention_mask, rope, past_key_value)?;
        let hidden_states = hidden_states.f_add(&self.attn_res_scale.f_mul(&attention)?)?;
        let ffn = self.ffn.forward(&self.ffn_norm.forward(&hidden_states)?)?;
        Ok((
            hidden_states.f_add(&self.ffn_res_scale.f_mul(&ffn)?)?,
            present,
        ))
    }
}

#[derive(Debug)]
struct GroupedQueryAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    query_heads: i64,
    key_value_heads: i64,
    head_dim: i64,
}

impl GroupedQueryAttention {
    fn new(path: &nn::Path<'_>, d_model: i64) -> Result<Self> {
        Ok(Self {
            w_q: Linear::new(
                path / "w_q",
                d_model,
                decoder::QUERY_HEADS * decoder::HEAD_DIM,
                false,
            )?,
            w_k: Linear::new(
                path / "w_k",
                d_model,
                decoder::KEY_VALUE_HEADS * decoder::HEAD_DIM,
                false,
            )?,
            w_v: Linear::new(
                path / "w_v",
                d_model,
                decoder::KEY_VALUE_HEADS * decoder::HEAD_DIM,
                false,
            )?,
            w_o: Linear::new(
                path / "w_o",
                decoder::QUERY_HEADS * decoder::HEAD_DIM,
                d_model,
                false,
            )?,
            q_norm: RmsNorm::new(&(path / "q_norm"), decoder::HEAD_DIM)?,
            k_norm: RmsNorm::new(&(path / "k_norm"), decoder::HEAD_DIM)?,
            query_heads: decoder::QUERY_HEADS,
            key_value_heads: decoder::KEY_VALUE_HEADS,
            head_dim: decoder::HEAD_DIM,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        (cos, sin): (&Tensor, &Tensor),
        past_key_value: Option<&LayerCache>,
    ) -> koharu_torch::Result<(Tensor, LayerCache)> {
        let size = hidden_states.size();
        let [batch_size, sequence_length] = [size[0], size[1]];
        let query_shape = [batch_size, sequence_length, self.query_heads, self.head_dim];
        let kv_shape = [
            batch_size,
            sequence_length,
            self.key_value_heads,
            self.head_dim,
        ];
        let query = apply_rotary_emb(
            &self
                .q_norm
                .forward(&self.w_q.forward(hidden_states)?.f_view(query_shape)?)?,
            cos,
            sin,
        )?
        .f_transpose(1, 2)?;
        let key = apply_rotary_emb(
            &self
                .k_norm
                .forward(&self.w_k.forward(hidden_states)?.f_view(kv_shape)?)?,
            cos,
            sin,
        )?
        .f_transpose(1, 2)?;
        let value = self
            .w_v
            .forward(hidden_states)?
            .f_view(kv_shape)?
            .f_transpose(1, 2)?;

        let (key, value) = match past_key_value {
            Some(cache) => (
                Tensor::f_cat(&[cache.key.f_alias()?, key], 2)?,
                Tensor::f_cat(&[cache.value.f_alias()?, value], 2)?,
            ),
            None => (key, value),
        };
        let present = LayerCache {
            key: key.f_alias()?,
            value: value.f_alias()?,
        };

        // Native GQA lets CUDA select flash or memory-efficient SDPA kernels
        // instead of physically repeating K/V heads like the reference code.
        let attention = Tensor::f_scaled_dot_product_attention(
            &query,
            &key,
            &value,
            attention_mask,
            0.0,
            false,
            (self.head_dim as f64).powf(-0.5),
            true,
        )?
        .f_transpose(1, 2)?
        .f_contiguous()?
        .f_reshape([
            batch_size,
            sequence_length,
            self.query_heads * self.head_dim,
        ])?;
        Ok((self.w_o.forward(&attention)?, present))
    }
}

#[derive(Debug)]
struct LayerCache {
    key: Tensor,
    value: Tensor,
}

#[derive(Debug)]
struct SwiGlu {
    w_gate: Linear,
    w_up: Linear,
    w_down: Linear,
}

impl SwiGlu {
    fn new(path: &nn::Path<'_>, d_model: i64, d_ffn: i64) -> Result<Self> {
        Ok(Self {
            w_gate: Linear::new(path / "w_gate", d_model, d_ffn, false)?,
            w_up: Linear::new(path / "w_up", d_model, d_ffn, false)?,
            w_down: Linear::new(path / "w_down", d_ffn, d_model, false)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> koharu_torch::Result<Tensor> {
        self.w_down.forward(
            &self
                .w_gate
                .forward(hidden_states)?
                .f_silu()?
                .f_mul(&self.w_up.forward(hidden_states)?)?,
        )
    }
}

#[derive(Debug)]
struct RmsNorm {
    weight: Tensor,
}

impl RmsNorm {
    fn new(path: &nn::Path<'_>, dim: i64) -> Result<Self> {
        Ok(Self {
            weight: weight(path, "weight", &[dim])?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> koharu_torch::Result<Tensor> {
        let input_kind = hidden_states.kind();
        let hidden_states = hidden_states.f_to_kind(Kind::Float)?;
        let variance = hidden_states.f_pow_tensor_scalar(2.0)?.f_mean_dim(
            &[-1i64][..],
            true,
            Some(Kind::Float),
        )?;
        self.weight
            .f_mul(&hidden_states)?
            .f_mul(&variance.f_add_scalar(RMS_NORM_EPS)?.f_rsqrt()?)?
            .f_to_kind(input_kind)
    }
}

// These inference-only layers use the fork's fallible LibTorch operators so
// allocation failures reach the pipeline's accelerator retry path.
fn weight(path: &nn::Path<'_>, name: &str, shape: &[i64]) -> koharu_torch::Result<Tensor> {
    let tensor = Tensor::f_empty(shape, (path.kind(), path.device()))?;
    Ok(path.add(name, tensor, false))
}

#[derive(Debug)]
struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn new(path: nn::Path<'_>, input: i64, output: i64, bias: bool) -> koharu_torch::Result<Self> {
        let bias = bias.then(|| weight(&path, "bias", &[output])).transpose()?;
        Ok(Self {
            weight: weight(&path, "weight", &[output, input])?,
            bias,
        })
    }

    fn forward(&self, input: &Tensor) -> koharu_torch::Result<Tensor> {
        input.f_linear(&self.weight, self.bias.as_ref())
    }
}

#[derive(Debug)]
struct LayerNorm {
    size: i64,
    eps: f64,
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    fn new(path: nn::Path<'_>, size: i64, eps: f64) -> koharu_torch::Result<Self> {
        Ok(Self {
            size,
            eps,
            weight: weight(&path, "weight", &[size])?,
            bias: weight(&path, "bias", &[size])?,
        })
    }

    fn forward(&self, input: &Tensor) -> koharu_torch::Result<Tensor> {
        input.f_layer_norm(
            [self.size],
            Some(&self.weight),
            Some(&self.bias),
            self.eps,
            true,
        )
    }
}
