//! Inference-only RT-DETRv2 (ResNet-50-vd) port for ogkalu's comic text and
//! bubble detector.
//!
//! The module tree, parameter paths and forward order follow transformers
//! v4.57.6:
//! https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py
//! https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/rt_detr/modeling_rt_detr_resnet.py
//! Training-only paths (denoising queries, auxiliary heads) are omitted.

use std::path::Path;

use anyhow::Result;
use koharu_torch::{
    Device, IndexOp, Kind, Tensor,
    nn::{self, Module, ModuleT},
};

pub(super) const NUM_LABELS: i64 = 3;
pub(super) const NUM_QUERIES: i64 = 300;
const D_MODEL: i64 = 256;
const ENCODER_HEADS: i64 = 8;
const ENCODER_FFN_DIM: i64 = 1024;
const DECODER_HEADS: i64 = 8;
const DECODER_FFN_DIM: i64 = 1024;
const DECODER_LAYERS: usize = 6;
const DECODER_LEVELS: i64 = 3;
const DECODER_POINTS: i64 = 4;
const DECODER_OFFSET_SCALE: f64 = 0.5;
const LAYER_NORM_EPS: f64 = 1e-5;
const POSITION_TEMPERATURE: f64 = 10000.0;
const BACKBONE_STAGES: [(i64, i64, i64, usize); 4] = [
    (64, 256, 1, 3),
    (256, 512, 2, 4),
    (512, 1024, 2, 6),
    (1024, 2048, 2, 3),
];
const ENCODER_IN_CHANNELS: [i64; 3] = [512, 1024, 2048];

#[derive(Debug)]
pub(super) struct Output {
    /// Final decoder layer class logits, `[batch, NUM_QUERIES, NUM_LABELS]`.
    pub logits: Tensor,
    /// Final decoder layer boxes as normalized `cx, cy, w, h`.
    pub pred_boxes: Tensor,
}

#[derive(Debug)]
pub(super) struct Model {
    var_store: nn::VarStore,
    backbone: ResNet,
    encoder_input_proj: Vec<ConvBatchNorm>,
    encoder: HybridEncoder,
    enc_output: (nn::Linear, nn::LayerNorm),
    enc_score_head: nn::Linear,
    enc_bbox_head: Mlp,
    decoder_input_proj: Vec<ConvBatchNorm>,
    decoder: Decoder,
}

impl Model {
    pub fn new(device: Device) -> Self {
        // Unlike the other models this one stays FP32 on CUDA: BF16 moves boxes
        // by several pixels and drops about 6% of the transformers FP32
        // detections on the hard-page set, and the 43M-parameter network is
        // cheap at FP32.
        let var_store = nn::VarStore::new(device);
        let root = var_store.root() / "model";
        let backbone = ResNet::new(&(&root / "backbone" / "model"));
        let encoder_input_proj = ENCODER_IN_CHANNELS
            .iter()
            .enumerate()
            .map(|(index, &channels)| {
                ConvBatchNorm::new(&(&root / "encoder_input_proj" / index), channels, D_MODEL)
            })
            .collect();
        let encoder = HybridEncoder::new(&(&root / "encoder"));
        let enc_output = (
            nn::linear(
                &root / "enc_output" / 0,
                D_MODEL,
                D_MODEL,
                Default::default(),
            ),
            layer_norm(&(&root / "enc_output" / 1)),
        );
        let enc_score_head = nn::linear(
            &root / "enc_score_head",
            D_MODEL,
            NUM_LABELS,
            Default::default(),
        );
        let enc_bbox_head = Mlp::new(&(&root / "enc_bbox_head"), D_MODEL, D_MODEL, 4, 3);
        let decoder_input_proj = (0..DECODER_LEVELS)
            .map(|index| {
                ConvBatchNorm::new(&(&root / "decoder_input_proj" / index), D_MODEL, D_MODEL)
            })
            .collect();
        let decoder = Decoder::new(&(&root / "decoder"));
        Self {
            var_store,
            backbone,
            encoder_input_proj,
            encoder,
            enc_output,
            enc_score_head,
            enc_bbox_head,
            decoder_input_proj,
            decoder,
        }
    }

    pub fn load(&mut self, path: impl AsRef<Path>) -> Result<()> {
        // The complete inference tree is registered before this strict load;
        // checkpoint-only tensors (denoising embeddings, BN counters) are ignored.
        self.var_store.load(path)?;
        Ok(())
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Output {
        let features = self.backbone.forward(pixel_values);
        let projected = features
            .iter()
            .zip(&self.encoder_input_proj)
            .map(|(feature, projection)| projection.forward(feature))
            .collect::<Vec<_>>();
        let encoded = self.encoder.forward(projected);

        let sources = encoded
            .iter()
            .zip(&self.decoder_input_proj)
            .map(|(feature, projection)| projection.forward(feature))
            .collect::<Vec<_>>();
        let spatial_shapes = sources
            .iter()
            .map(|source| (source.size()[2], source.size()[3]))
            .collect::<Vec<_>>();
        let source_flatten = Tensor::cat(
            &sources
                .iter()
                .map(|source| source.flatten(2, -1).transpose(1, 2))
                .collect::<Vec<_>>(),
            1,
        );

        let (anchors, valid_mask) = generate_anchors(&spatial_shapes, source_flatten.device());
        let memory = valid_mask.to_kind(source_flatten.kind()) * &source_flatten;
        let output_memory = self
            .enc_output
            .1
            .forward(&self.enc_output.0.forward(&memory));
        let enc_outputs_class = self.enc_score_head.forward(&output_memory);
        let enc_outputs_coord_logits = self.enc_bbox_head.forward(&output_memory) + anchors;
        let (_, topk_index) =
            enc_outputs_class
                .amax(&[-1i64][..], false)
                .topk(NUM_QUERIES, 1, true, true);
        let reference_points_unact =
            enc_outputs_coord_logits.gather(1, &topk_index.unsqueeze(-1).repeat([1, 1, 4]), false);
        let target =
            output_memory.gather(1, &topk_index.unsqueeze(-1).repeat([1, 1, D_MODEL]), false);

        self.decoder.forward(
            &target,
            &source_flatten,
            &reference_points_unact,
            &spatial_shapes,
        )
    }
}

// RTDetrV2Model.generate_anchors with grid_size 0.05 and no fixed anchor size.
fn generate_anchors(spatial_shapes: &[(i64, i64)], device: Device) -> (Tensor, Tensor) {
    let options = (Kind::Float, device);
    let anchors = spatial_shapes
        .iter()
        .enumerate()
        .map(|(level, &(height, width))| {
            let mesh = Tensor::meshgrid_indexing(
                &[
                    Tensor::arange(height, options),
                    Tensor::arange(width, options),
                ],
                "ij",
            );
            let grid_xy = Tensor::stack(&[&mesh[1], &mesh[0]], -1).unsqueeze(0) + 0.5;
            let grid_xy = grid_xy
                / Tensor::from_slice(&[width as f32, height as f32])
                    .to_device(device)
                    .view([1, 1, 1, 2]);
            let wh = grid_xy.ones_like() * (0.05 * 2f64.powi(level as i32));
            Tensor::cat(&[grid_xy, wh], -1).reshape([-1, height * width, 4])
        })
        .collect::<Vec<_>>();
    let anchors = Tensor::cat(&anchors, 1);
    let valid_mask = anchors
        .gt(1e-2)
        .logical_and(&anchors.lt(1.0 - 1e-2))
        .all_dim(-1, true);
    let anchors = (&anchors / (anchors.ones_like() - &anchors)).log();
    let anchors = anchors.where_self(&valid_mask, &anchors.full_like(f64::from(f32::MAX)));
    (anchors, valid_mask)
}

// RTDetrResNetBackbone: ResNet-50-vd with a deep stem and average-pool shortcuts.
#[derive(Debug)]
struct ResNet {
    embedder: Vec<ConvLayer>,
    stages: Vec<Vec<BottleNeckLayer>>,
}

impl ResNet {
    fn new(path: &nn::Path<'_>) -> Self {
        let embedder_path = path / "embedder" / "embedder";
        let embedder = vec![
            ConvLayer::new(&(&embedder_path / 0), 3, 32, 3, 2, true),
            ConvLayer::new(&(&embedder_path / 1), 32, 32, 3, 1, true),
            ConvLayer::new(&(&embedder_path / 2), 32, 64, 3, 1, true),
        ];
        let stages = BACKBONE_STAGES
            .iter()
            .enumerate()
            .map(|(stage, &(input, output, stride, depth))| {
                let path = path / "encoder" / "stages" / stage as i64 / "layers";
                (0..depth)
                    .map(|index| {
                        if index == 0 {
                            BottleNeckLayer::new(&(&path / 0), input, output, stride)
                        } else {
                            BottleNeckLayer::new(&(&path / index as i64), output, output, 1)
                        }
                    })
                    .collect()
            })
            .collect();
        Self { embedder, stages }
    }

    /// Returns the `stage2`, `stage3` and `stage4` feature maps.
    fn forward(&self, pixel_values: &Tensor) -> Vec<Tensor> {
        let mut hidden = pixel_values.shallow_clone();
        for layer in &self.embedder {
            hidden = layer.forward(&hidden);
        }
        hidden = hidden.max_pool2d([3, 3], [2, 2], [1, 1], [1, 1], false);
        let mut features = Vec::with_capacity(3);
        for (index, stage) in self.stages.iter().enumerate() {
            for layer in stage {
                hidden = layer.forward(&hidden);
            }
            if index > 0 {
                features.push(hidden.shallow_clone());
            }
        }
        features
    }
}

#[derive(Debug)]
struct ConvLayer {
    convolution: nn::Conv2D,
    normalization: nn::BatchNorm,
    relu: bool,
}

impl ConvLayer {
    fn new(
        path: &nn::Path<'_>,
        input: i64,
        output: i64,
        kernel: i64,
        stride: i64,
        relu: bool,
    ) -> Self {
        Self {
            convolution: nn::conv2d(
                path / "convolution",
                input,
                output,
                kernel,
                nn::ConvConfig {
                    stride,
                    padding: kernel / 2,
                    bias: false,
                    ..Default::default()
                },
            ),
            normalization: nn::batch_norm2d(path / "normalization", output, Default::default()),
            relu,
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self
            .normalization
            .forward_t(&self.convolution.forward(xs), false);
        if self.relu { xs.relu() } else { xs }
    }
}

#[derive(Debug)]
enum Shortcut {
    Identity,
    Projection(ConvLayer),
    PooledProjection(ConvLayer),
}

#[derive(Debug)]
struct BottleNeckLayer {
    shortcut: Shortcut,
    layer: [ConvLayer; 3],
}

impl BottleNeckLayer {
    fn new(path: &nn::Path<'_>, input: i64, output: i64, stride: i64) -> Self {
        let reduced = output / 4;
        let projects = input != output || stride != 1;
        let shortcut = if stride == 2 {
            Shortcut::PooledProjection(ConvLayer::new(
                &(path / "shortcut" / 1),
                input,
                output,
                1,
                1,
                false,
            ))
        } else if projects {
            Shortcut::Projection(ConvLayer::new(
                &(path / "shortcut"),
                input,
                output,
                1,
                stride,
                false,
            ))
        } else {
            Shortcut::Identity
        };
        let layer = path / "layer";
        Self {
            shortcut,
            layer: [
                ConvLayer::new(&(&layer / 0), input, reduced, 1, 1, true),
                ConvLayer::new(&(&layer / 1), reduced, reduced, 3, stride, true),
                ConvLayer::new(&(&layer / 2), reduced, output, 1, 1, false),
            ],
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let residual = match &self.shortcut {
            Shortcut::Identity => xs.shallow_clone(),
            Shortcut::Projection(projection) => projection.forward(xs),
            Shortcut::PooledProjection(projection) => {
                projection.forward(&xs.avg_pool2d([2, 2], [2, 2], [0, 0], true, true, None))
            }
        };
        let mut hidden = xs.shallow_clone();
        for layer in &self.layer {
            hidden = layer.forward(&hidden);
        }
        (hidden + residual).relu()
    }
}

/// `nn.Sequential(Conv2d(bias=False), BatchNorm2d)` input projection.
#[derive(Debug)]
struct ConvBatchNorm {
    conv: nn::Conv2D,
    norm: nn::BatchNorm,
}

impl ConvBatchNorm {
    fn new(path: &nn::Path<'_>, input: i64, output: i64) -> Self {
        Self {
            conv: nn::conv2d(
                path / 0,
                input,
                output,
                1,
                nn::ConvConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            norm: nn::batch_norm2d(path / 1, output, Default::default()),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        self.norm.forward_t(&self.conv.forward(xs), false)
    }
}

/// RTDetrV2ConvNormLayer with an optional SiLU.
#[derive(Debug)]
struct ConvNormLayer {
    conv: nn::Conv2D,
    norm: nn::BatchNorm,
    silu: bool,
}

impl ConvNormLayer {
    fn new(
        path: &nn::Path<'_>,
        input: i64,
        output: i64,
        kernel: i64,
        stride: i64,
        silu: bool,
    ) -> Self {
        Self {
            conv: nn::conv2d(
                path / "conv",
                input,
                output,
                kernel,
                nn::ConvConfig {
                    stride,
                    padding: (kernel - 1) / 2,
                    bias: false,
                    ..Default::default()
                },
            ),
            norm: nn::batch_norm2d(path / "norm", output, Default::default()),
            silu,
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self.norm.forward_t(&self.conv.forward(xs), false);
        if self.silu { xs.silu() } else { xs }
    }
}

#[derive(Debug)]
struct RepVggBlock {
    conv1: ConvNormLayer,
    conv2: ConvNormLayer,
}

impl RepVggBlock {
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            conv1: ConvNormLayer::new(&(path / "conv1"), D_MODEL, D_MODEL, 3, 1, false),
            conv2: ConvNormLayer::new(&(path / "conv2"), D_MODEL, D_MODEL, 1, 1, false),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        (self.conv1.forward(xs) + self.conv2.forward(xs)).silu()
    }
}

// hidden_expansion is 1.0, so `conv3` is the identity.
#[derive(Debug)]
struct CspRepLayer {
    conv1: ConvNormLayer,
    conv2: ConvNormLayer,
    bottlenecks: Vec<RepVggBlock>,
}

impl CspRepLayer {
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            conv1: ConvNormLayer::new(&(path / "conv1"), D_MODEL * 2, D_MODEL, 1, 1, true),
            conv2: ConvNormLayer::new(&(path / "conv2"), D_MODEL * 2, D_MODEL, 1, 1, true),
            bottlenecks: (0..3)
                .map(|index| RepVggBlock::new(&(path / "bottlenecks" / index)))
                .collect(),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut first = self.conv1.forward(xs);
        for block in &self.bottlenecks {
            first = block.forward(&first);
        }
        first + self.conv2.forward(xs)
    }
}

/// RTDetrV2HybridEncoder with AIFI on the stride-32 level (`encode_proj_layers = [2]`).
#[derive(Debug)]
struct HybridEncoder {
    encoder: EncoderLayer,
    lateral_convs: Vec<ConvNormLayer>,
    fpn_blocks: Vec<CspRepLayer>,
    downsample_convs: Vec<ConvNormLayer>,
    pan_blocks: Vec<CspRepLayer>,
}

impl HybridEncoder {
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            encoder: EncoderLayer::new(&(path / "encoder" / 0 / "layers" / 0)),
            lateral_convs: (0..2)
                .map(|index| {
                    ConvNormLayer::new(
                        &(path / "lateral_convs" / index),
                        D_MODEL,
                        D_MODEL,
                        1,
                        1,
                        true,
                    )
                })
                .collect(),
            fpn_blocks: (0..2)
                .map(|index| CspRepLayer::new(&(path / "fpn_blocks" / index)))
                .collect(),
            downsample_convs: (0..2)
                .map(|index| {
                    ConvNormLayer::new(
                        &(path / "downsample_convs" / index),
                        D_MODEL,
                        D_MODEL,
                        3,
                        2,
                        true,
                    )
                })
                .collect(),
            pan_blocks: (0..2)
                .map(|index| CspRepLayer::new(&(path / "pan_blocks" / index)))
                .collect(),
        }
    }

    fn forward(&self, mut hidden_states: Vec<Tensor>) -> Vec<Tensor> {
        let size = hidden_states[2].size();
        let (height, width) = (size[2], size[3]);
        let flattened = hidden_states[2].flatten(2, -1).permute([0, 2, 1]);
        let position = build_2d_sincos_position_embedding(width, height, &flattened);
        hidden_states[2] = self
            .encoder
            .forward(&flattened, &position)
            .permute([0, 2, 1])
            .reshape([-1, D_MODEL, height, width])
            .contiguous();

        let mut fpn = vec![hidden_states[2].shallow_clone()];
        for (index, (lateral, block)) in self.lateral_convs.iter().zip(&self.fpn_blocks).enumerate()
        {
            let backbone_feature = &hidden_states[1 - index];
            let top = lateral.forward(fpn.last().unwrap());
            let size = top.size();
            let upsampled = top.upsample_nearest2d([size[2] * 2, size[3] * 2], 2.0, 2.0);
            *fpn.last_mut().unwrap() = top;
            fpn.push(block.forward(&Tensor::cat(&[&upsampled, backbone_feature], 1)));
        }
        fpn.reverse();

        let mut pan = vec![fpn[0].shallow_clone()];
        for (index, (downsample, block)) in self
            .downsample_convs
            .iter()
            .zip(&self.pan_blocks)
            .enumerate()
        {
            let downsampled = downsample.forward(pan.last().unwrap());
            pan.push(block.forward(&Tensor::cat(&[&downsampled, &fpn[index + 1]], 1)));
        }
        pan
    }
}

// Upstream builds the grid with `meshgrid(grid_w, grid_h, indexing="ij")`,
// which flattens width-major while features flatten height-major. The
// checkpoint was trained with that ordering, so it is kept.
fn build_2d_sincos_position_embedding(width: i64, height: i64, like: &Tensor) -> Tensor {
    let options = (Kind::Float, like.device());
    let mesh = Tensor::meshgrid_indexing(
        &[
            Tensor::arange(width, options),
            Tensor::arange(height, options),
        ],
        "ij",
    );
    let position_dim = D_MODEL / 4;
    let omega = Tensor::arange(position_dim, options) / position_dim as f64;
    let omega = (omega * POSITION_TEMPERATURE.ln()).exp().reciprocal();
    let out_w = mesh[0]
        .flatten(0, -1)
        .unsqueeze(-1)
        .matmul(&omega.unsqueeze(0));
    let out_h = mesh[1]
        .flatten(0, -1)
        .unsqueeze(-1)
        .matmul(&omega.unsqueeze(0));
    Tensor::cat(&[out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], 1).unsqueeze(0)
}

#[derive(Debug)]
struct MultiheadAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    num_heads: i64,
}

impl MultiheadAttention {
    fn new(path: &nn::Path<'_>, num_heads: i64) -> Self {
        let linear = |name: &str| nn::linear(path / name, D_MODEL, D_MODEL, Default::default());
        Self {
            q_proj: linear("q_proj"),
            k_proj: linear("k_proj"),
            v_proj: linear("v_proj"),
            out_proj: linear("out_proj"),
            num_heads,
        }
    }

    /// Queries and keys see `hidden_states + position`; values see `hidden_states`.
    fn forward(&self, hidden_states: &Tensor, position: &Tensor) -> Tensor {
        let size = hidden_states.size();
        let (batch_size, length) = (size[0], size[1]);
        let head_dim = D_MODEL / self.num_heads;
        let positioned = hidden_states + position;
        let heads = |xs: Tensor| {
            xs.view([batch_size, length, self.num_heads, head_dim])
                .transpose(1, 2)
        };
        let query = heads(self.q_proj.forward(&positioned) * (head_dim as f64).powf(-0.5));
        let key = heads(self.k_proj.forward(&positioned));
        let value = heads(self.v_proj.forward(hidden_states));
        let weights = query
            .matmul(&key.transpose(-2, -1))
            .softmax(-1, None::<Kind>);
        let output = weights
            .matmul(&value)
            .transpose(1, 2)
            .reshape([batch_size, length, D_MODEL]);
        self.out_proj.forward(&output)
    }
}

#[derive(Debug)]
struct EncoderLayer {
    self_attn: MultiheadAttention,
    self_attn_layer_norm: nn::LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
    final_layer_norm: nn::LayerNorm,
}

impl EncoderLayer {
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            self_attn: MultiheadAttention::new(&(path / "self_attn"), ENCODER_HEADS),
            self_attn_layer_norm: layer_norm(&(path / "self_attn_layer_norm")),
            fc1: nn::linear(path / "fc1", D_MODEL, ENCODER_FFN_DIM, Default::default()),
            fc2: nn::linear(path / "fc2", ENCODER_FFN_DIM, D_MODEL, Default::default()),
            final_layer_norm: layer_norm(&(path / "final_layer_norm")),
        }
    }

    // normalize_before is false: post-norm residual blocks.
    fn forward(&self, hidden_states: &Tensor, position: &Tensor) -> Tensor {
        let hidden_states = self
            .self_attn_layer_norm
            .forward(&(hidden_states + self.self_attn.forward(hidden_states, position)));
        let feed_forward = self
            .fc2
            .forward(&self.fc1.forward(&hidden_states).gelu("none"));
        self.final_layer_norm
            .forward(&(hidden_states + feed_forward))
    }
}

#[derive(Debug)]
struct Decoder {
    layers: Vec<DecoderLayer>,
    query_pos_head: Mlp,
    class_embed: Vec<nn::Linear>,
    bbox_embed: Vec<Mlp>,
}

impl Decoder {
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            layers: (0..DECODER_LAYERS)
                .map(|index| DecoderLayer::new(&(path / "layers" / index as i64)))
                .collect(),
            query_pos_head: Mlp::new(&(path / "query_pos_head"), 4, 2 * D_MODEL, D_MODEL, 2),
            // Tied with RTDetrV2ForObjectDetection.class_embed / bbox_embed.
            class_embed: (0..DECODER_LAYERS)
                .map(|index| {
                    nn::linear(
                        path / "class_embed" / index as i64,
                        D_MODEL,
                        NUM_LABELS,
                        Default::default(),
                    )
                })
                .collect(),
            bbox_embed: (0..DECODER_LAYERS)
                .map(|index| {
                    Mlp::new(
                        &(path / "bbox_embed" / index as i64),
                        D_MODEL,
                        D_MODEL,
                        4,
                        3,
                    )
                })
                .collect(),
        }
    }

    fn forward(
        &self,
        target: &Tensor,
        memory: &Tensor,
        reference_points_unact: &Tensor,
        spatial_shapes: &[(i64, i64)],
    ) -> Output {
        let mut hidden_states = target.shallow_clone();
        let mut reference_points = reference_points_unact.sigmoid();
        for (index, layer) in self.layers.iter().enumerate() {
            let position = self.query_pos_head.forward(&reference_points);
            hidden_states = layer.forward(
                &hidden_states,
                &position,
                &reference_points.unsqueeze(2),
                memory,
                spatial_shapes,
            );
            reference_points = (self.bbox_embed[index].forward(&hidden_states)
                + inverse_sigmoid(&reference_points))
            .sigmoid();
        }
        Output {
            logits: self.class_embed[DECODER_LAYERS - 1].forward(&hidden_states),
            pred_boxes: reference_points,
        }
    }
}

fn inverse_sigmoid(xs: &Tensor) -> Tensor {
    let xs = xs.clamp(0.0, 1.0);
    (xs.clamp_min(1e-5) / (xs.ones_like() - &xs).clamp_min(1e-5)).log()
}

#[derive(Debug)]
struct DecoderLayer {
    self_attn: MultiheadAttention,
    self_attn_layer_norm: nn::LayerNorm,
    encoder_attn: MultiscaleDeformableAttention,
    encoder_attn_layer_norm: nn::LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
    final_layer_norm: nn::LayerNorm,
}

impl DecoderLayer {
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            self_attn: MultiheadAttention::new(&(path / "self_attn"), DECODER_HEADS),
            self_attn_layer_norm: layer_norm(&(path / "self_attn_layer_norm")),
            encoder_attn: MultiscaleDeformableAttention::new(&(path / "encoder_attn")),
            encoder_attn_layer_norm: layer_norm(&(path / "encoder_attn_layer_norm")),
            fc1: nn::linear(path / "fc1", D_MODEL, DECODER_FFN_DIM, Default::default()),
            fc2: nn::linear(path / "fc2", DECODER_FFN_DIM, D_MODEL, Default::default()),
            final_layer_norm: layer_norm(&(path / "final_layer_norm")),
        }
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position: &Tensor,
        reference_points: &Tensor,
        memory: &Tensor,
        spatial_shapes: &[(i64, i64)],
    ) -> Tensor {
        let hidden_states = self
            .self_attn_layer_norm
            .forward(&(hidden_states + self.self_attn.forward(hidden_states, position)));
        let hidden_states = self.encoder_attn_layer_norm.forward(
            &(&hidden_states
                + self.encoder_attn.forward(
                    &(&hidden_states + position),
                    reference_points,
                    memory,
                    spatial_shapes,
                )),
        );
        let feed_forward = self.fc2.forward(&self.fc1.forward(&hidden_states).relu());
        self.final_layer_norm
            .forward(&(hidden_states + feed_forward))
    }
}

/// RTDetrV2MultiscaleDeformableAttention with `decoder_method = "default"`.
#[derive(Debug)]
struct MultiscaleDeformableAttention {
    sampling_offsets: nn::Linear,
    attention_weights: nn::Linear,
    value_proj: nn::Linear,
    output_proj: nn::Linear,
    n_points_scale: Tensor,
}

impl MultiscaleDeformableAttention {
    fn new(path: &nn::Path<'_>) -> Self {
        let samples = DECODER_HEADS * DECODER_LEVELS * DECODER_POINTS;
        Self {
            sampling_offsets: nn::linear(
                path / "sampling_offsets",
                D_MODEL,
                samples * 2,
                Default::default(),
            ),
            attention_weights: nn::linear(
                path / "attention_weights",
                D_MODEL,
                samples,
                Default::default(),
            ),
            value_proj: nn::linear(path / "value_proj", D_MODEL, D_MODEL, Default::default()),
            output_proj: nn::linear(path / "output_proj", D_MODEL, D_MODEL, Default::default()),
            n_points_scale: path
                .zeros_no_train("n_points_scale", &[DECODER_LEVELS * DECODER_POINTS]),
        }
    }

    fn forward(
        &self,
        query: &Tensor,
        reference_points: &Tensor,
        memory: &Tensor,
        spatial_shapes: &[(i64, i64)],
    ) -> Tensor {
        let size = query.size();
        let (batch_size, num_queries) = (size[0], size[1]);
        let head_dim = D_MODEL / DECODER_HEADS;
        let samples = DECODER_LEVELS * DECODER_POINTS;
        let value = self
            .value_proj
            .forward(memory)
            .view([batch_size, -1, DECODER_HEADS, head_dim]);
        let offsets = self.sampling_offsets.forward(query).view([
            batch_size,
            num_queries,
            DECODER_HEADS,
            samples,
            2,
        ]);
        let weights = self
            .attention_weights
            .forward(query)
            .view([batch_size, num_queries, DECODER_HEADS, samples])
            .softmax(-1, None::<Kind>);
        // reference_points: [batch, queries, 1, 4] as cx, cy, w, h.
        let reference = reference_points.unsqueeze(2);
        let offset = offsets
            * self.n_points_scale.to_kind(query.kind()).unsqueeze(-1)
            * reference.i((.., .., .., .., 2..4))
            * DECODER_OFFSET_SCALE;
        let locations = reference.i((.., .., .., .., 0..2)) + offset;

        // multi_scale_deformable_attention_v2
        let lengths = spatial_shapes
            .iter()
            .map(|&(height, width)| height * width)
            .collect::<Vec<_>>();
        let values = value
            .permute([0, 2, 3, 1])
            .flatten(0, 1)
            .split_with_sizes(&lengths, -1);
        let grids = (locations * 2.0 - 1.0)
            .permute([0, 2, 1, 3, 4])
            .flatten(0, 1)
            .split_with_sizes([DECODER_POINTS; DECODER_LEVELS as usize], -2);
        let sampled = spatial_shapes
            .iter()
            .zip(values.iter().zip(&grids))
            .map(|(&(height, width), (value, grid))| {
                value
                    .reshape([batch_size * DECODER_HEADS, head_dim, height, width])
                    .grid_sampler_2d(grid, 0, 0, false)
            })
            .collect::<Vec<_>>();
        let weights = weights.permute([0, 2, 1, 3]).reshape([
            batch_size * DECODER_HEADS,
            1,
            num_queries,
            samples,
        ]);
        let output = (Tensor::cat(&sampled, -1) * weights)
            .sum_dim_intlist(&[-1i64][..], false, None::<Kind>)
            .view([batch_size, D_MODEL, num_queries])
            .transpose(1, 2)
            .contiguous();
        self.output_proj.forward(&output)
    }
}

#[derive(Debug)]
struct Mlp {
    layers: Vec<nn::Linear>,
}

impl Mlp {
    fn new(
        path: &nn::Path<'_>,
        input_dim: i64,
        hidden_dim: i64,
        output_dim: i64,
        num_layers: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|index| {
                let input = if index == 0 { input_dim } else { hidden_dim };
                let output = if index + 1 == num_layers {
                    output_dim
                } else {
                    hidden_dim
                };
                nn::linear(
                    path / "layers" / index as i64,
                    input,
                    output,
                    Default::default(),
                )
            })
            .collect();
        Self { layers }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.shallow_clone();
        for (index, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs);
            if index + 1 != self.layers.len() {
                xs = xs.relu();
            }
        }
        xs
    }
}

fn layer_norm(path: &nn::Path<'_>) -> nn::LayerNorm {
    nn::layer_norm(
        path,
        vec![D_MODEL],
        nn::LayerNormConfig {
            eps: LAYER_NORM_EPS,
            ..Default::default()
        },
    )
}
