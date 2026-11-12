#!/usr/bin/env python3
"""Export the exact Koharu RF-DETR Seg 2XL checkpoint to a portable ONNX graph.

Requires Python 3.12+, rfdetr==1.7.0, safetensors, and onnx. The exported
graph is an intermediate artifact, not a validated TensorRT replacement.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import onnx
from rfdetr import RFDETRSeg2XLarge
from safetensors.torch import load_file


def export(checkpoint_dir: Path, output_dir: Path) -> Path:
    config = json.loads((checkpoint_dir / "config.json").read_text())
    if config["architecture"] != "RFDETRSeg2XLarge":
        raise ValueError(f"unsupported RF-DETR architecture: {config['architecture']}")
    if config["rfdetr_version"] != "1.7.0":
        raise ValueError(f"unsupported checkpoint RF-DETR version: {config['rfdetr_version']}")
    version = importlib.metadata.version("rfdetr")
    if version != config["rfdetr_version"]:
        raise ValueError(f"RF-DETR {version} installed, expected {config['rfdetr_version']}")

    resolution = config["resolution"]
    weights_path = checkpoint_dir / config["checkpoint"]
    with weights_path.open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    if digest != config["checkpoint_sha256"]:
        raise ValueError(f"checkpoint SHA-256 mismatch: {digest}")

    weights = load_file(str(weights_path), device="cpu")
    detector = RFDETRSeg2XLarge(
        pretrain_weights=None,
        resolution=resolution,
        num_classes=len(config["classes"]),
    )
    model = detector.model.model
    expected = model.state_dict()
    missing = sorted(expected.keys() - weights.keys())
    unexpected = sorted(weights.keys() - expected.keys())
    mismatched = [
        (name, tuple(expected[name].shape), tuple(weights[name].shape))
        for name in expected.keys() & weights.keys()
        if expected[name].shape != weights[name].shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            f"checkpoint is not an exact architecture match: "
            f"missing={missing}, unexpected={unexpected}, shapes={mismatched}"
        )
    model.load_state_dict(weights, strict=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(
        detector.export(
            format="onnx",
            output_dir=str(output_dir),
            shape=(resolution, resolution),
            batch_size=1,
            verbose=False,
        )
    )
    onnx.checker.check_model(str(output))
    graph = onnx.load(str(output), load_external_data=False)
    tensors = {
        tensor.name: [
            dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim
        ]
        for tensor in [*graph.graph.input, *graph.graph.output]
    }
    expected_tensors = {
        "input": [1, 3, resolution, resolution],
        "dets": [1, 300, 4],
        "labels": [1, 300, len(config["classes"]) + 1],
        "masks": [1, 300, resolution // 4, resolution // 4],
    }
    if tensors != expected_tensors:
        raise ValueError(f"unexpected ONNX inputs/outputs: {tensors}")
    with output.open("rb") as onnx_file:
        output_digest = hashlib.file_digest(onnx_file, "sha256").hexdigest()
    print(f"checkpoint: {digest}")
    print(f"ONNX: {output}")
    print(f"ONNX SHA-256: {output_digest}")
    print(f"tensors: {tensors}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="directory containing the pinned model.safetensors and config.json",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="directory for the generated ONNX artifact",
    )
    args = parser.parse_args()
    export(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
