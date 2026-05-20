"""
Model checkpoint inspector for UTAI v2.
Dumps all parameter names, shapes, dtypes, and metadata from .pth files.
Zero model-architecture dependency — reads raw state_dict only.

Usage:
    python inspect.py --input model.pth --type rvc|sovits [--output report.json]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def inspect_rvc(path: Path) -> dict:
    import torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

    report = {
        "model_type": "rvc",
        "source_file": path.name,
        "top_level_keys": list(checkpoint.keys()),
    }

    for key in checkpoint:
        if key == "weight":
            continue
        val = checkpoint[key]
        if isinstance(val, (str, int, float, bool)):
            report[key] = val
        elif isinstance(val, (list, tuple)):
            report[key] = list(val)
        elif isinstance(val, dict) and len(val) < 50:
            report[key] = {k: str(v) for k, v in val.items()}

    state_dict = checkpoint.get("weight", {})
    if not isinstance(state_dict, dict):
        print(f"ERROR: 'weight' key is {type(state_dict).__name__}, expected dict", file=sys.stderr)
        return report

    params = []
    modules = defaultdict(lambda: {"count": 0, "total_elements": 0, "weight_norm_pairs": []})
    weight_v_keys = set()
    weight_g_keys = set()

    for name, tensor in state_dict.items():
        if not hasattr(tensor, "shape"):
            params.append({"name": name, "type": type(tensor).__name__})
            continue

        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        numel = tensor.numel()
        module_prefix = name.split(".")[0]

        params.append({
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "numel": numel,
        })

        modules[module_prefix]["count"] += 1
        modules[module_prefix]["total_elements"] += numel

        if name.endswith(".weight_v"):
            weight_v_keys.add(name[:-len(".weight_v")])
        elif name.endswith(".weight_g"):
            weight_g_keys.add(name[:-len(".weight_g")])

    weight_norm_layers = sorted(weight_v_keys & weight_g_keys)
    for layer in weight_norm_layers:
        prefix = layer.split(".")[0]
        modules[prefix]["weight_norm_pairs"].append(layer)

    total_elements = sum(p.get("numel", 0) for p in params)

    report["parameters"] = params
    report["total_params"] = len(params)
    report["total_elements"] = total_elements
    report["estimated_size_mb_fp32"] = round(total_elements * 4 / (1024 * 1024), 2)
    report["estimated_size_mb_fp16"] = round(total_elements * 2 / (1024 * 1024), 2)
    report["modules"] = dict(modules)
    report["weight_norm_layers"] = weight_norm_layers
    report["weight_norm_count"] = len(weight_norm_layers)

    return report


def inspect_sovits(path: Path) -> dict:
    import torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

    report = {
        "model_type": "sovits",
        "source_file": path.name,
        "top_level_keys": list(checkpoint.keys()),
    }

    for key in checkpoint:
        if key in ("model", "optimizer"):
            continue
        val = checkpoint[key]
        if isinstance(val, (str, int, float, bool)):
            report[key] = val
        elif isinstance(val, (list, tuple)):
            report[key] = list(val)

    has_optimizer = "optimizer" in checkpoint
    report["has_optimizer"] = has_optimizer
    if has_optimizer:
        opt = checkpoint["optimizer"]
        if isinstance(opt, dict) and "param_groups" in opt:
            report["optimizer_type"] = "Adam (inferred)"
            report["optimizer_param_count"] = len(opt.get("state", {}))

    state_dict = checkpoint.get("model", {})
    if not isinstance(state_dict, dict):
        print(f"ERROR: 'model' key is {type(state_dict).__name__}, expected dict", file=sys.stderr)
        return report

    params = []
    modules = defaultdict(lambda: {"count": 0, "total_elements": 0, "weight_norm_pairs": []})
    weight_v_keys = set()
    weight_g_keys = set()

    for name, tensor in state_dict.items():
        if not hasattr(tensor, "shape"):
            params.append({"name": name, "type": type(tensor).__name__})
            continue

        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        numel = tensor.numel()
        module_prefix = name.split(".")[0]

        params.append({
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "numel": numel,
        })

        modules[module_prefix]["count"] += 1
        modules[module_prefix]["total_elements"] += numel

        if name.endswith(".weight_v"):
            weight_v_keys.add(name[:-len(".weight_v")])
        elif name.endswith(".weight_g"):
            weight_g_keys.add(name[:-len(".weight_g")])

    weight_norm_layers = sorted(weight_v_keys & weight_g_keys)
    for layer in weight_norm_layers:
        prefix = layer.split(".")[0]
        modules[prefix]["weight_norm_pairs"].append(layer)

    total_elements = sum(p.get("numel", 0) for p in params)

    report["parameters"] = params
    report["total_params"] = len(params)
    report["total_elements"] = total_elements
    report["estimated_size_mb_fp32"] = round(total_elements * 4 / (1024 * 1024), 2)
    report["modules"] = dict(modules)
    report["weight_norm_layers"] = weight_norm_layers
    report["weight_norm_count"] = len(weight_norm_layers)
    report["inference_modules"] = [m for m in modules if m != "enc_q"]
    report["training_only_modules"] = [m for m in modules if m == "enc_q"]

    return report


def main():
    parser = argparse.ArgumentParser(description="Inspect .pth model checkpoint")
    parser.add_argument("--input", required=True, help="Path to .pth file")
    parser.add_argument("--type", required=True, choices=["rvc", "sovits"], help="Model type")
    parser.add_argument("--output", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Inspecting {args.type} model: {input_path}", file=sys.stderr)
    print(f"File size: {input_path.stat().st_size / (1024*1024):.1f} MB", file=sys.stderr)

    if args.type == "rvc":
        report = inspect_rvc(input_path)
    else:
        report = inspect_sovits(input_path)

    output = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Report saved to: {out_path}", file=sys.stderr)
    else:
        print(output)

    print(f"\nSummary:", file=sys.stderr)
    print(f"  Total parameters: {report['total_params']}", file=sys.stderr)
    print(f"  Total elements: {report['total_elements']:,}", file=sys.stderr)
    print(f"  Estimated size (fp32): {report.get('estimated_size_mb_fp32', '?')} MB", file=sys.stderr)
    print(f"  Weight-norm layers: {report.get('weight_norm_count', 0)}", file=sys.stderr)
    print(f"  Modules: {list(report.get('modules', {}).keys())}", file=sys.stderr)


if __name__ == "__main__":
    main()
