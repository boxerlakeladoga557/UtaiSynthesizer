"""so-vits-svc companion-asset converter: cluster kmeans / feature-retrieval.

Two asset kinds ship alongside SoVITS models (both optional, both feed the
"聚类/检索混合比" feature at inference time):

  1. cluster kmeans .pt — torch.save'd {speaker_name: {"n_features_in_": int,
     "_n_threads": int, "cluster_centers_": ndarray [K, dim]}} (see so-vits
     cluster/__init__.py get_cluster_model). Speaker keys are NAMES (may be
     Chinese).
       -> <outdir>/<speaker>.centers.npy          float32 [K, dim]

  2. feature-retrieval .pkl — pickle'd {speaker_id: faiss.Index} (see so-vits
     utils.train_index: "IVF{n},Flat" over the training features; infer_tool
     reads the raw vectors back via index.reconstruct_n(0, ntotal)). Speaker
     keys are the spk2id INTS.
       -> <outdir>/<speaker_id>.index_vectors.npy float32 [N, dim]

The .npy vectors are byte-identical to what the original runtime sees
(reconstruct_n returns the stored Flat vectors in add order) — gated by
verify/voice/gate1_sovits.py round-trip. The Rust side implements the k=8
inverse-square-distance blend from infer_tool.py on these raw vectors.

Usage:
    python export_cluster.py --input kmeans_10000.pt            --outdir out/
    python export_cluster.py --input feature_and_index.pkl      --outdir out/
    (--type cluster|retrieval overrides the extension-based auto-detection)

Requires: torch, numpy; faiss-cpu for retrieval .pkl (pip install faiss-cpu).
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

# Windows console default codepage chokes on Chinese speaker names.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_ILLEGAL = re.compile(r'[\\/:*?"<>|]')


def _safe_name(key) -> str:
    """Speaker key -> filename stem. Chinese survives; path-illegal chars are
    replaced (with a warning) so a hostile key cannot escape outdir."""
    name = str(key)
    safe = _ILLEGAL.sub("_", name)
    if safe != name:
        print(f"WARNING: speaker key {name!r} contains path-illegal chars, "
              f"writing as {safe!r}", file=sys.stderr)
    return safe


def convert_cluster(input_path: Path, outdir: Path) -> list:
    import torch
    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(f"不是 so-vits 聚类模型（期望 speaker->KMeans dict）: {input_path}")
    written = []
    for spk, ckpt in checkpoint.items():
        try:
            centers = ckpt["cluster_centers_"]
        except (TypeError, KeyError):
            raise ValueError(
                f"speaker {spk!r} 缺少 cluster_centers_ — 不是 so-vits 聚类模型")
        centers = np.ascontiguousarray(np.asarray(centers, dtype=np.float32))
        out = outdir / f"{_safe_name(spk)}.centers.npy"
        np.save(str(out), centers)
        print(f"  {spk} -> {out}  {centers.shape} {centers.dtype}")
        written.append(out)
    return written


def convert_retrieval(input_path: Path, outdir: Path) -> list:
    import faiss
    with open(input_path, "rb") as f:
        index_map = pickle.load(f)
    if not isinstance(index_map, dict) or not index_map:
        raise ValueError(f"不是 so-vits 检索索引（期望 speaker_id->faiss.Index dict）: {input_path}")
    written = []
    for spk, index in index_map.items():
        try:
            vectors = index.reconstruct_n(0, index.ntotal)
        except RuntimeError:
            # IVF indexes need a direct map before reconstruct (faiss raises
            # "direct map not initialized"); identical vectors either way.
            faiss.extract_index_ivf(index).make_direct_map()
            vectors = index.reconstruct_n(0, index.ntotal)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        out = outdir / f"{_safe_name(spk)}.index_vectors.npy"
        np.save(str(out), vectors)
        print(f"  {spk} -> {out}  {vectors.shape} {vectors.dtype}")
        written.append(out)
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Convert so-vits cluster .pt / feature-retrieval .pkl to .npy")
    parser.add_argument("--input", type=str, required=True,
                        help="kmeans_*.pt or feature_and_index.pkl")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory for the .npy files")
    parser.add_argument("--type", type=str, default="auto",
                        choices=["auto", "cluster", "retrieval"],
                        help="auto: .pt -> cluster, .pkl -> retrieval")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    outdir.mkdir(parents=True, exist_ok=True)

    kind = args.type
    if kind == "auto":
        suffix = input_path.suffix.lower()
        if suffix == ".pt":
            kind = "cluster"
        elif suffix in (".pkl", ".pickle"):
            kind = "retrieval"
        else:
            print(f"Error: cannot auto-detect type from suffix {suffix!r}; "
                  f"pass --type cluster|retrieval", file=sys.stderr)
            sys.exit(1)

    if kind == "cluster":
        written = convert_cluster(input_path, outdir)
    else:
        written = convert_retrieval(input_path, outdir)
    print(f"Converted {kind}: {input_path} -> {len(written)} file(s) in {outdir}")


if __name__ == "__main__":
    main()
