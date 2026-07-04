"""Legacy UVR MDX-Net (KUIELab Conv-TDF lineage) — .onnx passthrough registry.

These models ship as ready-made fp32 ONNX (opset 13, input "input"
[B, 4, dim_f, dim_t] = [L.re, L.im, R.re, R.im] CaC planes, output "output"
same shape = the PRIMARY stem's spectrogram directly, no mask). No torch
conversion happens — "converting" one means: validate the graph, cross-check
its I/O against this registry, and write the <stem>.json the Rust pipeline
needs (n_fft/compensate are NOT recoverable from the graph — dim_f/dim_t are,
and are asserted against the registry).

Identification follows UVR: md5 of the last 10_000*1024 file bytes
(uvr_vr.uvr_tail_hash), keys matching UVR's mdx_model_data/model_data_new.json.

Inference recipe the registry params feed (UVR SeperateMDX semantics):
hop=1024, chunk = hop*(dim_t-1), trim = n_fft//2 zeros lead/trail, hann OLA,
input bins [0..3) zeroed, primary = net_output * compensate (time domain after
iSTFT), secondary = mix - primary  ==> Rust num_stems=1 + residual_name.
"""

from architectures.uvr_vr import uvr_tail_hash

MDX_HOP = 1024

MDX_NET_REGISTRY = {
    # "UVR-MDX-NET Karaoke": primary = LEAD vocals; instrumental+backing = residual.
    "2f5501189a2f6db6349916fabe8c90de": {
        "name": "UVR_MDXNET_KARA", "n_fft": 6144, "dim_f": 2048, "dim_t": 256,
        "compensate": 1.035, "primary_stem": "Vocals",
        "stem_names": ["vocals"], "residual_name": "instrumental",
    },
    # "UVR-MDX-NET Karaoke 2": primary = instrumental+backing (the karaoke
    # track); LEAD vocals = residual. Mirror-image of KARA — not a typo.
    "1d64a6d2c30f709b8c9b4ce1366d96ee": {
        "name": "UVR_MDXNET_KARA_2", "n_fft": 5120, "dim_f": 2048, "dim_t": 256,
        "compensate": 1.065, "primary_stem": "Instrumental",
        "stem_names": ["instrumental"], "residual_name": "vocals",
    },
}


def detect_config(onnx_path: str) -> dict:
    h = uvr_tail_hash(onnx_path)
    entry = MDX_NET_REGISTRY.get(h)
    if entry is None:
        known = ", ".join(sorted(e["name"] for e in MDX_NET_REGISTRY.values()))
        raise ValueError(
            f"Unknown legacy MDX-Net model (tail-md5 {h}). Its n_fft/compensate "
            f"cannot be read from the graph; add the model to MDX_NET_REGISTRY in "
            f"architectures/mdx_net.py (params from UVR's mdx_model_data registry). "
            f"Known models: {known}"
        )
    return dict(entry)
