# gate_contentvec.py — ContentVec (vec768l12 / vec256l9) gates vs the reference.
#
# REFERENCE CHOICE (converter\verify\README.md: gate vs ORIGINAL, never self):
#   The true original is fairseq 0.12.2 HubertModel.extract_features (so-vits
#   vencoder\ContentVec768L12.py / ContentVec256L9.py and RVC pipeline.py:212-220
#   make the IDENTICAL call: all-False padding_mask, output_layer 12|9, final_proj
#   only for the 256 variant). fairseq 0.12.2 has NO Windows wheel (sdist needs an
#   MSVC Cython build) and fights torch 2.12, so the oracle here is:
#     transformers HubertModel(+final_proj) loaded from
#     D:\MyDev\Much-Better-S2H\pretrained\content-vec-best  (lengyue233/content-vec-best)
#   That conversion was verified against REAL fairseq twice:
#     * its own convert.py sanity check (fairseq vs HF on random input), and
#     * Much-Better-S2H scripts/extract_contentvec256.py header: "Verified
#       bit-identical to so-vits's fairseq ContentVec256L9 (cos=1.0)".
#   Gate 0 below closes the remaining hole by proving the HF weights are EXACTLY
#   the tensors of OUR source checkpoint (checkpoint_best_legacy_500.pt) under the
#   documented key mapping — so the oracle runs original-weight math from an
#   independent implementation (transformers eager attention), not our code.
#
# Gates (run: ..\..\..\converter\.venv\Scripts\python.exe gate_contentvec.py):
#   (0) weight identity : fairseq ckpt tensors (fp16->fp32 cast) vs loaded HF model
#                         state_dict under the mapping. Must be EXACTLY 0 for every
#                         tensor (incl. the weight-normed pos_conv g/v and the fused
#                         effective weight).
#   (a) torch port      : our export_contentvec.HubertModel (fp32) vs HF reference
#                         (fp32, attn_implementation="eager") on REAL audio:
#                         max_abs_diff < 1e-5 for both variants.
#   (a2) f64 structure  : same comparison with both models forced to f64 —
#                         max_abs_diff < 1e-9. Removes fp32 rounding entirely, so
#                         any algorithmic drift (norm order, scaling, gelu flavor,
#                         pos-conv trim...) cannot hide inside fp noise.
#   (b) onnx vs ref     : contentvec_768l12.onnx / contentvec_256l9.onnx (ORT CPU)
#                         vs HF reference on REAL audio, both test lengths:
#                         min per-frame cosine > 0.9999 AND max_abs_diff < 1e-4.
#   (c) dynamic shape   : two different lengths (odd + even frame count) both
#                         return T == (N-400)//320+1.
#   (bisect aid)        : per-encoder-layer max_abs (ours vs HF) printed for the
#                         record — must grow smoothly (fp accumulation), no jumps.
#
# Real audio: D:\MyDev\TESTING\ikanaiteyo\vocal.wav (vocal stem), two slices
# resampled to 16 kHz mono via librosa/soxr. Both sides consume the same array.

import os
import sys

import numpy as np
import torch

# ---------------- paths ----------------
CKPT = r"D:\MyDev\so-vits-svc\so-vits-svc\pretrain\checkpoint_best_legacy_500.pt"
HF_DIR = r"D:\MyDev\Much-Better-S2H\pretrained\content-vec-best"
CONVERTER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX_768 = os.path.join(CONVERTER, "test_output", "contentvec_768l12.onnx")
ONNX_256 = os.path.join(CONVERTER, "test_output", "contentvec_256l9.onnx")
WAV = r"D:\MyDev\TESTING\ikanaiteyo\vocal.wav"
SR = 16000
# two slices: N=160000 -> T=499 (odd), N=100000 -> T=312 (even)
SLICES = [(30.0, 160000), (60.0, 100000)]

sys.path.insert(0, CONVERTER)
from export_contentvec import HubertModel, build_model, expected_frames  # noqa: E402


# ---------------- fairseq-key -> HF-key mapping ----------------
# Vendored from the conversion that produced HF_DIR (content-vec-best convert.py,
# i.e. lengyue233/content-vec-best). HF side uses the torch>=2.1 parametrized
# weight-norm key names for pos_conv (original0=weight_g, original1=weight_v).
def hf_to_fairseq_mapping():
    mapping = {
        "masked_spec_embed": "mask_emb",
        "encoder.layer_norm.bias": "encoder.layer_norm.bias",
        "encoder.layer_norm.weight": "encoder.layer_norm.weight",
        "encoder.pos_conv_embed.conv.bias": "encoder.pos_conv.0.bias",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0":
            "encoder.pos_conv.0.weight_g",
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1":
            "encoder.pos_conv.0.weight_v",
        "feature_projection.layer_norm.bias": "layer_norm.bias",
        "feature_projection.layer_norm.weight": "layer_norm.weight",
        "feature_projection.projection.bias": "post_extract_proj.bias",
        "feature_projection.projection.weight": "post_extract_proj.weight",
        "final_proj.bias": "final_proj.bias",
        "final_proj.weight": "final_proj.weight",
    }
    for layer in range(12):
        for j in ["q", "k", "v"]:
            mapping[f"encoder.layers.{layer}.attention.{j}_proj.weight"] = \
                f"encoder.layers.{layer}.self_attn.{j}_proj.weight"
            mapping[f"encoder.layers.{layer}.attention.{j}_proj.bias"] = \
                f"encoder.layers.{layer}.self_attn.{j}_proj.bias"
        mapping[f"encoder.layers.{layer}.final_layer_norm.bias"] = \
            f"encoder.layers.{layer}.final_layer_norm.bias"
        mapping[f"encoder.layers.{layer}.final_layer_norm.weight"] = \
            f"encoder.layers.{layer}.final_layer_norm.weight"
        mapping[f"encoder.layers.{layer}.layer_norm.bias"] = \
            f"encoder.layers.{layer}.self_attn_layer_norm.bias"
        mapping[f"encoder.layers.{layer}.layer_norm.weight"] = \
            f"encoder.layers.{layer}.self_attn_layer_norm.weight"
        mapping[f"encoder.layers.{layer}.attention.out_proj.bias"] = \
            f"encoder.layers.{layer}.self_attn.out_proj.bias"
        mapping[f"encoder.layers.{layer}.attention.out_proj.weight"] = \
            f"encoder.layers.{layer}.self_attn.out_proj.weight"
        mapping[f"encoder.layers.{layer}.feed_forward.intermediate_dense.bias"] = \
            f"encoder.layers.{layer}.fc1.bias"
        mapping[f"encoder.layers.{layer}.feed_forward.intermediate_dense.weight"] = \
            f"encoder.layers.{layer}.fc1.weight"
        mapping[f"encoder.layers.{layer}.feed_forward.output_dense.bias"] = \
            f"encoder.layers.{layer}.fc2.bias"
        mapping[f"encoder.layers.{layer}.feed_forward.output_dense.weight"] = \
            f"encoder.layers.{layer}.fc2.weight"
    for layer in range(7):
        mapping[f"feature_extractor.conv_layers.{layer}.conv.weight"] = \
            f"feature_extractor.conv_layers.{layer}.0.weight"
        if layer == 0:
            mapping[f"feature_extractor.conv_layers.{layer}.layer_norm.weight"] = \
                f"feature_extractor.conv_layers.{layer}.2.weight"
            mapping[f"feature_extractor.conv_layers.{layer}.layer_norm.bias"] = \
                f"feature_extractor.conv_layers.{layer}.2.bias"
    return mapping


def load_reference():
    import warnings
    warnings.simplefilter("ignore")
    from torch import nn
    from transformers import HubertModel as HFHubertModel

    class HubertModelWithFinalProj(HFHubertModel):
        def __init__(self, config):
            super().__init__(config)
            self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    m, info = HubertModelWithFinalProj.from_pretrained(
        HF_DIR, attn_implementation="eager", output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"] \
        and not info["mismatched_keys"], info
    return m.eval()


def load_audio_slice(offset_s, n_samples):
    import librosa
    y, _ = librosa.load(WAV, sr=SR, mono=True, offset=offset_s,
                        duration=n_samples / SR + 0.5)
    y = y[:n_samples].astype(np.float32)
    assert len(y) == n_samples, len(y)
    rms = float(np.sqrt(np.mean(y ** 2)))
    assert rms > 1e-3, f"slice at {offset_s}s is silent (rms={rms})"
    return y


def per_frame_cos_min(a, b):
    # a, b: [T, D]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.min(num / den))


@torch.no_grad()
def ref_features(ref, wav_t):
    out = ref(wav_t, output_hidden_states=True)
    f768 = out.hidden_states[12]                       # [1, T, 768]
    f256 = ref.final_proj(out.hidden_states[9])        # [1, T, 256]
    return f768, f256


@torch.no_grad()
def our_layerwise(model, wav_t):
    # orchestration-only replay of HubertModel.extract using the model's own
    # submodules, collecting every encoder layer output for the bisect report.
    x = model.feature_extractor(wav_t).transpose(1, 2)
    x = model.layer_norm(x)
    x = model.post_extract_proj(x)
    x = x + model.encoder.pos_conv(x.transpose(1, 2)).transpose(1, 2)
    x = model.encoder.layer_norm(x)
    outs = []
    for layer in model.encoder.layers:
        x = layer(x)
        outs.append(x)
    return outs  # outs[i] == encoder output at fairseq output_layer=i+1


def main():
    failures = []

    # ---------------- gate 0: weight identity ----------------
    print("== gate 0: weight identity (fairseq ckpt vs HF reference) ==")
    from export_contentvec import load_fairseq_checkpoint
    fair_sd, _ = load_fairseq_checkpoint(CKPT)
    ref = load_reference()
    hf_sd = ref.state_dict()
    mapping = hf_to_fairseq_mapping()
    worst = 0.0
    n_cmp = 0
    for hf_k, fair_k in mapping.items():
        a = hf_sd[hf_k].float()
        b = fair_sd[fair_k].float()
        assert a.shape == b.shape, (hf_k, a.shape, b.shape)
        d = float((a - b).abs().max())
        worst = max(worst, d)
        n_cmp += 1
    # fairseq-only key not present in HF (training-time nce embeddings)
    leftover = set(fair_sd) - set(mapping.values())
    assert leftover == {"label_embs_concat"}, leftover
    # effective (fused) pos_conv weight: HF parametrization vs torch._weight_norm
    fused_hf = ref.encoder.pos_conv_embed.conv.weight.detach()
    fused_fair = torch._weight_norm(fair_sd["encoder.pos_conv.0.weight_v"].float(),
                                    fair_sd["encoder.pos_conv.0.weight_g"].float(), 2)
    d_fused = float((fused_hf - fused_fair).abs().max())
    print(f"  {n_cmp} mapped tensors: max_abs_diff = {worst:.3e}")
    print(f"  fused pos_conv weight: max_abs_diff = {d_fused:.3e}")
    if worst != 0.0 or d_fused != 0.0:
        failures.append(f"gate0 weight identity: {worst:.3e} / fused {d_fused:.3e}")

    # ---------------- our model ----------------
    model = build_model(CKPT)  # strict=True inside

    # ---------------- audio ----------------
    wavs = [load_audio_slice(off, n) for off, n in SLICES]
    for (off, n), w in zip(SLICES, wavs):
        print(f"audio slice @{off}s N={n} rms={np.sqrt(np.mean(w**2)):.4f} "
              f"-> expect T={expected_frames(n)}")

    # ---------------- gate a + a2 + bisect (slice 0) ----------------
    print("== gate a: our torch port vs HF reference (fp32, real audio) ==")
    wav_t = torch.from_numpy(wavs[0]).unsqueeze(0)
    f768_ref, f256_ref = ref_features(ref, wav_t)
    with torch.no_grad():
        f768_ours = model.extract(wav_t, output_layer=12)
        f256_ours = model.final_proj(model.extract(wav_t, output_layer=9))
    a768 = float((f768_ours - f768_ref).abs().max())
    a256 = float((f256_ours - f256_ref).abs().max())
    print(f"  vec768l12: max_abs_diff = {a768:.3e}  (threshold 1e-5)")
    print(f"  vec256l9 : max_abs_diff = {a256:.3e}  (threshold 1e-5)")
    if a768 >= 1e-5 or a256 >= 1e-5:
        failures.append(f"gate a: 768={a768:.3e} 256={a256:.3e}")

    # bisect aid: per-layer drift (must grow smoothly = fp accumulation only)
    outs = our_layerwise(model, wav_t)
    ref_all = ref(wav_t, output_hidden_states=True).hidden_states
    per_layer = [float((outs[i] - ref_all[i + 1]).abs().max()) for i in range(12)]
    print("  per-layer max_abs: " + " ".join(f"{d:.1e}" for d in per_layer))

    print("== gate a2: f64 structural comparison ==")
    # both sides must recompute the pos_conv weight-norm in f64 (the exported /
    # gate-a model fuses it as an fp32 constant — that fp32 rounding would
    # otherwise masquerade as ~1e-5 "structural" drift here)
    import copy
    model64 = build_model(CKPT, fuse_pos_conv=False).double()
    ref64 = copy.deepcopy(ref).double()
    wav64 = wav_t.double()
    with torch.no_grad():
        o768 = model64.extract(wav64, output_layer=12)
        o256 = model64.final_proj(model64.extract(wav64, output_layer=9))
        r = ref64(wav64, output_hidden_states=True)
        r768 = r.hidden_states[12]
        r256 = ref64.final_proj(r.hidden_states[9])
    s768 = float((o768 - r768).abs().max())
    s256 = float((o256 - r256).abs().max())
    print(f"  vec768l12 f64: max_abs_diff = {s768:.3e}  (threshold 1e-9)")
    print(f"  vec256l9  f64: max_abs_diff = {s256:.3e}  (threshold 1e-9)")
    if s768 >= 1e-9 or s256 >= 1e-9:
        failures.append(f"gate a2: 768={s768:.3e} 256={s256:.3e}")
    del model64, ref64

    # ---------------- gates b + c: onnx vs reference, both lengths ----------------
    print("== gates b/c: onnx (ORT CPU) vs HF reference, dynamic shapes ==")
    import onnxruntime as ort
    sess768 = ort.InferenceSession(ONNX_768, providers=["CPUExecutionProvider"])
    sess256 = ort.InferenceSession(ONNX_256, providers=["CPUExecutionProvider"])
    for (off, n), w in zip(SLICES, wavs):
        wt = torch.from_numpy(w).unsqueeze(0)
        r768, r256 = ref_features(ref, wt)
        r768 = r768.squeeze(0).numpy()
        r256 = r256.squeeze(0).numpy()
        o768 = sess768.run(None, {"waveform": wt.numpy()})[0]
        o256 = sess256.run(None, {"waveform": wt.numpy()})[0]
        t_want = expected_frames(n)
        for name, o, rr, dim in (("vec768l12", o768, r768, 768),
                                 ("vec256l9", o256, r256, 256)):
            ok_shape = o.shape == (1, t_want, dim)
            if not ok_shape:
                failures.append(f"gate c: {name} N={n} shape {o.shape} != (1,{t_want},{dim})")
            of = o.squeeze(0)
            mad = float(np.abs(of - rr).max())
            cmin = per_frame_cos_min(of, rr)
            print(f"  [{name}] N={n} T={o.shape[1]} (want {t_want}) "
                  f"max_abs={mad:.3e} (<1e-4) min_frame_cos={cmin:.7f} (>0.9999)")
            if mad >= 1e-4 or cmin <= 0.9999:
                failures.append(f"gate b: {name} N={n} max_abs={mad:.3e} cos={cmin:.7f}")

    print("=" * 60)
    if failures:
        print("FAILED:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("ALL GATES PASSED")


if __name__ == "__main__":
    main()
