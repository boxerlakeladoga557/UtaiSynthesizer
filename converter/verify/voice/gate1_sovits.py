"""Gate 1 — SoVITS synthesizer equivalence: ORIGINAL (so-vits-svc 4.1-Stable)
vs our port (architectures/sovits_v4.py).

a.  4.0 REAL weights (akiko_320000.pth, vec256l9/gin256): original
    SynthesizerTrn built from config.json hparams, strict=True both sides,
    infer() vs our forward with all randomness zeroed AND with a fixed z_p
    noise tensor. PASS: audio max_abs_diff < 1e-5. Per-module diffs
    (m_p/logs_p/z_p/z/audio) always printed so a failure bisects itself.
a2. random-weight transplant with flow_share_parameter=True + vol_embedding
    (structure not covered by the real checkpoints): original state_dict() ->
    ours strict=True, same comparison (flow post convs re-randomized so the
    coupling math is exercised, not identity).
b.  4.1 REAL weights (东雪莲, vec768l12/vol_embedding/gin768, compressed — no
    enc_q, Chinese paths): same comparison, vol from the ORIGINAL
    utils.Volume_Extractor on synthetic audio.
b2. refusals: use_depthwise_conv / use_transformer_flow / exotic
    speech_encoder / exotic vocoder_name -> Chinese ValueError; config/weights
    mismatches (ssl_dim, hop_length) -> "不匹配" ValueError; and the
    missing-config fallback must build akiko bit-identically to the
    config-driven build.
c.  torch vs ONNX parity: deterministic export via convert.convert_sovits for
    BOTH versions, ORT vs our torch at the export T (200, < 1e-4) and a
    dynamic-T sweep down to min_frames=6 (< 5e-4, same rationale as the RVC
    sweep: ORT/torch fp32 conv rounding through the 30-conv dec stack).
d.  real shipping exports via the convert.py CLI (deterministic OFF, noise
    live in-graph) to converter/test_output/ — akiko_320000.onnx +
    Sovits4.1东雪莲主模型.onnx (Chinese filename survives end-to-end) +
    sidecar json schema checks + ORT sanity at two T + live-noise proof
    (two identical runs must differ).
e.  cluster / feature-retrieval asset round-trip via export_cluster.py CLI:
    synthetic kmeans .pt (Chinese speaker key) and faiss "IVF,Flat" .pkl
    (utils.train_index recipe) -> .npy == original arrays EXACTLY.

Run:  converter/.venv/Scripts/python.exe converter/verify/voice/gate1_sovits.py
"""

import copy
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Windows console default codepage (cp932/936) chokes on the report glyphs.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SOVITS_REPO = r"D:\MyDev\so-vits-svc\so-vits-svc"
CONVERTER = r"D:\MyDev\Utai_v2-dev\converter"
V40_PTH = r"D:\MyDev\TESTING\Sovits-SVC\MinamiyaAkiko-Sovits4.0\akiko_320000.pth"
V41_PTH = r"D:\MyDev\TESTING\Sovits-SVC\东雪莲\Sovits4.1东雪莲主模型.pth"
SCRATCH = (r"C:\Users\admin\AppData\Local\Temp\claude\D--MyDev-Utai-v2-dev"
           r"\d475bbb0-dad6-40d9-97c1-631befaef2b0\scratchpad")
TEST_OUTPUT = os.path.join(CONVERTER, "test_output")

TOL_TORCH = 1e-5   # torch-vs-torch (gates a, a2, b)
TOL_ONNX = 1e-4    # torch-vs-ORT fp32 at the export T (gate c, mandated)
# Dynamic-T sweep tolerance — same rationale as gate1_rvc.py: ORT's fp32
# conv/matmul rounding differs from torch's and the dec stack can amplify an
# unlucky input a few x; -66 dB on full-scale audio.
TOL_ONNX_SWEEP = 5e-4
EXPORT_T = 200
MIN_FRAMES = 6     # window_size(4) + 2; must match the sidecar min_frames
NOISE_SCALE = 0.4  # original inference default (infer_tool.py noice_scale)

sys.path.insert(0, SOVITS_REPO)
sys.path.insert(0, CONVERTER)

import models as orig_models          # noqa: E402 — so-vits models.py
import utils as so_utils              # noqa: E402 — so-vits utils.py
from architectures import sovits_v4   # noqa: E402

os.makedirs(SCRATCH, exist_ok=True)
torch.set_grad_enabled(False)

failures = []


def check(cond, label):
    tag = "PASS" if cond else "FAIL"
    print(f"    [{tag}] {label}")
    if not cond:
        failures.append(label)


def mad(a, b):
    return (a - b).abs().max().item()


# --- noise patching for the ORIGINAL model ---------------------------------
# original infer(): z_p noise = torch.randn_like(m_p) * noice_scale (enc_p)
# original SineGen (non-onnx path): rand_ini = torch.rand(...),
#   noise = amp * randn_like(sines); SourceModuleHnNSF: randn_like(uv) (unused)
# Ours runs with deterministic=True (SineGen randomness zeroed internally) and
# the z_p noise fed explicitly, so patches only wrap ORIGINAL forwards.

class PatchedNoise:
    """torch.rand -> zeros; torch.randn_like -> `fixed` for tensors matching
    fixed.shape (the z_p noise), zeros otherwise (the SineGen noise)."""

    def __init__(self, fixed=None):
        self.fixed = fixed

    def __enter__(self):
        self._rand, self._randn_like = torch.rand, torch.randn_like

        def zrand(*size, **kw):
            return torch.zeros(*size, device=kw.get("device"),
                               dtype=kw.get("dtype"))

        def zrandn_like(t, **kw):
            if self.fixed is not None and tuple(t.shape) == tuple(self.fixed.shape):
                return self.fixed
            return torch.zeros_like(t)

        torch.rand, torch.randn_like = zrand, zrandn_like
        return self

    def __exit__(self, *exc):
        torch.rand, torch.randn_like = self._rand, self._randn_like


def make_inputs(t, ssl_dim, seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(1, t, ssl_dim, generator=g)
    f0 = 150.0 + 250.0 * torch.rand(1, t, generator=g)
    if t >= 20:  # unvoiced stretch to exercise the uv path
        u0 = t // 5
        f0[:, u0:u0 + 10] = 0.0
    uv = (f0 > 0).float()
    sid = torch.zeros(1, dtype=torch.int64)
    return c, f0, uv, sid


def ours_intermediates(ours, c, f0, uv, noise, sid, vol=None):
    """Replicates sovits_v4.SynthesizerTrn.forward, returning every stage."""
    g = ours.emb_g(sid.unsqueeze(0)).transpose(1, 2)
    x_mask = torch.unsqueeze(torch.ones_like(f0), 1).to(c.dtype)
    volt = (ours.emb_vol(vol[:, :, None]).transpose(1, 2)
            if vol is not None and ours.vol_embedding else 0)
    x = (ours.pre(c.transpose(1, 2)) * x_mask
         + ours.emb_uv(uv.long()).transpose(1, 2) + volt)
    z_p, m_p, logs_p, c_mask = ours.enc_p(
        x, x_mask, f0=sovits_v4.f0_to_coarse(f0), z=noise)
    z = ours.flow(z_p, c_mask, g=g, reverse=True)
    o = ours.dec(z * c_mask, g=g, f0=f0)
    return {"m_p": m_p, "logs_p": logs_p, "z_p": z_p, "z": z, "audio": o}


def orig_intermediates(orig, c_cf, f0, uv, sid, noice_scale, vol=None):
    """Re-runs the ORIGINAL submodules along models.py infer()'s exact steps
    (diagnosis only — the gate compares orig.infer()'s audio)."""
    g = orig.emb_g(sid.unsqueeze(0)).transpose(1, 2)
    x_mask = torch.unsqueeze(torch.ones_like(f0), 1).to(c_cf.dtype)
    volt = (orig.emb_vol(vol[:, :, None]).transpose(1, 2)
            if vol is not None and orig.vol_embedding else 0)
    x = (orig.pre(c_cf) * x_mask
         + orig.emb_uv(uv.long()).transpose(1, 2) + volt)
    z_p, m_p, logs_p, c_mask = orig.enc_p(
        x, x_mask, f0=so_utils.f0_to_coarse(f0), noice_scale=noice_scale)
    z = orig.flow(z_p, c_mask, g=g, reverse=True)
    o = orig.dec(z * c_mask, g=g, f0=f0)
    return {"m_p": m_p, "logs_p": logs_p, "z_p": z_p, "z": z, "audio": o}


def compare_full(tag, orig, ours, ssl_dim, inter, t=EXPORT_T, seed=1234,
                 vol=None):
    """Zero-noise + fixed-noise full-forward comparison with per-module diffs.
    The GATE is orig.infer() audio vs ours forward audio."""
    c, f0, uv, sid = make_inputs(t, ssl_dim, seed)
    c_cf = c.transpose(1, 2)  # original infer takes channels-first c

    for mode in ("zero-noise", "fixed-noise"):
        if mode == "zero-noise":
            fixed = None
            noise = torch.zeros(1, inter, t)
        else:
            g = torch.Generator().manual_seed(seed + 1)
            fixed = torch.randn(1, inter, t, generator=g)
            noise = fixed * NOISE_SCALE

        with PatchedNoise(fixed):
            o_audio, _ = orig.infer(c_cf, f0, uv, g=sid,
                                    noice_scale=NOISE_SCALE,
                                    predict_f0=False, vol=vol)
        with PatchedNoise(fixed):
            o_st = orig_intermediates(orig, c_cf, f0, uv, sid, NOISE_SCALE,
                                      vol=vol)

        m = ours_intermediates(ours, c, f0, uv, noise, sid, vol=vol)

        diffs = {
            "m_p": mad(o_st["m_p"], m["m_p"]),
            "logs_p": mad(o_st["logs_p"], m["logs_p"]),
            "z_p": mad(o_st["z_p"], m["z_p"]),
            "z (flow.reverse)": mad(o_st["z"], m["z"]),
            "audio (dec)": mad(o_st["audio"], m["audio"]),
        }
        for name, d in diffs.items():
            print(f"      {mode:>11} {name:<16} max_abs_diff = {d:.3e}")
        d_infer = mad(o_audio, m["audio"])
        check(d_infer < TOL_TORCH,
              f"{tag} {mode}: orig.infer() audio max_abs_diff "
              f"{d_infer:.3e} < {TOL_TORCH:.0e}")


def build_orig(cfg, strict_state, allow_missing_enc_q=False):
    orig = orig_models.SynthesizerTrn(
        cfg["data"]["filter_length"] // 2 + 1,
        cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
        **cfg["model"])
    if allow_missing_enc_q:
        missing, unexpected = orig.load_state_dict(strict_state, strict=False)
        check(not unexpected, f"orig load: no unexpected keys ({len(unexpected)})")
        check(all(k.startswith("enc_q.") for k in missing),
              f"orig load: missing keys are enc_q-only ({len(missing)})")
    else:
        orig.load_state_dict(strict_state, strict=True)
        print("    orig strict=True load OK")
    orig.eval()
    return orig


# ===========================================================================
print("=== gate (a) — 4.0 real weights: akiko_320000.pth ===")
ck40 = torch.load(V40_PTH, map_location="cpu", weights_only=False)
cfg40, cfg40_path = sovits_v4.load_sovits_config(V40_PTH)
print(f"  config: {cfg40_path}")
state40 = {k: v.float() for k, v in ck40["model"].items()}

orig40 = build_orig(cfg40, state40)
ours40, meta40 = sovits_v4.build_from_checkpoint(ck40, cfg40)
sovits_v4.set_deterministic(ours40, True)
check(meta40["version"] == "4.0" and meta40["features_dim"] == 256
      and meta40["speech_encoder"] == "vec256l9"
      and not meta40["vol_embedding"],
      f"meta: 4.0 / 256 / vec256l9 / no vol (got {meta40['version']}, "
      f"{meta40['features_dim']}, {meta40['speech_encoder']})")
compare_full("4.0", orig40, ours40, ssl_dim=256, inter=meta40["inter_channels"])

# ===========================================================================
print("=== gate (a2) — random transplant: flow_share_parameter + vol ===")
TRANS_CFG = {
    "train": {"segment_size": 5120},
    "data": {"sampling_rate": 16000, "hop_length": 16, "filter_length": 254,
             "unit_interpolate_mode": "nearest"},
    "model": {"inter_channels": 64, "hidden_channels": 64,
              "filter_channels": 128, "n_heads": 2, "n_layers": 2,
              "kernel_size": 3, "p_dropout": 0.1, "resblock": "1",
              "resblock_kernel_sizes": [3, 7],
              "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
              "upsample_rates": [4, 4], "upsample_initial_channel": 64,
              "upsample_kernel_sizes": [8, 8], "gin_channels": 32,
              "ssl_dim": 256, "n_speakers": 3, "sampling_rate": 16000,
              "vol_embedding": True, "flow_share_parameter": True,
              "use_automatic_f0_prediction": True,
              "speech_encoder": "vec256l9"},
    "spk": {"a": 0, "b": 1, "c": 2},
}
torch.manual_seed(20260704)
orig_tr = orig_models.SynthesizerTrn(
    TRANS_CFG["data"]["filter_length"] // 2 + 1,
    TRANS_CFG["train"]["segment_size"] // TRANS_CFG["data"]["hop_length"],
    **TRANS_CFG["model"])
# coupling `post` convs are zero-init (coupling == identity) — randomize them
# so a wrong shared-WN wiring cannot hide.
for i in range(0, len(orig_tr.flow.flows), 2):
    orig_tr.flow.flows[i].post.weight.data.normal_(0, 0.05)
    orig_tr.flow.flows[i].post.bias.data.normal_(0, 0.05)
orig_tr.eval()

ck_tr = {"model": orig_tr.state_dict()}
ours_tr, meta_tr = sovits_v4.build_from_checkpoint(ck_tr, TRANS_CFG)
print("    ours strict=True transplant load OK "
      f"({len(ck_tr['model'])} tensors, shared flow WN + emb_vol)")
sovits_v4.set_deterministic(ours_tr, True)
check(meta_tr["hop_size"] == 16 and meta_tr["vol_embedding"],
      f"meta: hop 16, vol_embedding (got {meta_tr['hop_size']}, "
      f"{meta_tr['vol_embedding']})")
vol_tr = 0.05 + 0.1 * torch.rand(1, 60)
compare_full("transplant", orig_tr, ours_tr, ssl_dim=256, inter=64, t=60,
             seed=5678, vol=vol_tr)

# ===========================================================================
print("=== gate (b) — 4.1 real weights: 东雪莲 (vol_embedding, no enc_q) ===")
ck41 = torch.load(V41_PTH, map_location="cpu", weights_only=False)
cfg41, cfg41_path = sovits_v4.load_sovits_config(V41_PTH)
print(f"  config: {cfg41_path}")
state41 = {k: v.float() for k, v in ck41["model"].items()}

orig41 = build_orig(cfg41, state41, allow_missing_enc_q=True)
ours41, meta41 = sovits_v4.build_from_checkpoint(ck41, cfg41)
sovits_v4.set_deterministic(ours41, True)
check(meta41["version"] == "4.1" and meta41["features_dim"] == 768
      and meta41["speech_encoder"] == "vec768l12" and meta41["vol_embedding"]
      and meta41["unit_interpolate_mode"] == "nearest"
      and meta41["speakers"] == {"AzumaVocal": 0},
      f"meta: 4.1 / 768 / vec768l12 / vol / nearest (got {meta41})")
# real vol vector from the ORIGINAL Volume_Extractor on synthetic audio
gen = torch.Generator().manual_seed(41)
audio_syn = 0.2 * torch.randn(1, EXPORT_T * 512, generator=gen)
vol41 = so_utils.Volume_Extractor(512).extract(audio_syn)[None, :]
check(vol41.shape == (1, EXPORT_T) and (vol41 >= 0).all().item(),
      f"Volume_Extractor vol: shape {tuple(vol41.shape)}, non-negative")
compare_full("4.1", orig41, ours41, ssl_dim=768,
             inter=meta41["inter_channels"], vol=vol41)

# ===========================================================================
print("=== gate (b2) — refusals + missing-config fallback ===")


def expect_refusal(cfg_mutator, expect_substr, label, ckpt=ck40):
    cfg = copy.deepcopy(cfg40)
    cfg_mutator(cfg)
    try:
        sovits_v4.build_from_checkpoint(ckpt, cfg)
        check(False, f"{label}: refused")
    except ValueError as e:
        check(expect_substr in str(e), f"{label}: {e}")


expect_refusal(lambda c: c["model"].update(use_depthwise_conv=True),
               "暂不支持 use_depthwise_conv", "use_depthwise_conv=true")
expect_refusal(lambda c: c["model"].update(use_transformer_flow=True),
               "暂不支持 use_transformer_flow", "use_transformer_flow=true")
expect_refusal(lambda c: c["model"].update(speech_encoder="whisper-ppg"),
               "暂不支持 speech_encoder=whisper-ppg", "exotic speech_encoder")
expect_refusal(lambda c: c["model"].update(vocoder_name="nsf-snake-hifigan"),
               "暂不支持 vocoder_name=nsf-snake-hifigan", "exotic vocoder_name")
expect_refusal(lambda c: c["model"].update(speech_encoder="vec768l12"),
               "配置文件与模型不匹配", "speech_encoder/ssl_dim mismatch")
expect_refusal(lambda c: c["data"].update(hop_length=480),
               "配置文件与模型不匹配", "hop_length/upsample mismatch")

ours40_nc, meta40_nc = sovits_v4.build_from_checkpoint(ck40, None)
sovits_v4.set_deterministic(ours40_nc, True)
check(meta40_nc["version"] == "4.0" and meta40_nc["features_dim"] == 256
      and meta40_nc["hop_size"] == 512 and meta40_nc["speakers"] == {},
      f"no-config fallback meta: 4.0/256/hop512 (got {meta40_nc})")
c, f0, uv, sid = make_inputs(64, 256, 99)
noise = torch.randn(1, meta40["inter_channels"], 64,
                    generator=torch.Generator().manual_seed(100)) * NOISE_SCALE
d_nc = mad(ours40(c, f0, uv, noise, sid), ours40_nc(c, f0, uv, noise, sid))
check(d_nc == 0.0, f"no-config fallback == config build bit-exact ({d_nc:.1e})")

# ===========================================================================
print("=== gate (c) — torch vs ONNX parity (deterministic exports) ===")
import convert as convert_mod  # noqa: E402
import onnxruntime as ort      # noqa: E402


def make_session(path):
    """ORT session from BYTES: the python onnxruntime build cannot open
    session paths with Chinese characters on Windows (locale ACP issue) —
    exercised here on purpose because SoVITS filenames routinely are Chinese.
    (Rust's ort crate uses wide-string paths and is unaffected.)"""
    return ort.InferenceSession(Path(path).read_bytes(),
                                providers=["CPUExecutionProvider"])


def ort_vs_torch(sess, ours, t, ssl_dim, inter, seed, zero_noise=False,
                 with_vol=False):
    c, f0, uv, sid = make_inputs(t, ssl_dim, seed)
    if zero_noise:
        noise = torch.zeros(1, inter, t)
    else:
        g = torch.Generator().manual_seed(seed + 9)
        noise = torch.randn(1, inter, t, generator=g) * NOISE_SCALE
    vol = (0.05 + 0.1 * torch.rand(1, t, generator=torch.Generator()
                                   .manual_seed(seed + 7))) if with_vol else None
    t_out = (ours(c, f0, uv, noise, sid, vol) if with_vol
             else ours(c, f0, uv, noise, sid))
    feeds = {"c": c.numpy(), "f0": f0.numpy(), "uv": uv.numpy(),
             "noise": noise.numpy(), "sid": sid.numpy()}
    if with_vol:
        feeds["vol"] = vol.numpy()
    o_out = sess.run(None, feeds)[0]
    return mad(t_out, torch.from_numpy(o_out))


for tag, pth, ours, meta in (("4.0 akiko", V40_PTH, ours40, meta40),
                             ("4.1 东雪莲", V41_PTH, ours41, meta41)):
    det_onnx = Path(SCRATCH) / (Path(pth).stem + ".det.onnx")
    convert_mod.convert_sovits(Path(pth), det_onnx, deterministic=True)
    sess = make_session(det_onnx)
    onnx_inputs = [i.name for i in sess.get_inputs()]
    with_vol = meta["vol_embedding"]
    check(("vol" in onnx_inputs) == with_vol,
          f"{tag}: vol input present iff vol_embedding ({onnx_inputs})")

    dz = ort_vs_torch(sess, ours, EXPORT_T, meta["features_dim"],
                      meta["inter_channels"], 42, zero_noise=True,
                      with_vol=with_vol)
    check(dz < TOL_ONNX,
          f"{tag} T={EXPORT_T} noise=0: max_abs_diff {dz:.3e} < {TOL_ONNX:.0e}")
    df = ort_vs_torch(sess, ours, EXPORT_T, meta["features_dim"],
                      meta["inter_channels"], 42, with_vol=with_vol)
    check(df < TOL_ONNX,
          f"{tag} T={EXPORT_T} fixed noise: max_abs_diff {df:.3e} < {TOL_ONNX:.0e}")

    print(f"    {tag} dynamic-T sweep (graph exported at T={EXPORT_T}):")
    for t in (311, 137, 57, 20, 10, MIN_FRAMES):
        d = ort_vs_torch(sess, ours, t, meta["features_dim"],
                         meta["inter_channels"], 100 + t, with_vol=with_vol)
        check(d < TOL_ONNX_SWEEP,
              f"{tag} T={t}: max_abs_diff {d:.3e} < {TOL_ONNX_SWEEP:.0e}")

# ===========================================================================
print("=== gate (d) — shipping exports via convert.py CLI ===")
os.makedirs(TEST_OUTPUT, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

EXPECT_COMMON = {"type": "sovits", "sample_rate": 44100, "hop_size": 512,
                 "min_frames": MIN_FRAMES}
SHIP = [
    (V40_PTH, "akiko_320000.onnx",
     dict(EXPECT_COMMON, version="4.0", features_dim=256,
          speech_encoder="vec256l9", vol_embedding=False,
          unit_interpolate_mode="left", n_speakers=200,
          speakers={"akiko4.0": 0},
          inputs=["c", "f0", "uv", "noise", "sid"])),
    (V41_PTH, "Sovits4.1东雪莲主模型.onnx",
     dict(EXPECT_COMMON, version="4.1", features_dim=768,
          speech_encoder="vec768l12", vol_embedding=True,
          unit_interpolate_mode="nearest", n_speakers=1,
          speakers={"AzumaVocal": 0},
          inputs=["c", "f0", "uv", "noise", "sid", "vol"])),
]
for pth, out_name, expect in SHIP:
    ship_onnx = os.path.join(TEST_OUTPUT, out_name)
    proc = subprocess.run(
        [sys.executable, "convert.py", "--input", pth,
         "--output", ship_onnx, "--type", "sovits"],
        cwd=CONVERTER, capture_output=True, env=env)
    stdout_txt = proc.stdout.decode("utf-8", errors="replace")
    print("    " + "\n    ".join(stdout_txt.strip().splitlines()[-3:]))
    check(proc.returncode == 0, f"convert.py CLI exit code {proc.returncode} "
                                f"({out_name})")
    if proc.returncode != 0:
        print(proc.stderr.decode("utf-8", errors="replace"))
        continue

    sidecar = json.loads(Path(ship_onnx).with_suffix(".json")
                         .read_text(encoding="utf-8"))
    for k, v in expect.items():
        check(sidecar.get(k) == v, f"{out_name} sidecar {k} == {v!r} "
                                   f"(got {sidecar.get(k)!r})")
    check(sidecar.get("noise") == {"noise_input": [1, 192, "T"],
                                   "default_scale": NOISE_SCALE},
          f"{out_name} sidecar noise block (got {sidecar.get('noise')})")

    sess = make_session(ship_onnx)
    for t in (EXPORT_T, 333):
        c, f0, uv, sid = make_inputs(t, expect["features_dim"], 7000 + t)
        feeds = {"c": c.numpy(), "f0": f0.numpy(), "uv": uv.numpy(),
                 "noise": (torch.randn(1, 192, t) * NOISE_SCALE).numpy(),
                 "sid": sid.numpy()}
        if expect["vol_embedding"]:
            feeds["vol"] = (0.05 + 0.1 * torch.rand(1, t)).numpy()
        audio = sess.run(None, feeds)[0]
        ok = (audio.shape == (1, 1, t * 512) and np.isfinite(audio).all()
              and np.abs(audio).max() <= 1.0)
        check(ok, f"{out_name} T={t}: shape {audio.shape} == (1,1,{t * 512}), "
                  f"finite, |max| {np.abs(audio).max():.3f} <= 1")
    # the shipping graph must keep SineGen noise LIVE (RandomNormalLike):
    # two identical runs must differ.
    a1 = sess.run(None, feeds)[0]
    a2 = sess.run(None, feeds)[0]
    check(np.abs(a1 - a2).max() > 0,
          f"{out_name}: in-graph noise is live (two runs differ by "
          f"{np.abs(a1 - a2).max():.1e})")

# ===========================================================================
print("=== gate (e) — cluster / feature-retrieval asset round-trip ===")
import faiss  # noqa: E402

cluster_pt = os.path.join(SCRATCH, "kmeans_synth.pt")
rng = np.random.default_rng(20260704)
centers_a = rng.standard_normal((64, 768)).astype(np.float32)
centers_b = rng.standard_normal((32, 256)).astype(np.float32)
torch.save({"东雪莲": {"n_features_in_": 768, "_n_threads": 4,
                       "cluster_centers_": centers_a},
            "akiko4.0": {"n_features_in_": 256, "_n_threads": 4,
                         "cluster_centers_": centers_b}}, cluster_pt)
cluster_out = os.path.join(SCRATCH, "cluster_out")
proc = subprocess.run(
    [sys.executable, "export_cluster.py", "--input", cluster_pt,
     "--outdir", cluster_out],
    cwd=CONVERTER, capture_output=True, env=env)
check(proc.returncode == 0, f"export_cluster.py (cluster) exit {proc.returncode}")
if proc.returncode != 0:
    print(proc.stderr.decode("utf-8", errors="replace"))
else:
    got_a = np.load(os.path.join(cluster_out, "东雪莲.centers.npy"))
    got_b = np.load(os.path.join(cluster_out, "akiko4.0.centers.npy"))
    check(np.array_equal(got_a, centers_a) and got_a.dtype == np.float32,
          f"cluster centers round-trip EXACT (东雪莲 {got_a.shape})")
    check(np.array_equal(got_b, centers_b),
          f"cluster centers round-trip EXACT (akiko4.0 {got_b.shape})")

# faiss retrieval — the utils.train_index recipe ("IVF{n},Flat")
big = rng.standard_normal((500, 768)).astype(np.float32)
n_ivf = min(int(16 * np.sqrt(big.shape[0])), big.shape[0] // 39)
index = faiss.index_factory(big.shape[1], "IVF%s,Flat" % n_ivf)
faiss.extract_index_ivf(index).nprobe = 1
index.train(big)
index.add(big)
retrieval_pkl = os.path.join(SCRATCH, "feature_and_index_synth.pkl")
with open(retrieval_pkl, "wb") as f:
    pickle.dump({0: index}, f)
retrieval_out = os.path.join(SCRATCH, "retrieval_out")
proc = subprocess.run(
    [sys.executable, "export_cluster.py", "--input", retrieval_pkl,
     "--outdir", retrieval_out],
    cwd=CONVERTER, capture_output=True, env=env)
check(proc.returncode == 0, f"export_cluster.py (retrieval) exit {proc.returncode}")
if proc.returncode != 0:
    print(proc.stderr.decode("utf-8", errors="replace"))
else:
    got = np.load(os.path.join(retrieval_out, "0.index_vectors.npy"))
    check(np.array_equal(got, big) and got.dtype == np.float32,
          f"retrieval vectors round-trip EXACT (IVF{n_ivf},Flat, {got.shape})")

# ===========================================================================
print()
if failures:
    print(f"GATE FAILED — {len(failures)} check(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL SOVITS GATES PASSED")
