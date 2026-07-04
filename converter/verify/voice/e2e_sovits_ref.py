"""关卡 2 — SoVITS 4.x E2E python reference (ORIGINAL single-segment semantics vs Rust).

Reproduces the ORIGINAL so-vits-svc single-segment path (infer_tool.slice_inference around
ONE segment + Svc.infer/get_unit_f0), transcribed faithfully, driving:
  * the ORIGINAL torch synthesizer models.SynthesizerTrn (real .pth weights, config.json
    hparams — strict for 4.0, missing⊆enc_q for the compressed 4.1); torch.randn/​randn_like
    monkeypatched to zeros (kills z_p noise + SineGen noise) == our Rust det build + noise_scale=0;
  * the ORIGINAL so-vits utils.repeat_expand_2d + utils.Volume_Extractor;
  * INPUT pre-resampled to the model sr (44100) MONO — kills the native→target input resampler.

TWO resampler variants for the pipeline-internal 44100→16000 resamples (isolate the
S34 REFPOLY residue):
  A (original): torchaudio Resample — f0 path width=128 (so-vits rmvpe), hubert path default
                width=6 (Svc.get_unit_f0). TWO different 16k signals, exactly like the original.
  B (Rust):     scipy.signal.resample_poly — ONE wav16k fed to BOTH f0 and hubert, exactly
                like src\inference\sovits.rs. Rust-vs-B should be markedly higher than Rust-vs-A.

DOCUMENTED substitutions (each certified against the true ORIGINAL elsewhere):
  * ContentVec: fairseq unimportable on Windows — inject contentvec_{768l12,256l9}.onnx
    (gate_contentvec.py: min cos>0.9999 vs fairseq). Rust uses the SAME onnx.
  * f0 backbone: inject rmvpe_e2e.onnx behind a python log-mel front-end mirroring f0.rs;
    the ORIGINAL RMVPEF0Predictor.post_process (resize/uv/gap-interp) is transcribed verbatim.
    gate_rmvpe.py certifies the onnx vs the original rmvpe.pt.

Run: converter\.venv\Scripts\python.exe converter\verify\voice\e2e_sovits_ref.py \
        --input <44k.wav> --pth <...\akiko_320000.pth> --out-a <a.wav> --out-b <b.wav>
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SOVITS_REPO = r"D:\MyDev\so-vits-svc\so-vits-svc"
CONVERTER = r"D:\MyDev\Utai_v2-dev\converter"
REPO_ROOT = r"D:\MyDev\Utai_v2-dev"

AUX = os.path.join(REPO_ROOT, "data", "models", "aux")
RMVPE_ONNX = os.path.join(AUX, "rmvpe_e2e.onnx")
MEL_FILTERS = os.path.join(AUX, "rmvpe_mel_filters.npy")

sys.path.insert(0, SOVITS_REPO)
sys.path.insert(0, CONVERTER)

import models as orig_models          # noqa: E402  so-vits models.py (torch only)
import utils as so_utils              # noqa: E402  so-vits utils.py
from architectures import sovits_v4   # noqa: E402  converter port (config loader + meta)

torch.set_grad_enabled(False)
PAD_SECONDS = 0.5
NOISE_SCALE = 0.4
RMVPE_THRED = 0.05                     # so-vits infer() cr_threshold default


class ZeroNoise:
    def __enter__(self):
        self._r, self._rl, self._rn = torch.rand, torch.randn_like, torch.randn

        def zrand(*size, **kw):
            return torch.zeros(*size, device=kw.get("device"), dtype=kw.get("dtype"))

        torch.rand = zrand
        torch.randn_like = lambda t, **kw: torch.zeros_like(t)
        torch.randn = zrand
        return self

    def __exit__(self, *exc):
        torch.rand, torch.randn_like, torch.randn = self._r, self._rl, self._rn


# ── our onnx front-ends (contentvec + rmvpe, ORT CPU) ──
_CV_CACHE = {}


def contentvec_session(dim):
    if dim not in _CV_CACHE:
        name = "contentvec_768l12.onnx" if dim == 768 else "contentvec_256l9.onnx"
        _CV_CACHE[dim] = ort.InferenceSession(os.path.join(AUX, name),
                                              providers=["CPUExecutionProvider"])
    return _CV_CACHE[dim]


class RmvpeFront:
    def __init__(self):
        self.sess = ort.InferenceSession(RMVPE_ONNX, providers=["CPUExecutionProvider"])
        self.mel = np.load(MEL_FILTERS).astype(np.float32)
        k = np.arange(1024)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * k / 1024.0)).astype(np.float32)

    def f0_100fps(self, wav16k, thred):
        x = np.asarray(wav16k, dtype=np.float32)
        n = len(x)
        if n < 513:
            x = np.pad(x, (0, 513 - n)); n = 513
        t_frames = 1 + n // 160
        padded = np.pad(x, (512, 512), mode="reflect")
        mag = np.empty((513, t_frames), dtype=np.float32)
        for t in range(t_frames):
            mag[:, t] = np.abs(np.fft.rfft(padded[t*160:t*160+1024] * self.window, n=1024))
        logmel = np.log(np.clip(self.mel @ mag, 1e-5, None)).astype(np.float32)
        f0 = self.sess.run(None, {"mel": logmel[None],
                                  "threshold": np.array([thred], np.float32)})[0]
        return np.asarray(f0, dtype=np.float32).reshape(-1)


def torch_interp_nearest(src, dst_len):
    """torch F.interpolate 1-D 'nearest' (RMVPEF0Predictor.repeat_expand)."""
    src = np.asarray(src, dtype=np.float64)
    n = len(src)
    idx = np.minimum((np.arange(dst_len) * (n / dst_len)).astype(np.int64), n - 1)
    return src[idx]


def sovits_post_process(f0_100, pad_to, hop, sr):
    """Verbatim RMVPEF0Predictor.compute_f0_uv + post_process (nearest resize / uv /
    np.interp gap fill). Short-circuits an all-zero rmvpe result, like the original."""
    if np.all(f0_100 == 0):
        return np.zeros(pad_to, np.float32), np.zeros(pad_to, np.float32)
    f0 = torch_interp_nearest(f0_100, pad_to)
    uv = (f0 > 0.0).astype(np.float32)
    nz = np.nonzero(f0)[0]
    if nz.size == 0:
        return np.zeros(pad_to, np.float32), uv
    if nz.size == 1:
        return np.full(pad_to, f0[nz[0]], np.float32), uv
    time_org = hop / sr * nz
    time_frame = np.arange(pad_to) * hop / sr
    vals = f0[nz]
    f0i = np.interp(time_frame, time_org, vals, left=vals[0], right=vals[-1])
    return f0i.astype(np.float32), uv


def build_orig_synth(pth, cfg):
    ck = torch.load(pth, map_location="cpu", weights_only=False)
    state = {k: v.float() for k, v in ck["model"].items()}
    orig = orig_models.SynthesizerTrn(
        cfg["data"]["filter_length"] // 2 + 1,
        cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
        **cfg["model"])
    missing, unexpected = orig.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {len(unexpected)}"
    assert all(k.startswith("enc_q.") for k in missing), f"non-enc_q missing: {missing[:3]}"
    orig.eval()
    return orig


def pad_array_center(arr, target_length):
    cur = len(arr)
    if cur >= target_length:
        return arr
    pad = target_length - cur
    return np.concatenate([np.zeros(pad // 2, arr.dtype), arr,
                           np.zeros(pad - pad // 2, arr.dtype)])


def run_variant(variant, orig, meta, rmvpe, dim, hop, ssl_mode, vol_embedding,
                input_audio, model_sr):
    native_sr = model_sr                       # input pre-resampled to model sr
    pad_native = int(native_sr * PAD_SECONDS)
    padded = np.concatenate([np.zeros(pad_native, np.float32),
                             input_audio.astype(np.float32),
                             np.zeros(pad_native, np.float32)])
    wav_m = padded                             # native==target → identity resample
    n_frames = len(wav_m) // hop

    if variant == "A":
        # original torchaudio internal resamples (two distinct kernels)
        wm = torch.from_numpy(wav_m)[None, :]
        wav16k_f0 = torchaudio.transforms.Resample(
            native_sr, 16000, lowpass_filter_width=128)(wm)[0].numpy()
        wav16k_hub = torchaudio.transforms.Resample(native_sr, 16000)(wm)[0].numpy()
    else:
        # Rust choice: ONE resample_poly wav16k for BOTH f0 and hubert
        wav16k = resample_poly(wav_m, 16000, native_sr).astype(np.float32)
        wav16k_f0 = wav16k_hub = wav16k

    # f0 (our rmvpe onnx + original post_process); tran=0 → no shift
    f0_100 = rmvpe.f0_100fps(wav16k_f0, RMVPE_THRED)
    if np.all(f0_100 == 0):
        f0 = np.zeros(n_frames, np.float32); uv = np.zeros(n_frames, np.float32)
    else:
        f0, uv = sovits_post_process(f0_100, n_frames, hop, model_sr)

    # content (our contentvec onnx) → [ssl_dim, T_hub] → repeat_expand → [1, ssl_dim, n_frames]
    c_raw = contentvec_session(dim).run(None, {"waveform": wav16k_hub[None]})[0]  # [1,T,ssl]
    c_hub = torch.from_numpy(c_raw[0].T.copy())                                   # [ssl, T]
    c = so_utils.repeat_expand_2d(c_hub, n_frames, ssl_mode).unsqueeze(0)         # [1,ssl,nf]

    f0t = torch.from_numpy(f0)[None].float()
    uvt = torch.from_numpy(uv)[None].float()
    sid = torch.LongTensor([0]).unsqueeze(0)
    vol = None
    if vol_embedding:
        vol = so_utils.Volume_Extractor(hop).extract(
            torch.FloatTensor(wav_m)[None, :])[None, :]

    with ZeroNoise():
        audio = orig.infer(c, f0t, uvt, g=sid, noice_scale=NOISE_SCALE,
                           predict_f0=False, vol=vol)[0][0, 0].cpu().numpy()

    pad_tgt = int(model_sr * PAD_SECONDS)
    trimmed = audio[pad_tgt: -pad_tgt]
    per_length = int(np.ceil(len(input_audio) / native_sr * model_sr))
    return pad_array_center(trimmed.astype(np.float32), per_length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-a", required=True)
    ap.add_argument("--out-b", required=True)
    args = ap.parse_args()

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    ck = torch.load(args.pth, map_location="cpu", weights_only=False)
    cfg, cfg_src = sovits_v4.load_sovits_config(
        Path(args.pth), Path(args.config) if args.config else None)
    _, meta = sovits_v4.build_from_checkpoint(ck, cfg)
    model_sr = meta["sample_rate"]
    assert sr == model_sr, f"input must be {model_sr} Hz mono (got {sr})"
    dim = meta["features_dim"]; hop = meta["hop_size"]
    ssl_mode = meta["unit_interpolate_mode"]; vol_emb = meta["vol_embedding"]
    print(f"[ref] {Path(args.pth).name}: v{meta['version']} dim={dim} hop={hop} "
          f"sr={model_sr} vol={vol_emb} mode={ssl_mode} (config {cfg_src})")
    print(f"[ref] input n={len(audio)} rms={np.sqrt(np.mean(audio**2)):.4f}")

    orig = build_orig_synth(args.pth, cfg)
    rmvpe = RmvpeFront()

    for variant, out in (("A", args.out_a), ("B", args.out_b)):
        y = run_variant(variant, orig, meta, rmvpe, dim, hop, ssl_mode, vol_emb,
                        audio, model_sr)
        print(f"[ref] variant {variant}: n={len(y)} peak={np.abs(y).max():.4f} "
              f"rms={np.sqrt(np.mean(y**2)):.4f}")
        sf.write(out, y, model_sr, subtype="FLOAT")
    print("[ref] done")


if __name__ == "__main__":
    main()
