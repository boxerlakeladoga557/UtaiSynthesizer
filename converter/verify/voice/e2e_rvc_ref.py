"""关卡 2 — RVC E2E python reference (ORIGINAL pipeline semantics vs our Rust chain).

Drives the ORIGINAL RVC Pipeline orchestration end-to-end in-process:
  * the REAL `infer.modules.vc.pipeline.Pipeline` object (config stub is_half=False,
    device=cpu, x_pad=1/x_query=6/x_center=38/x_max=41 = our Rust constants) — its REAL
    `vc()` and `get_f0()` methods carry the chunking / KNN / 2x-upsample / protect /
    change_rms / trim logic under test;
  * the ORIGINAL torch synthesizer SynthesizerTrnMs768NSFsid (lengv2.3.pth real weights),
    passed straight into vc() as net_g;  torch.randn/randn_like monkeypatched to zeros
    (kills z_p noise AND SineGen noise) == our Rust det build + noise_scale=0;
  * index BOTH off and on — for index-on the faiss index is an EXACT IndexFlatL2 over the
    SAME vectors the Rust harness loads (data\models\rvc\lengv2.3.npy) so both sides do
    exact top-8 KNN over identical vectors.

DOCUMENTED substitutions (each already certified against the true ORIGINAL elsewhere):
  * ContentVec feature extractor: fairseq hubert_base.pt is unimportable on Windows
    (no wheel) — we inject our contentvec_768l12.onnx via a shim `model`. gate_contentvec.py
    proved this ONNX bit-identical to the fairseq original (min cos > 0.9999, <1e-4).
    The Rust harness uses the SAME onnx → this stage contributes ~0 dB to the gap.
  * f0: the original get_f0 rmvpe branch calls model_rmvpe.infer_from_audio — we inject
    our rmvpe_e2e.onnx behind that call via a python log-mel front-end that mirrors
    src\inference\f0.rs verbatim (reflect-512 / periodic-hann / rFFT-1024 / mel@ / ln-clamp).
    gate_rmvpe.py certifies this onnx vs the original rmvpe.pt (f0 rel<0.1%, uv>99%).
    The original mel-coarse quantization (np.rint) stays REAL (get_f0). Rust uses
    round-half-away f0_to_coarse — a measure-zero difference on real f0.
  * output is kept FLOAT (the original's trailing int16 quantization is skipped — our Rust
    stays f32 for the DAW; a documented deviation). Everything else is the verbatim
    pipeline.py orchestration.

INPUT is a pre-resampled 16 kHz MONO wav (both sides consume identical samples → no
input-resampler variance; our Rust 16k→16k resample is a no-op, verified in rvc.rs).

Run: converter\.venv\Scripts\python.exe converter\verify\voice\e2e_rvc_ref.py \
        --input <16k.wav> --out-off <voc_off.wav> --out-on <voc_on.wav>
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

RVC_REPO = r"D:\MyDev\RVC\RVC20240604Nvidia"
CONVERTER = r"D:\MyDev\Utai_v2-dev\converter"
REPO_ROOT = r"D:\MyDev\Utai_v2-dev"
V2_PTH = r"D:\MyDev\TESTING\RVC\lenglengv2\lengv2.3.pth"

AUX = os.path.join(REPO_ROOT, "data", "models", "aux")
CONTENTVEC_768 = os.path.join(AUX, "contentvec_768l12.onnx")
RMVPE_ONNX = os.path.join(AUX, "rmvpe_e2e.onnx")
MEL_FILTERS = os.path.join(AUX, "rmvpe_mel_filters.npy")
INDEX_NPY = os.path.join(REPO_ROOT, "data", "models", "rvc", "lengv2.3.npy")

sys.path.insert(0, RVC_REPO)
sys.path.insert(0, CONVERTER)

# pipeline.py imports these at module level for the pm/harvest/crepe f0 methods we never
# use (not installed on this box) — stub them so the import succeeds. rmvpe is injected.
import types  # noqa: E402
for _m in ("parselmouth", "pyworld", "torchcrepe"):
    sys.modules.setdefault(_m, types.ModuleType(_m))

from infer.lib.infer_pack import models as orig_models  # noqa: E402
from infer.modules.vc.pipeline import Pipeline, change_rms, bh, ah  # noqa: E402
from scipy import signal  # noqa: E402

torch.set_grad_enabled(False)


# ── noise patch: torch.rand/​randn_like → zeros (deterministic; == Rust noise_scale=0) ──
class ZeroNoise:
    def __enter__(self):
        self._rand, self._randn_like, self._randn = (
            torch.rand, torch.randn_like, torch.randn)

        def zrand(*size, **kw):
            return torch.zeros(*size, device=kw.get("device"), dtype=kw.get("dtype"))

        torch.rand = zrand
        torch.randn_like = lambda t, **kw: torch.zeros_like(t)
        torch.randn = zrand
        return self

    def __exit__(self, *exc):
        torch.rand, torch.randn_like, torch.randn = (
            self._rand, self._randn_like, self._randn)


# ── our rmvpe onnx behind the original get_f0 rmvpe call (mel front-end mirrors f0.rs) ──
class RmvpeShim:
    """Stands in for RVC's RMVPE: .infer_from_audio(x, thred) → f0[Hz] @100fps."""

    def __init__(self):
        self.sess = ort.InferenceSession(RMVPE_ONNX, providers=["CPUExecutionProvider"])
        self.mel_filters = np.load(MEL_FILTERS).astype(np.float32)  # [128, 513]
        k = np.arange(1024)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * k / 1024.0)).astype(np.float32)

    def infer_from_audio(self, audio, thred=0.03):
        x = np.asarray(audio, dtype=np.float32)
        n = len(x)
        if n < 513:
            x = np.pad(x, (0, 513 - n))
            n = 513
        t_frames = 1 + n // 160
        padded = np.pad(x, (512, 512), mode="reflect")
        mag = np.empty((513, t_frames), dtype=np.float32)
        for t in range(t_frames):
            fr = padded[t * 160: t * 160 + 1024] * self.window
            mag[:, t] = np.abs(np.fft.rfft(fr, n=1024))
        mel = self.mel_filters @ mag                      # [128, T]
        logmel = np.log(np.clip(mel, 1e-5, None)).astype(np.float32)
        f0 = self.sess.run(None, {"mel": logmel[None],
                                  "threshold": np.array([thred], np.float32)})[0]
        # original RMVPE.infer_from_audio returns a 1-D f0 track (get_f0 slices it as 1-D);
        # our onnx emits [1, T] — flatten so the downstream [:p_len] slices the time axis.
        return np.asarray(f0, dtype=np.float32).reshape(-1)


# ── our contentvec onnx behind the original vc() extract_features call ──
class HubertShim:
    def __init__(self):
        self.sess = ort.InferenceSession(CONTENTVEC_768, providers=["CPUExecutionProvider"])

    def extract_features(self, source, padding_mask=None, output_layer=12):
        # source: [1, N] torch (padded 16k chunk). Our onnx = raw 16k f32 → [1,T,768].
        wav = source.detach().cpu().float().numpy().astype(np.float32)
        feats = self.sess.run(None, {"waveform": wav})[0]
        return (torch.from_numpy(feats),)                 # logits[0] used for v2

    def final_proj(self, x):                              # v1 only (unused here)
        return x


class ExpandedIndex:
    """Diagnostic index whose .search mirrors Rust's expanded-norm L2 EXACTLY
    (d² = |q|² − 2q·v + |v|², fp32, clamp 1e-9) instead of faiss's direct ‖q−v‖².
    Used only to attribute the index-on gap (faiss-L2 vs expanded-norm cancellation)."""

    def __init__(self, big):
        self.big = big
        self.norms = (big * big).sum(1).astype(np.float32)

    def search(self, npy, k):
        npy = np.asarray(npy, dtype=np.float32)
        qn = (npy * npy).sum(1, keepdims=True).astype(np.float32)
        d2 = (qn - 2.0 * (npy @ self.big.T) + self.norms[None]).astype(np.float32)
        ix = np.argpartition(d2, k, axis=1)[:, :k]
        score = np.maximum(np.take_along_axis(d2, ix, axis=1), 1e-9).astype(np.float32)
        return score, ix


def build_net_g():
    cpt = torch.load(V2_PTH, map_location="cpu", weights_only=False)
    config = list(cpt["config"])
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]   # official loader patch
    state = {k: v.float() for k, v in cpt["weight"].items()}
    net_g = orig_models.SynthesizerTrnMs768NSFsid(*config, is_half=False)
    missing, unexpected = net_g.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith("enc_q.") for k in missing), \
        (len(unexpected), len(missing))
    net_g.eval()
    return net_g, int(cpt.get("sr", 48000)) if not isinstance(cpt.get("sr"), str) else 48000


class Cfg:
    x_pad, x_query, x_center, x_max, is_half, device = 1, 6, 38, 41, False, "cpu"


def run_reference(pl, net_g, hub, sid, audio, tgt_sr, index, big_npy,
                  index_rate, f0_up_key=0, protect=0.33, rms_mix_rate=0.25):
    """Verbatim transcription of Pipeline.pipeline() (rmvpe f0, if_f0=1), using the REAL
    pl.get_f0 / pl.vc, our shim `hub` as the feature model, EXACT `index`, and skipping
    the original int16 quantization (Rust stays float)."""
    audio = signal.filtfilt(bh, ah, audio)
    audio_pad = np.pad(audio, (pl.window // 2, pl.window // 2), mode="reflect")
    opt_ts = []
    if audio_pad.shape[0] > pl.t_max:
        audio_sum = np.zeros_like(audio)
        for i in range(pl.window):
            audio_sum += np.abs(audio_pad[i: i - pl.window])
        for t in range(pl.t_center, audio.shape[0], pl.t_center):
            opt_ts.append(
                t - pl.t_query
                + np.where(audio_sum[t - pl.t_query: t + pl.t_query]
                           == audio_sum[t - pl.t_query: t + pl.t_query].min())[0][0])
    s = 0
    audio_opt = []
    t = None
    audio_pad = np.pad(audio, (pl.t_pad, pl.t_pad), mode="reflect")
    p_len = audio_pad.shape[0] // pl.window
    sid_t = torch.tensor(sid, device="cpu").unsqueeze(0).long()
    pitch, pitchf = pl.get_f0("ref", audio_pad, p_len, f0_up_key, "rmvpe", 3, None)
    pitch = pitch[:p_len]
    pitchf = pitchf[:p_len].astype(np.float32)
    pitch = torch.tensor(pitch, device="cpu").unsqueeze(0).long()
    pitchf = torch.tensor(pitchf, device="cpu").unsqueeze(0).float()
    times = [0, 0, 0]
    for t in opt_ts:
        t = t // pl.window * pl.window
        audio_opt.append(pl.vc(
            hub, net_g, sid_t, audio_pad[s: t + pl.t_pad2 + pl.window],
            pitch[:, s // pl.window: (t + pl.t_pad2) // pl.window],
            pitchf[:, s // pl.window: (t + pl.t_pad2) // pl.window],
            times, index, big_npy, index_rate, "v2", protect
        )[pl.t_pad_tgt: -pl.t_pad_tgt])
        s = t
    audio_opt.append(pl.vc(
        hub, net_g, sid_t, audio_pad[t:],
        pitch[:, t // pl.window:] if t is not None else pitch,
        pitchf[:, t // pl.window:] if t is not None else pitchf,
        times, index, big_npy, index_rate, "v2", protect
    )[pl.t_pad_tgt: -pl.t_pad_tgt])
    audio_opt = np.concatenate(audio_opt)
    if rms_mix_rate != 1:
        audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
    # NO int16 quantization (Rust stays f32) — return float at model sr.
    return np.asarray(audio_opt, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-off", required=True)
    ap.add_argument("--out-on", required=True)
    ap.add_argument("--index-rate", type=float, default=0.75)
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument("--rms-mix-rate", type=float, default=0.25)
    ap.add_argument("--knn", choices=["faiss", "expanded"], default="faiss",
                    help="index-on retrieval: faiss IndexFlatL2 (original, default) or "
                         "expanded-norm L2 matching Rust (diagnostic for attribution)")
    args = ap.parse_args()

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"input must be 16 kHz mono (got {sr})"
    print(f"[ref] input {args.input}: sr={sr} n={len(audio)} "
          f"rms={np.sqrt(np.mean(audio**2)):.4f}")

    net_g, tgt_sr = build_net_g()
    pl = Pipeline(tgt_sr, Cfg())
    pl.model_rmvpe = RmvpeShim()
    hub = HubertShim()
    print(f"[ref] net_g SynthesizerTrnMs768NSFsid, tgt_sr={tgt_sr}, upp={net_g.dec.upp}")

    big_npy = np.load(INDEX_NPY).astype(np.float32)
    print(f"[ref] index vectors {big_npy.shape} from {INDEX_NPY}")
    if args.knn == "expanded":
        index_flat = ExpandedIndex(big_npy)               # Rust's expanded-norm L2 (diagnostic)
        print("[ref] index-ON KNN = expanded-norm L2 (Rust-matching, diagnostic)")
    else:
        import faiss
        index_flat = faiss.IndexFlatL2(big_npy.shape[1])
        index_flat.add(big_npy)                           # EXACT faiss L2 (original)

    with ZeroNoise():
        off = run_reference(pl, net_g, hub, 0, audio.copy(), tgt_sr, None, None, 0.0,
                            protect=args.protect, rms_mix_rate=args.rms_mix_rate)
        print(f"[ref] index-OFF: n={len(off)} peak={np.abs(off).max():.4f} "
              f"rms={np.sqrt(np.mean(off**2)):.4f}")
        sf.write(args.out_off, off, tgt_sr, subtype="FLOAT")

        on = run_reference(pl, net_g, hub, 0, audio.copy(), tgt_sr,
                           index_flat, big_npy, args.index_rate,
                           protect=args.protect, rms_mix_rate=args.rms_mix_rate)
        print(f"[ref] index-ON ({args.index_rate}): n={len(on)} "
              f"peak={np.abs(on).max():.4f} rms={np.sqrt(np.mean(on**2)):.4f}")
        sf.write(args.out_on, on, tgt_sr, subtype="FLOAT")

    print("[ref] done")


if __name__ == "__main__":
    main()
