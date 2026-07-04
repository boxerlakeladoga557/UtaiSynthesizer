# gate_rmvpe.py — RMVPE f0 chain gates vs the ORIGINAL RVC implementation.
#
# REF chain  : ORIGINAL infer\lib\rmvpe.py RMVPE(rmvpe.pt, is_half=False, cpu)
#              .infer_from_audio(audio, thred=0.03)   (torch mel + torch E2E + numpy decode)
# OUR chain  : numpy "Rust-simulated" DSP (reflect pad + periodic hann + rFFT +
#              rmvpe_mel_filters.npy matmul + ln clamp)  ->  rmvpe_e2e.onnx (ORT CPU)
#              (pad-to-32 + E2E + decode are IN-GRAPH)
#
# Gates (converter\verify\README.md methodology — compare vs ORIGINAL, never self):
#   (b1) mel STRUCTURE : f64 sim vs ORIGINAL MelSpectrogram code forced to f64
#                        max_abs_diff < 1e-4 (log domain). Both sides f64 removes fp32
#                        rounding, so this isolates ALGORITHM drift (pad/window/fft/
#                        filterbank/log). Expected ~1e-9; any structural bug -> >> 1e-4.
#   (b2) mel fp32 noise: f32 sim (what Rust actually computes) vs ORIGINAL fp32
#                        MelSpectrogram: LINEAR-domain max_abs_diff < 1e-5.
#                        NOTE: a global log-domain gate vs a fp32 reference is
#                        unattainable by construction — at bins just above the 1e-5
#                        clamp, log amplifies the reference's own fp32 rounding
#                        (~1e-6 absolute) to ~1e-3. Log-domain diff is reported
#                        per-level for the record; the f0 impact is bounded by (a).
#   (a) full chain     : f32 sim mel -> our onnx, on >=2 real vocal wavs (mono 16k):
#         voiced-frame f0 relative error < 0.1% for >= 99% of voiced frames
#         uv agreement > 99%     (worst case reported)
#   (c) dynamic T      : the two wavs have different, non-multiple-of-32 frame counts;
#                        output length must equal 1 + n_samples//160 for each
#   (extra) onnx isolation: ORIGINAL torch mel -> our onnx  vs  REF f0
#                        (separates model+decode drift from mel drift for bisection)
#
# Run: ..\..\.venv\Scripts\python.exe gate_rmvpe.py

import os
import sys
import types

import numpy as np
import torch

# ---------------- paths ----------------
RVC_ROOT = r"D:\MyDev\RVC\RVC20240604Nvidia"
RMVPE_PT = os.path.join(RVC_ROOT, "assets", "rmvpe", "rmvpe.pt")
CONVERTER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX_PATH = os.path.join(CONVERTER, "test_output", "rmvpe_e2e.onnx")
MEL_FILTERS_NPY = os.path.join(CONVERTER, "test_output", "rmvpe_mel_filters.npy")
WAVS = [
    (r"D:\MyDev\TESTING\ikanaiteyo\vocal.wav", 15.0),               # full-song vocal, first 15 s
    (r"D:\MyDev\TESTING\MSST\leak_fix_ab\FIXED_vocals_20s.wav", None),  # separated vocals, 20 s
]
THRED = 0.03  # RVC pipeline default (pipeline.py get_f0 rmvpe path)

# mel/STFT constants — MUST mirror RMVPE_CONTRACT.md
SR = 16000
N_FFT = 1024
WIN_LENGTH = 1024
HOP = 160
LOG_CLAMP = 1e-5

# ---------------- import ORIGINAL rmvpe (read-only) ----------------
# infer.lib.jit imports tqdm (not in venv) — shim it, it is unused on our path.
if "tqdm" not in sys.modules:
    _tqdm = types.ModuleType("tqdm")
    _tqdm.tqdm = lambda x, *a, **k: x
    sys.modules["tqdm"] = _tqdm
sys.path.insert(0, RVC_ROOT)
from infer.lib.rmvpe import RMVPE  # noqa: E402


# ---------------- python simulation of the Rust DSP ----------------
def rust_mel_sim(audio: np.ndarray, mel_basis: np.ndarray, f64: bool = False) -> np.ndarray:
    """Exactly the steps RMVPE_CONTRACT.md prescribes for Rust.

    audio: f32 [N] mono 16 kHz. returns log-mel [128, T], T = 1 + N//160.
    f64=False -> everything in f32 (what Rust computes); f64=True -> f64 pipeline
    (used by gate b1 to compare structure against the f64-forced original).
    """
    import scipy.fft

    assert audio.ndim == 1
    dt = np.float64 if f64 else np.float32
    n = len(audio)
    pad = N_FFT // 2  # 512
    # torch.stft(center=True) semantics: reflect-pad n_fft//2 both sides
    x = np.pad(audio.astype(dt), (pad, pad), mode="reflect")
    # periodic hann (torch.hann_window default): 0.5 - 0.5*cos(2*pi*k/N),
    # tabulated in f64 then cast (Rust: same)
    k = np.arange(WIN_LENGTH)
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * k / WIN_LENGTH)).astype(dt)
    n_frames = 1 + n // HOP
    frames = np.stack(
        [x[i * HOP : i * HOP + N_FFT] * window for i in range(n_frames)]
    )  # [T, 1024], dtype dt
    spec = scipy.fft.rfft(frames, axis=1)  # [T, 513] complex64/complex128 (dtype-preserving)
    magnitude = np.abs(spec).T.astype(dt)  # [513, T]
    mel_out = mel_basis.astype(dt) @ magnitude  # [128, T]  (filterbank values ARE f32)
    log_mel = np.log(np.maximum(mel_out, dt(LOG_CLAMP)))
    return log_mel


def original_mel_f64(audio: np.ndarray, mel_extractor) -> np.ndarray:
    """ORIGINAL MelSpectrogram forward forced to float64 (same code path, higher
    precision) — reference for the structure gate b1."""
    me = mel_extractor.double()
    # pre-seed the window cache so forward() does not create a f32 window;
    # torch.hann_window(dtype=f64) computes natively in f64 (== our sim's table)
    me.hann_window["0_cpu"] = torch.hann_window(WIN_LENGTH, dtype=torch.float64)
    with torch.no_grad():
        out = me(torch.from_numpy(audio.astype(np.float64)).unsqueeze(0), center=True)
    me.float()  # restore for subsequent fp32 use
    me.hann_window.clear()
    return out.squeeze(0).numpy()


def load_wav_16k_mono(path: str, seconds) -> np.ndarray:
    import librosa
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    if seconds is not None:
        audio = audio[: int(seconds * SR)]
    return audio.astype(np.float32)


def main():
    import onnxruntime as ort

    mel_basis = np.load(MEL_FILTERS_NPY)
    assert mel_basis.shape == (128, N_FFT // 2 + 1), mel_basis.shape

    print("loading ORIGINAL RMVPE (torch, cpu, fp32) ...")
    ref = RMVPE(RMVPE_PT, is_half=False, device="cpu")

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    thr = np.array([THRED], dtype=np.float32)

    all_pass = True
    frame_counts = []
    for path, seconds in WAVS:
        name = os.path.basename(path)
        audio = load_wav_16k_mono(path, seconds)
        n = len(audio)
        expect_T = 1 + n // HOP
        print(f"\n=== {name}: {n} samples ({n/SR:.1f}s), expect T={expect_T} "
              f"(mult32={expect_T % 32 == 0}) ===")
        frame_counts.append(expect_T)

        # ---- REF: original full chain ----
        f0_ref = ref.infer_from_audio(audio, thred=THRED)  # numpy [T]

        # ---- gate (b1): mel STRUCTURE, f64 vs f64 ----
        mel_ref64 = original_mel_f64(audio, ref.mel_extractor)  # [128, T] f64
        mel_ours64 = rust_mel_sim(audio, mel_basis, f64=True)
        assert mel_ours64.shape == mel_ref64.shape, (mel_ours64.shape, mel_ref64.shape)
        b1_diff = np.abs(mel_ours64 - mel_ref64).max()
        b1_pass = b1_diff < 1e-4
        all_pass &= b1_pass
        print(f"[gate b1] mel STRUCTURE f64-vs-f64 max_abs_diff (log) = {b1_diff:.3e}  "
              f"{'PASS' if b1_pass else 'FAIL'} (< 1e-4)")

        # ---- gate (b2): mel fp32 noise, f32 sim vs ORIGINAL fp32 ----
        mel_ref = (
            ref.mel_extractor(torch.from_numpy(audio).float().unsqueeze(0), center=True)
            .squeeze(0)
            .numpy()
        )  # [128, T] f32
        mel_ours = rust_mel_sim(audio, mel_basis)  # f32
        lin_diff = np.abs(np.exp(mel_ours) - np.exp(mel_ref)).max()
        b2_pass = lin_diff < 1e-5
        all_pass &= b2_pass
        log_diff = np.abs(mel_ours - mel_ref)
        lvl = mel_ref
        by_level = {t: (log_diff[lvl > t].max() if (lvl > t).any() else 0.0)
                    for t in (-12.0, -7.0, -3.0)}
        print(f"[gate b2] mel fp32 LINEAR max_abs_diff = {lin_diff:.3e}  "
              f"{'PASS' if b2_pass else 'FAIL'} (< 1e-5)")
        print(f"[gate b2] (info) log-domain max diff by level: "
              + ", ".join(f">{t:g}: {v:.2e}" for t, v in by_level.items())
              + "  (near-clamp amplification, see header)")

        # ---- OUR chain: simulated-Rust mel -> onnx ----
        f0_ours = sess.run(
            None, {"mel": mel_ours[None], "threshold": thr}
        )[0][0]  # [T]

        # ---- gate (c): dynamic T / frame alignment ----
        c_pass = (len(f0_ours) == expect_T) and (len(f0_ref) == expect_T)
        all_pass &= c_pass
        print(f"[gate c] T ours={len(f0_ours)} ref={len(f0_ref)} expect={expect_T}  "
              f"{'PASS' if c_pass else 'FAIL'}")

        # ---- extra: onnx isolation (ORIGINAL mel -> our onnx vs REF f0) ----
        f0_iso = sess.run(None, {"mel": mel_ref[None], "threshold": thr})[0][0]
        iso_v = (f0_ref > 0) & (f0_iso > 0)
        iso_rel = (np.abs(f0_iso[iso_v] - f0_ref[iso_v]) / f0_ref[iso_v]).max() if iso_v.any() else 0.0
        iso_uv = float(np.mean((f0_ref > 0) == (f0_iso > 0)))
        print(f"[isolate] onnx(model+decode) vs ref: worst voiced rel={iso_rel:.3e}, "
              f"uv agree={iso_uv*100:.3f}%")

        # ---- gate (a): full chain ----
        uv_agree = float(np.mean((f0_ref > 0) == (f0_ours > 0)))
        voiced = (f0_ref > 0) & (f0_ours > 0)
        n_voiced = int(voiced.sum())
        rel = np.abs(f0_ours[voiced] - f0_ref[voiced]) / f0_ref[voiced]
        pct_ok = float(np.mean(rel < 1e-3)) if n_voiced else 1.0
        worst = float(rel.max()) if n_voiced else 0.0
        a_pass = (pct_ok >= 0.99) and (uv_agree > 0.99)
        all_pass &= a_pass
        print(f"[gate a] voiced frames={n_voiced}/{expect_T}")
        print(f"[gate a] rel err < 0.1% on {pct_ok*100:.3f}% of voiced frames "
              f"(need >= 99%), worst-case rel = {worst:.3e}")
        print(f"[gate a] uv agreement = {uv_agree*100:.3f}% (need > 99%)  "
              f"{'PASS' if a_pass else 'FAIL'}")

    # dynamic-T sanity: the wavs must exercise two different lengths
    dyn_ok = len(set(frame_counts)) >= 2
    all_pass &= dyn_ok
    print(f"\n[gate c] distinct T values: {frame_counts}  "
          f"{'PASS' if dyn_ok else 'FAIL (need 2 different lengths)'}")

    print("\n" + ("ALL GATES PASS" if all_pass else "GATE FAILURE"))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
