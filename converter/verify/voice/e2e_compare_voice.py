"""关卡 2 voice — SNR/corr compare of Rust pipeline output vs python reference (ref = truth).

Usage: e2e_compare_voice.py <label> <ref.wav> <rust.wav> [more label ref rust ...]
Prints a per-case table: lengths, max_abs_diff, SNR(dB), corr, peaks/rms, plus per-band
error ratios for the WORST few cases (helps attribute a mid-range SNR to a frequency zone).
SNR = 10·log10( Σref² / Σ(ref-rust)² ) over the common length (ref = signal, diff = noise).
Gate line (verify\README.md): SNR > 40 dB = pipeline faithful.
"""
import struct
import sys

import numpy as np


def read_wav_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", path
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        csz = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + csz]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            raw = body
        pos += 8 + csz + (csz & 1)
    audio_fmt, ch, sr = fmt[0], fmt[1], fmt[2]
    bits = fmt[5]
    if audio_fmt in (3, 65534) and bits == 32:
        x = np.frombuffer(raw, dtype="<f4").reshape(-1, ch)
    elif audio_fmt == 1 and bits == 16:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, ch) / 32768.0
    else:
        raise ValueError(f"{path}: fmt={audio_fmt} bits={bits}")
    return x.astype(np.float64).mean(axis=1), sr


def compare(label, ref_path, rust_path):
    ref, sr1 = read_wav_f32(ref_path)
    rust, sr2 = read_wav_f32(rust_path)
    n = min(len(ref), len(rust))
    lenflag = "" if len(ref) == len(rust) else f"  !!LEN py={len(ref)} rs={len(rust)}"
    r, e = ref[:n], rust[:n]
    d = r - e
    snr = 10 * np.log10((r ** 2).sum() / max((d ** 2).sum(), 1e-30))
    corr = float(np.corrcoef(r, e)[0, 1]) if n > 1 else float("nan")
    print(f"{label:<20}{sr1:>7}{len(ref):>10}{len(rust):>10}{np.abs(d).max():>13.3e}"
          f"{snr:>9.2f}{corr:>10.6f}{np.abs(r).max():>8.3f}{np.sqrt((r**2).mean()):>8.4f}"
          f"{np.abs(e).max():>8.3f}{np.sqrt((e**2).mean()):>8.4f}{lenflag}")
    return snr, sr1


def band_report(label, ref_path, rust_path, sr):
    ref, _ = read_wav_f32(ref_path)
    rust, _ = read_wav_f32(rust_path)
    n = min(len(ref), len(rust))
    R = np.fft.rfft(ref[:n]); X = np.fft.rfft(rust[:n])
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    E = X - R
    nyq = sr / 2
    bands = [(0, 250), (250, 1000), (1000, 4000), (4000, 10000), (10000, nyq)]
    print(f"  band err/ref [{label}]:")
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        er = (np.abs(R[m]) ** 2).sum(); ee = (np.abs(E[m]) ** 2).sum()
        ratio = ee / max(er, 1e-30)
        print(f"    {int(lo):>6}-{int(hi):<6} Hz: {10*np.log10(max(ratio,1e-30)):>7.1f} dB")


def main():
    args = sys.argv[1:]
    assert len(args) % 3 == 0 and args, "need <label> <ref> <rust> triples"
    print(f"{'case':<20}{'sr':>7}{'len_py':>10}{'len_rs':>10}{'max_diff':>13}"
          f"{'SNR_dB':>9}{'corr':>10}{'py_pk':>8}{'py_rms':>8}{'rs_pk':>8}{'rs_rms':>8}")
    print("-" * 118)
    results = []
    for i in range(0, len(args), 3):
        label, ref, rust = args[i], args[i + 1], args[i + 2]
        snr, sr = compare(label, ref, rust)
        results.append((label, ref, rust, snr, sr))
    print()
    for label, ref, rust, snr, sr in results:
        if snr < 65:
            band_report(label, ref, rust, sr)
    worst = min(r[3] for r in results)
    print(f"\nWORST SNR = {worst:.2f} dB  "
          f"({'PASS >40' if worst > 40 else 'BELOW 40 — bisect'})")


if __name__ == "__main__":
    main()
