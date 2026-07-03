# Compare Rust pipeline stems vs Python reference stems (python = ref).
import os, struct
import numpy as np

SCRATCH = r"C:\Users\admin\AppData\Local\Temp\claude\D--MyDev-Utai-v2-dev\01a1a6a4-09b0-4416-8ddb-91c217acc8a8\scratchpad"

def read_wav_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", path
    pos = 12; fmt = None; raw = None
    while pos + 8 <= len(data):
        cid = data[pos:pos+4]
        csz = struct.unpack("<I", data[pos+4:pos+8])[0]
        body = data[pos+8:pos+8+csz]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            raw = body
        pos += 8 + csz + (csz & 1)
    audio_fmt, ch, sr, _, _, bits = fmt
    if audio_fmt in (3, 65534) and bits == 32:
        # 65534 = WAVE_FORMAT_EXTENSIBLE; Rust hound writes float32 under it
        x = np.frombuffer(raw, dtype="<f4").reshape(-1, ch)
    elif audio_fmt == 1 and bits == 16:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, ch) / 32768.0
    else:
        raise ValueError(f"{path}: fmt={audio_fmt} bits={bits}")
    return x.astype(np.float32), sr, (audio_fmt, bits)

mix, _, _ = read_wav_f32(os.path.join(SCRATCH, "e2e_mix.wav"))
print(f"[mix] len={mix.shape[0]}")

pairs = [
    ("vocals",       "py_ref_vocals.wav", "e2e_mix.wav.vocals.wav"),
    ("instrumental", "py_ref_instr.wav",  "e2e_mix.wav.instrumental.wav"),
]

print(f"\n{'stem':<14}{'len_py':>9}{'len_rs':>9}{'max_abs_diff':>14}{'SNR_dB':>9}{'corr':>10}"
      f"{'py_peak':>9}{'py_rms':>8}{'rs_peak':>9}{'rs_rms':>8}")
for name, pyf, rsf in pairs:
    ref, sr1, f1 = read_wav_f32(os.path.join(SCRATCH, pyf))
    x, sr2, f2 = read_wav_f32(os.path.join(SCRATCH, rsf))
    n = min(len(ref), len(x))
    if len(ref) != len(x):
        print(f"  !! LENGTH MISMATCH {name}: py={len(ref)} rust={len(x)} (trimming to {n})")
    r = ref[:n].astype(np.float64); e = x[:n].astype(np.float64)
    d = r - e
    snr = 10*np.log10((r**2).sum() / max((d**2).sum(), 1e-30))
    corr = float(np.corrcoef(r.flatten(), e.flatten())[0, 1])
    print(f"{name:<14}{len(ref):>9}{len(x):>9}{np.abs(d).max():>14.6e}{snr:>9.2f}{corr:>10.6f}"
          f"{np.abs(r).max():>9.4f}{np.sqrt((r**2).mean()):>8.4f}"
          f"{np.abs(e).max():>9.4f}{np.sqrt((e**2).mean()):>8.4f}")
    print(f"    rust wav fmt={f2} sr={sr2}")

# sanity: py vocals + py instr == mix
pv, _, _ = read_wav_f32(os.path.join(SCRATCH, "py_ref_vocals.wav"))
pi, _, _ = read_wav_f32(os.path.join(SCRATCH, "py_ref_instr.wav"))
n = min(len(mix), len(pv))
res = mix[:n] - (pv[:n] + pi[:n])
print(f"\n[sanity] |mix - (py_voc+py_instr)| max = {np.abs(res).max():.2e} (should be ~0)")

rv, _, _ = read_wav_f32(os.path.join(SCRATCH, "e2e_mix.wav.vocals.wav"))
ri, _, _ = read_wav_f32(os.path.join(SCRATCH, "e2e_mix.wav.instrumental.wav"))
n = min(len(mix), len(rv), len(ri))
res = mix[:n] - (rv[:n] + ri[:n])
print(f"[sanity] |mix - (rs_voc+rs_instr)| max = {np.abs(res).max():.2e}")

# per-band error (only meaningful if SNR is mid-range)
BANDS = [(0, 250), (250, 1000), (1000, 4000), (4000, 10000), (10000, 22050)]
for name, pyf, rsf in pairs:
    ref, _, _ = read_wav_f32(os.path.join(SCRATCH, pyf))
    x, _, _ = read_wav_f32(os.path.join(SCRATCH, rsf))
    n = min(len(ref), len(x))
    R = np.fft.rfft(ref[:n].astype(np.float64), axis=0)
    X = np.fft.rfft(x[:n].astype(np.float64), axis=0)
    freqs = np.fft.rfftfreq(n, d=1.0/44100)
    E = X - R
    print(f"\n  band-wise err/ref ({name}):")
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        e_ref = (np.abs(R[m])**2).sum()
        e_err = (np.abs(E[m])**2).sum()
        ratio = e_err / max(e_ref, 1e-30)
        print(f"    {lo:>5}-{hi:<5} Hz: {ratio:>10.4e} ({10*np.log10(max(ratio,1e-30)):>7.1f} dB)")
print("\nDONE")
