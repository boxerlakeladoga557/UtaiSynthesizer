"""Generate scipy/librosa reference values for the Rust A/B tests (deterministic input)."""
import numpy as np
import scipy.signal as sps
import librosa

# deterministic multi-tone test signal (f64 for resample, f32 for stft)
n = 2000
t = np.arange(n)
x64 = np.sin(0.05 * t) + 0.5 * np.sin(0.31 * t + 1.0) + 0.25 * np.sin(1.7 * t + 2.0)
x32 = x64.astype(np.float32)

print("== resample_poly (window=('kaiser',5.0)) ==")
for up, down in [(1, 3), (1, 2), (2, 1), (3, 1)]:
    y = sps.resample_poly(x64, up, down, window=("kaiser", 5.0))
    print(f"up={up} down={down} len={len(y)}")
    print("  head:", ", ".join(f"{v:.12e}" for v in y[:6]))
    print(f"  sum={y.sum():.12e}  absmax={np.abs(y).max():.12e}  mid[{len(y)//2}]={y[len(y)//2]:.12e}")

print("== librosa.stft (n_fft=512, hop=128) ==")
S = librosa.stft(x32, n_fft=512, hop_length=128)
print("shape:", S.shape, "dtype:", S.dtype)
for (f, fr) in [(0, 0), (10, 5), (100, 7), (256, 10)]:
    v = S[f, fr]
    print(f"  S[{f},{fr}] = {v.real:.9e} + {v.imag:.9e}j")
print(f"  abs sum = {np.abs(S).sum():.9e}")

print("== librosa.istft (dtype f64) ==")
y = librosa.istft(S, hop_length=128, dtype=np.float64)
print(f"len={len(y)} sum={y.sum():.12e} mid[{len(y)//2}]={y[len(y)//2]:.12e}")
