"""关卡2 MDX-Net reference — UVR SeperateMDX semantics driving the OFFICIAL onnx
via onnxruntime (torch only for the STFT class, replicated verbatim from
uvr_lib_v5/stft.py). Geometry matches the Rust default: step = chunk_size / 2
(num_overlap 2). Primary = net output × compensate; secondary = mix − primary.
"""

import os
import sys

import numpy as np
import torch
import librosa
import soundfile as sf
import onnxruntime as ort

SCRATCH = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "UVR_MDXNET_KARA": {"n_fft": 6144, "compensate": 1.035},
    "UVR_MDXNET_KARA_2": {"n_fft": 5120, "compensate": 1.065},
}
HOP = 1024
DIM_F = 2048
DIM_T = 256


class STFT:
    """uvr_lib_v5/stft.py semantics (torch.stft/istft, hann periodic, center)."""

    def __init__(self, n_fft, hop_length, dim_f):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=n_fft, periodic=True)
        self.dim_f = dim_f

    def __call__(self, x):
        b, c, t = x.shape
        x = x.reshape([-1, t])
        s = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                       window=self.window, center=True, return_complex=True)
        s = torch.view_as_real(s)
        s = s.permute([0, 3, 1, 2])
        s = s.reshape([b, c, 2, self.n_fft // 2 + 1, -1]).reshape([b, c * 2, self.n_fft // 2 + 1, -1])
        return s[:, :, : self.dim_f]

    def inverse(self, s):
        b, c4, f, t = s.shape
        n = self.n_fft // 2 + 1
        pad = n - f
        s = torch.cat([s, torch.zeros([b, c4, pad, t])], -2)
        c = c4 // 2
        s = s.reshape([b, c, 2, n, t]).reshape([-1, 2, n, t])
        s = s.permute([0, 2, 3, 1])
        s = s[..., 0] + 1j * s[..., 1]
        x = torch.istft(s, n_fft=self.n_fft, hop_length=self.hop_length,
                        window=self.window, center=True)
        return x.reshape([b, c, -1])


def run(name, mix, sess, stft):
    p = MODELS[name]
    n_fft = p["n_fft"]
    trim = n_fft // 2
    chunk_size = HOP * (DIM_T - 1)
    gen_size = chunk_size - 2 * trim
    L = mix.shape[1]
    pad = gen_size + trim - (L % gen_size)
    mixture = np.concatenate(
        [np.zeros((2, trim), np.float32), mix, np.zeros((2, pad), np.float32)], axis=1)
    step = chunk_size // 2  # num_overlap 2

    result = np.zeros((2, mixture.shape[1]), np.float64)
    divider = np.zeros(mixture.shape[1], np.float64)
    for start in range(0, mixture.shape[1], step):
        end = min(start + chunk_size, mixture.shape[1])
        actual = end - start
        part = np.zeros((2, chunk_size), np.float32)
        part[:, :actual] = mixture[:, start:end]
        spek = stft(torch.from_numpy(part[None]))
        spek[:, :, :3, :] *= 0
        out = sess.run(None, {"input": spek.numpy()})[0]
        tar = stft.inverse(torch.from_numpy(out)).numpy()[0]
        window = np.hanning(actual)
        result[:, start:end] += tar[:, :actual] * window
        divider[start:end] += window

    with np.errstate(invalid="ignore"):
        result = np.where(divider > 0, result / divider, 0.0)
    est = result[:, trim:trim + L].astype(np.float32)
    primary = est * p["compensate"]
    secondary = mix - primary
    return primary, secondary


mix_path = os.path.join(SCRATCH, "mix_20s.wav")
mix, _ = librosa.load(mix_path, sr=44100, mono=False)
mix = mix.astype(np.float32)

for name in (sys.argv[1:] or list(MODELS)):
    sess = ort.InferenceSession(os.path.join(SCRATCH, "onnx", name + ".onnx"))
    stft = STFT(MODELS[name]["n_fft"], HOP, DIM_F)
    primary, secondary = run(name, mix, sess, stft)
    sf.write(os.path.join(SCRATCH, f"{name}_REF_primary.wav"),
             primary.T, 44100, subtype="FLOAT")
    sf.write(os.path.join(SCRATCH, f"{name}_REF_secondary.wav"),
             secondary.T, 44100, subtype="FLOAT")
    print(f"{name}: primary peak {np.abs(primary).max():.4f}, "
          f"secondary peak {np.abs(secondary).max():.4f}")
print("MDX REF DONE")
