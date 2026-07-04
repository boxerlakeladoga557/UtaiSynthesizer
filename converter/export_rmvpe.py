# export_rmvpe.py — RMVPE f0 estimator -> single ONNX "rmvpe_e2e.onnx"
#
# Input : "mel"       f32 [1, 128, T]  log-mel (see RMVPE_CONTRACT.md for the exact
#                                      Rust-side DSP: torch-lineage STFT 1024/160 +
#                                      librosa HTK mel filterbank + ln(clamp(., 1e-5)))
#         "threshold" f32 [1]          salience gate (original RVC uses 0.03)
# Output: "f0"        f32 [1, T]       f0 in Hz @ 100 fps; unvoiced frames == 0.0
#
# What is baked IN-GRAPH (so Rust does none of it):
#   * pad T to the next multiple of 32 (constant 0.0, same as original mel2hidden)
#   * E2E model (DeepUnet + BiGRU) -> salience [1, T, 360]
#   * slice back to T frames
#   * to_local_average_cents decode (argmax + 9-bin weighted average + threshold
#     gating) + cents -> Hz + the f0==10 -> 0 unvoiced rule
#
# Model classes below are VERBATIM ports from the ORIGINAL
# D:\MyDev\RVC\RVC20240604Nvidia\infer\lib\rmvpe.py (pure tensor math, only the
# decode is re-expressed as batched torch ops — gated numerically in
# converter\verify\voice\gate_rmvpe.py against the original numpy decode).
#
# Also dumps converter\test_output\rmvpe_mel_filters.npy — the [128, 513] f32 mel
# filterbank (librosa, htk=True, slaney-norm) that Rust loads verbatim.
#
# Usage:
#   .venv\Scripts\python.exe export_rmvpe.py [--pt PATH] [--out PATH]
#
# torch.onnx.export: dynamo=False, opset 17 (project-wide rule).

import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel

DEFAULT_PT = r"D:\MyDev\RVC\RVC20240604Nvidia\assets\rmvpe\rmvpe.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "test_output", "rmvpe_e2e.onnx")
MEL_FILTERS_NPY = os.path.join(HERE, "test_output", "rmvpe_mel_filters.npy")

# Mel/STFT constants (from original RMVPE.__init__:
#   MelSpectrogram(is_half, 128, 16000, 1024, 160, None, 30, 8000))
SR = 16000
N_MELS = 128
N_FFT = 1024  # n_fft=None -> win_length
WIN_LENGTH = 1024
HOP_LENGTH = 160
MEL_FMIN = 30
MEL_FMAX = 8000
LOG_CLAMP = 1e-5

N_CLASS = 360
CENTS_OFFSET = 1997.3794084376191  # cents_mapping = 20*arange(360) + this


# ------------------------------------------------------------------
# VERBATIM model classes from infer\lib\rmvpe.py
# ------------------------------------------------------------------
class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor):
        concat_tensors: List[torch.Tensor] = []
        x = self.bn(x)
        for i, layer in enumerate(self.layers):
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class ResEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01
    ):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i, conv in enumerate(self.conv):
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for i in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i, conv2 in enumerate(self.conv2):
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: List[torch.Tensor]):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(E2E, self).__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError("rmvpe.pt uses n_gru=1")

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


# ------------------------------------------------------------------
# Export wrapper: pad-to-32 + E2E + to_local_average_cents decode
# ------------------------------------------------------------------
class RmvpeF0(nn.Module):
    """mel [1,128,T] log-mel + threshold [1] -> f0 [1,T] Hz (unvoiced = 0)."""

    def __init__(self, model: E2E):
        super().__init__()
        self.model = model
        # ORIGINAL: cents_mapping = 20*arange(360)+1997.3794084376191, padded (4,4)
        cents_mapping = 20 * np.arange(N_CLASS) + CENTS_OFFSET
        cents_mapping = np.pad(cents_mapping, (4, 4))  # [368]
        self.register_buffer(
            "cents_mapping", torch.from_numpy(cents_mapping).float(), persistent=False
        )
        self.register_buffer(
            "window_offsets", torch.arange(-4, 5, dtype=torch.long), persistent=False
        )

    def forward(self, mel: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        # --- ORIGINAL mel2hidden: pad T to multiple of 32 with zeros ---
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        mel = F.pad(mel, (0, n_pad), mode="constant")
        hidden = self.model(mel)  # [1, T_pad, 360]
        hidden = hidden[:, :n_frames]  # [1, T, 360]

        # --- ORIGINAL to_local_average_cents, batched torch port ---
        center = torch.argmax(hidden, dim=2)  # [1, T]  (first max on ties)
        salience = F.pad(hidden, (4, 4))  # [1, T, 368]
        center = center + 4
        idx = center.unsqueeze(-1) + self.window_offsets  # [1, T, 9]
        todo_salience = torch.gather(salience, 2, idx)  # [1, T, 9]
        cents_full = self.cents_mapping.unsqueeze(0).unsqueeze(0).expand_as(salience)
        todo_cents_mapping = torch.gather(cents_full, 2, idx)  # [1, T, 9]
        product_sum = torch.sum(todo_salience * todo_cents_mapping, dim=2)  # [1, T]
        weight_sum = torch.sum(todo_salience, dim=2)  # [1, T]
        devided = product_sum / weight_sum  # [1, T]
        maxx = torch.max(salience, dim=2).values  # [1, T] (over padded, as original)
        devided = torch.where(
            maxx > threshold, devided, torch.zeros_like(devided)
        )  # ORIGINAL: devided[maxx <= thred] = 0

        # --- ORIGINAL decode: f0 = 10*2^(cents/1200); f0[f0==10] = 0 ---
        f0 = 10.0 * torch.pow(2.0, devided / 1200.0)
        f0 = torch.where(devided == 0.0, torch.zeros_like(f0), f0)
        return f0


def dump_mel_filters(path: str) -> np.ndarray:
    # ORIGINAL MelSpectrogram.__init__: librosa.filters.mel(sr, n_fft, n_mels,
    # fmin, fmax, htk=True) -> htk mel scale + slaney area-normalization (default)
    mel_basis = mel(
        sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX, htk=True
    ).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, mel_basis)
    print(f"mel filterbank {mel_basis.shape} {mel_basis.dtype} -> {path}")
    return mel_basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=DEFAULT_PT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    torch.manual_seed(0)

    dump_mel_filters(MEL_FILTERS_NPY)

    model = E2E(4, 1, (2, 2))  # ORIGINAL get_default_model()
    ckpt = torch.load(args.pt, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    wrapper = RmvpeF0(model).eval()

    T = 417  # deliberately NOT a multiple of 32 -> pad path is traced
    mel_in = torch.randn(1, N_MELS, T)
    thr = torch.tensor([0.03], dtype=torch.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (mel_in, thr),
            args.out,
            input_names=["mel", "threshold"],
            output_names=["f0"],
            dynamic_axes={"mel": {2: "T"}, "f0": {1: "T"}},
            opset_version=17,
            dynamo=False,
        )
    print(f"exported -> {args.out}")

    # smoke: ORT parity vs torch wrapper on two DIFFERENT lengths (incl. mult-of-32)
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    for t in (T, 256, 100):
        m = torch.randn(1, N_MELS, t)
        with torch.no_grad():
            ref = wrapper(m, thr).numpy()
        got = sess.run(
            None, {"mel": m.numpy(), "threshold": thr.numpy()}
        )[0]
        assert got.shape == (1, t), (got.shape, t)
        d = np.abs(got - ref)
        rel = d / np.maximum(np.abs(ref), 1e-9)
        print(f"T={t}: shape ok, max_abs_diff={d.max():.3e}, max_rel={rel.max():.3e}")


if __name__ == "__main__":
    main()
