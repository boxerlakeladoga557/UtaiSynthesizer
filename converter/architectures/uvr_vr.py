"""UVR VR-arch (vocal-remover v5 / v5.1) reimplementation for ONNX export.

Two network generations, selected per model by UVR's registry:
- v5.0 `CascadedASPPNet` (UVR nets.py / layers.py lineage) — e.g. 5_HP/6_HP-Karaoke
- v5.1 `CascadedNet`     (UVR nets_new.py / layers_new.py lineage, has a BiLSTM)

Reference implementation (gate-1 comparison target):
D:\\MyDev\\ARCHIVE\\MSSTRVCv2\\MSST\\modules\\vocal_remover\\uvr_lib_v5\\vr_network
(audio-separator-derived; verified identical to Anjok07/ultimatevocalremovergui@master).

State-dict key parity with the originals is CONTRACTUAL — gate 1 loads real UVR
checkpoints with strict=True into these classes, so attribute names/structure must
match the original classes exactly. The only intentional deviations (parameter-free,
math-identical, ONNX-friendliness only):
- `nn.AdaptiveAvgPool2d((1, None))` -> `FreqMean` (mean over the freq axis with
  keepdim; exactly equivalent for output_size (1, None) and exports cleanly).
- The offset time-crop of `predict_mask()` lives in the export wrapper
  (`VrMaskModel`), and v4 forward's dead `mix = detach(); clone()` prologue is
  dropped (eval-mode upstream returns the bare mask too — `return mask  # * mix`).
- v5.1 LSTMModule reshapes use -1 at the batch position so the exported graph
  keeps a dynamic batch axis (T is fixed at export, so all other dims are static).

Model identification follows UVR itself: md5 of the LAST 10_000*1024 bytes of the
.pth (whole file if smaller) -> VR_REGISTRY. Band-split DSP constants come from
UVR's modelparams JSONs, embedded VERBATIM below (VR models do not carry their DSP
params in the checkpoint — mispairing produces garbage, so unknown hashes are
refused instead of guessed).
"""

import hashlib
import os

import torch
from torch import nn
import torch.nn.functional as F

# Inference geometry: UVR default window (frames per crop fed to the net). The
# export bakes T=WINDOW_SIZE static (DML-friendly); only batch is dynamic.
WINDOW_SIZE = 512

# UVR hashes the tail of the file, not the whole file (UVR.py get_model_hash).
BYTES_TO_HASH = 10_000 * 1024

# UVR gui_data/constants.py NON_ACCOM_STEMS — decides the aggressiveness flip
# (adjust_aggr's is_non_accom_stem). None of the 9 registry models hit it
# (e.g. 17_HP's primary is "No Woodwinds", not "Woodwinds"), but keep the exact
# list so future registry additions inherit correct behavior.
NON_ACCOM_STEMS = (
    "Vocals", "Other", "Bass", "Drums", "Guitar", "Piano",
    "Synthesizer", "Strings", "Woodwinds", "Brass", "Wind Instrument",
)


def uvr_tail_hash(path) -> str:
    """md5 of the last BYTES_TO_HASH bytes (whole file if smaller) — UVR's model id."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size >= BYTES_TO_HASH:
            f.seek(-BYTES_TO_HASH, 2)
        return hashlib.md5(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# Band-split DSP params — verbatim copies of UVR lib_v5/vr_network/modelparams/
# (local authoritative copies: ARCHIVE\MSSTRVCv2\MSST\configs\vr_modelparams).
# Band keys are ints (ModelParameters parses JSON string keys back to int).
# 4band_v4_ms_fullband ships with "n_bins"/"stable_bins" key names — normalized
# to "bins" here exactly like ModelParameters does.
# ---------------------------------------------------------------------------
VR_MODELPARAMS = {
    "4band_v2_sn": {
        "bins": 672, "unstable_bins": 8, "reduction_bins": 637,
        "band": {
            1: {"sr": 7350, "hl": 80, "n_fft": 640, "crop_start": 0, "crop_stop": 85,
                "lpf_start": 25, "lpf_stop": 53, "res_type": "polyphase"},
            2: {"sr": 7350, "hl": 80, "n_fft": 320, "crop_start": 4, "crop_stop": 87,
                "hpf_start": 25, "hpf_stop": 12, "lpf_start": 31, "lpf_stop": 62,
                "res_type": "polyphase"},
            3: {"sr": 14700, "hl": 160, "n_fft": 512, "crop_start": 17, "crop_stop": 216,
                "hpf_start": 48, "hpf_stop": 24, "lpf_start": 139, "lpf_stop": 210,
                "res_type": "polyphase"},
            4: {"sr": 44100, "hl": 480, "n_fft": 960, "crop_start": 78, "crop_stop": 383,
                "hpf_start": 130, "hpf_stop": 86, "convert_channels": "stereo_n",
                "res_type": "kaiser_fast"},
        },
        "sr": 44100, "pre_filter_start": 668, "pre_filter_stop": 672,
    },
    "4band_v3": {
        "bins": 672, "unstable_bins": 8, "reduction_bins": 530,
        "band": {
            1: {"sr": 7350, "hl": 80, "n_fft": 640, "crop_start": 0, "crop_stop": 85,
                "lpf_start": 25, "lpf_stop": 53, "res_type": "polyphase"},
            2: {"sr": 7350, "hl": 80, "n_fft": 320, "crop_start": 4, "crop_stop": 87,
                "hpf_start": 25, "hpf_stop": 12, "lpf_start": 31, "lpf_stop": 62,
                "res_type": "polyphase"},
            3: {"sr": 14700, "hl": 160, "n_fft": 512, "crop_start": 17, "crop_stop": 216,
                "hpf_start": 48, "hpf_stop": 24, "lpf_start": 139, "lpf_stop": 210,
                "res_type": "polyphase"},
            4: {"sr": 44100, "hl": 480, "n_fft": 960, "crop_start": 78, "crop_stop": 383,
                "hpf_start": 130, "hpf_stop": 86, "res_type": "kaiser_fast"},
        },
        "sr": 44100, "pre_filter_start": 668, "pre_filter_stop": 672,
    },
    "3band_44100_msb2": {
        "mid_side_b2": True,
        "bins": 640, "unstable_bins": 7, "reduction_bins": 565,
        "band": {
            1: {"sr": 11025, "hl": 108, "n_fft": 1024, "crop_start": 0, "crop_stop": 187,
                "lpf_start": 92, "lpf_stop": 186, "res_type": "polyphase"},
            2: {"sr": 22050, "hl": 216, "n_fft": 768, "crop_start": 0, "crop_stop": 212,
                "hpf_start": 68, "hpf_stop": 34, "lpf_start": 174, "lpf_stop": 209,
                "res_type": "polyphase"},
            3: {"sr": 44100, "hl": 432, "n_fft": 640, "crop_start": 66, "crop_stop": 307,
                "hpf_start": 86, "hpf_stop": 72, "res_type": "kaiser_fast"},
        },
        "sr": 44100, "pre_filter_start": 639, "pre_filter_stop": 640,
    },
    "1band_sr44100_hl1024": {
        "bins": 1024, "unstable_bins": 0, "reduction_bins": 0,
        "band": {
            1: {"sr": 44100, "hl": 1024, "n_fft": 2048, "crop_start": 0, "crop_stop": 1024,
                "hpf_start": -1, "res_type": "sinc_best"},
        },
        "sr": 44100, "pre_filter_start": 1023, "pre_filter_stop": 1024,
    },
    # Source JSON uses n_bins/stable_bins; bins normalized per ModelParameters.
    "4band_v4_ms_fullband": {
        "bins": 896, "unstable_bins": 9, "stable_bins": 530,
        "band": {
            1: {"sr": 7350, "hl": 96, "n_fft": 768, "crop_start": 0, "crop_stop": 102,
                "lpf_start": 30, "lpf_stop": 62, "res_type": "polyphase",
                "convert_channels": "mid_side"},
            2: {"sr": 7350, "hl": 96, "n_fft": 384, "crop_start": 5, "crop_stop": 104,
                "hpf_start": 30, "hpf_stop": 14, "lpf_start": 37, "lpf_stop": 73,
                "res_type": "polyphase", "convert_channels": "mid_side"},
            3: {"sr": 14700, "hl": 192, "n_fft": 640, "crop_start": 20, "crop_stop": 259,
                "hpf_start": 58, "hpf_stop": 29, "lpf_start": 191, "lpf_stop": 262,
                "res_type": "polyphase", "convert_channels": "mid_side"},
            4: {"sr": 44100, "hl": 576, "n_fft": 1152, "crop_start": 119, "crop_stop": 575,
                "hpf_start": 157, "hpf_stop": 110, "res_type": "kaiser_fast",
                "convert_channels": "mid_side"},
        },
        "sr": 44100, "pre_filter_start": -1, "pre_filter_stop": -1,
    },
}

# ---------------------------------------------------------------------------
# Model registry — keyed by UVR tail-hash (ground truth: hashes computed locally
# from the official TRvlvr/model_repo downloads, matching UVR's
# vr_model_data/model_data_new.json; De-Reverb-aufr33-jarredou is only in
# audio-separator's overlay registry).
# stem_names = TRUE output order [primary(=mask target), secondary(=1-mask)].
# NOTE DeNoise: primary IS "Noise" (the useful stem is the secondary) — order
# must stay content-true (S31 port-label lesson).
# For v5.0 entries nn_arch_size feeds determine_model_capacity (capacity tier +
# ASPP layer-count branches); for v5.1 entries nout is PRE-RESOLVED here (the
# original forces nout=64 when nn_arch_size==218409 — 17_HP + DeEcho-DeReverb).
# ---------------------------------------------------------------------------
VR_REGISTRY = {
    "f6ea8473ff86017b5ebd586ccacf156b": {
        "name": "5_HP-Karaoke-UVR", "vr_model_param": "4band_v2_sn",
        "is_v51": False, "nn_arch_size": 123812,
        "primary_stem": "Instrumental", "stem_names": ["instrumental", "vocals"],
    },
    "6b5916069a49be3fe29d4397ecfd73fa": {
        "name": "6_HP-Karaoke-UVR", "vr_model_param": "3band_44100_msb2",
        "is_v51": False, "nn_arch_size": 123812,
        "primary_stem": "Instrumental", "stem_names": ["instrumental", "vocals"],
    },
    "0ec76fd9e65f81d8b4fbd13af4826ed8": {
        "name": "17_HP-Wind_Inst-UVR", "vr_model_param": "4band_v3",
        "is_v51": True, "nout": 64, "nout_lstm": 128,
        "primary_stem": "No Woodwinds", "stem_names": ["no woodwinds", "woodwinds"],
    },
    "6857b2972e1754913aad0c9a1678c753": {
        "name": "UVR-De-Echo-Aggressive", "vr_model_param": "4band_v3",
        "is_v51": True, "nout": 48, "nout_lstm": 128,
        "primary_stem": "No Echo", "stem_names": ["no echo", "echo"],
    },
    "f200a145434efc7dcf0cd093f517ed52": {
        "name": "UVR-De-Echo-Normal", "vr_model_param": "4band_v3",
        "is_v51": True, "nout": 48, "nout_lstm": 128,
        "primary_stem": "No Echo", "stem_names": ["no echo", "echo"],
    },
    "0fb9249ffe4ffc38d7b16243f394c0ff": {
        "name": "UVR-DeEcho-DeReverb", "vr_model_param": "4band_v3",
        "is_v51": True, "nout": 64, "nout_lstm": 128,
        "primary_stem": "No Reverb", "stem_names": ["no reverb", "reverb"],
    },
    "51ea8c43a6928ed3c10ef5cb2707d57b": {
        "name": "UVR-DeNoise-Lite", "vr_model_param": "1band_sr44100_hl1024",
        "is_v51": True, "nout": 16, "nout_lstm": 128,
        "primary_stem": "Noise", "stem_names": ["noise", "no noise"],
    },
    "44c55d8b5d2e3edea98c2b2bf93071c7": {
        "name": "UVR-DeNoise", "vr_model_param": "4band_v3",
        "is_v51": True, "nout": 48, "nout_lstm": 128,
        "primary_stem": "Noise", "stem_names": ["noise", "no noise"],
    },
    "97dc361a7a88b2c4542f68364b32c7f6": {
        "name": "UVR-De-Reverb-aufr33-jarredou", "vr_model_param": "4band_v4_ms_fullband",
        "is_v51": True, "nout": 32, "nout_lstm": 128,
        "primary_stem": "Dry", "stem_names": ["dry", "reverb"],
    },
}


class FreqMean(nn.Module):
    """Parameter-free stand-in for nn.AdaptiveAvgPool2d((1, None)) — mean over the
    freq axis, keepdim. Exactly equivalent and exports as ReduceMean."""

    def forward(self, x):
        return x.mean(dim=2, keepdim=True)


def crop_center(h1, h2):
    """spec_utils.crop_center — center-crop h1's TIME axis (dim 3) to h2's."""
    t1, t2 = h1.size(3), h2.size(3)
    if t1 == t2:
        return h1
    if t1 < t2:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")
    s = (t1 - t2) // 2
    return h1[:, :, :, s : s + t2]


# ---------------------------------------------------------------------------
# v5.0 layers (UVR layers.py) — attribute names are state-dict contract
# ---------------------------------------------------------------------------
class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=ksize, stride=stride, padding=pad,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def forward(self, x):
        return self.conv(x)


class SeperableConv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nin, nin, kernel_size=ksize, stride=stride, padding=pad,
                      dilation=dilation, groups=nin, bias=False),
            nn.Conv2d(nin, nout, kernel_size=1, bias=False),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def forward(self, x):
        return self.conv(x)


class EncoderV4(nn.Module):
    """v4 Encoder: conv1 stride-1 (its output is the skip), conv2 downsamples."""

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)

    def forward(self, x):
        skip = self.conv1(x)
        hidden = self.conv2(skip)
        return hidden, skip


class DecoderV4(nn.Module):
    """v4 Decoder: attr name `conv` (v5's is `conv1` — different state keys)."""

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False):
        super().__init__()
        self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            skip = crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)
        h = self.conv(x)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class ASPPModuleV4(nn.Module):
    def __init__(self, nn_architecture, nin, nout, dilations=(4, 8, 16), activ=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Sequential(
            FreqMean(),  # original: nn.AdaptiveAvgPool2d((1, None)) — no params
            Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ),
        )
        self.nn_architecture = nn_architecture
        self.six_layer = [129605]
        self.seven_layer = [537238, 537227, 33966]
        # Original assigns ONE instance to both conv6 and conv7 (tied weights);
        # replicated for state-dict + numeric parity on seven_layer archs.
        extra_conv = SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[2], dilations[2], activ=activ)
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        self.conv3 = SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[2], dilations[2], activ=activ)
        if self.nn_architecture in self.six_layer:
            self.conv6 = extra_conv
            nin_x = 6
        elif self.nn_architecture in self.seven_layer:
            self.conv6 = extra_conv
            self.conv7 = extra_conv
            nin_x = 7
        else:
            nin_x = 5
        self.bottleneck = nn.Sequential(
            Conv2DBNActiv(nin * nin_x, nout, 1, 1, 0, activ=activ),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode="bilinear", align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        if self.nn_architecture in self.six_layer:
            feat6 = self.conv6(x)
            out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6), dim=1)
        elif self.nn_architecture in self.seven_layer:
            feat6 = self.conv6(x)
            feat7 = self.conv7(x)
            out = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6, feat7), dim=1)
        else:
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        return self.bottleneck(out)


class BaseASPPNet(nn.Module):
    def __init__(self, nn_architecture, nin, ch, dilations=(4, 8, 16)):
        super().__init__()
        self.nn_architecture = nn_architecture
        self.enc1 = EncoderV4(nin, ch, 3, 2, 1)
        self.enc2 = EncoderV4(ch, ch * 2, 3, 2, 1)
        self.enc3 = EncoderV4(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = EncoderV4(ch * 4, ch * 8, 3, 2, 1)
        if self.nn_architecture == 129605:
            self.enc5 = EncoderV4(ch * 8, ch * 16, 3, 2, 1)
            self.aspp = ASPPModuleV4(nn_architecture, ch * 16, ch * 32, dilations)
            self.dec5 = DecoderV4(ch * (16 + 32), ch * 16, 3, 1, 1)
        else:
            self.aspp = ASPPModuleV4(nn_architecture, ch * 8, ch * 16, dilations)
        self.dec4 = DecoderV4(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = DecoderV4(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = DecoderV4(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = DecoderV4(ch * (1 + 2), ch, 3, 1, 1)

    def forward(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)
        if self.nn_architecture == 129605:
            h, e5 = self.enc5(h)
            h = self.aspp(h)
            h = self.dec5(h, e5)
        else:
            h = self.aspp(h)
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)
        return h


def determine_model_capacity(n_fft_bins, nn_architecture):
    """UVR nets.determine_model_capacity — capacity tier by nn_arch_size."""
    sp_model_arch = [31191, 33966, 129605]
    hp_model_arch = [123821, 123812]
    hp2_model_arch = [537238, 537227]
    if nn_architecture in sp_model_arch:
        model_capacity_data = [(2, 16), (2, 16), (18, 8, 1, 1, 0), (8, 16),
                               (34, 16, 1, 1, 0), (16, 32), (32, 2, 1), (16, 2, 1), (16, 2, 1)]
    elif nn_architecture in hp_model_arch:
        model_capacity_data = [(2, 32), (2, 32), (34, 16, 1, 1, 0), (16, 32),
                               (66, 32, 1, 1, 0), (32, 64), (64, 2, 1), (32, 2, 1), (32, 2, 1)]
    elif nn_architecture in hp2_model_arch:
        model_capacity_data = [(2, 64), (2, 64), (66, 32, 1, 1, 0), (32, 64),
                               (130, 64, 1, 1, 0), (64, 128), (128, 2, 1), (64, 2, 1), (64, 2, 1)]
    else:
        raise ValueError(f"Unknown v5.0 VR nn_architecture size: {nn_architecture}")
    return CascadedASPPNet(n_fft_bins, model_capacity_data, nn_architecture)


class CascadedASPPNet(nn.Module):
    """v5.0 net. Constructor n_fft == modelparam bins*2 (NOT the STFT n_fft)."""

    def __init__(self, n_fft, model_capacity_data, nn_architecture):
        super().__init__()
        self.stg1_low_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[0])
        self.stg1_high_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[1])
        self.stg2_bridge = Conv2DBNActiv(*model_capacity_data[2])
        self.stg2_full_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[3])
        self.stg3_bridge = Conv2DBNActiv(*model_capacity_data[4])
        self.stg3_full_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[5])
        self.out = nn.Conv2d(*model_capacity_data[6], bias=False)
        # Training-only aux heads — declared for state-dict parity.
        self.aux1_out = nn.Conv2d(*model_capacity_data[7], bias=False)
        self.aux2_out = nn.Conv2d(*model_capacity_data[8], bias=False)
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.offset = 128

    def forward(self, x):
        x = x[:, :, : self.max_bin]
        bandw = x.size(2) // 2
        aux1 = torch.cat([
            self.stg1_low_band_net(x[:, :, :bandw]),
            self.stg1_high_band_net(x[:, :, bandw:]),
        ], dim=2)
        h = torch.cat([x, aux1], dim=1)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))
        h = torch.cat([x, aux1, aux2], dim=1)
        h = self.stg3_full_band_net(self.stg3_bridge(h))
        mask = torch.sigmoid(self.out(h))
        mask = F.pad(mask, (0, 0, 0, self.output_bin - mask.size(2)), mode="replicate")
        return mask


# ---------------------------------------------------------------------------
# v5.1 layers (UVR layers_new.py / nets_new.py)
# ---------------------------------------------------------------------------
class EncoderV5(nn.Module):
    """v5 Encoder: conv1 carries the stride (downsample first), conv2 stride 1;
    returns a single tensor (skips are taken from encoder outputs upstream)."""

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return h


class DecoderV5(nn.Module):
    """v5 Decoder: attr name `conv1` (state-dict differs from v4's `conv`)."""

    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False):
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            skip = crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)
        h = self.conv1(x)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class ASPPModuleV5(nn.Module):
    """v5 ASPP: plain (non-separable) convs with 2-D (freq, time) dilations."""

    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            FreqMean(),  # original: nn.AdaptiveAvgPool2d((1, None))
            Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ),
        )
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        self.conv3 = Conv2DBNActiv(nin, nout, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = Conv2DBNActiv(nin, nout, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = Conv2DBNActiv(nin, nout, 3, 1, dilations[2], dilations[2], activ=activ)
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode="bilinear", align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.bottleneck(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LSTMModule(nn.Module):
    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super().__init__()
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
        self.lstm = nn.LSTM(input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(nout_lstm, nin_lstm), nn.BatchNorm1d(nin_lstm), nn.ReLU()
        )

    def forward(self, x):
        # nbins/nframes are static at export (T fixed); -1 keeps batch dynamic.
        nbins, nframes = x.size(2), x.size(3)
        h = self.conv(x)[:, 0]            # [N, nbins, nframes]
        h = h.permute(2, 0, 1)            # [nframes, N, nbins]
        h, _ = self.lstm(h)
        h = self.dense(h.reshape(-1, h.size(-1)))  # [nframes*N, nbins]
        h = h.reshape(nframes, -1, 1, nbins)
        h = h.permute(1, 2, 3, 0)         # [N, 1, nbins, nframes]
        return h


class BaseNet(nn.Module):
    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super().__init__()
        self.enc1 = Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = EncoderV5(nout, nout * 2, 3, 2, 1)
        self.enc3 = EncoderV5(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = EncoderV5(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = EncoderV5(nout * 6, nout * 8, 3, 2, 1)
        self.aspp = ASPPModuleV5(nout * 8, nout * 8, dilations, dropout=True)
        self.dec4 = DecoderV5(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = DecoderV5(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = DecoderV5(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = DecoderV5(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        h = self.aspp(e5)
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)
        return h


class CascadedNet(nn.Module):
    """v5.1 net. Constructor n_fft == modelparam bins*2. nout must arrive
    PRE-RESOLVED (the original's `nout = 64 if nn_arch_size == 218409` rule is
    applied in VR_REGISTRY, not here)."""

    def __init__(self, n_fft, nout=32, nout_lstm=128):
        super().__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64
        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm),
            Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0),
        )
        self.stg1_high_band_net = BaseNet(2, nout // 4, self.nin_lstm // 2, nout_lstm // 2)
        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm),
            Conv2DBNActiv(nout, nout // 2, 1, 1, 0),
        )
        self.stg2_high_band_net = BaseNet(nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2)
        self.stg3_full_band_net = BaseNet(3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm)
        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        # Training-only aux head — declared for state-dict parity.
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)

    def forward(self, x):
        x = x[:, :, : self.max_bin]
        bandw = x.size(2) // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)
        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)
        f3_in = torch.cat([x, aux1, aux2], dim=1)
        f3 = self.stg3_full_band_net(f3_in)
        mask = torch.sigmoid(self.out(f3))
        mask = F.pad(mask, (0, 0, 0, self.output_bin - mask.size(2)), mode="replicate")
        return mask


class VrMaskModel(nn.Module):
    """Export wrapper: [B, 2, bins+1, WINDOW_SIZE] normalized magnitude window ->
    sigmoid mask [B, 2, bins+1, WINDOW_SIZE - 2*offset] (predict_mask semantics;
    the offset time-crop is baked into the graph)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.offset = net.offset

    def forward(self, mag):
        mask = self.net(mag)
        return mask[:, :, :, self.offset : -self.offset]


def detect_config(ckpt_path: str) -> dict:
    """Identify the model by UVR tail-hash and return its full config
    (registry entry + embedded modelparams). Refuses unknown models — VR DSP
    params cannot be inferred from the checkpoint."""
    h = uvr_tail_hash(ckpt_path)
    entry = VR_REGISTRY.get(h)
    if entry is None:
        known = ", ".join(sorted(e["name"] for e in VR_REGISTRY.values()))
        raise ValueError(
            f"Unknown UVR VR model (tail-md5 {h}, file {os.path.basename(ckpt_path)}). "
            f"VR models need exact band-split params from UVR's registry; add the model "
            f"to VR_REGISTRY in architectures/uvr_vr.py. Known models: {known}"
        )
    config = dict(entry)
    config["model_params"] = VR_MODELPARAMS[entry["vr_model_param"]]
    return config


def build_network(config: dict):
    bins = config["model_params"]["bins"]
    if config["is_v51"]:
        return CascadedNet(bins * 2, nout=config["nout"], nout_lstm=config["nout_lstm"])
    return determine_model_capacity(bins * 2, config["nn_arch_size"])


def load_from_checkpoint(ckpt_path: str, config: dict = None):
    if config is None:
        config = detect_config(ckpt_path)
    net = build_network(config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net
