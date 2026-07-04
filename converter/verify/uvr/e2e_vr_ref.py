"""关卡2 VR reference — ORIGINAL torch net + ORIGINAL ARCHIVE DSP, end-to-end.

Orchestration below is copied line-for-line from ARCHIVE vr_separator.py
(loading_mix / inference_vr); all DSP calls go to the ARCHIVE spec_utils.
One documented deviation: cmb_spectrogram_to_wave allocates its per-band
scratch with np.zeros (audio-separator master behavior) instead of ARCHIVE's
uninitialized np.ndarray — garbage rows are a bug, not a semantic.

Outputs per model into scratchpad: <name>_REF_primary.wav / _REF_secondary.wav
plus <name>_REF_intermediates.npz (combined spec, mask) for stage-level bisecting.
"""

import json
import math
import os
import sys
import types

for name, attrs in (("audioread", {}), ("six", {"PY2": False})):
    try:
        __import__(name)
    except ImportError:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

import numpy as np
import torch
import librosa
import soundfile as sf

MSST_ROOT = r"D:\MyDev\ARCHIVE\MSSTRVCv2\MSST"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MSST_ROOT)

from modules.vocal_remover.uvr_lib_v5 import spec_utils                                # noqa: E402
from modules.vocal_remover.uvr_lib_v5.vr_network import nets as orig_nets              # noqa: E402
from modules.vocal_remover.uvr_lib_v5.vr_network import nets_new as orig_nets_new      # noqa: E402
from modules.vocal_remover.uvr_lib_v5.vr_network.model_param_init import ModelParameters  # noqa: E402

with open(os.path.join(SCRATCH, "model_data_new.json"), encoding="utf-8") as f:
    uvr_raw = json.load(f)
with open(os.path.join(SCRATCH, "audiosep_model_data.json"), encoding="utf-8") as f:
    uvr_raw.update(json.load(f).get("vr_model_data", {}))

NN_ARCH_SIZES = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
VR_51_SIZES = [56817, 218409]
WINDOW_SIZE = 512
BATCH = 8


def tail_hash(path):
    import hashlib
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size >= 10_000 * 1024:
            f.seek(-10_000 * 1024, 2)
        return hashlib.md5(f.read()).hexdigest()


def build_original(pth_path):
    entry = uvr_raw[tail_hash(pth_path)]
    mp = ModelParameters(os.path.join(MSST_ROOT, "configs", "vr_modelparams",
                                      entry["vr_model_param"] + ".json"))
    model_capacity = (32, 128)
    is_vr_51 = False
    if "nout" in entry and "nout_lstm" in entry:
        model_capacity = (entry["nout"], entry["nout_lstm"])
        is_vr_51 = True
    model_size = math.ceil(os.stat(pth_path).st_size / 1024)
    nn_arch_size = min(NN_ARCH_SIZES, key=lambda x: abs(x - model_size))
    if nn_arch_size in VR_51_SIZES or is_vr_51:
        net = orig_nets_new.CascadedNet(mp.param["bins"] * 2, nn_arch_size,
                                        nout=model_capacity[0], nout_lstm=model_capacity[1])
        is_vr_51 = True
    else:
        net = orig_nets.determine_model_capacity(mp.param["bins"] * 2, nn_arch_size)
    net.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    net.eval()
    return net, mp, is_vr_51, entry


def loading_mix(path, mp, is_v51):
    X_wave, X_spec_s = {}, {}
    bands_n = len(mp.param["band"])
    for d in range(bands_n, 0, -1):
        bp = mp.param["band"][d]
        wav_resolution = bp["res_type"]
        if d == bands_n:
            X_wave[d], _ = librosa.load(path, sr=bp["sr"], mono=False,
                                        dtype=np.float32, res_type=wav_resolution)
            if X_wave[d].ndim == 1:
                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
        else:
            X_wave[d] = librosa.resample(X_wave[d + 1],
                                         orig_sr=mp.param["band"][d + 1]["sr"],
                                         target_sr=bp["sr"], res_type=wav_resolution)
        X_spec_s[d] = spec_utils.wave_to_spectrogram(
            X_wave[d], bp["hl"], bp["n_fft"], mp, band=d, is_v51_model=is_v51)
    return spec_utils.combine_spectrograms(X_spec_s, mp, is_v51_model=is_v51)


def inference_vr(X_spec, model, aggressiveness, is_non_accom):
    def _execute(X_mag_pad, roi_size):
        X_dataset = []
        patches = (X_mag_pad.shape[2] - 2 * model.offset) // roi_size
        for i in range(patches):
            start = i * roi_size
            X_dataset.append(X_mag_pad[:, :, start:start + WINDOW_SIZE])
        X_dataset = np.asarray(X_dataset)
        model.eval()
        with torch.no_grad():
            mask = []
            for i in range(0, patches, BATCH):
                X_batch = torch.from_numpy(X_dataset[i:i + BATCH])
                pred = model.predict_mask(X_batch).detach().cpu().numpy()
                pred = np.concatenate(pred, axis=2)
                mask.append(pred)
            mask = np.concatenate(mask, axis=2)
        return mask

    X_mag, X_phase = spec_utils.preprocess(X_spec)
    n_frame = X_mag.shape[2]
    pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, WINDOW_SIZE, model.offset)
    X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
    X_mag_pad /= X_mag_pad.max()
    mask = _execute(X_mag_pad, roi_size)
    mask = mask[:, :, :n_frame]

    mask = spec_utils.adjust_aggr(mask, is_non_accom, aggressiveness)
    y_spec = mask * X_mag * np.exp(1.0j * X_phase)
    v_spec = (1 - mask) * X_mag * np.exp(1.0j * X_phase)
    return y_spec, v_spec, mask


def cmb_zeros(spec_m, mp, is_v51):
    """ARCHIVE cmb_spectrogram_to_wave, verbatim except np.zeros scratch."""
    spec_m = np.where(np.isnan(spec_m), 0, spec_m)
    bands_n = len(mp.param["band"])
    offset = 0
    wave = None
    for d in range(1, bands_n + 1):
        bp = mp.param["band"][d]
        spec_s = np.zeros(shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex)
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"]:bp["crop_stop"], :] = spec_m[:, offset:offset + h, :]
        offset += h
        if d == bands_n:
            if bp["hpf_start"] > 0:
                if is_v51:
                    spec_s *= spec_utils.get_hp_filter_mask(spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1)
                else:
                    spec_s = spec_utils.fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            if bands_n == 1:
                wave = spec_utils.spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51)
            else:
                wave = np.add(wave, spec_utils.spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51))
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:
                if is_v51:
                    spec_s *= spec_utils.get_lp_filter_mask(spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"])
                else:
                    spec_s = spec_utils.fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave = librosa.resample(spec_utils.spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51),
                                        orig_sr=bp["sr"], target_sr=sr,
                                        res_type=spec_utils.wav_resolution)
            else:
                if is_v51:
                    spec_s *= spec_utils.get_hp_filter_mask(spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s *= spec_utils.get_lp_filter_mask(spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"])
                else:
                    spec_s = spec_utils.fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s = spec_utils.fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave2 = np.add(wave, spec_utils.spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51))
                wave = librosa.resample(wave2, orig_sr=bp["sr"], target_sr=sr,
                                        res_type=spec_utils.wav_resolution)
    return wave


NON_ACCOM_STEMS = ("Vocals", "Other", "Bass", "Drums", "Guitar", "Piano",
                   "Synthesizer", "Strings", "Woodwinds", "Brass", "Wind Instrument")

# ── prepare 20s stereo excerpt ──
mix_path = os.path.join(SCRATCH, "mix_20s.wav")
if not os.path.exists(mix_path):
    y, _ = librosa.load(r"D:\MyDev\TESTING\MSST\perf_mix_120s.wav",
                        sr=44100, mono=False, duration=20.0)
    sf.write(mix_path, y.T.astype(np.float32), 44100, subtype="FLOAT")
    print(f"wrote {mix_path}")

models = sys.argv[1:] or ["6_HP-Karaoke-UVR", "UVR-DeEcho-DeReverb"]
for name in models:
    pth = os.path.join(SCRATCH, "pth", name + ".pth")
    net, mp, is51, entry = build_original(pth)
    print(f"=== {name}: v5.1={is51}, param={entry['vr_model_param']}, "
          f"primary={entry['primary_stem']}")
    X_spec = loading_mix(mix_path, mp, is51)
    aggr = {"value": 0.05, "split_bin": mp.param["band"][1]["crop_stop"], "aggr_correction": None}
    is_non_accom = entry["primary_stem"] in NON_ACCOM_STEMS
    y_spec, v_spec, mask = inference_vr(X_spec, net, aggr, is_non_accom)
    y_spec = np.nan_to_num(y_spec, nan=0.0, posinf=0.0, neginf=0.0)
    v_spec = np.nan_to_num(v_spec, nan=0.0, posinf=0.0, neginf=0.0)
    wy = cmb_zeros(y_spec, mp, is51)
    wv = cmb_zeros(v_spec, mp, is51)
    sf.write(os.path.join(SCRATCH, f"{name}_REF_primary.wav"),
             wy.T.astype(np.float32), 44100, subtype="FLOAT")
    sf.write(os.path.join(SCRATCH, f"{name}_REF_secondary.wav"),
             wv.T.astype(np.float32), 44100, subtype="FLOAT")
    # Isolation variant: synthesis upsampler forced to polyphase (= our Rust choice,
    # = UVR-on-macOS-ARM). Rust vs REFPOLY isolates our DSP correctness from the
    # sinc_fastest-vs-polyphase resampler delta (REF vs REFPOLY quantifies that).
    orig_res = spec_utils.wav_resolution
    spec_utils.wav_resolution = "polyphase"
    try:
        wy_p = cmb_zeros(y_spec, mp, is51)
        wv_p = cmb_zeros(v_spec, mp, is51)
    finally:
        spec_utils.wav_resolution = orig_res
    sf.write(os.path.join(SCRATCH, f"{name}_REFPOLY_primary.wav"),
             wy_p.T.astype(np.float32), 44100, subtype="FLOAT")
    sf.write(os.path.join(SCRATCH, f"{name}_REFPOLY_secondary.wav"),
             wv_p.T.astype(np.float32), 44100, subtype="FLOAT")
    np.savez_compressed(os.path.join(SCRATCH, f"{name}_REF_intermediates.npz"),
                        combined=X_spec.astype(np.complex64), mask=mask.astype(np.float32))
    print(f"  stems: {wy.shape} / {wv.shape}, mask {mask.shape}, "
          f"peak_y={np.abs(wy).max():.4f} peak_v={np.abs(wv).max():.4f}")
print("REF DONE")
