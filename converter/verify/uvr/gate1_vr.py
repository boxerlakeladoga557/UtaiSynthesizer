"""Gate 1 — UVR VR arch equivalence: ORIGINAL (ARCHIVE) vs our reimplementation.

Per model (all 9 registry .pth, real weights):
  0. tail-hash must hit BOTH our VR_REGISTRY and UVR's own registry (model_data_new
     + audio-separator overlay), and our registry fields must match UVR's raw entry
     (vr_model_param, nout-resolution rule).
  0b. our embedded VR_MODELPARAMS must equal the ARCHIVE modelparams JSON verbatim.
  1. build ORIGINAL net via the original selection logic (file-size nearest +
     raw model_data nout), load real weights strict.
  2. build OURS via uvr_vr.detect_config/load_from_checkpoint (strict=True).
  3. same random input -> forward mask max_abs_diff (expect ~0.0), and
     predict_mask vs VrMaskModel (offset crop) parity.
"""

import json
import math
import os
import sys
import types

# --- optional-dependency stubs (import-time only; none of their code runs) ---
for name, attrs in (("audioread", {}), ("soundfile", {}), ("six", {"PY2": False})):
    try:
        __import__(name)
    except ImportError:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

import numpy as np
import torch

MSST_ROOT = r"D:\MyDev\ARCHIVE\MSSTRVCv2\MSST"
CONVERTER = r"D:\MyDev\Utai_v2-dev\converter"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
PTH_DIR = os.path.join(SCRATCH, "pth")

sys.path.insert(0, MSST_ROOT)
sys.path.insert(0, CONVERTER)

from modules.vocal_remover.uvr_lib_v5.vr_network import nets as orig_nets            # noqa: E402
from modules.vocal_remover.uvr_lib_v5.vr_network import nets_new as orig_nets_new    # noqa: E402
from modules.vocal_remover.uvr_lib_v5.vr_network.model_param_init import ModelParameters  # noqa: E402
from architectures import uvr_vr                                                     # noqa: E402

# UVR's raw registries (fetched from TRvlvr/application_data + audio-separator)
with open(os.path.join(SCRATCH, "model_data_new.json"), encoding="utf-8") as f:
    uvr_raw = json.load(f)
with open(os.path.join(SCRATCH, "audiosep_model_data.json"), encoding="utf-8") as f:
    asep = json.load(f)
uvr_raw.update(asep.get("vr_model_data", {}))

NN_ARCH_SIZES = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
VR_51_SIZES = [56817, 218409]

failures = []


def check(cond, label):
    tag = "PASS" if cond else "FAIL"
    print(f"    [{tag}] {label}")
    if not cond:
        failures.append(label)


print("=== modelparams cross-check (embedded vs ARCHIVE JSON) ===")
for mp_name in uvr_vr.VR_MODELPARAMS:
    with open(os.path.join(MSST_ROOT, "configs", "vr_modelparams", mp_name + ".json"),
              encoding="utf-8") as f:
        raw = json.load(f)
    # normalize like ModelParameters: int band keys, n_bins -> bins
    raw["band"] = {int(k): v for k, v in raw["band"].items()}
    if "n_bins" in raw:
        raw["bins"] = raw["n_bins"]
        del raw["n_bins"]
    ours = dict(uvr_vr.VR_MODELPARAMS[mp_name])
    # stable_bins/reduction_bins are training-only; compare everything anyway
    check(ours == raw, f"{mp_name}: embedded == ARCHIVE json")

print("\n=== per-model equivalence ===")
torch.manual_seed(20260704)
for fname in sorted(os.listdir(PTH_DIR)):
    if not fname.endswith(".pth"):
        continue
    path = os.path.join(PTH_DIR, fname)
    print(f"\n--- {fname} ({os.path.getsize(path):,} B)")

    h = uvr_vr.uvr_tail_hash(path)
    ours_entry = uvr_vr.VR_REGISTRY.get(h)
    raw_entry = uvr_raw.get(h)
    check(ours_entry is not None, f"tail-hash {h} in our VR_REGISTRY")
    check(raw_entry is not None, f"tail-hash {h} in UVR raw registry")
    if ours_entry is None or raw_entry is None:
        continue
    check(ours_entry["vr_model_param"] == raw_entry["vr_model_param"],
          f"vr_model_param match ({raw_entry['vr_model_param']})")
    check(ours_entry["primary_stem"] == raw_entry["primary_stem"],
          f"primary_stem match ({raw_entry['primary_stem']})")

    # --- ORIGINAL build, replicating vr_separator.load_model exactly ---
    mp = ModelParameters(os.path.join(MSST_ROOT, "configs", "vr_modelparams",
                                      raw_entry["vr_model_param"] + ".json"))
    model_capacity = (32, 128)
    is_vr_51 = False
    if "nout" in raw_entry and "nout_lstm" in raw_entry:
        model_capacity = (raw_entry["nout"], raw_entry["nout_lstm"])
        is_vr_51 = True
    model_size = math.ceil(os.stat(path).st_size / 1024)
    nn_arch_size = min(NN_ARCH_SIZES, key=lambda x: abs(x - model_size))
    if nn_arch_size in VR_51_SIZES or is_vr_51:
        orig = orig_nets_new.CascadedNet(mp.param["bins"] * 2, nn_arch_size,
                                         nout=model_capacity[0], nout_lstm=model_capacity[1])
        orig_is51 = True
    else:
        orig = orig_nets.determine_model_capacity(mp.param["bins"] * 2, nn_arch_size)
        orig_is51 = False
    state = torch.load(path, map_location="cpu", weights_only=True)
    orig.load_state_dict(state)  # strict=True default
    orig.eval()
    check(orig_is51 == ours_entry["is_v51"], f"generation match (v5.1={orig_is51})")

    # --- OURS ---
    config = uvr_vr.detect_config(path)
    net = uvr_vr.load_from_checkpoint(path, config)  # strict=True inside
    wrapper = uvr_vr.VrMaskModel(net)
    wrapper.eval()
    check(net.offset == orig.offset, f"offset match ({orig.offset})")

    bins = mp.param["bins"]
    x = torch.rand(2, 2, bins + 1, uvr_vr.WINDOW_SIZE)

    # Tier 1 — TRUE architecture equivalence: with FreqMean temporarily swapped
    # back to adaptive_avg_pool2d (the original op), outputs must be BIT-EXACT.
    # This pins any residual diff in tier 2 on the pooling op's summation order
    # alone, not on a structural mistake.
    orig_freqmean_fwd = uvr_vr.FreqMean.forward
    uvr_vr.FreqMean.forward = lambda self, t: torch.nn.functional.adaptive_avg_pool2d(
        t, (1, t.size(3)))
    try:
        with torch.no_grad():
            m_ours_aap = net(x)
            m_orig = orig(x)
        d_exact = (m_orig - m_ours_aap).abs().max().item()
    finally:
        uvr_vr.FreqMean.forward = orig_freqmean_fwd
    print(f"    [tier1 adaptive-pool] forward max_abs_diff = {d_exact:.3e}")
    check(d_exact == 0.0, "tier1 BIT-EXACT (proves only deviation = pooling op)")

    # Tier 2 — shipped graph semantics (FreqMean = ReduceMean, what ONNX runs):
    # playbook threshold < 1e-5 (converter/verify/README.md 关卡 1).
    with torch.no_grad():
        m_ours = net(x)
        pm_orig = orig.predict_mask(x)
        pm_ours = wrapper(x)
    d_fwd = (m_orig - m_ours).abs().max().item()
    d_pm = (pm_orig - pm_ours).abs().max().item()
    print(f"    [tier2 shipped/mean]  forward max_abs_diff = {d_fwd:.3e}")
    print(f"    predict_mask max_abs_diff = {d_pm:.3e}  shapes {list(pm_orig.shape)} vs {list(pm_ours.shape)}")
    check(d_fwd < 1e-5, "tier2 forward parity < 1e-5 (playbook)")
    check(d_pm < 1e-5 and pm_orig.shape == pm_ours.shape, "tier2 predict_mask parity")

print("\n=== SUMMARY ===")
if failures:
    print(f"GATE 1 FAILED ({len(failures)}):")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
print("GATE 1 PASSED: all models bit-equivalent to the original implementation.")
