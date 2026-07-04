"""DML smoke for the new archs (playbook S33): DmlExecutionProvider vs CPU —
SNR + wall-time ratio. speedup ~1x = silent per-node CPU fallback (DML does not
error on unsupported nodes); tens of x = genuinely on the GPU.
Runs 2 representative VR models (v5.0 conv-only + v5.1 with BiLSTM) + both MDX.
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

SCRATCH = os.path.dirname(os.path.abspath(__file__))
CASES = [
    # (file, input_name, shape)
    ("6_HP-Karaoke-UVR.onnx", "mag", (4, 2, 641, 512)),          # v5.0 CascadedASPPNet
    ("UVR-DeEcho-DeReverb.onnx", "mag", (4, 2, 673, 512)),       # v5.1 CascadedNet + LSTM
    ("UVR_MDXNET_KARA.onnx", "input", (1, 4, 2048, 256)),
    ("UVR_MDXNET_KARA_2.onnx", "input", (1, 4, 2048, 256)),
]

rng = np.random.default_rng(20260704)
print(f"onnxruntime {ort.__version__}, providers: {ort.get_available_providers()}")
for fname, iname, shape in CASES:
    path = os.path.join(SCRATCH, "onnx", fname)
    x = rng.random(shape, dtype=np.float32)
    out = {}
    times = {}
    for label, providers in (("cpu", ["CPUExecutionProvider"]),
                             ("dml", ["DmlExecutionProvider"])):
        sess = ort.InferenceSession(path, providers=providers)
        used = sess.get_providers()
        sess.run(None, {iname: x})  # warmup (graph upload/compile)
        t0 = time.perf_counter()
        n = 3
        for _ in range(n):
            y = sess.run(None, {iname: x})[0]
        times[label] = (time.perf_counter() - t0) / n
        out[label] = y
        del sess
    err = np.square(out["cpu"] - out["dml"]).sum()
    sig = np.square(out["cpu"]).sum()
    snr = float("inf") if err == 0 else 10 * np.log10(sig / err)
    speedup = times["cpu"] / times["dml"]
    tell = "GPU-REAL" if speedup > 3 else ("SUSPECT-FALLBACK" if speedup < 1.5 else "CHECK")
    print(f"{fname:36s} cpu {times['cpu']*1e3:8.1f} ms  dml {times['dml']*1e3:8.1f} ms  "
          f"speedup {speedup:5.1f}x  SNR {snr:6.1f} dB  [{tell}]")
print("DML SMOKE DONE")
