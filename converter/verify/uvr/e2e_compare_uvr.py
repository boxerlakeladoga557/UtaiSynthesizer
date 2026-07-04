"""关卡2 compare: REF stems (python original) vs Rust harness stems. SNR + corr."""
import os
import sys

import numpy as np
import soundfile as sf

SCRATCH = os.path.dirname(os.path.abspath(__file__))


def load(p):
    x, sr = sf.read(p, dtype="float64")
    assert sr == 44100, p
    return x.T if x.ndim == 2 else x[None]


def snr_db(ref, x):
    n = min(ref.shape[1], x.shape[1])
    r, y = ref[:, :n], x[:, :n]
    err = ((r - y) ** 2).sum()
    sig = (r ** 2).sum()
    if err == 0:
        return float("inf")
    return 10 * np.log10(sig / err)


def corr(ref, x):
    n = min(ref.shape[1], x.shape[1])
    a, b = ref[:, :n].ravel(), x[:, :n].ravel()
    return float(np.corrcoef(a, b)[0, 1])


pairs = []
i = 1
while i + 1 < len(sys.argv):
    pairs.append((sys.argv[i], sys.argv[i + 1]))
    i += 2

print(f"{'pair':<70} {'SNR dB':>9} {'corr':>10}")
for ref_p, rust_p in pairs:
    ref = load(os.path.join(SCRATCH, ref_p) if not os.path.isabs(ref_p) else ref_p)
    rust = load(rust_p)
    print(f"{os.path.basename(ref_p)+' vs '+os.path.basename(rust_p):<70} "
          f"{snr_db(ref, rust):>9.2f} {corr(ref, rust):>10.6f}")
