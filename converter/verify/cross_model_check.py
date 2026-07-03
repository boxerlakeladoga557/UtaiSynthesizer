import numpy as np, struct, sys
def read_wav(path):
    with open(path, "rb") as f: data = f.read()
    pos = 12; raw = None; ch = 2
    while pos + 8 <= len(data):
        cid = data[pos:pos+4]; csz = struct.unpack("<I", data[pos+4:pos+8])[0]
        body = data[pos+8:pos+8+csz]
        if cid == b"fmt ": ch = struct.unpack("<HHIIHH", body[:16])[1]
        elif cid == b"data": raw = body
        pos += 8 + csz + (csz & 1)
    x = np.frombuffer(raw, dtype="<f4")
    return x[::ch]  # left channel
base = "D:/MyDev/Utai_v2-dev/data/cache/"
files = {
 "MDX23C_voc":  base+"05a0f7f4-1cd8-43bb-b486-b1cfe88934d3/rmr4olx3y0/separation-5d405a90/vocals.wav",
 "BS_voc":      base+"12cade3d-5393-4b09-a704-402a77566a51/rmr4qg0n82/separation-bda5186d/vocals.wav",
 "Mel_voc":     base+"12cade3d-5393-4b09-a704-402a77566a51/rmr4qg0n82/separation-4f1c8f54/vocals.wav",
 "rerun_voc":   base+"12cade3d-5393-4b09-a704-402a77566a51/rmr4qocpz3/separation-bda5186d/vocals.wav",
}
sig = {k: read_wav(v) for k, v in files.items()}
n = min(len(v) for v in sig.values())
for k in sig: sig[k] = sig[k][:n].astype(np.float64)
keys = list(sig)
print(f"len={n} samples ({n/44100:.1f}s)")
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a, b = sig[keys[i]], sig[keys[j]]
        corr = np.corrcoef(a, b)[0,1]
        snr = 10*np.log10(np.sum(a**2)/max(np.sum((a-b)**2),1e-12))
        print(f"{keys[i]:12s} vs {keys[j]:12s}  corr={corr:.4f}  SNR={snr:6.2f} dB")
