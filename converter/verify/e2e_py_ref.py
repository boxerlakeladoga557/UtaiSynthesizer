# Python REFERENCE for the E2E pipeline parity test.
# Reproduces the ORIGINAL MSST demix() (D:\MyDev\ARCHIVE\MSSTRVCv2\MSST\utils\utils.py)
# EXACTLY, with the model forward = torch.stft -> ONNX mask -> complex multiply -> torch.istft
# (the procedure already proven equivalent to the original PyTorch BSRoformer in exp_e).
#
# Params (must match the deployed JSON the Rust pipeline uses):
#   chunk_size C=352800, num_overlap N=4, step=C//4, border=C-step, fade_size=C//10
#   n_fft=2048 hop=441 win=2048 periodic hann, normalized=False, center=True (torch default)
#
# Per-chunk outputs are cached to SCRATCH/py_chunks/ so a killed run resumes.
import os, struct, time
import numpy as np
import torch

SCRATCH = r"C:\Users\admin\AppData\Local\Temp\claude\D--MyDev-Utai-v2-dev\01a1a6a4-09b0-4416-8ddb-91c217acc8a8\scratchpad"
ONNX = r"D:\MyDev\Utai_v2-dev\data\models\msst\model_bs_roformer_ep_317_sdr_12.9755.onnx"
MIX = os.path.join(SCRATCH, "e2e_mix.wav")
CHUNK_DIR = os.path.join(SCRATCH, "py_chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

SR = 44100
C = 352800          # chunk_size
N = 4               # num_overlap
STEP = C // N       # 88200
BORDER = C - STEP   # 264600
FADE = C // 10      # 35280

torch.set_num_threads(8)

# ---------- WAV I/O (float32) ----------
def read_wav_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
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
    assert audio_fmt == 3 and bits == 32, (audio_fmt, bits)
    x = np.frombuffer(raw, dtype="<f4").reshape(-1, ch)
    return x, sr

def write_wav_f32(path, x, sr):  # x: (T, ch)
    x = np.ascontiguousarray(x.astype("<f4")); ch = x.shape[1]; raw = x.tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 3, ch, sr, sr*ch*4, ch*4, 32)
    hdr += b"data" + struct.pack("<I", len(raw))
    with open(path, "wb") as f:
        f.write(hdr + raw)

mix_np, sr = read_wav_f32(MIX)
assert sr == SR and mix_np.shape[1] == 2, (sr, mix_np.shape)
mix = torch.from_numpy(mix_np.T.copy())  # (2, T)
length_init = mix.shape[-1]
print(f"[audio] {length_init} samples ({length_init/SR:.2f}s) stereo", flush=True)

# ---------- demix() reflect border padding ----------
assert length_init > 2 * BORDER and BORDER > 0
mix_p = torch.nn.functional.pad(mix, (BORDER, BORDER), mode="reflect")
L = mix_p.shape[-1]
print(f"[pad] border={BORDER} padded length={L}", flush=True)

# ---------- trapezoid windows ----------
fadein = torch.linspace(0, 1, FADE)
fadeout = torch.linspace(1, 0, FADE)
window_start = torch.ones(C); window_start[-FADE:] *= fadeout   # first chunk: no fade-in
window_finish = torch.ones(C); window_finish[:FADE] *= fadein   # last chunk: no fade-out
window_middle = torch.ones(C)
window_middle[-FADE:] *= fadeout
window_middle[:FADE] *= fadein

# ---------- model forward: stft -> onnx mask -> complex mul -> istft ----------
import onnxruntime as ort
from einops import rearrange

hann = torch.hann_window(2048)  # periodic (torch default)
sess = None  # lazy: skip session load if all chunks cached
inp_name = None

def model_forward(part):  # part: (2, C) -> (2, C) vocals
    global sess, inp_name
    if sess is None:
        t0 = time.time()
        sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        print(f"[onnx] session loaded in {time.time()-t0:.1f}s input={inp_name}", flush=True)
    stft_c = torch.stft(part, n_fft=2048, hop_length=441, win_length=2048,
                        window=hann, normalized=False, return_complex=True)  # (2,1025,Tf) center=True default
    stft_repr = rearrange(torch.view_as_real(stft_c).unsqueeze(0), "b s f t c -> b (f s) t c")  # (1,2050,Tf,2)
    mask = torch.from_numpy(sess.run(None, {inp_name: stft_repr.numpy()})[0])  # (1,1,2050,Tf,2)
    sc = torch.view_as_complex(stft_repr.contiguous())      # (1,2050,Tf)
    mc = torch.view_as_complex(mask.contiguous())           # (1,1,2050,Tf)
    spec = rearrange(sc.unsqueeze(1) * mc, "b n (f s) t -> (b n s) f t", s=2)  # (2,1025,Tf) complex
    out = torch.istft(spec, n_fft=2048, hop_length=441, win_length=2048,
                      window=hann, normalized=False, return_complex=False, length=C)  # (2,C)
    return out

# ---------- demix() chunk loop (batch_size=1 flush semantics) ----------
result = torch.zeros((1, 2, L), dtype=torch.float32)
counter = torch.zeros((1, 2, L), dtype=torch.float32)

starts = list(range(0, L, STEP))
print(f"[demix] {len(starts)} chunks, step={STEP}", flush=True)

for idx, i in enumerate(starts):
    part = mix_p[:, i:i+C]
    length = part.shape[-1]
    if length < C:
        if length > C // 2 + 1:
            part = torch.nn.functional.pad(part, (0, C - length), mode="reflect")
            padmode = "reflect"
        else:
            part = torch.nn.functional.pad(part, (0, C - length, 0, 0), mode="constant", value=0)
            padmode = "zeros"
    else:
        padmode = "none"

    cache = os.path.join(CHUNK_DIR, f"chunk_{i}.npy")
    if os.path.exists(cache):
        x = torch.from_numpy(np.load(cache))
        print(f"[chunk {idx+1}/{len(starts)}] start={i} len={length} pad={padmode} CACHED", flush=True)
    else:
        t0 = time.time()
        x = model_forward(part)
        np.save(cache, x.numpy())
        print(f"[chunk {idx+1}/{len(starts)}] start={i} len={length} pad={padmode} "
              f"forward={time.time()-t0:.1f}s rms={float(x.pow(2).mean().sqrt()):.4f}", flush=True)

    # window selection: flush-time i in the original = i_start + STEP (batch_size=1)
    i_after = i + STEP
    if i == 0:
        window = window_start
    elif i_after >= L:
        window = window_finish
    else:
        window = window_middle

    result[..., i:i+length] += x[..., :length] * window[..., :length]
    counter[..., i:i+length] += window[..., :length]

est = (result / counter).numpy()
np.nan_to_num(est, copy=False, nan=0.0)
est = est[..., BORDER:-BORDER]  # strip borders
vocals = est[0]                 # (2, T)
assert vocals.shape[-1] == length_init

instr = mix.numpy() - vocals

vp = os.path.join(SCRATCH, "py_ref_vocals.wav")
ip = os.path.join(SCRATCH, "py_ref_instr.wav")
write_wav_f32(vp, vocals.T, SR)
write_wav_f32(ip, instr.T, SR)

def stats(name, x):
    print(f"[{name}] peak={np.abs(x).max():.4f} rms={np.sqrt((x**2).mean()):.4f}")

stats("py vocals", vocals)
stats("py instr ", instr)
print("saved", vp)
print("saved", ip)
print("DONE")
