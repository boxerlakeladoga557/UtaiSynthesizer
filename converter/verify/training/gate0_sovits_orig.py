"""SoVITS 关卡0（原版侧，ground truth）：在「原版时代环境」= RVC 整合包 runtime
（python3.9 + torch 2.0.0 + torchaudio 2.0.1+cpu + fairseq 0.12.2 + librosa 0.9.1
—— 与 so-vits-svc requirements 钉的 librosa==0.9.1 / fairseq==0.12.2 同代）里，
runpy 原样执行 so-vits-svc 4.1-Stable 的三个预处理脚本 + vencoder ContentVec
双维度 oracle。全程 CPU fp32（CUDA_VISIBLE_DEVICES=-1 在 torch import 前设置）。

运行（cwd 任意）：
    D:\\MyDev\\RVC\\RVC20240604Nvidia\\runtime\\python.exe ^
        converter\\verify\\training\\gate0_sovits_orig.py

Harness 补丁（零数值影响，全部登记）：
  - loguru 桩（RVC runtime 没有 loguru；纯日志）
  - concurrent.futures.ProcessPoolExecutor -> 串行内联执行器（runpy 的 __main__
    命名空间没法被 spawn 子进程 unpickle；逐文件数学不变——与我方管线的串行化
    偏离同类）
  - configs/config.json + configs/diffusion.yaml 运行前快照、运行后恢复
    （flist 脚本硬编码写 repo 内路径；恢复保证参照 repo 保持原样）
  - pretrain/rmvpe.pt 若缺失则从本项目 aux 拷入（上游本来就要求用户自放；
    双方使用同一份权重文件，对拍只剩代码轴）
"""
import os
import shutil
import sys

# ---- CPU fp32: before ANY torch import ----
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SOVITS = r"D:\MyDev\so-vits-svc\so-vits-svc"
UTAI = r"D:\MyDev\Utai_v2-dev"
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
SHIM = os.path.join(TESTING, "pyshim")
SLICES_ROOT = os.path.join(TESTING, "sovits_slices")  # contains speaker dir "gate"
ORIG = os.path.join(TESTING, "sovits_orig")
D44K = os.path.join(ORIG, "dataset44k")
ORACLE = os.path.join(ORIG, "oracle")

sys.path.insert(0, SHIM)
sys.path.insert(0, SOVITS)


def install_inline_executor():
    import concurrent.futures as cf

    class _Future:
        def __init__(self, fn, *a, **kw):
            self._exc = None
            try:
                self._res = fn(*a, **kw)
            except BaseException as e:  # surface in .result() like a real future
                self._exc = e
                self._res = None

        def result(self, timeout=None):
            if self._exc is not None:
                raise self._exc
            return self._res

        def done(self):
            return True

    class InlineExecutor:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *a, **kw):
            return _Future(fn, *a, **kw)

    cf.ProcessPoolExecutor = InlineExecutor
    # as_completed over already-done futures
    orig_as_completed = cf.as_completed

    def _as_completed(fs, timeout=None):
        return iter(list(fs))

    cf.as_completed = _as_completed


def run_script(name, argv):
    import runpy

    sys.argv = [name] + argv
    print("== running %s %s" % (name, " ".join(argv)))
    runpy.run_path(os.path.join(SOVITS, name), run_name="__main__")


def main():
    os.chdir(SOVITS)
    install_inline_executor()

    # rmvpe.pt: both sides must use the same weight file. NB the so-vits rmvpe is
    # the yxlllc/RMVPE fork (E2E0, {'model': sd}) — NOT interchangeable with the
    # RVC-lineage aux/rmvpe.pt (raw sd, no unet.tf.* layers)
    rmvpe_dst = os.path.join(SOVITS, "pretrain", "rmvpe.pt")
    if not os.path.exists(rmvpe_dst):
        shutil.copyfile(
            os.path.join(UTAI, "data", "models", "training", "sovits", "rmvpe.pt"),
            rmvpe_dst,
        )
        print("copied rmvpe.pt into pretrain/")

    # snapshot repo configs (flist writes them at hardcoded paths)
    cfg_dir = os.path.join(SOVITS, "configs")
    snaps = {}
    for n in ("config.json", "diffusion.yaml"):
        p = os.path.join(cfg_dir, n)
        snaps[n] = open(p, "rb").read() if os.path.exists(p) else None

    try:
        if os.path.isdir(D44K):
            shutil.rmtree(D44K)
        os.makedirs(D44K, exist_ok=True)
        flists = os.path.join(ORIG, "filelists")
        os.makedirs(flists, exist_ok=True)

        # ① resample.py — upstream defaults (loudnorm ON)
        run_script("resample.py", ["--in_dir", SLICES_ROOT, "--out_dir2", D44K])

        # ② flist + config (vec768l12 + vol_aug, matching the gate's ours-side run)
        run_script(
            "preprocess_flist_config.py",
            [
                "--source_dir", D44K,
                "--train_list", os.path.join(flists, "train.txt"),
                "--val_list", os.path.join(flists, "val.txt"),
                "--speech_encoder", "vec768l12",
                "--vol_aug",
            ],
        )
        # keep the generated config for the semantic comparison
        shutil.copyfile(os.path.join(cfg_dir, "config.json"), os.path.join(ORIG, "config.json"))

        # ③ hubert/f0/spec/vol — CPU (env already hides the GPU), rmvpe default
        run_script(
            "preprocess_hubert_f0.py",
            ["-d", "cpu", "--in_dir", D44K, "--num_processes", "1"],
        )

        # ④ ContentVec oracle, BOTH dims, on the era-authentic 16k resample of the
        #    ORIGINAL 44k wavs (librosa 0.9.1) — saved so the compare script can
        #    feed our ONNX extractors the IDENTICAL 16k input (isolates the
        #    extractor axis from the resampler axis)
        import librosa
        import numpy as np
        import torch

        from vencoder.ContentVec256L9 import ContentVec256L9
        from vencoder.ContentVec768L12 import ContentVec768L12

        os.makedirs(ORACLE, exist_ok=True)
        enc768 = ContentVec768L12(device="cpu")
        enc256 = ContentVec256L9(device="cpu")
        spk_dir = os.path.join(D44K, "gate")
        wavs = sorted(n for n in os.listdir(spk_dir) if n.endswith(".wav"))
        for n in wavs:
            wav, _sr = librosa.load(os.path.join(spk_dir, n), sr=44100)
            wav16k = librosa.resample(wav, orig_sr=44100, target_sr=16000)
            np.save(os.path.join(ORACLE, n + ".wav16k.npy"), wav16k.astype(np.float32))
            t = torch.from_numpy(wav16k)
            with torch.no_grad():
                c768 = enc768.encoder(t)  # [1, 768, T]
                c256 = enc256.encoder(t)  # [1, 256, T]
            np.save(os.path.join(ORACLE, n + ".venc768.npy"), c768.cpu().numpy())
            np.save(os.path.join(ORACLE, n + ".venc256.npy"), c256.cpu().numpy())
        print("oracle saved for %d wavs" % len(wavs))
    finally:
        for n, data in snaps.items():
            p = os.path.join(cfg_dir, n)
            if data is None:
                if os.path.exists(p):
                    os.remove(p)
            else:
                with open(p, "wb") as f:
                    f.write(data)
        print("repo configs restored")

    print("GATE0 SOVITS ORIG SIDE DONE")


if __name__ == "__main__":
    main()
