# Ported from so-vits-svc 4.1-Stable preprocess_hubert_f0.py process_one().
# Per-file products (all skip-if-exists, same names/layout as upstream):
#   <wav>.soft.pt   ContentVec features, torch tensor [1, dim, T]
#   <wav>.f0.npy    np object array (f0, uv) from RMVPEF0Predictor (thr 0.05)
#   <stem>.spec.pt  linear spectrogram_torch 2048/512/2048 center=False, [1025, T]
#   <wav>.vol.npy   Volume_Extractor(hop) frame RMS — only when vol_embedding
# Deviations (deliberate):
#   - ContentVec via the project's ONNX extractors instead of fairseq
#     (aux/contentvec_768l12.onnx | contentvec_256l9.onnx — gate-verified against
#     real fairseq: max 7.7e-4 / cos 1-1e-9; kills the fairseq dependency and
#     guarantees training feature space == inference feature space), CPU EP like
#     the RVC trainer
#   - the RMVPE predictor is constructed ONCE per run (upstream re-loads the
#     180MB checkpoint per file inside process_one); math identical
#   - sequential over sorted files instead of shuffle + ProcessPoolExecutor
#     (per-file products are independent; upstream's spawn workers each loading
#     an encoder copy is a Windows minefield)
#   - --use_diff mel/aug_mel products are NOT produced here (the shallow
#     diffusion trainer is a later backend; its products slot into this same
#     stage incrementally)
import logging
import os
import traceback

import librosa
import numpy as np
import torch

from .f0.RMVPEF0Predictor import RMVPEF0Predictor
from .modules.mel_processing import spectrogram_torch
from .utils import Volume_Extractor

logger = logging.getLogger(__name__)

# ContentVec conv frontend needs at least 400 samples @16k (S35 aux contract)
MIN_SAMPLES_16K = 400


def extract_all(
    dataset_44k_dir,
    hps,               # HParams from the workspace config.json
    contentvec_onnx,
    rmvpe_pt,
    device,            # "cuda" | "cpu" (for the f0 predictor)
    reporter,
    stop,
):
    import onnxruntime as ort

    sampling_rate = hps.data.sampling_rate
    hop_length = hps.data.hop_length
    vol_embedding = bool(hps.model.vol_embedding)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(contentvec_onnx, so, providers=["CPUExecutionProvider"])

    f0_predictor = RMVPEF0Predictor(
        hop_length=hop_length,
        sampling_rate=sampling_rate,
        dtype=torch.float32,
        device=device,
        threshold=0.05,
        model_path=rmvpe_pt,
    )
    volume_extractor = Volume_Extractor(hop_length)

    filenames = []
    for spk in sorted(os.listdir(dataset_44k_dir)):
        spk_dir = os.path.join(dataset_44k_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        for name in sorted(os.listdir(spk_dir)):
            if name.endswith(".wav"):
                filenames.append(os.path.join(spk_dir, name))

    # fail-fast on ANY file: the filelists are already written — a tolerated
    # failure here would surface 800 steps later as a raw FileNotFoundError
    # inside a DataLoader worker (upstream preprocess aborts loudly too)
    for n, filename in enumerate(filenames):
        stop.check()
        reporter.stage(
            "extract", done=n, total=len(filenames), message=os.path.basename(filename)
        )
        try:
            _process_one(
                filename,
                sess,
                f0_predictor,
                volume_extractor,
                sampling_rate,
                hps,
                vol_embedding,
            )
        except Exception:
            logger.error("extract failed for %s\n%s", filename, traceback.format_exc())
            raise RuntimeError(
                "切片 %s 特征提取失败（详见日志）" % os.path.basename(filename)
            )
    reporter.stage("extract", done=len(filenames), total=len(filenames))


def _atomic_torch_save(obj, path):
    # products live under skip-if-exists caches: a kill mid-write must not leave
    # a truncated file that every later run treats as a valid cache hit
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_np_save(arr, path):
    tmp = path + ".tmp.npy"  # np.save appends .npy to extension-less paths
    np.save(tmp, arr)
    os.replace(tmp, path)


def _process_one(filename, sess, f0_predictor, volume_extractor, sampling_rate, hps, vol_embedding):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)

    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        if len(wav16k) < MIN_SAMPLES_16K:
            raise RuntimeError("切片过短（<400 采样点 @16k），无法提取特征")
        feats = sess.run(
            ["features"], {"waveform": wav16k.astype(np.float32)[None, :]}
        )[0][0]  # [T, dim]
        if np.isnan(feats).sum() > 0:
            raise RuntimeError("ContentVec 特征包含 NaN")
        # upstream layout: [1, dim, T] cpu tensor
        c = torch.from_numpy(np.ascontiguousarray(feats.T))[None, :, :].float()
        _atomic_torch_save(c, soft_path)

    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0, uv = f0_predictor.compute_f0_uv(wav)
        _atomic_np_save(np.asanyarray((f0, uv), dtype=object), f0_path)

    spec_path = filename.replace(".wav", ".spec.pt")
    if not os.path.exists(spec_path):
        if sr != hps.data.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(sr, hps.data.sampling_rate)
            )
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        _atomic_torch_save(spec, spec_path)

    if vol_embedding:
        volume_path = filename + ".vol.npy"
        if not os.path.exists(volume_path):
            volume = volume_extractor.extract(audio_norm)
            _atomic_np_save(volume.to("cpu").numpy(), volume_path)
