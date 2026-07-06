"""SoVITS 关卡0 准备：用我们的 slicer（rvc/slicer2.py = openvpi slicer-v2，默认
参数）把 gate 数据集切成 float32 wav 切片，作为**双方共同的输入**。

上游 so-vits-svc 没有切片器（README 要求用户用同一个 openvpi 工具自切），所以
切片轴不在本关卡对拍范围内 —— slicer2 本体已在 RVC 关卡0 对原版逐位验证过。
float32 写盘保证两侧读入完全一致（librosa.load 对 float wav 无量化）。

    training/.venv/Scripts/python.exe converter/verify/training/gate0_sovits_prepare.py
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "training"))

TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
DATASET = os.path.join(TESTING, "gate_dataset")
SLICES = os.path.join(TESTING, "sovits_slices", "gate")

import numpy as np
from scipy.io import wavfile

from utai_train.sovits.preprocess import _decode
from utai_train.rvc.slicer2 import Slicer


def main():
    os.makedirs(SLICES, exist_ok=True)
    for n in os.listdir(SLICES):
        os.remove(os.path.join(SLICES, n))
    ffmpeg = r"D:\MyDev\RVC\RVC20240604Nvidia\ffmpeg.exe"
    names = sorted(os.listdir(DATASET))
    count = 0
    for i, name in enumerate(names):
        wav, sr = _decode(os.path.join(DATASET, name), ffmpeg)
        slicer = Slicer(sr=sr)  # openvpi defaults
        for idx1, chunk in enumerate(slicer.slice(wav)):
            out = os.path.join(SLICES, "%03d_%03d.wav" % (i, idx1))
            wavfile.write(out, sr, chunk.astype(np.float32))
            count += 1
    print("sliced %d files -> %d slices at %s" % (len(names), count, SLICES))


if __name__ == "__main__":
    main()
