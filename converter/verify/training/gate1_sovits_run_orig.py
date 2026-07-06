"""SoVITS 关卡1（原版侧）：在我们的 training venv（torch 2.5.1 —— 与我方侧同一个
torch，隔离代码轴；RVC 关卡1 同款方法）里运行**未改动的**原版 so-vits-svc
train.py，CPU fp32 确定性。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_sovits_run_orig.py

原版 train.py 硬 assert CUDA + mp.spawn/DDP + 裸 .cuda(rank)。shim 只动执行环境、
不动数学（全部登记）：
  - CUDA_VISIBLE_DEVICES=-1（torch import 前）—— CPU fp32；库内教训：训练侧
    CUDA 无确定性背书（htdemucs 先例），bitwise 级对拍必须 CPU
  - 绕过 main()（其中只有 assert + mp.spawn），直接调 run(rank=0, n_gpus=1, hps)
    —— world_size=1 的 DDP 梯度 == 单进程
  - faiss 桩模块（原版 utils.py 顶层 import faiss，训练路径从不调用）
  - torch.Tensor.cuda / torch.nn.Module.cuda -> 恒等；torch.cuda.set_device -> no-op
  - train.DDP -> 剥掉 device_ids 的真 DDP（CPU 模块 device_ids 必须为 None）
  - gloo env:// 所需 MASTER_ADDR/PORT + USE_LIBUV=0（torch>=2.4 Windows TCPStore）
"""
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_LIBUV"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "8001"

SOVITS = r"D:\MyDev\so-vits-svc\so-vits-svc"
sys.path.insert(0, SOVITS)
sys.modules["faiss"] = types.ModuleType("faiss")  # top-level import in utils.py

os.chdir(SOVITS)  # get_hparams uses ./logs/<model>; filelists are absolute

import torch  # noqa: E402

torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
torch.cuda.set_device = lambda *a, **k: None

import train  # noqa: E402  (original file, unmodified)
import utils  # noqa: E402

_RealDDP = train.DDP


class ShimDDP(_RealDDP):
    def __init__(self, module, device_ids=None, **kw):
        super().__init__(module, **kw)


train.DDP = ShimDDP

GATE_CFG = r"D:\MyDev\TESTING\utai-v2-testing\gate1_sovits_config.json"


def main():
    sys.argv = ["train.py", "-c", GATE_CFG, "-m", "gate1_sovits"]
    hps = utils.get_hparams()
    train.run(0, 1, hps)
    print("GATE1 SOVITS ORIG SIDE DONE")


if __name__ == "__main__":
    main()
