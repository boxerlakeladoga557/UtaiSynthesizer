"""SoVITS 关卡1 布置：两侧共用关卡0 我方侧的预处理产物（sovits_ours 的 filelists
指向绝对路径，双方直接读同一批文件 —— batch 组成由 filelist 行序 + seed 决定）。

  原版侧 model_dir = <so-vits repo>/logs/gate1_sovits（train.py 硬编码 ./logs/<name>）
  我方侧 exp_dir   = TESTING/gate1_sovits_ours

两侧各放同一份 config（log_interval=1 逐步记录）+ 同一对底模 G_0/D_0。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_sovits_prepare.py
"""
import json
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TESTING = r"D:\MyDev\TESTING\utai-v2-testing"
SOVITS = r"D:\MyDev\so-vits-svc\so-vits-svc"
OURS_G0 = os.path.join(REPO, "data", "models", "training", "sovits", "vec768")

SRC_CFG = os.path.join(TESTING, "sovits_ours", "config.json")
ORIG_DIR = os.path.join(SOVITS, "logs", "gate1_sovits")
OURS_DIR = os.path.join(TESTING, "gate1_sovits_ours")
GATE_CFG = os.path.join(TESTING, "gate1_sovits_config.json")


def main():
    with open(SRC_CFG, encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["train"]["all_in_mem"] is True, "gate 需要 all_in_mem（双侧 num_workers=0）"
    assert cfg["train"]["fp16_run"] is False
    assert cfg["train"]["epochs"] == 2 and cfg["train"]["batch_size"] == 4
    cfg["train"]["log_interval"] = 1  # 原版侧 TB 逐 step 记录
    with open(GATE_CFG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    for d in (ORIG_DIR, OURS_DIR):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        shutil.copyfile(os.path.join(OURS_G0, "G_0.pth"), os.path.join(d, "G_0.pth"))
        shutil.copyfile(os.path.join(OURS_G0, "D_0.pth"), os.path.join(d, "D_0.pth"))
    # ours side reads exp_dir/config.json
    shutil.copyfile(GATE_CFG, os.path.join(OURS_DIR, "config.json"))

    # sanity: filelist entries must exist (both sides will read them)
    missing = []
    for lst in ("training_files", "validation_files"):
        for line in open(cfg["data"][lst], encoding="utf-8"):
            p = line.strip()
            if p and not os.path.exists(p):
                missing.append(p)
    assert not missing, "filelist 路径缺失: %s" % missing[:3]
    print("prepared:", ORIG_DIR, "and", OURS_DIR)
    sys.exit(0)


if __name__ == "__main__":
    main()
