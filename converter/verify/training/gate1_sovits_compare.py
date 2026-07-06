"""SoVITS 关卡1 对拍：逐 step loss 轨迹 —— 原版 so-vits train.py vs 我们 vendored
sovits/train.py。

    training/.venv/Scripts/python.exe converter/verify/training/gate1_sovits_compare.py

两侧同 torch(2.5.1)/同数据/同 filelist 行序/同 seed(1234)/同底模(G_0/D_0 vec768)/
fp32 CPU（确定性）。原版侧取 tensorboard events（全精度），我方侧取协议 JSONL。
与 RVC 关卡1 的差异：so-vits **没有** mel>75/kl>9 的显示夹取（TB 写的就是原始值），
且多一个 loss/g/lf0 分量（自动 f0 预测，模板默认开）。
通过线 max 相对差 ≤1e-3（结构性移植错误 = O(0.1~1)；实测期望 ~1e-6 级）。
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ORIG_TB_DIR = r"D:\MyDev\so-vits-svc\so-vits-svc\logs\gate1_sovits"
OURS_JSONL = r"D:\MyDev\TESTING\utai-v2-testing\gate1_sovits_ours_steps.jsonl"

PAIRS = [  # (TB tag, ours key) — NO clamps (upstream writes raw values)
    ("loss/g/total", "g_total"),
    ("loss/d/total", "d_total"),
    ("loss/g/fm", "fm"),
    ("loss/g/mel", "mel"),
    ("loss/g/kl", "kl"),
    ("loss/g/lf0", "lf0"),
]


def load_orig():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(ORIG_TB_DIR, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag, _ in PAIRS:
        out[tag] = {e.step: e.value for e in acc.Scalars(tag)}
    return out


def load_ours():
    steps = {}
    with open(OURS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "step" and "g_total" in obj.get("losses", {}):
                steps[obj["step"]] = obj["losses"]
    return steps


def main():
    orig = load_orig()
    ours = load_ours()
    common = sorted(set(orig["loss/g/total"]) & set(ours))
    print(f"orig steps={len(orig['loss/g/total'])} ours steps={len(ours)} common={len(common)}")
    if len(common) < 10:
        print("GATE1 SOVITS: FAIL — 对齐步数不足")
        sys.exit(1)

    failures = []
    for tag, key in PAIRS:
        rels = []
        for s in common:
            a = orig[tag][s]
            b = ours[s][key]
            denom = max(abs(a), 1e-6)
            rels.append(abs(a - b) / denom)
        rels = np.array(rels)
        worst = common[int(rels.argmax())]
        ok = rels.max() <= 1e-3
        print(
            f"[{'PASS' if ok else 'FAIL'}] {tag:>14} vs {key:>7}: max_rel={rels.max():.3e} @step {worst}, mean_rel={rels.mean():.3e}"
        )
        if not ok:
            failures.append(tag)

    print()
    if failures:
        print("GATE1 SOVITS: FAIL —", ", ".join(failures))
        sys.exit(1)
    print(f"GATE1 SOVITS: ALL PASS ({len(common)} steps compared)")


if __name__ == "__main__":
    main()
