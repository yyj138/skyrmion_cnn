# -*- coding: utf-8 -*-
"""
冒烟测试：用合成数据把整条流水线（读数据->增强->训练->评估->出图）跑通。

目的不是得到有意义的指标，而是验证代码在真实数据到来前即可运行。
合成数据仅用于自测，不会污染正式实验。

运行: python smoke_test.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).parent
TMP = ROOT / "_smoke"
N_IMAGES = 120  # 小样本即可验证流程


def make_synthetic():
    """生成 300x300 灰度图 + 论文格式的 labels.csv。"""
    (TMP / "imgs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    classes = [23, 24, 25, 26, 28, 30, 32, 34, 36, 38, 40]  # 论文 Fig.2h 的 11 类
    rows = []
    for i in range(N_IMAGES):
        eta = classes[i % len(classes)]
        U = round(rng.uniform(0, 3.5), 3)
        fe = rng.uniform(8.9e6, 9.3e6)
        arr = (rng.random((300, 300)) * 255).astype(np.uint8)
        name = f"img_{i:04d}.png"
        Image.fromarray(arr, mode="L").save(TMP / "imgs" / name)
        rows.append({"filename": name, "eta": eta, "U": U, "free_energy": fe})
    pd.DataFrame(rows).to_csv(TMP / "labels.csv", index=False)
    print(f"synthetic data: {N_IMAGES} images -> {TMP}")


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    make_synthetic()
    py = sys.executable
    csv = str(TMP / "labels.csv")
    imgs = str(TMP / "imgs")

    # 三个任务各训 2 个 epoch，仅验证代码可运行
    for task in ["pitch", "voltage", "energy"]:
        run([py, "train.py", "--task", task, "--csv", csv, "--images", imgs,
             "--out", str(TMP / "runs" / task), "--epochs", "2", "--batch-size", "16"])
        run([py, "evaluate.py", "--run", str(TMP / "runs" / task)])

    # 校验产物齐全
    expected = []
    for task in ["pitch", "voltage", "energy"]:
        r = TMP / "runs" / task
        expected += [r / "model.keras", r / "training_log.csv", r / "meta.json",
                     r / "metrics.json", r / "loss_curve.png"]
    expected.append(TMP / "runs" / "pitch" / "confusion_matrix.png")
    expected.append(TMP / "runs" / "voltage" / "pred_vs_true.png")
    expected.append(TMP / "runs" / "energy" / "pred_vs_true.png")
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        print("MISSING:", missing)
        sys.exit(1)
    print("\nSMOKE TEST PASSED - all artifacts produced.")


if __name__ == "__main__":
    if TMP.exists():
        shutil.rmtree(TMP)
    main()
