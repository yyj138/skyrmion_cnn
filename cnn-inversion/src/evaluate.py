# -*- coding: utf-8 -*-
"""
评估与绘图 —— 复刻论文 Figure 2g/2h, 3c/3d, 3f/3g, 4f/4g, 5b-e 的图。

论文依据：
- 训练/验证 loss 随 epoch 曲线（Fig.2g, 3c, 3f, 4f, 5b）
- 分类混淆矩阵（Fig.2h, 5e），类别轴取数据集实际 eta 类别（11 类）
- 回归画预测值-真值散点 + 1:1 参考线 + 每点标准差阴影
  （"the crosses represent the 1:1 relationship. The blue shaded area
    represents the standard deviation for each point." Fig.3d/3g 图注, p.5）
- 指标：分类平均准确率（"average accuracy is approximately 0.96", p.3）；
  回归决定系数 R^2（"coefficient of determination of approximately
  0.94 / 0.96 / 0.999", p.5-6）

测试集从 train.py 保存的 meta.json 精确复现（含 A 的划分表或自划分的
文件名清单、类别表、回归标签标准化统计量），预测反变换回真实物理量后
再计算指标与绘图。

用法:
    python evaluate.py --run runs/pitch   (meta.json 已含 csv/image_dir)
    python evaluate.py --run runs/energy
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from data import load_dataframe, _make_labels, make_dataset

BATCH_SIZE = 32  # 与 train.py 保持一致


def plot_loss_curves(log_csv, out_path):
    """Fig.2g/3c/3f/4f/5b 样式：train/val loss 随 epoch。"""
    log = pd.read_csv(log_csv)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(log["epoch"], log["loss"], "o-", color="tab:blue", label="Training Loss")
    ax.plot(log["epoch"], log["val_loss"], "x--", color="tab:green", label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def collect_test_predictions(model, meta):
    """按 meta.json 复现测试集并返回 (y_true, y_pred)，回归任务反标准化。"""
    paths, df = load_dataframe(meta["csv"], meta["image_dir"])
    task = meta["task"]
    # 用 csv 的类别顺序重建标签（与训练时一致）
    if task == "pitch":
        classes = meta["classes"]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        labels = df["eta"].map(class_to_idx).to_numpy(dtype=np.int32)
    else:
        labels = _make_labels(df, task)

    test_files = set(meta["test_filenames"])
    te = np.array([i for i, fn in enumerate(df["filename"]) if fn in test_files])
    test_ds = make_dataset(paths[te], labels[te], BATCH_SIZE, training=False)
    preds = model.predict(test_ds)

    y_true = labels[te].astype(np.float64)
    if task == "pitch":
        # softmax 输出 (N,11) -> 类别索引
        y_pred = preds.argmax(axis=1).astype(np.float64)
        return y_true.ravel(), y_pred
    y_true = y_true.ravel()
    y_pred = preds.ravel().astype(np.float64)
    if task in ("voltage", "energy"):
        # 反 z-score 变换回真实物理量（ASSUMPTION 3.11，见 train.py 文件头）
        y_pred = y_pred * meta["label_std"] + meta["label_mean"]
    return y_true, y_pred


def eval_classification(model, meta, out_dir):
    """Fig.2h/5e：混淆矩阵 + 平均准确率。"""
    y_true, y_pred = collect_test_predictions(model, meta)
    y_hat = y_pred.astype(int)
    classes = meta["classes"]
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true.astype(int), y_hat):
        cm[t, p] += 1
    acc = (y_true == y_hat).mean()  # "average accuracy" (p.3)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n), classes)
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("Predicted Pitch")
    ax.set_ylabel("True Pitch")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)
    return {"accuracy": float(acc)}


def eval_regression(model, meta, out_dir):
    """Fig.3d/3g/4g/5c/5d：预测-真值散点 + 1:1 线 + 每点标准差阴影。"""
    task = meta["task"]
    y_true, y_pred = collect_test_predictions(model, meta)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot  # "coefficient of determination" (p.5-6)
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    # 论文 Fig.3d/3g: "The blue shaded area represents the standard deviation
    # for each point." —— 对相同真值的预测画 mean(pred) ± std(pred)。这样不会
    # 在单 toron 的 4 个测试样本上强制拆出超过样本数的空分箱。
    order = np.argsort(y_true)
    x, y = y_true[order], y_pred[order]
    bx = np.unique(x)
    grouped_pred = [y[x == value] for value in bx]
    by = np.array([values.mean() for values in grouped_pred])
    bs = np.array([values.std() for values in grouped_pred])

    unit = "Voltage" if task == "voltage" else "Free Energy"
    fig, ax = plt.subplots(figsize=(5, 4))
    line_min = min(x.min(), y.min())
    line_max = max(x.max(), y.max())
    ax.plot([line_min, line_max], [line_min, line_max], "--",
            color="black", linewidth=1.2, label="1:1 Reference")
    ax.fill_between(bx, by - bs, by + bs, color="tab:blue", alpha=0.2)
    ax.plot(bx, by, "o-", color="tab:blue", markersize=3,
            linewidth=1, label="Mean Prediction")
    ax.scatter(x, y, marker="x", color="tab:green", alpha=0.55,
               label="Test Samples")
    ax.set_xlabel(f"True {unit}")
    ax.set_ylabel(f"Predicted {unit}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_true.png", dpi=200)
    plt.close(fig)
    return {"R2": float(r2), "MSE": float(mse), "MAE": float(mae)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train.py 的输出目录")
    args = ap.parse_args()

    run = Path(args.run)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    model = keras.models.load_model(run / "model.keras")

    plot_loss_curves(run / "training_log.csv", run / "loss_curve.png")
    if meta["task"] == "pitch":
        metrics = eval_classification(model, meta, run)
    else:
        metrics = eval_regression(model, meta, run)

    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
