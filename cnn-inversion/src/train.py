# -*- coding: utf-8 -*-
"""
训练入口 —— 优化器/损失/早停全部按论文设置。

论文依据：
- Adam，学习率 0.001：
    "The network parameters have been optimised using the Adam algorithm50
     (learning rate of 0.001)" (p.3)
- 分类损失 categorical cross-entropy：
    "the loss function is categorical cross-entropy (commonly used in
     multinomial classification)" (p.3)
- 回归损失 MSE：
    "We have trained this network by optimising the mean square error
     (loss function)" (p.5)
- 早停，patience=5，作用于所有卷积层与全连接层：
    "we have applied an early stopping regularisation procedure (with
     patience set to 5 epochs) over all convolutional and fully connected
     layers" (p.3)

ASSUMPTION（论文未写明，见 PAPER_BASIS.md 3.7/3.8/3.11）：
- batch_size=32；最大 epoch 数 200（早停会提前终止）；
- 早停监控 val_loss、restore_best_weights=True（论文只说 patience=5）；
- 回归标签 z-score 标准化（用训练集统计量），预测时反变换回真实物理量。
  论文未提及标签标准化；数值上必要（free_energy ~1e-13 J，不标准化
  MSE ~1e-26，float32 梯度过小无法收敛，成员 A 的 dataset_readme.md 亦建议）。

成员 A 数据集用法（推荐，使用 A 交付的划分表保证两端一致）：
    python train.py --task pitch --csv labels/labels_single.csv \
        --images pom_images \
        --split-csv splits/split_pitch_classification_single.csv \
        --out runs/pitch

无划分表时（自划分，比例仍按论文）：
    python train.py --task pitch --csv labels.csv --images ./imgs --out runs/pitch
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models import build_pitch_classifier, build_voltage_regressor, build_energy_regressor
from data import build_data, make_dataset

# --- 论文固定超参 ---
LEARNING_RATE = 0.001        # "learning rate of 0.001" (p.3)
EARLY_STOP_PATIENCE = 5      # "patience set to 5 epochs" (p.3)
# --- ASSUMPTION ---
BATCH_SIZE = 32              # 见 PAPER_BASIS.md 3.7
MAX_EPOCHS = 200             # 见 PAPER_BASIS.md 3.8


def compile_model(model, task):
    """按任务设置优化器/损失（依据见文件头）。"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy" if task == "pitch" else "mse",
        metrics=["accuracy"] if task == "pitch" else ["mae"],
    )
    return model


def train_one(task, csv_path, image_dir, out_dir, seed, batch_size=BATCH_SIZE,
              max_epochs=MAX_EPOCHS, split_task=None, split_csv=None,
              patience=EARLY_STOP_PATIENCE, det_aug=True,
              fixed_det_aug=False, augment_probability=1.0):
    """训练单个 (任务, 模型) 组合，保存权重与训练历史。

    split_csv: 成员 A 的划分表路径（含 split 列），优先使用；保证 A/B 两端
    划分一致可复现（A 的 dataset_readme.md 要求勿重新随机划分）。
    split_task: 自划分时的规则名，mixed 模式传 "mixed"（Fig.5b 图注）。
    patience: 早停 patience。论文为 5（p.3，默认值）；成员 A 的稀疏纹理
    数据上学习较慢，实测需 30 才能收敛（2026-09-08 对照实验：patience=5
    在 epoch 13 停住、test acc 0.077；patience=30 test acc 1.000，
    详见 PAPER_BASIS.md 3.12）。该偏差属数据驱动调整，报告中应说明。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 固定全局随机种子：权重初始化 + 增强 + shuffle 可复现。
    # 注意：小数据集（88 张）上不同 seed 的测试表现波动很大（实测
    # seed 间 test acc 从 0.62 到 1.00 不等，2026-09-09），报告中应
    # 用多种子 mean±std 或声明所用 seed。
    tf.keras.utils.set_random_seed(seed)

    if task == "pitch":
        model = build_pitch_classifier()
    elif task == "voltage":
        model = build_voltage_regressor()
    else:  # energy / mixed-energy
        model = build_energy_regressor()

    paths, labels, (tr, va, te), meta = build_data(
        csv_path, image_dir, task, split_task=split_task,
        split_csv=split_csv, seed=seed)
    print(f"[{task}] total={meta['n_total']} train={meta['n_train']} "
          f"val={meta['n_val']} test={meta['n_test']}"
          + (f" (split: {Path(split_csv).name})" if split_csv else " (自划分)"))

    # --- 回归标签 z-score 标准化（ASSUMPTION，见文件头 3.11） ---
    if task in ("voltage", "energy"):
        y_train = labels[tr].astype(np.float64)
        label_mean = float(y_train.mean())
        label_std = float(y_train.std())
        labels = ((labels.astype(np.float64) - label_mean) / label_std).astype(np.float32)
        meta["label_mean"] = label_mean
        meta["label_std"] = label_std
        print(f"[{task}] 标签标准化: mean={label_mean:.6g} std={label_std:.6g}")

    train_ds = make_dataset(paths[tr], labels[tr], batch_size, training=True,
                            deterministic=det_aug, seed=seed,
                            fixed_deterministic=fixed_det_aug,
                            augment_probability=augment_probability)
    val_ds = make_dataset(paths[va], labels[va], batch_size, training=False)

    # 分类标签转 one-hot 以配合 categorical_crossentropy
    if task == "pitch":
        n_classes = len(meta["classes"])
        def to_onehot(x, y):
            return x, tf.one_hot(y, depth=n_classes)
        train_ds = train_ds.map(to_onehot)
        val_ds = val_ds.map(to_onehot)

    model = compile_model(model, task)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,  # ASSUMPTION，见文件头
        ),
        keras.callbacks.CSVLogger(out / "training_log.csv"),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
    )
    model.save(out / "model.keras")

    # meta 已含 filenames/划分/类别/标签统计，供 evaluate.py 精确复现测试集
    meta["task"] = task
    meta["csv"] = str(csv_path)
    meta["image_dir"] = str(image_dir)
    meta["stopped_epoch"] = len(history.history["loss"])
    meta["det_aug"] = bool(det_aug)   # 是否启用确定性增强（复现追溯用）
    meta["fixed_det_aug"] = bool(fixed_det_aug)
    meta["augment_probability"] = float(augment_probability)
    meta["epochs_run"] = max_epochs
    meta.pop("filenames", None)  # 全量列表冗余，删去减小体积
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["pitch", "voltage", "energy", "mixed"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-csv", default=None,
                    help="成员 A 的划分表（含 split 列），优先使用")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE,
                    help="早停 patience，论文为 5；A 的稀疏纹理数据建议 30（见 PAPER_BASIS 3.12）")
    aug_group = ap.add_mutually_exclusive_group()
    aug_group.add_argument(
        "--det-aug", dest="det_aug", action="store_true",
        help="使用可复现且逐 epoch 更新的正式增强路径（默认）")
    aug_group.add_argument(
        "--legacy-random-aug", dest="det_aug", action="store_false",
        help="使用历史非确定增强路径，仅用于复核旧实验")
    ap.add_argument(
        "--fixed-det-aug", action="store_true",
        help="使用 mixed_pitch_v2 的固定逐样本增强，仅用于精确复核该历史成功运行")
    ap.add_argument(
        "--aug-prob", type=float, default=1.0,
        help="动态增强应用概率（0~1）；混合分类正式运行使用 0.1 以保留原图分布")
    ap.set_defaults(det_aug=True)
    args = ap.parse_args()
    if not 0.0 <= args.aug_prob <= 1.0:
        ap.error("--aug-prob 必须在 0 到 1 之间")

    if args.task == "mixed":
        # Fig.5: 混合数据集上分别训练 free_energy / voltage / pitch 三个模型，
        # 划分规则统一（Fig.5b 图注）。
        for sub in ["energy", "voltage", "pitch"]:
            print(f"\n===== mixed / {sub} =====")
            train_one(sub, args.csv, args.images, Path(args.out) / sub, args.seed,
                      batch_size=args.batch_size, max_epochs=args.epochs,
                      split_task="mixed", split_csv=args.split_csv,
                      patience=args.patience, det_aug=args.det_aug,
                      fixed_det_aug=args.fixed_det_aug,
                      augment_probability=args.aug_prob)
    else:
        train_one(args.task, args.csv, args.images, args.out, args.seed,
                  batch_size=args.batch_size, max_epochs=args.epochs,
                  split_csv=args.split_csv, patience=args.patience,
                  det_aug=args.det_aug, fixed_det_aug=args.fixed_det_aug,
                  augment_probability=args.aug_prob)


if __name__ == "__main__":
    main()
