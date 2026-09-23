"""
Task 2-4 & 2-5: 批量数据集生成 + 数据集工程
=====================================================
扫描参数 (η, U, [双 toron 间距])，批量生成 POM 图像 + csv 标签，
严格按论文规定比例划分 train/val/test。

【标签含义声明，报告/答辩必读】
csv 中 free_energy 标签是【对 ansatz 构型直接做 Frank-Oseen 体积积分的结果】，
NOT 论文中 PDE 松弛收敛后的平衡稳态自由能。
指向矢场来自解析 ansatz（含手性扭转注入 + 经验电压扰动），未做 PDE 松弛。
正确表述: "对解析 ansatz 构型做 Frank-Oseen 自由能体积积分，作为标签"；
错误表述: "得到 Frank-Oseen 极小化的自由能"。

【电场简化声明】
论文原文使用脉冲宽度调制 (pulse-width modulated) 电场（不同幅值/频率/占空比）
生成训练数据；本项目简化为静态直流电压 U 扫描（0-3.5V 等间隔 8 档），
数据集多样性低于原文，属于已知简化。

参数范围（论文 Section: Single-skyrmion / Two-skyrmion data generation）:
  - 单 toron: η ∈ {30, 31, ..., 40} (丢弃 η<30 的无 toron 样本)
              U ∈ {0.0, 0.5, 1.0, ..., 3.5} V
  - 双 toron: 同上 + 分离距离扫描

划分比例（论文 Fig.2-5 caption 严格规定）:
  - 螺距分类:    85% train / 15% test
  - 回归任务:    80% train / 15% val / 5% test
  - 混合数据集:  10% test, 余下 90% 再分 85% train / 15% val
"""
import os
import csv
import time
import numpy as np
from tqdm import tqdm

from config import (L, Lz, PITCH_MIN, PITCH_MAX, PITCH_VALID_MIN,
                    VOLTAGE_MIN, VOLTAGE_MAX,
                    DATASET_DIR, POM_DIR, LABELS_DIR, SPLIT_DIR,
                    SPLIT_PITCH_CLASSIFICATION, SPLIT_REGRESSION, SPLIT_MIXED)
from ansatz import (generate_single_toron, generate_double_toron,
                    apply_voltage_perturbation)
from frank_osseen import compute_free_energy
from pom_imaging import compute_pom_image


def _voltage_grid(n_voltages=8):
    """等间距电压网格 [0, 3.5]V"""
    return np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, n_voltages)


def _pitch_grid():
    """螺距网格: 整数 30~40 (丢弃 <30)"""
    return list(range(max(PITCH_MIN, PITCH_VALID_MIN), PITCH_MAX + 1))


def _separation_grid():
    """双 toron 分离距离网格 (仿真单位)"""
    # R = 14.4，所以两 toron 不重叠要求 separation > 2R ≈ 29
    # 取 30~75 范围，步长 5
    return list(range(30, 80, 5))


def _filename_prefix(kind, eta, U, sep=None, idx=None):
    """统一文件名: kind_etaXX_UXpX[_sepXX].png"""
    u_str = f"{U:.2f}".replace(".", "p")
    name = f"{kind}_eta{eta:02d}_U{u_str}"
    if sep is not None:
        name += f"_sep{int(sep):02d}"
    if idx is not None:
        name += f"_{idx:04d}"
    return name + ".png"


def generate_single_toron_dataset(n_voltages=8, output_dir=POM_DIR,
                                  labels_dir=LABELS_DIR,
                                  verbose=True):
    """
    生成单 toron 数据集。
    对每个 (η, U) 组合生成 1 张 POM 图像。

    返回:
        labels: list of dict, 每个 dict 含 {filename, eta, U, free_energy, kind}
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    eta_list = _pitch_grid()
    U_list = _voltage_grid(n_voltages)
    labels = []

    total = len(eta_list) * len(U_list)
    desc = "单 toron 数据集生成" if verbose else None
    for eta in (tqdm(eta_list, desc=desc) if verbose else eta_list):
        for U in U_list:
            # 生成指向矢场 (含 η 手性扭转 + U 电场扰动)
            nx, ny, nz = generate_single_toron(eta=eta)
            nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)

            # 计算 POM 图像
            I = compute_pom_image(nx, ny, nz)

            # 计算自由能标签
            F = compute_free_energy(nx, ny, nz, eta=eta, U=U)

            # 保存 POM 图像
            fname = _filename_prefix("single", eta, U)
            # 保存为 uint8 PNG (0-255 灰度)
            img_uint8 = (I * 255).clip(0, 255).astype(np.uint8)
            try:
                from PIL import Image
                Image.fromarray(img_uint8, mode="L").save(
                    os.path.join(output_dir, fname))
            except ImportError:
                # 无 PIL，用 matplotlib 保存
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.imsave(os.path.join(output_dir, fname), img_uint8,
                           cmap="gray", vmin=0, vmax=255)

            labels.append({
                "filename": fname,
                "eta": int(eta),
                "U": float(U),
                "free_energy": float(F),
                "kind": "single",
                "n_torons": 1,
            })

    # 保存 csv 标签
    csv_path = os.path.join(labels_dir, "labels_single.csv")
    _write_csv(labels, csv_path)
    if verbose:
        print(f"  单 toron: 生成 {len(labels)} 张图像，输出到 {output_dir}")
        print(f"  标签 csv: {csv_path}")
    return labels


def generate_double_toron_dataset(n_voltages=8, output_dir=POM_DIR,
                                  labels_dir=LABELS_DIR,
                                  verbose=True):
    """
    生成双 toron 数据集。
    对每个 (η, U, separation) 组合生成 1 张 POM 图像。
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    eta_list = _pitch_grid()
    U_list = _voltage_grid(n_voltages)
    sep_list = _separation_grid()
    labels = []

    total = len(eta_list) * len(U_list) * len(sep_list)
    desc = "双 toron 数据集生成" if verbose else None
    for eta in (tqdm(eta_list, desc=desc) if verbose else eta_list):
        for U in U_list:
            for sep in sep_list:
                nx, ny, nz, _ = generate_double_toron(separation=sep, eta=eta)
                nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
                I = compute_pom_image(nx, ny, nz)
                F = compute_free_energy(nx, ny, nz, eta=eta, U=U)

                fname = _filename_prefix("double", eta, U, sep=sep)
                img_uint8 = (I * 255).clip(0, 255).astype(np.uint8)
                try:
                    from PIL import Image
                    Image.fromarray(img_uint8, mode="L").save(
                        os.path.join(output_dir, fname))
                except ImportError:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    plt.imsave(os.path.join(output_dir, fname), img_uint8,
                               cmap="gray", vmin=0, vmax=255)

                labels.append({
                    "filename": fname,
                    "eta": int(eta),
                    "U": float(U),
                    "free_energy": float(F),
                    "kind": "double",
                    "n_torons": 2,
                    "separation": float(sep),
                })

    csv_path = os.path.join(labels_dir, "labels_double.csv")
    _write_csv(labels, csv_path, include_sep=True)
    if verbose:
        print(f"  双 toron: 生成 {len(labels)} 张图像，输出到 {output_dir}")
        print(f"  标签 csv: {csv_path}")
    return labels


def generate_mixed_dataset(single_labels, double_labels,
                           labels_dir=LABELS_DIR, verbose=True):
    """
    合并单 + 双 toron 标签为混合数据集 csv。
    """
    all_labels = single_labels + double_labels
    csv_path = os.path.join(labels_dir, "labels_mixed.csv")
    _write_csv(all_labels, csv_path, include_sep=True)
    if verbose:
        print(f"  混合数据集: {len(all_labels)} 张图像")
        print(f"  标签 csv: {csv_path}")
    return all_labels


def _write_csv(labels, path, include_sep=False):
    """写入标签 csv。表头: filename,eta,U,free_energy,kind,n_torons[,separation]"""
    fieldnames = ["filename", "eta", "U", "free_energy",
                  "kind", "n_torons"]
    if include_sep:
        fieldnames.append("separation")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in labels:
            writer.writerow(row)


def split_dataset(labels, task="regression", seed=42, verbose=True):
    """
    按论文规定的比例划分数据集。

    task:
        "classification" : 85% train / 15% test (无 val)
        "regression"     : 80% train / 15% val / 5% test
        "mixed"          : 10% test, 余下 90% 再分 85% train / 15% val
                          (≈ 76.5% train / 13.5% val / 10% test)
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    idx = np.arange(n)
    rng.shuffle(idx)

    if task == "classification":
        split = SPLIT_PITCH_CLASSIFICATION
        n_test = int(round(n * split["test"]))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        val_idx = np.array([], dtype=int)
    elif task == "regression":
        split = SPLIT_REGRESSION
        n_test = int(round(n * split["test"]))
        n_val = int(round(n * split["val"]))
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
    elif task == "mixed":
        split = SPLIT_MIXED
        n_test = int(round(n * split["test"]))
        test_idx = idx[:n_test]
        remaining = idx[n_test:]
        n_remaining = len(remaining)
        n_val = int(round(n_remaining * 0.15))   # 余下 90% 中 15% 作为 val
        val_idx = remaining[:n_val]
        train_idx = remaining[n_val:]
    else:
        raise ValueError(f"未知 task: {task}")

    if verbose:
        print(f"  [{task}] 划分: train={len(train_idx)}, "
              f"val={len(val_idx)}, test={len(test_idx)} "
              f"(共 {n})")

    splits = {
        "train": [labels[i] for i in train_idx],
        "val":   [labels[i] for i in val_idx] if len(val_idx) > 0 else [],
        "test":  [labels[i] for i in test_idx],
    }
    return splits


def save_splits(single_labels, double_labels, mixed_labels,
                split_dir=SPLIT_DIR, verbose=True):
    """
    为所有任务生成划分文件。
    每个任务一个 csv: split_<task>_<dataset>.csv
    列: filename, eta, U, free_energy, kind, n_torons, separation, split
    """
    os.makedirs(split_dir, exist_ok=True)

    tasks = [
        # (任务名, 数据集 labels, task 类型)
        ("pitch_classification_single", single_labels, "classification"),
        ("voltage_regression_single", single_labels, "regression"),
        ("free_energy_regression_single", single_labels, "regression"),
        ("free_energy_regression_double", double_labels, "regression"),
        ("free_energy_regression_mixed", mixed_labels, "mixed"),
        ("voltage_regression_mixed", mixed_labels, "mixed"),
        ("pitch_classification_mixed", mixed_labels, "mixed"),
    ]

    for task_name, labels, split_type in tasks:
        splits = split_dataset(labels, task=split_type,
                                seed=42, verbose=verbose)
        out_path = os.path.join(split_dir, f"split_{task_name}.csv")
        fieldnames = ["filename", "eta", "U", "free_energy",
                      "kind", "n_torons", "separation", "split"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                     extrasaction="ignore")
            writer.writeheader()
            for split_name, rows in splits.items():
                for row in rows:
                    row = dict(row)
                    row["split"] = split_name
                    writer.writerow(row)
        if verbose:
            print(f"  保存划分: {out_path}")


# ==================== 主入口 ====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="批量生成 toron 数据集")
    parser.add_argument("--mode", choices=["single", "double", "all", "demo"],
                        default="demo",
                        help="single=仅单 toron; double=仅双 toron; "
                             "all=两者都生成并划分; demo=最小验证 (6 组小样本)")
    parser.add_argument("--n_voltages", type=int, default=8,
                        help="电压离散点数 (默认 8: 0, 0.5, ..., 3.5)")
    args = parser.parse_args()

    t0 = time.time()
    if args.mode == "demo":
        # 最小验证 (论文模块1 最小 demo): 6 组样本 η∈{32,36,40}, U∈{0,3.5}
        print("=== 最小验证 demo (6 组样本) ===")
        os.makedirs(POM_DIR, exist_ok=True)
        os.makedirs(LABELS_DIR, exist_ok=True)
        labels = []
        for eta in [32, 36, 40]:
            for U in [0.0, 3.5]:
                nx, ny, nz = generate_single_toron(eta=eta)
                nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
                I = compute_pom_image(nx, ny, nz)
                F = compute_free_energy(nx, ny, nz, eta=eta, U=U)
                fname = _filename_prefix("single", eta, U)
                img_uint8 = (I * 255).clip(0, 255).astype(np.uint8)
                try:
                    from PIL import Image
                    Image.fromarray(img_uint8, mode="L").save(
                        os.path.join(POM_DIR, fname))
                except ImportError:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    plt.imsave(os.path.join(POM_DIR, fname), img_uint8,
                               cmap="gray", vmin=0, vmax=255)
                labels.append({"filename": fname, "eta": int(eta),
                               "U": float(U), "free_energy": float(F),
                               "kind": "single", "n_torons": 1})
        _write_csv(labels, os.path.join(LABELS_DIR, "labels_demo.csv"))
        print(f"  生成 {len(labels)} 张 demo 图像")
        print(f"  耗时 {time.time()-t0:.1f}s")
    elif args.mode in ("single", "all"):
        single_labels = generate_single_toron_dataset(n_voltages=args.n_voltages)
    if args.mode in ("double", "all"):
        double_labels = generate_double_toron_dataset(n_voltages=args.n_voltages)
    if args.mode == "all":
        # 混合数据集 + 全部划分
        mixed_labels = generate_mixed_dataset(single_labels, double_labels)
        print("\n=== 数据集划分 ===")
        save_splits(single_labels, double_labels, mixed_labels)
    print(f"\n总耗时: {time.time()-t0:.1f}s")
