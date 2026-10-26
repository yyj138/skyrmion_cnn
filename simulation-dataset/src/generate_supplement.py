"""
补充测试数据集生成（成员 A 交付补充）
=====================================================
在原数据集 (968 张) 之外，新增 440 张模拟 POM 灰度图作为测试数据，
不加入训练集、不覆盖原有图片与划分文件。

生成原则（严格遵循原论文方法，仅在 ansatz 参数上做小幅物理变异）:
  1. 指向矢场: 论文 Eq.(3)-(6) 解析 ansatz（同 ansatz.py），
     手性扭转 q0*(z-Cz) 注入、经验电压扰动，与原数据集完全一致；
  2. 为使新图与原 968 张不重复，对 ansatz 几何参数 R (toron 半径)、
     B (扭转壁宽度) 做小幅变异（原论文固定 R=0.45*Lz=14.4, B=0.5）。
     变异幅度: R ∈ {13.4, 14.4, 15.4}, B ∈ {0.45, 0.5, 0.55}，
     均选取与原值 (14.4, 0.5) 不同的组合，保证每张图构型全新；
  3. 自由能标签: 对每个新构型重新做 Frank-Oseen 体积积分（同 frank_osseen.py）；
  4. POM 成像: Jones 矩阵正交偏振（同 pom_imaging.py），300×300 单通道灰度。

参数覆盖（满足 补充图需求清单.md）:
  - 单 toron 220 张: η=30~40 (11 类) × 每类 20 张；电压 U=0~3.5V 8 档均匀覆盖；
  - 双 toron 220 张: η=30~40 (11 类) × 每类 20 张；均衡覆盖 8 档 U 与 10 档间距 (30~75, 步长 5)；
  - 混合: 合并上述两部分标签表为 440 行，不另生成图片。

输出目录: simulation-dataset/outputs/supplement/
  ├── pom_images/     # 440 张灰度 PNG
  └── labels/
      ├── labels_supplement_single.csv   # 220 行
      ├── labels_supplement_double.csv   # 220 行
      └── labels_supplement_mixed.csv    # 440 行

运行:  cd simulation-dataset/src && python generate_supplement.py
"""
import os
import csv
import time
import numpy as np
from tqdm import tqdm

from config import (L, Lz, VOLTAGE_MIN, VOLTAGE_MAX)
from ansatz import (generate_single_toron, generate_double_toron,
                    apply_voltage_perturbation)
from frank_osseen import compute_free_energy
from pom_imaging import compute_pom_image

# === 输出目录（独立于原 outputs/dataset，不覆盖任何原文件）===
SUPPLEMENT_DIR = "../outputs/supplement"
SUP_POM_DIR = os.path.join(SUPPLEMENT_DIR, "pom_images")
SUP_LABELS_DIR = os.path.join(SUPPLEMENT_DIR, "labels")

# === 随机种子（确定性参数网格，无随机抽样；记录用于可复现）===
SEED = 20261022

# === 论文原始 ansatz 参数 ===
R_DEFAULT = 0.45 * Lz   # 14.4
B_DEFAULT = 0.5

# === 新增构型的 (R, B) 变异组合（均与原值 (14.4, 0.5) 不同，保证全新）===
# 选取 4 组，覆盖 R ±1 格子、B ±0.05 的小幅物理变异
VARIANTS = [
    {"R": 13.4, "B": 0.45, "vid": "v1"},   # R-1, B-0.05
    {"R": 15.4, "B": 0.55, "vid": "v2"},   # R+1, B+0.05
    {"R": 14.4, "B": 0.45, "vid": "v3"},   # R 不变, B-0.05
    {"R": 13.4, "B": 0.55, "vid": "v4"},   # R-1, B+0.05
]


def _voltage_grid(n_voltages=8):
    """等间距电压网格 [0, 3.5]V，与原数据集一致"""
    return np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, n_voltages)


def _pitch_grid():
    """螺距网格: 整数 30~40"""
    return list(range(30, 41))


def _separation_grid():
    """双 toron 分离距离: 30~75, 步长 5 (10 档)"""
    return list(range(30, 80, 5))


def _u_str(U):
    """电压字符串: 0.5 -> '0p50'"""
    return f"{U:.2f}".replace(".", "p")


def _save_png(intensity, filepath):
    """保存为 300×300 单通道灰度 PNG (mode=L, 0-255)"""
    img_uint8 = (np.asarray(intensity) * 255.0).clip(0, 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(img_uint8, mode="L").save(filepath)
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(filepath, img_uint8, cmap="gray", vmin=0, vmax=255)


def generate_supplement_single():
    """
    生成单 toron 补充集: 11 η × 20 = 220 张。
    每类 20 张 = 8 U × V1 (8) + 8 U × V2 (8) + 4 U × V3 (4)
    8 档电压全部覆盖；4 个偶数电压 (0,1.0,2.0,3.0V) 额外多一张。
    所有 (R,B) 均非原值 (14.4,0.5)，与原 88 张不重复。
    """
    eta_list = _pitch_grid()
    U_list = _voltage_grid(8)
    # V1, V2 用于全部 8 档 U; V3 仅用于偶数索引 U (0, 1.0, 2.0, 3.0)
    v1, v2, v3 = VARIANTS[0], VARIANTS[1], VARIANTS[2]
    even_u_idx = [0, 2, 4, 6]   # U = 0, 1.0, 2.0, 3.0

    labels = []
    for eta in tqdm(eta_list, desc="单 toron 补充集"):
        for ui, U in enumerate(U_list):
            for variant in (v1, v2):
                fname = f"single_eta{eta:02d}_U{_u_str(U)}_{variant['vid']}.png"
                nx, ny, nz = generate_single_toron(
                    eta=eta, R=variant["R"], B=variant["B"])
                nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
                I = compute_pom_image(nx, ny, nz)
                F = compute_free_energy(nx, ny, nz, eta=eta, U=U)
                _save_png(I, os.path.join(SUP_POM_DIR, fname))
                labels.append({
                    "filename": fname,
                    "eta": int(eta),
                    "U": float(U),
                    "free_energy": float(F),
                    "kind": "single",
                    "n_torons": 1,
                    "R": variant["R"],
                    "B": variant["B"],
                })
            # 偶数 U 追加 V3 一张
            if ui in even_u_idx:
                fname = f"single_eta{eta:02d}_U{_u_str(U)}_{v3['vid']}.png"
                nx, ny, nz = generate_single_toron(
                    eta=eta, R=v3["R"], B=v3["B"])
                nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
                I = compute_pom_image(nx, ny, nz)
                F = compute_free_energy(nx, ny, nz, eta=eta, U=U)
                _save_png(I, os.path.join(SUP_POM_DIR, fname))
                labels.append({
                    "filename": fname,
                    "eta": int(eta),
                    "U": float(U),
                    "free_energy": float(F),
                    "kind": "single",
                    "n_torons": 1,
                    "R": v3["R"],
                    "B": v3["B"],
                })
    return labels


def generate_supplement_double():
    """
    生成双 toron 补充集: 11 η × 20 = 220 张。
    每类 20 张: 10 档间距各 2 张 (不同 U)，8 档电压各 2~3 张，均衡覆盖。
    全部使用 V1=(R=13.4, B=0.45)，与原 880 张 (R=14.4, B=0.5) 不重复。
    """
    eta_list = _pitch_grid()
    U_list = _voltage_grid(8)
    sep_list = _separation_grid()
    v1 = VARIANTS[0]

    labels = []
    for eta in tqdm(eta_list, desc="双 toron 补充集"):
        for si, sep in enumerate(sep_list):
            # 每个间距配 2 个电压: (si % 8) 和 ((si+4) % 8)
            # 这样 8 档电压中 4 档出现 3 次、4 档出现 2 次，10 档间距各 2 次
            u1 = U_list[si % 8]
            u2 = U_list[(si + 4) % 8]
            for U in (u1, u2):
                fname = (f"double_eta{eta:02d}_U{_u_str(U)}"
                         f"_sep{int(sep):02d}_{v1['vid']}.png")
                nx, ny, nz, _ = generate_double_toron(
                    separation=sep, eta=eta, R=v1["R"], B=v1["B"])
                nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
                I = compute_pom_image(nx, ny, nz)
                F = compute_free_energy(nx, ny, nz, eta=eta, U=U)
                _save_png(I, os.path.join(SUP_POM_DIR, fname))
                labels.append({
                    "filename": fname,
                    "eta": int(eta),
                    "U": float(U),
                    "free_energy": float(F),
                    "kind": "double",
                    "n_torons": 2,
                    "separation": float(sep),
                    "R": v1["R"],
                    "B": v1["B"],
                })
    return labels


def _write_csv(labels, path, include_sep=False):
    """写入标签 csv。列: filename,eta,U,free_energy,kind,n_torons[,separation][,R][,B]"""
    fieldnames = ["filename", "eta", "U", "free_energy",
                  "kind", "n_torons"]
    if include_sep:
        fieldnames.append("separation")
    fieldnames.extend(["R", "B"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in labels:
            writer.writerow(row)


def write_generation_log(single_labels, double_labels, elapsed):
    """记录生成方法、参数与种子，便于可复现追溯"""
    log_path = os.path.join(SUPPLEMENT_DIR, "generation_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("补充测试数据集生成日志\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"随机种子 (确定性参数网格): {SEED}\n")
        f.write(f"总耗时: {elapsed:.1f} s\n\n")
        f.write("方法声明:\n")
        f.write("  指向矢场: 论文 Eq.(3)-(6) 解析 ansatz (ansatz.py)，含手性扭转\n")
        f.write("            q0*(z-Cz) 注入 + 经验电压扰动，与原数据集一致。\n")
        f.write("  自由能: Frank-Oseen 体积积分 (frank_osseen.py Eq.(1))。\n")
        f.write("  POM: Jones 矩阵正交偏振成像 (pom_imaging.py)，300x300 灰度。\n\n")
        f.write(f"原数据集 ansatz 参数: R={R_DEFAULT}, B={B_DEFAULT}\n")
        f.write("新增构型 (R,B) 变异组合 (均与原值不同，保证构型全新):\n")
        for v in VARIANTS:
            f.write(f"  {v['vid']}: R={v['R']}, B={v['B']}\n")
        f.write("\n")
        f.write(f"单 toron: {len(single_labels)} 张 "
                f"(11 eta x 20 = 220)\n")
        f.write(f"双 toron: {len(double_labels)} 张 "
                f"(11 eta x 20 = 220)\n")
        f.write(f"合计: {len(single_labels) + len(double_labels)} 张\n\n")
        f.write("单 toron 每类 20 张结构: 8 U x V1 (8) + 8 U x V2 (8) + 4 U x V3 (4)\n")
        f.write("双 toron 每类 20 张结构: 10 sep x 2 U (V1)，均衡覆盖 8 U / 10 sep\n")
    return log_path


def main():
    t0 = time.time()
    os.makedirs(SUP_POM_DIR, exist_ok=True)
    os.makedirs(SUP_LABELS_DIR, exist_ok=True)

    print("=" * 60)
    print("补充测试数据集生成 (440 张)")
    print("=" * 60)
    print(f"原 ansatz 参数: R={R_DEFAULT}, B={B_DEFAULT}")
    print(f"新增 (R,B) 变异: {[(v['R'], v['B']) for v in VARIANTS]}")
    print(f"输出目录: {SUPPLEMENT_DIR}")
    print()

    # 1. 单 toron 220 张
    single_labels = generate_supplement_single()
    _write_csv(single_labels,
               os.path.join(SUP_LABELS_DIR, "labels_supplement_single.csv"))
    print(f"  单 toron: {len(single_labels)} 张")

    # 2. 双 toron 220 张
    double_labels = generate_supplement_double()
    _write_csv(double_labels,
               os.path.join(SUP_LABELS_DIR, "labels_supplement_double.csv"),
               include_sep=True)
    print(f"  双 toron: {len(double_labels)} 张")

    # 3. 混合标签表 440 行 (仅合并，不另生成图片)
    mixed_labels = single_labels + double_labels
    _write_csv(mixed_labels,
               os.path.join(SUP_LABELS_DIR, "labels_supplement_mixed.csv"),
               include_sep=True)
    print(f"  混合: {len(mixed_labels)} 行")

    elapsed = time.time() - t0
    log_path = write_generation_log(single_labels, double_labels, elapsed)
    print(f"\n生成日志: {log_path}")
    print(f"总耗时: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
