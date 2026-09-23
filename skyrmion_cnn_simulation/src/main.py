"""
主入口: 最小验证 demo (模块 1 任务)
=====================================================
生成 6 组小样本 (η=32,36,40; U=0V,3.5V)，产出:
  - 三维指向矢场 n (截面图)
  - POM 图像 (复现论文 Fig.2a-d 样式)
  - Frank-Oseen 自由能标签
  - 小 csv 标签文件

确认整条链路 (ansatz - 自由能 - POM - csv) 可以跑通。

运行:  python main.py
"""
import os
import csv
import time
import numpy as np

from config import (L, Lz, VIZ_DIR, POM_DIR, LABELS_DIR, OUTPUT_DIR)
from ansatz import generate_single_toron, apply_voltage_perturbation
from frank_osseen import compute_free_energy
from pom_imaging import compute_pom_image, save_pom_png


def run_demo():
    print("=" * 60)
    print("最小验证 demo: 6 组样本 (η=32,36,40 × U=0,3.5V)")
    print("=" * 60)

    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(POM_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    labels = []
    t_total = time.time()

    for eta in [32, 36, 40]:
        for U in [0.0, 3.5]:
            t0 = time.time()
            print(f"\n--- η={eta}, U={U}V ---")

            # 1. 生成三维指向矢场 (ansatz + 手性扭转 + U 扰动)
            nx, ny, nz = generate_single_toron(eta=eta)
            nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
            print(f"  指向矢场形状: {nx.shape}, |n| 范围: "
                  f"[{np.sqrt(nx**2+ny**2+nz**2).min():.4f}, "
                  f"{np.sqrt(nx**2+ny**2+nz**2).max():.4f}]")

            # 2. 计算 Frank-Oseen 自由能
            F, comps = compute_free_energy(nx, ny, nz, eta=eta, U=U,
                                            return_components=True)
            print(f"  自由能 F = {F:.4e} J")
            print(f"    (splay={comps[0]:.2e}, twist={comps[1]:.2e}, "
                  f"bend={comps[2]:.2e}, elec={comps[3]:.2e})")

            # 3. 计算 POM 图像
            I = compute_pom_image(nx, ny, nz)
            print(f"  POM 图像: shape={I.shape}, "
                  f"I_mean={I.mean():.4f}, I_max={I.max():.3f}")

            # 4. 保存 POM 图像
            u_str = str(U).replace(".", "p")
            fname = f"single_eta{eta:02d}_U{u_str}.png"
            pom_path = os.path.join(POM_DIR, fname)
            save_pom_png(I, pom_path)
            print(f"  保存 POM: {pom_path}")

            # 5. 保存指向矢截面图 (z=Cz 处的 nz 分布)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
                ax.imshow(nz[:, :, Lz//2], cmap="RdBu_r", origin="lower",
                          vmin=-1, vmax=1)
                ax.set_title(f"nz at z=Cz, η={eta}, U={U}V")
                plt.tight_layout()
                director_path = os.path.join(
                    VIZ_DIR, f"director_eta{eta:02d}_U{u_str}.png")
                plt.savefig(director_path, bbox_inches="tight")
                plt.close()
                print(f"  保存指向矢截面: {director_path}")
            except Exception as e:
                print(f"  (跳过指向矢截面图: {e})")

            labels.append({
                "filename": fname,
                "eta": int(eta),
                "U": float(U),
                "free_energy": float(F),
                "kind": "single",
                "n_torons": 1,
            })
            print(f"  耗时 {time.time()-t0:.1f}s")

    # 保存 csv 标签
    csv_path = os.path.join(LABELS_DIR, "labels_demo.csv")
    fieldnames = ["filename", "eta", "U", "free_energy", "kind", "n_torons"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels)
    print(f"\n=== 完成 ===")
    print(f"  生成 {len(labels)} 组样本")
    print(f"  标签 csv: {csv_path}")
    print(f"  总耗时: {time.time()-t_total:.1f}s")

    # 打印标签汇总表
    print(f"\n标签汇总:")
    print(f"{'filename':<35} {'eta':>4} {'U':>5} {'F (J)':>14}")
    for row in labels:
        print(f"{row['filename']:<35} {row['eta']:>4} {row['U']:>5.1f} "
              f"{row['free_energy']:>14.4e}")

    print(f"\n链路验证: ansatz - 自由能 - POM - csv 全部跑通")


if __name__ == "__main__":
    run_demo()
