"""
Task 2-6: 数据校验与可视化
=====================================================
1. 指向矢场截面图 (复现论文 Fig.1 样式: nz 在 xz/yz 截面的 color-coded 图)
2. 抽样 POM 纹理可视化 (复现论文 Fig.2a-d 样式)
3. 标签分布校验 (η, U, F 的统计直方图)
4. POM 图像随 η, U 变化趋势校验
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from config import (L, Lz, CX, CY, CZ, R_SKY, PITCH_VALID_MIN, PITCH_MAX,
                    VOLTAGE_MIN, VOLTAGE_MAX, VIZ_DIR,
                    POM_DIR, LABELS_DIR, DATASET_DIR)
from ansatz import (generate_single_toron, generate_double_toron,
                    apply_voltage_perturbation)
from pom_imaging import compute_pom_image


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================================================
# 1. 指向矢场截面图 (复现论文 Fig.1)
# =========================================================
def plot_director_cross_section(eta=40, U=0.0, savepath=None, show=False):
    """
    绘制单 toron 指向矢场在 xz 平面 (y=Cy) 的 nz 截面图。
    复现论文 Fig.1: color-coded nz 在 x-z 平面的分布。
    """
    nx, ny, nz = generate_single_toron(eta=eta)
    nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    # xz 截面 (y = Cy)
    xz = nz[:, L // 2, :].T  # shape (Lz, L)
    im0 = axes[0].imshow(xz, cmap="RdBu_r", origin="lower",
                          extent=[0, L, 0, Lz], aspect="auto",
                          vmin=-1, vmax=1)
    axes[0].set_title(f"nz in xz plane (y=Cy), η={eta}, U={U}V")
    axes[0].set_xlabel("x (lattice units)")
    axes[0].set_ylabel("z (lattice units)")
    fig.colorbar(im0, ax=axes[0], label="nz")

    # yz 截面 (x = Cx)
    yz = nz[L // 2, :, :].T
    im1 = axes[1].imshow(yz, cmap="RdBu_r", origin="lower",
                          extent=[0, L, 0, Lz], aspect="auto",
                          vmin=-1, vmax=1)
    axes[1].set_title(f"nz in yz plane (x=Cx), η={eta}, U={U}V")
    axes[1].set_xlabel("y (lattice units)")
    axes[1].set_ylabel("z (lattice units)")
    fig.colorbar(im1, ax=axes[1], label="nz")

    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath) or ".")
        plt.savefig(savepath, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close()


def plot_director_quiver(eta=40, U=0.0, savepath=None, show=False):
    """
    绘制中平面 (z=Cz) 的指向矢横向分量 (nx, ny) 矢量图 (quiver)，
    叠加 nz 的 color map。复现论文 Fig.1 的指向矢结构可视化。
    """
    nx, ny, nz = generate_single_toron(eta=eta)
    nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    z_mid = Lz // 2
    nz_slice = nz[:, :, z_mid]
    # 下采样以便 quiver 不太密
    step = 8
    X, Y = np.meshgrid(np.arange(0, L, step), np.arange(0, L, step),
                        indexing="ij")
    nx_q = nx[::step, ::step, z_mid]
    ny_q = ny[::step, ::step, z_mid]

    im = ax.imshow(nz_slice, cmap="RdBu_r", origin="lower",
                    extent=[0, L, 0, Lz], aspect="equal",
                    vmin=-1, vmax=1, alpha=0.6)
    ax.quiver(X, Y, nx_q, ny_q, color="k", scale=30, width=0.002,
              headwidth=3, headlength=4)
    ax.set_title(f"Director field at z=Cz={CZ}, η={eta}, U={U}V")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    fig.colorbar(im, ax=ax, label="nz")
    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath) or ".")
        plt.savefig(savepath, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close()


# =========================================================
# 2. POM 纹理样例 (复现论文 Fig.2a-d)
# =========================================================
def plot_pom_gallery(eta_list=None, U_list=None, savepath=None, show=False):
    """
    绘制不同 (η, U) 组合下的 POM 图像 gallery，复现论文 Fig.2a-d。
    """
    if eta_list is None:
        eta_list = [32, 36]
    if U_list is None:
        U_list = [0.0, 3.5]

    n_eta = len(eta_list)
    n_U = len(U_list)
    fig, axes = plt.subplots(n_eta, n_U, figsize=(4 * n_U, 4 * n_eta),
                              dpi=100)
    if n_eta == 1 and n_U == 1:
        axes = np.array([[axes]])
    elif n_eta == 1 or n_U == 1:
        axes = axes.reshape(n_eta, n_U)

    for i, eta in enumerate(eta_list):
        for j, U in enumerate(U_list):
            nx, ny, nz = generate_single_toron(eta=eta)
            nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
            I = compute_pom_image(nx, ny, nz)
            ax = axes[i, j]
            ax.imshow(I, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.set_title(f"η={eta}, U={U}V")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("POM textures vs (η, U) — single toron (复现论文 Fig.2a-d)",
                  fontsize=12)
    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath) or ".")
        plt.savefig(savepath, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close()


def plot_double_toron_pom_gallery(sep_list=None, eta=32, U=0.0,
                                   savepath=None, show=False):
    """双 toron POM 图像随分离距离变化。"""
    if sep_list is None:
        sep_list = [20.0, 35.0, 50.0, 65.0]
    n = len(sep_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=100)
    if n == 1:
        axes = [axes]
    for i, sep in enumerate(sep_list):
        nx, ny, nz, _ = generate_double_toron(separation=sep, eta=eta)
        nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
        I = compute_pom_image(nx, ny, nz)
        axes[i].imshow(I, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[i].set_title(f"sep={sep:.0f}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.suptitle(f"Double-toron POM vs separation (η={eta}, U={U}V)",
                  fontsize=12)
    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath) or ".")
        plt.savefig(savepath, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close()


# =========================================================
# 3. 标签分布校验
# =========================================================
def plot_label_distribution(csv_path, savepath=None, show=False):
    """
    从 csv 标签文件读取，绘制 η, U, free_energy 的分布直方图。
    用于校验数据集标签是否合理、是否覆盖参数空间。
    """
    etas, Us, Fs = [], [], []
    kinds = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            etas.append(int(row["eta"]))
            Us.append(float(row["U"]))
            Fs.append(float(row["free_energy"]))
            kinds.append(row.get("kind", "single"))

    etas = np.array(etas)
    Us = np.array(Us)
    Fs = np.array(Fs)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=110)
    axes[0].hist(etas, bins=range(int(etas.min()), int(etas.max()) + 2),
                  edgecolor="black", color="steelblue")
    axes[0].set_xlabel("η (pitch, lattice units)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Pitch distribution")

    axes[1].hist(Us, bins=10, edgecolor="black", color="darkorange")
    axes[1].set_xlabel("U (V)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Voltage distribution")

    axes[2].hist(Fs, bins=30, edgecolor="black", color="seagreen")
    axes[2].set_xlabel("Free energy F (J)")
    axes[2].set_ylabel("count")
    axes[2].set_title("Free energy distribution")
    # 科学计数法
    axes[2].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath) or ".")
        plt.savefig(savepath, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close()


# =========================================================
# 4. POM 强度随 η, U 趋势校验
# =========================================================
def plot_pom_intensity_trend(savepath=None, show=False):
    """
    计算并绘制 POM 平均强度随 η 和 U 的变化曲线。
    用于校验图像纹理随参数单调变化（论文 Fig.2 的定性趋势）。
    """
    eta_list = list(range(PITCH_VALID_MIN, PITCH_MAX + 1))
    U_list = np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, 8)

    # mean intensity matrix [eta, U]
    mean_I = np.zeros((len(eta_list), len(U_list)))
    for i, eta in enumerate(eta_list):
        for j, U in enumerate(U_list):
            nx, ny, nz = generate_single_toron(eta=eta)
            nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
            I = compute_pom_image(nx, ny, nz)
            mean_I[i, j] = I.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=110)

    # 左图: I_mean vs η (每条曲线对应一个 U)
    ax = axes[0]
    for j, U in enumerate(U_list):
        ax.plot(eta_list, mean_I[:, j], "o-", label=f"U={U:.2f}V",
                markersize=4)
    ax.set_xlabel("η (pitch, lattice units)")
    ax.set_ylabel("Mean POM intensity")
    ax.set_title("POM mean intensity vs η (各 U 曲线)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # 右图: I_mean vs U (每条曲线对应一个 η)
    ax = axes[1]
    for i, eta in enumerate(eta_list[::2]):  # 每 2 个 η 画一条
        ax.plot(U_list, mean_I[i * 2, :], "s-", label=f"η={eta}",
                markersize=4)
    ax.set_xlabel("U (V)")
    ax.set_ylabel("Mean POM intensity")
    ax.set_title("POM mean intensity vs U (各 η 曲线)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath) or ".")
        plt.savefig(savepath, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close()


# =========================================================
# 2b. 仿论文紫调展示图 (仅报告/PPT用，绝不进训练集)
# =========================================================
def save_presentation_gallery(eta_list=None, U_list=None,
                               out_dir=None, sep_list=None):
    """
    用 save_pom_for_presentation() 输出一套仿论文紫调展示图，
    单独存放在 visualizations/presentation/，与训练灰度图严格分离。
    同时校验第 3 部分的定性趋势 (返回 I_mean 矩阵)。
    """
    from pom_imaging import save_pom_for_presentation

    if out_dir is None:
        out_dir = os.path.join(VIZ_DIR, "presentation")
    _ensure_dir(out_dir)

    if eta_list is None:
        eta_list = [32, 36, 40]
    if U_list is None:
        U_list = [0.0, 3.5]

    print(f"\n=== 仿论文展示图，输出目录 {out_dir} (仅报告用，不进训练集) ===")
    for eta in eta_list:
        for U in U_list:
            nx, ny, nz = generate_single_toron(eta=eta)
            nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
            I = compute_pom_image(nx, ny, nz)
            u_str = str(U).replace(".", "p")
            pres_path = os.path.join(out_dir,
                                      f"single_eta{eta:02d}_U{u_str}_pres.png")
            save_pom_for_presentation(I, pres_path)
            print(f"  保存展示图: {pres_path}")

    # 双 toron 展示图
    if sep_list is None:
        sep_list = [30, 45, 60]
    for sep in sep_list:
        nx, ny, nz, _ = generate_double_toron(separation=float(sep), eta=32)
        nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, 0.0)
        I = compute_pom_image(nx, ny, nz)
        pres_path = os.path.join(out_dir, f"double_eta32_sep{sep:02d}_pres.png")
        save_pom_for_presentation(I, pres_path)
        print(f"  保存展示图: {pres_path}")


# =========================================================
# 主入口
# =========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="数据校验与可视化")
    parser.add_argument("--all", action="store_true",
                        help="生成全部可视化图")
    parser.add_argument("--sample-only", action="store_true",
                        help="只生成基于 ansatz 的样例图 (不读 csv)")
    args = parser.parse_args()

    _ensure_dir(VIZ_DIR)

    if args.all or args.sample_only or True:
        print("=== 1. 指向矢场截面图 (复现论文 Fig.1) ===")
        for eta, U in [(40, 0.0), (40, 3.5)]:
            savepath = os.path.join(VIZ_DIR,
                                     f"director_xz_eta{eta}_U{str(U).replace('.','p')}.png")
            plot_director_cross_section(eta=eta, U=U, savepath=savepath)
            print(f"  保存: {savepath}")

        print("\n=== 2. 指向矢矢量图 (quiver) ===")
        plot_director_quiver(eta=40, U=0.0,
                               savepath=os.path.join(VIZ_DIR, "director_quiver_eta40.png"))
        print(f"  保存: {os.path.join(VIZ_DIR, 'director_quiver_eta40.png')}")

        print("\n=== 3. POM 纹理 gallery (复现论文 Fig.2a-d) ===")
        plot_pom_gallery(eta_list=[32, 36], U_list=[0.0, 3.5],
                          savepath=os.path.join(VIZ_DIR, "pom_gallery_single.png"))
        print(f"  保存: {os.path.join(VIZ_DIR, 'pom_gallery_single.png')}")

        print("\n=== 4. 双 toron POM gallery ===")
        plot_double_toron_pom_gallery(sep_list=[20, 35, 50, 65], eta=32, U=0.0,
                                       savepath=os.path.join(VIZ_DIR,
                                                              "pom_gallery_double.png"))
        print(f"  保存: {os.path.join(VIZ_DIR, 'pom_gallery_double.png')}")

        print("\n=== 5. POM 强度随 η, U 趋势 ===")
        plot_pom_intensity_trend(
            savepath=os.path.join(VIZ_DIR, "pom_intensity_trend.png"))
        print(f"  保存: {os.path.join(VIZ_DIR, 'pom_intensity_trend.png')}")

        # 仿论文紫调展示图 (与训练图严格分离)
        save_presentation_gallery(eta_list=[32, 36, 40], U_list=[0.0, 3.5])

    # 若有 csv 标签，绘制标签分布
    for csv_name in ["labels_single.csv", "labels_double.csv",
                     "labels_mixed.csv", "labels_demo.csv"]:
        csv_path = os.path.join(LABELS_DIR, csv_name)
        if os.path.exists(csv_path):
            print(f"\n=== 6. 标签分布 ({csv_name}) ===")
            out = os.path.join(VIZ_DIR,
                                csv_name.replace(".csv", "_dist.png"))
            plot_label_distribution(csv_path, savepath=out)
            print(f"  保存: {out}")

    print("\n所有可视化完成。")
