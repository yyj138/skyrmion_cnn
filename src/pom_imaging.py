"""
Task 2-3: 3D 琼斯矩阵 POM 成像 (向量化加速)
=====================================================
复现论文 Methods: Generation of POM textures 章节。

光学模型:
  - 单色光 λ = 500 nm
  - 双折射 Δn = 0.06
  - 正交偏振器 (polarizer + analyzer, 角度差 90°)
  - LC 在 z 方向被离散为 Lz=32 层，每层等效一块相位延迟片
  - 每层 Jones 矩阵: J_k = R(-θ_k) * diag(e^(-iΓ/2), e^(iΓ/2)) * R(θ_k)
        其中 θ_k(x,y) = atan2(ny, nx) 是该层指向矢在 xy 平面投影的方位角
        Γ = 2π * Δn * dz_phys / λ 是单层相位延迟
  - 像素 (x,y) 的总 Jones 矩阵:
        J(x,y) = J_Lz * J_{Lz-1} * ... * J_1
  - 输出光强: I(x,y) = |J_ana * J * J_pol * E_0|²

向量化: 仅沿 z 轴循环 (Lz=32 次)，每个 z 层所有 (x,y) 像素并行做 2x2 矩阵乘。
"""
import numpy as np
from config import L, Lz, DX, DELTA_N, WAVELENGTH


def compute_pom_image(nx, ny, nz=None,
                      delta_n=DELTA_N, wavelength=WAVELENGTH,
                      dz_phys=DX, polarizer_angle=0.0,
                      analyzer_angle=np.pi/2):
    """
    计算单波长 POM 图像。

    每层 Jones 矩阵:
        J_k = R(-θ_k) * diag(e^(-iΓ_k/2), e^(iΓ_k/2)) * R(θ_k)
        其中 θ_k(x,y) = atan2(ny_k, nx_k)  是该层指向矢的横向方位角
             Γ_k(x,y) = (2π Δn dz / λ) * (1 - nz_k²)  是该层有效相位延迟
               (当指向矢沿 z 轴时 nz²=1, Γ=0 无双折射信号；
                当指向矢在 xy 平面时 nz=0, Γ 最大)

    参数:
        nx, ny, nz: 三维指向矢场，形状 (L, L, Lz)
        delta_n: 双折射 (无量纲)
        wavelength: 光波长 (m)
        dz_phys: 每层厚度 (m) = 仿真单位 = 0.3125 μm
        polarizer_angle: 起偏器方位角 (rad, 0=沿x轴)
        analyzer_angle: 检偏器方位角 (rad)，默认 π/2 (与起偏器正交)

    返回:
        intensity: 归一化到 [0,1] 的 POM 灰度图像，形状 (L, L)
    """
    shape = nx.shape
    assert shape[0] == L and shape[1] == L and shape[2] == Lz, \
        f"形状应为 ({L},{L},{Lz})，实际 {shape}"
    Lx, Ly, Lz_ = shape

    if nz is None:
        # 若只给 nx, ny，按 |n|=1 推出 nz² = 1 - nx² - ny²
        nz2 = np.maximum(1.0 - nx * nx - ny * ny, 0.0)
        nz = np.sqrt(nz2)  # 符号不重要，只参与 Γ_eff 的计算

    # 基础相位延迟 Γ_0 = 2π Δn dz / λ
    gamma0 = 2.0 * np.pi * delta_n * dz_phys / wavelength
    # 每层有效相位延迟 Γ_k(x,y) = Γ_0 * (1 - nz_k²)
    # 形状 (L, L, Lz)
    gamma_eff = gamma0 * (1.0 - nz * nz)

    # 每层指向矢在 xy 平面投影的方位角 θ_k(x,y) = atan2(ny, nx)
    theta = np.arctan2(ny, nx)  # (L, L, Lz)

    # 起偏器/检偏器 Jones 矩阵 (沿角度 α 的线偏振)
    def _jones_polarizer(alpha):
        c, s = np.cos(alpha), np.sin(alpha)
        return np.array([[c * c, c * s],
                         [c * s, s * s]], dtype=complex)

    J_pol = _jones_polarizer(polarizer_angle)
    J_ana = _jones_polarizer(analyzer_angle)

    # 初始化每个像素的总 Jones 矩阵为单位阵
    J_total = np.zeros((2, 2, Lx, Ly), dtype=complex)
    J_total[0, 0] = 1.0
    J_total[1, 1] = 1.0

    # === 沿 z 逐层累乘 (向量化在 xy 平面) ===
    # J_k = R(-θ) * diag(e^(-iΓ_k/2), e^(iΓ_k/2)) * R(θ)
    # 展开后 (c=cos θ, s=sin θ):
    #   j11 = c² e^(-iΓ/2) + s² e^(iΓ/2)
    #   j12 = c*s (e^(iΓ/2) - e^(-iΓ/2)) = i * 2 c s sin(Γ/2)
    #   j22 = s² e^(-iΓ/2) + c² e^(iΓ/2)
    for k in range(Lz_):
        th = theta[:, :, k]              # (L, L)
        g = gamma_eff[:, :, k]           # (L, L) 有效相位延迟
        c = np.cos(th)
        s = np.sin(th)
        cc = c * c
        ss = s * s
        cs = c * s

        # 该层复相位
        cos_g = np.cos(g / 2.0)
        sin_g = np.sin(g / 2.0)
        e_neg = cos_g - 1j * sin_g  # e^(-iΓ/2)，逐像素
        e_pos = cos_g + 1j * sin_g  # e^(+iΓ/2)，逐像素

        j11 = cc * e_neg + ss * e_pos
        j12 = cs * (e_pos - e_neg)         # 逐像素
        j22 = ss * e_neg + cc * e_pos

        # J_total_new = J_k @ J_total (2x2 @ 2x2，逐像素)
        new00 = j11 * J_total[0, 0] + j12 * J_total[1, 0]
        new01 = j11 * J_total[0, 1] + j12 * J_total[1, 1]
        new10 = j12 * J_total[0, 0] + j22 * J_total[1, 0]
        new11 = j12 * J_total[0, 1] + j22 * J_total[1, 1]
        J_total[0, 0] = new00
        J_total[0, 1] = new01
        J_total[1, 0] = new10
        J_total[1, 1] = new11

    # 输入光场: 通过起偏器后的光 (沿起偏器方向偏振)
    E_in = J_pol[:, 0].copy()

    # 输出 = J_ana @ J_total @ E_in
    field_xy = np.empty((2, Lx, Ly), dtype=complex)
    field_xy[0] = J_total[0, 0] * E_in[0] + J_total[0, 1] * E_in[1]
    field_xy[1] = J_total[1, 0] * E_in[0] + J_total[1, 1] * E_in[1]

    out0 = J_ana[0, 0] * field_xy[0] + J_ana[0, 1] * field_xy[1]
    out1 = J_ana[1, 0] * field_xy[0] + J_ana[1, 1] * field_xy[1]

    intensity = np.abs(out0) ** 2 + np.abs(out1) ** 2

    # 归一化到 [0, 1]
    # 阈值: 当最大强度 < 1e-12 (数值噪声/完全暗场) 时不做归一化，保持 0
    max_val = intensity.max()
    if max_val > 1e-12:
        intensity = intensity / max_val
    else:
        intensity = np.zeros_like(intensity)
    return intensity


def compute_pom_image_45polarizers(nx, ny, nz=None, **kwargs):
    """
    使用 45°/-45° 正交偏振器 (常用于最大化双折射信号)。
    适用场景: 检验偏振器角度对纹理对比度的影响。
    """
    return compute_pom_image(nx, ny, nz,
                             polarizer_angle=np.pi/4,
                             analyzer_angle=-np.pi/4 + np.pi/2,  # = π/4
                             **kwargs)


def save_pom_png(intensity, filepath, cmap="gray"):
    """
    把 POM 强度图保存为灰度 PNG —— 仅用于 CNN 训练数据集！
    保持原始 0-1 强度，强制单通道灰度 (PIL mode L)，
    不做任何对比度拉伸或伪彩处理。cmap 参数仅为兼容保留。
    """
    import numpy as _np
    from PIL import Image
    img_uint8 = (_np.asarray(intensity) * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(img_uint8, mode="L").save(filepath)


def save_pom_for_presentation(intensity, filepath):
    """
    仅用于报告、PPT 展示，不用于 CNN 训练！
    仿论文附图风格: 紫色 colormap + 对比度裁剪 (2%-98% 分位 clip)，
    放大纹理细节；无坐标轴刻度。输出文件用 *_pres.png 后缀，
    单独存放在 visualizations/presentation/，绝不进入 pom_images/ 训练目录。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    # 对比度裁剪，突出纹理，模拟论文图
    vmin, vmax = np.percentile(intensity, (2, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    ax.imshow(intensity, cmap="Purples", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0.02)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.02)
    plt.close()


# ==================== 自检 ====================
if __name__ == "__main__":
    import time
    from ansatz import (generate_single_toron, generate_double_toron,
                       apply_voltage_perturbation)

    print("=== POM 成像自检 ===")

    # 1. 均匀指向矢场 (n = ẑ): nz=1 时 Γ_eff=0，应完全暗场
    nx_u = np.zeros((L, L, Lz)); ny_u = np.zeros((L, L, Lz))
    nz_u = np.ones((L, L, Lz))
    I_uniform = compute_pom_image(nx_u, ny_u, nz_u)
    print(f"  均匀场 POM: 最大强度 = {I_uniform.max():.3e} (应≈0, 完全暗场)")

    # 2. 单 toron 在不同 η, U 下的 POM (验证 η, U 依赖性)
    print("\n--- 单 toron POM 图像随 η, U 变化 (复现论文 Fig.2a-d) ---")
    print(f"{'eta':>4} {'U':>5} {'I_max':>8} {'I_mean':>8} {'I_std':>8} "
          f"{'elapsed (s)':>12}")
    for eta in [32, 36, 40]:
        for U in [0.0, 3.5]:
            nx, ny, nz = generate_single_toron(eta=eta)
            nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
            t0 = time.time()
            I = compute_pom_image(nx, ny, nz)
            dt = time.time() - t0
            print(f"{eta:>4} {U:>5.2f} {I.max():>8.3f} {I.mean():>8.4f} "
                  f"{I.std():>8.4f} {dt:>12.3f}")

    # 3. 双 toron POM
    print("\n--- 双 toron POM 图像 (η=32, U=0, 不同间距) ---")
    for sep in [20.0, 35.0, 50.0]:
        nx2, ny2, nz2, _ = generate_double_toron(separation=sep, eta=32)
        I = compute_pom_image(nx2, ny2, nz2)
        print(f"  sep={sep:>4.0f}: I_max={I.max():.3f}, I_mean={I.mean():.4f}, "
              f"I_std={I.std():.4f}")

    # 4. 保存样例: 4 张 POM 图 (η=32,36 × U=0,3.5)，对照论文 Fig.2
    import os
    os.makedirs("../outputs", exist_ok=True)
    for eta in [32, 36]:
        for U in [0.0, 3.5]:
            nx, ny, nz = generate_single_toron(eta=eta)
            nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
            I = compute_pom_image(nx, ny, nz)
            fname = f"../outputs/pom_sample_eta{eta}_U{str(U).replace('.','p')}.png"
            save_pom_png(I, fname)
            print(f"  保存: {fname}")
    print("OK")
