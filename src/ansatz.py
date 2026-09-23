"""
Task 2-1: Toron 三维解析 ansatz 指向矢场生成
=====================================================
严格复现论文 Eq.(3)-(6) 的解析 ansatz，输出 (300,300,32) 的 nx,ny,nz 数组。
支持单 toron 与双 toron 构型（双 toron 通过改变两 toron 中心位置生成）。

注意（简化假设声明，答辩/报告必读）:
1. 本文件生成的 toron 解析 ansatz【仅为论文 PDE 求解器的初始条件】。
   本项目主流程不运行 PDE 松弛动力学，直接将 ansatz 场作为等效平衡态指向矢场。
2. 手性螺旋扭转 q0*(z-Cz) 为人为注入（模拟 PDE 松弛后 twist 项驱动的螺旋特征），
   并非 Frank-Oseen 自由能极小化自然产生的结果。
3. 电压效应 apply_voltage_perturbation() 使用经验扰动模型（缩放 nz 分量），
   并非完整 Frank-Oseen 介电耦合求解；形变模式为定性模拟。
4. 双 toron 为两个 ansatz 场线性叠加后归一化，不含弹性相互作用；
   自由能随间距的变化趋势与论文 Fig.4e 不完全一致。
5. pde_relaxation.py 为可选增强模块（完整松弛求解器），本批量数据集生成不使用。

论文 ansatz（Methods 节）:
    n_x(r) = sin(a(r)) * sin(b(r) + π/2)
    n_y(r) = sin(a(r)) * cos(b(r) + π/2)
    n_z(r) = -cos(a(r))

其中 (z' = z - C_z):
    a(r) = (π/2) * [1 - tanh((B/2)*(r - sqrt(R² - z'²)))]   当 |z'| < R
         = 0                                                 当 |z'| ≥ R
    b(r) = arctan((x - Cx) / (y - Cy))
    r    = sqrt((x - Cx)² + (y - Cy)²)

参数（论文给定）: R = 0.45*Lz = 14.4, B = 0.5, Cx = Cy = L/2 = 150, Cz = Lz/2 = 16
"""
import numpy as np
from config import (L, Lz, R_SKY, B_WALL, CX, CY, CZ)


def _build_grid():
    """构建 3D 仿真网格坐标 (单位: 仿真格子)"""
    x = np.arange(L, dtype=np.float64)
    y = np.arange(L, dtype=np.float64)
    z = np.arange(Lz, dtype=np.float64)
    # 形状 (L, L, Lz) 对应 (x, y, z)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z


def _single_toron_field(X, Y, Z, center_xy=(CX, CY), center_z=CZ,
                        R=R_SKY, B=B_WALL, eta=None, q0=None):
    """
    在指定中心生成单个 toron 的指向矢场（不归一化前的中间量）。
    返回 nx, ny, nz 三个数组，形状 (L, L, Lz)。

    参数:
        center_xy: (Cx, Cy) toron 在 xy 平面中心
        center_z: toron 在 z 方向中心
        R, B: ansatz 几何参数
        eta: 螺距 (仿真单位)。若不为 None，则在方位角 b 中加入手性螺旋扭转
             q0*(z-Cz) 项，使 POM 图像随 η 变化（兜底方案，复现 PDE 松弛后的
             手性特征）。
        q0: 手性波矢 2π/η；若提供则覆盖 eta。
    """
    Cx, Cy = center_xy
    Cz = center_z

    # 极径 r 与方位角 b
    dx = X - Cx
    dy = Y - Cy
    r = np.sqrt(dx * dx + dy * dy)  # (L, L, Lz) 广播
    # b = arctan((x-Cx)/(y-Cy))，用 arctan2 处理 y=Cy 的奇点
    b = np.arctan2(dx, dy)

    # 加入手性螺旋扭转项 q0*(z - Cz) 使指向矢沿 z 轴扭转，pitch=η
    # （论文中该扭转由 PDE 松弛自然产生；此处作为兜底方案直接注入）
    if q0 is None and eta is not None:
        q0 = 2.0 * np.pi / eta
    if q0 is not None and q0 != 0:
        b = b + q0 * (Z - Cz)

    # z' = z - Cz，并判断 |z'| < R 的掩码
    zp = Z - Cz
    in_tube = (np.abs(zp) < R)  # bool 掩码

    # 计算 tube 半径 r0(z) = sqrt(R² - z'²)，仅在 |z'|<R 处有效
    r0 = np.zeros_like(Z)
    r0[in_tube] = np.sqrt(R * R - zp[in_tube] ** 2)

    # a(r) = (π/2) * [1 - tanh((B/2)*(r - r0(z)))]
    # 注: 仅在管内 (|z'|<R) 用此公式，管外 a=0 (均匀背景)
    a = np.zeros_like(Z)
    # 避免在 r0 处出现 0/0；tanh 自变量 = (B/2)*(r - r0)
    arg = (B / 2.0) * (r - r0)
    a_tube = (np.pi / 2.0) * (1.0 - np.tanh(arg))
    a = np.where(in_tube, a_tube, 0.0)

    # 指向矢分量
    sin_a = np.sin(a)
    cos_a = np.cos(a)
    # sin(b + π/2) = cos(b),  cos(b + π/2) = -sin(b)
    nx = sin_a * np.cos(b)
    ny = -sin_a * np.sin(b)
    nz = -cos_a

    return nx, ny, nz


def _normalize(nx, ny, nz, eps=1e-12):
    """逐点归一化 |n| = 1（双 toron 叠加后必须重新归一化）"""
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    norm = np.maximum(norm, eps)
    return nx / norm, ny / norm, nz / norm


def generate_single_toron(center_xy=(CX, CY), center_z=CZ,
                          R=R_SKY, B=B_WALL, normalize=True,
                          eta=None, q0=None):
    """
    生成单 toron 三维指向矢场。

    参数:
        center_xy: toron 中心 (Cx, Cy)，默认 (150, 150) 居中
        center_z: toron 中心 z 坐标，默认 16
        R, B: ansatz 几何参数（默认论文值）
        normalize: 是否归一化为单位矢量
        eta: 螺距 (仿真单位)；提供时加入手性螺旋扭转 q0*(z-Cz)
        q0: 手性波矢，提供时覆盖 eta
    返回:
        nx, ny, nz: 形状 (L, L, Lz) = (300, 300, 32) 的 numpy 数组
    """
    X, Y, Z = _build_grid()
    nx, ny, nz = _single_toron_field(X, Y, Z, center_xy, center_z, R, B,
                                     eta=eta, q0=q0)
    if normalize:
        nx, ny, nz = _normalize(nx, ny, nz)
    return nx, ny, nz


def generate_double_toron(separation=30.0,
                          center_xy=(CX, CY), center_z=CZ,
                          R=R_SKY, B=B_WALL, normalize=True,
                          eta=None, q0=None):
    """
    生成双 toron 三维指向矢场。
    两 toron 沿 x 方向相对中心对称分布，间距为 `separation`（仿真单位）。
    叠加后重新归一化以保持 |n|=1（论文采用类似的初始化策略）。

    参数:
        separation: 两 toron 中心距离 (仿真格子)
        center_xy: 双 toron 系统的整体中心
        eta, q0: 手性参数，同 generate_single_toron
    返回:
        nx, ny, nz, (c1_xy, c2_xy): 指向矢场 + 两 toron 中心坐标
    """
    Cx0, Cy0 = center_xy
    half = separation / 2.0
    c1_xy = (Cx0 - half, Cy0)
    c2_xy = (Cx0 + half, Cy0)

    X, Y, Z = _build_grid()
    n1x, n1y, n1z = _single_toron_field(X, Y, Z, c1_xy, center_z, R, B,
                                         eta=eta, q0=q0)
    n2x, n2y, n2z = _single_toron_field(X, Y, Z, c2_xy, center_z, R, B,
                                         eta=eta, q0=q0)

    # 叠加（论文为线性叠加后通过 PDE 松弛自动归一化；这里直接相加再归一化作为兜底）
    nx = n1x + n2x
    ny = n1y + n2y
    nz = n1z + n2z
    if normalize:
        nx, ny, nz = _normalize(nx, ny, nz)
    return nx, ny, nz, (c1_xy, c2_xy)


def _electric_field_ansatz_perturbation(nx, ny, nz, U, Lz_phys=10e-6,
                                        delta_eps= -3.7, eps0=8.854e-12):
    """
    【经验模型，非物理求解】简化处理电场对指向矢的影响（兜底方案）。
    论文: 电场效应来自 Eq.(1) 的 ε0·Δε·(E·n)² 项，经 PDE 松弛动力学
    改变全场指向矢分布。本项目不做该求解，改用经验缩放:
        |nz| 变为 |nz| * (1 - 0.15*(U/3.5)^2)
    Δε < 0 时电场确实使指向矢倾向 xy 平面，故缩放方向定性正确；
    但形变的空间模式与幅度是人为设定的，仅用于让 POM 图像随 U 变化。
    """
    if U == 0:
        return nx, ny, nz
    # 简化: 降低 |nz|，并放大横向分量；不做严格 PDE 松弛
    # 这只是为了让 POM 图像随 U 变化，定性复现论文 Fig.2(a-d) 的趋势
    factor = 1.0 - 0.15 * (U / 3.5) ** 2  # U=3.5V 时 |nz| 减小约 15%
    nz_new = nz * factor
    scale_lat = np.sqrt(np.maximum(1.0 - nz_new ** 2, 0) /
                        np.maximum(nx * nx + ny * ny, 1e-12))
    nx_new = nx * scale_lat
    ny_new = ny * scale_lat
    nx_new, ny_new, nz_new = _normalize(nx_new, ny_new, nz_new)
    return nx_new, ny_new, nz_new


def apply_voltage_perturbation(nx, ny, nz, U):
    """
    对外暴露的电场扰动接口。
    【经验模型】并非 Frank-Oseen 介电耦合求解；如需物理求解请用
    pde_relaxation.py 的 relax_director()（可选增强，批量生成不使用）。
    """
    return _electric_field_ansatz_perturbation(nx, ny, nz, U)


# ==================== 自检 ====================
if __name__ == "__main__":
    print(f"生成单 toron 指向矢场 (L={L}, Lz={Lz})...")
    nx, ny, nz = generate_single_toron()
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    print(f"  形状: {nx.shape}, |n| 范围: [{norm.min():.4f}, {norm.max():.4f}]")
    print(f"  nz 中心点 (150,150,16) = {nz[L//2, L//2, Lz//2]:.4f} (应≈+1)")
    print(f"  nz 边缘   (0,0,16) = {nz[0, 0, Lz//2]:.4f} (应≈-1, 背景)")
    print(f"  nz 顶面   (150,150,0) = {nz[L//2, L//2, 0]:.4f} (应≈-1, 边界)")

    print(f"\n生成双 toron 指向矢场 (separation=30)...")
    nx2, ny2, nz2, centers = generate_double_toron(separation=30.0)
    print(f"  形状: {nx2.shape}, 两 toron 中心: {centers}")
    print(f"  |n| 范围: [{np.sqrt(nx2**2+ny2**2+nz2**2).min():.4f}, "
          f"{np.sqrt(nx2**2+ny2**2+nz2**2).max():.4f}]")
    print("OK")
