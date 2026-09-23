"""
Task 2-2: Frank-Oseen 自由能数值积分
=====================================================
严格复现论文 Eq.(1) 的 Frank-Oseen 弹性自由能:

F = (1/2) ∫_V [ K1 (∇·n)² + K2 (n·∇×n − 2π/η)²
              + K3 (n×∇×n)² + ε0·Δε·(E·n)² ] d³r

输入: 三维指向矢数组 (nx, ny, nz), 形状 (L, L, Lz) = (300, 300, 32)
      螺距 η (仿真单位), 外加电压 U (V)
输出: 体系总自由能 F (单位: 焦耳，物理单位)

数值方法:
  - 空间导数采用中心差分
  - x,y 方向周期边界条件 (论文规定)
  - z 方向 (z=0 与 z=Lz-1) 采用单侧差分，边界指向矢保持垂直 (固定边界条件)
  - 积分采用梯形/中点求和 (体积元 dV = dx_phys³)
"""
import numpy as np
from config import (L, Lz, DX, K1, K2, K3, EPS0, DELTA_EPS)


def _grad_n(nx, ny, nz, dx=DX):
    """
    计算指向矢场各分量的空间梯度 (∂_x, ∂_y, ∂_z)。
    返回 dict: dnxdx, dnxdy, dnxdz, dnydx, dnydy, dnydz, dnzdx, dnzdy, dnzdz
    全部形状 (L, L, Lz)，单位: 1/m (物理)。
    边界处理: x,y 周期；z 单侧差分。
    """
    def periodic_axis(a, axis):
        """沿指定轴 (0=x, 1=y) 做周期中心差分"""
        rolled_p = np.roll(a, -1, axis=axis)
        rolled_m = np.roll(a, 1, axis=axis)
        return (rolled_p - rolled_m) / (2.0 * dx)

    def one_sided_z(a):
        """沿 z (axis=2) 做中心差分；z=0 与 z=Lz-1 用单侧差分"""
        out = np.zeros_like(a)
        # 内部用中心差分
        out[:, :, 1:-1] = (a[:, :, 2:] - a[:, :, :-2]) / (2.0 * dx)
        # 边界单侧二阶精度差分
        out[:, :, 0] = (-3 * a[:, :, 0] + 4 * a[:, :, 1] - a[:, :, 2]) / (2.0 * dx)
        out[:, :, -1] = (3 * a[:, :, -1] - 4 * a[:, :, -2] + a[:, :, -3]) / (2.0 * dx)
        return out

    grads = {
        "dnxdx": periodic_axis(nx, 0),
        "dnxdy": periodic_axis(nx, 1),
        "dnxdz": one_sided_z(nx),
        "dnydx": periodic_axis(ny, 0),
        "dnydy": periodic_axis(ny, 1),
        "dnydz": one_sided_z(ny),
        "dnzdx": periodic_axis(nz, 0),
        "dnzdy": periodic_axis(nz, 1),
        "dnzdz": one_sided_z(nz),
    }
    return grads


def _div_and_curl(nx, ny, nz, dx=DX):
    """
    计算散度 ∇·n 与旋度 ∇×n。
    返回: div_n (L,L,Lz), curl_x, curl_y, curl_z (均 L,L,Lz)
    """
    g = _grad_n(nx, ny, nz, dx)
    div_n = g["dnxdx"] + g["dnydy"] + g["dnzdz"]
    # curl n = (∂_y nz - ∂_z ny,  ∂_z nx - ∂_x nz,  ∂_x ny - ∂_y nx)
    curl_x = g["dnzdy"] - g["dnydz"]
    curl_y = g["dnxdz"] - g["dnzdx"]
    curl_z = g["dnydx"] - g["dnxdy"]
    return div_n, curl_x, curl_y, curl_z


def compute_free_energy(nx, ny, nz, eta, U=0.0,
                        K1=K1, K2=K2, K3=K3,
                        eps0=EPS0, delta_eps=DELTA_EPS,
                        dx_phys=DX, Lz_phys=10e-6,
                        return_components=False):
    """
    计算 Frank-Oseen 总自由能 (物理单位: 焦耳)。

    参数:
        nx, ny, nz: 三维指向矢场，形状 (L, L, Lz)
        eta: 螺距 (仿真单位，整数)
        U: 外加电压 (V)，电场 E = -U/Lz_phys * ẑ (V/m)
        其余物理常数见 config.py

    返回:
        F: 总自由能 (J)
        若 return_components=True, 额外返回 (F_splay, F_twist, F_bend, F_elec)
    """
    # 物理螺距
    eta_phys = eta * dx_phys

    # 散度与旋度
    div_n, curl_x, curl_y, curl_z = _div_and_curl(nx, ny, nz, dx_phys)

    # === Splay 项: K1 * (∇·n)² ===
    F_splay_dens = K1 * div_n * div_n

    # === Twist 项: K2 * (n·curl n - 2π/η)² ===
    n_dot_curl = nx * curl_x + ny * curl_y + nz * curl_z
    q0 = 2.0 * np.pi / eta_phys  # 螺旋波矢
    F_twist_dens = K2 * (n_dot_curl - q0) ** 2

    # === Bend 项: K3 * |n × curl n|² ===
    cross_x = ny * curl_z - nz * curl_y
    cross_y = nz * curl_x - nx * curl_z
    cross_z = nx * curl_y - ny * curl_x
    F_bend_dens = K3 * (cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)

    # === 电场项: ε0 Δε (E·n)² ===
    # E = -U/Lz_phys * ẑ，故 E·n = -U/Lz_phys * nz
    Ez = -U / Lz_phys
    E_dot_n = Ez * nz
    F_elec_dens = eps0 * delta_eps * (E_dot_n * E_dot_n)

    # === 体积积分 (中点求和) ===
    dV = dx_phys ** 3
    F_splay = 0.5 * np.sum(F_splay_dens) * dV
    F_twist = 0.5 * np.sum(F_twist_dens) * dV
    F_bend = 0.5 * np.sum(F_bend_dens) * dV
    F_elec = 0.5 * np.sum(F_elec_dens) * dV

    F_total = F_splay + F_twist + F_bend + F_elec

    if return_components:
        return F_total, (F_splay, F_twist, F_bend, F_elec)
    return F_total


# ==================== 自检 ====================
if __name__ == "__main__":
    from ansatz import generate_single_toron, generate_double_toron

    print("=== Frank-Oseen 自由能自检 ===")

    # 1. 均匀指向矢场 (n=ẑ) 应该只有 twist 项 (非零)，且接近 K2*(2π/η)² * V
    nz_uniform = np.ones((L, L, Lz))
    nx_uniform = np.zeros((L, L, Lz))
    ny_uniform = np.zeros((L, L, Lz))
    # 注意: 真实均匀场应满足 n·curl n = 2π/η 来消除 twist 张力，但均匀场 curl=0，
    # 故 n·curl - 2π/η = -2π/η，twist 项 = K2*(2π/η)² * V (有限值)

    for eta in [32, 36, 40]:
        for U in [0.0, 3.5]:
            F_uniform = compute_free_energy(nx_uniform, ny_uniform, nz_uniform,
                                             eta=eta, U=U)
            print(f"  均匀场 (n=ẑ) η={eta}, U={U}V: F = {F_uniform:.4e} J")

    print("\n--- 单 toron 不同 η, U 下的自由能 ---")
    print(f"{'eta':>4} {'U':>5} {'F_total (J)':>14} {'F_splay':>12} "
          f"{'F_twist':>12} {'F_bend':>12} {'F_elec':>12}")
    for eta in [32, 36, 40]:
        for U in [0.0, 1.75, 3.5]:
            nx, ny, nz = generate_single_toron()
            from ansatz import apply_voltage_perturbation
            nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
            F, comps = compute_free_energy(nx, ny, nz, eta=eta, U=U,
                                            return_components=True)
            print(f"{eta:>4} {U:>5.2f} {F:>14.4e} {comps[0]:>12.4e} "
                  f"{comps[1]:>12.4e} {comps[2]:>12.4e} {comps[3]:>12.4e}")

    print("\n--- 双 toron 不同间距下的自由能 ---")
    for sep in [20.0, 30.0, 50.0]:
        nx2, ny2, nz2, _ = generate_double_toron(separation=sep)
        F = compute_free_energy(nx2, ny2, nz2, eta=32, U=0.0)
        print(f"  separation={sep:>4.0f}: F = {F:.4e} J")
    print("OK")
