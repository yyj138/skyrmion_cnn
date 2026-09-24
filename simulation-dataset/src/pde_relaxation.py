"""
兜底方案: PDE 松弛求解器（可选）
=====================================================
复现论文 Methods: Minimising Frank-Oseen free energy 节的 PDE 松弛动力学:

    γ ∂n_μ/∂t = -δF/δn_μ  (Eq.2)

数值方法:
  - 空间导数: 中心差分 (x,y 周期；z 单侧)
  - 时间积分: 4 阶 Runge-Kutta (RK4)
  - 约束 |n|=1: 每步投影到垂直于 n 的平面 (I - n⊗n)h

分子场 h = -δF/δn 的分量:
  h_splay = K1 ∇(∇·n)
  h_twist = -K2 [2 f curl n + ∇f × n]   其中 f = n·curl n - q0, q0 = 2π/η
  h_bend  = -K3 [curl n × g + curl(g × n)]  其中 g = n × curl n
  h_elec  = -ε0 Δε (E·n) E

调用:
  nx, ny, nz = generate_single_toron(eta=eta)  # ansatz 初值
  nx, ny, nz = relax_director(nx, ny, nz, eta=eta, U=U, n_steps=200)
  # 若 PDE 松弛调不通 (报错/发散)，直接使用 ansatz 结果即可。
"""
import numpy as np
from config import (L, Lz, DX, K1, K2, K3, EPS0, DELTA_EPS, GAMMA)


def _grad(a, dx=DX):
    """3D 标量/矢量场的梯度。返回 ∂a/∂x, ∂a/∂y, ∂a/∂z。
    x,y 周期；z 单侧差分。"""
    def periodic_axis(a, axis):
        return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / (2.0 * dx)
    def one_sided_z(a):
        out = np.zeros_like(a)
        out[:, :, 1:-1] = (a[:, :, 2:] - a[:, :, :-2]) / (2.0 * dx)
        out[:, :, 0] = (-3*a[:, :, 0] + 4*a[:, :, 1] - a[:, :, 2]) / (2.0 * dx)
        out[:, :, -1] = (3*a[:, :, -1] - 4*a[:, :, -2] + a[:, :, -3]) / (2.0 * dx)
        return out
    return periodic_axis(a, 0), periodic_axis(a, 1), one_sided_z(a)


def _div(a_x, a_y, a_z, dx=DX):
    """散度 ∇·a"""
    ax_dx, _, _ = _grad(a_x, dx)
    _, ay_dy, _ = _grad(a_y, dx)
    _, _, az_dz = _grad(a_z, dx)
    return ax_dx + ay_dy + az_dz


def _curl(a_x, a_y, a_z, dx=DX):
    """旋度 ∇×a。返回 (cx, cy, cz)。"""
    _, ay_dz = _grad(a_y, dx)[2], None
    # 重新计算每个分量的导数
    ax_dx, ax_dy, ax_dz = _grad(a_x, dx)
    ay_dx, ay_dy, ay_dz = _grad(a_y, dx)
    az_dx, az_dy, az_dz = _grad(a_z, dx)
    cx = az_dy - ay_dz
    cy = ax_dz - az_dx
    cz = ay_dx - ax_dy
    return cx, cy, cz


def _molecular_field(nx, ny, nz, eta, U=0.0,
                     K1=K1, K2=K2, K3=K3,
                     eps0=EPS0, delta_eps=DELTA_EPS,
                     dx_phys=DX, Lz_phys=10e-6):
    """
    计算 Frank-Oseen 自由能的分子场 h = -δF/δn。
    返回 h_x, h_y, h_z，形状同 nx。
    """
    eta_phys = eta * dx_phys
    q0 = 2.0 * np.pi / eta_phys
    Ez = -U / Lz_phys  # 电场 z 分量 (V/m)

    # 1) 散度与旋度
    div_n = _div(nx, ny, nz, dx_phys)
    curl_x, curl_y, curl_z = _curl(nx, ny, nz, dx_phys)

    # 2) Splay: h_splay = K1 ∇(∇·n)
    h_splay_x, h_splay_y, h_splay_z = _grad(div_n, dx_phys)
    h_splay_x *= K1; h_splay_y *= K1; h_splay_z *= K1

    # 3) Twist: h_twist = -K2 [2 f curl n + ∇f × n]
    #    f = n·curl n - q0
    f = nx * curl_x + ny * curl_y + nz * curl_z - q0
    # ∇f
    f_dx, f_dy, f_dz = _grad(f, dx_phys)
    # ∇f × n = (f_dy * nz - f_dz * ny, f_dz * nx - f_dx * nz, f_dx * ny - f_dy * nx)
    cross_x = f_dy * nz - f_dz * ny
    cross_y = f_dz * nx - f_dx * nz
    cross_z = f_dx * ny - f_dy * nx
    h_twist_x = -K2 * (2.0 * f * curl_x + cross_x)
    h_twist_y = -K2 * (2.0 * f * curl_y + cross_y)
    h_twist_z = -K2 * (2.0 * f * curl_z + cross_z)

    # 4) Bend: h_bend = -K3 [curl n × g + curl(g × n)]
    #    g = n × curl n
    gx = ny * curl_z - nz * curl_y
    gy = nz * curl_x - nx * curl_z
    gz = nx * curl_y - ny * curl_x
    # curl n × g
    cxg_x = curl_y * gz - curl_z * gy
    cxg_y = curl_z * gx - curl_x * gz
    cxg_z = curl_x * gy - curl_y * gx
    # g × n
    gn_x = gy * nz - gz * ny
    gn_y = gz * nx - gx * nz
    gn_z = gx * ny - gy * nx
    # curl(g × n)
    cgn_x, cgn_y, cgn_z = _curl(gn_x, gn_y, gn_z, dx_phys)
    h_bend_x = -K3 * (cxg_x + cgn_x)
    h_bend_y = -K3 * (cxg_y + cgn_y)
    h_bend_z = -K3 * (cxg_z + cgn_z)

    # 5) Electric: h_elec = -ε0 Δε (E·n) E
    # E = (0, 0, Ez), E·n = Ez * nz
    En = Ez * nz
    h_elec_x = 0.0
    h_elec_y = 0.0
    h_elec_z = -eps0 * delta_eps * En * Ez

    hx = h_splay_x + h_twist_x + h_bend_x + h_elec_x
    hy = h_splay_y + h_twist_y + h_bend_y + h_elec_y
    hz = h_splay_z + h_twist_z + h_bend_z + h_elec_z
    return hx, hy, hz


def _project(nx, ny, nz, hx, hy, hz):
    """投影到垂直于 n 的平面: h_perp = h - (n·h) n"""
    nh = nx * hx + ny * hy + nz * hz
    hx_p = hx - nh * nx
    hy_p = hy - nh * ny
    hz_p = hz - nh * nz
    return hx_p, hy_p, hz_p


def _normalize(nx, ny, nz, eps=1e-12):
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    norm = np.maximum(norm, eps)
    return nx / norm, ny / norm, nz / norm


def _apply_bc(nx, ny, nz, bc_sign=-1.0):
    """
    强制边界条件: z=0 与 z=Lz-1 处 n = bc_sign * ẑ
    （ansatz 背景为 -ẑ，故默认 bc_sign=-1 以匹配 ansatz；
      论文 "perpendicular to surfaces" 在向列液晶中 n 与 -n 等价）
    """
    nx[:, :, 0] = 0.0; ny[:, :, 0] = 0.0; nz[:, :, 0] = bc_sign
    nx[:, :, -1] = 0.0; ny[:, :, -1] = 0.0; nz[:, :, -1] = bc_sign
    return nx, ny, nz


def _euler_step(nx, ny, nz, eta, U, dt, gamma=GAMMA, bc_sign=-1.0, **kwargs):
    """单步 Euler: n_new = n + (dt/γ) h_perp，然后归一化。"""
    hx, hy, hz = _molecular_field(nx, ny, nz, eta, U, **kwargs)
    hx, hy, hz = _project(nx, ny, nz, hx, hy, hz)
    nx_new = nx + (dt / gamma) * hx
    ny_new = ny + (dt / gamma) * hy
    nz_new = nz + (dt / gamma) * hz
    # 强制边界条件: z=0 与 z=Lz-1 处 n = bc_sign * ẑ (匹配 ansatz 背景)
    nx_new, ny_new, nz_new = _apply_bc(nx_new, ny_new, nz_new, bc_sign)
    return _normalize(nx_new, ny_new, nz_new)


def _rk4_step(nx, ny, nz, eta, U, dt, gamma=GAMMA, bc_sign=-1.0, **kwargs):
    """4 阶 Runge-Kutta 步进 (论文 Eq.2 数值方法)。"""
    # k1
    hx1, hy1, hz1 = _molecular_field(nx, ny, nz, eta, U, **kwargs)
    hx1, hy1, hz1 = _project(nx, ny, nz, hx1, hy1, hz1)
    k1x = (dt / gamma) * hx1; k1y = (dt / gamma) * hy1; k1z = (dt / gamma) * hz1

    # k2 (n + k1/2)
    nx2, ny2, nz2 = _normalize(nx + 0.5 * k1x, ny + 0.5 * k1y, nz + 0.5 * k1z)
    nx2, ny2, nz2 = _apply_bc(nx2, ny2, nz2, bc_sign)
    hx2, hy2, hz2 = _molecular_field(nx2, ny2, nz2, eta, U, **kwargs)
    hx2, hy2, hz2 = _project(nx2, ny2, nz2, hx2, hy2, hz2)
    k2x = (dt / gamma) * hx2; k2y = (dt / gamma) * hy2; k2z = (dt / gamma) * hz2

    # k3 (n + k2/2)
    nx3, ny3, nz3 = _normalize(nx + 0.5 * k2x, ny + 0.5 * k2y, nz + 0.5 * k2z)
    nx3, ny3, nz3 = _apply_bc(nx3, ny3, nz3, bc_sign)
    hx3, hy3, hz3 = _molecular_field(nx3, ny3, nz3, eta, U, **kwargs)
    hx3, hy3, hz3 = _project(nx3, ny3, nz3, hx3, hy3, hz3)
    k3x = (dt / gamma) * hx3; k3y = (dt / gamma) * hy3; k3z = (dt / gamma) * hz3

    # k4 (n + k3)
    nx4, ny4, nz4 = _normalize(nx + k3x, ny + k3y, nz + k3z)
    nx4, ny4, nz4 = _apply_bc(nx4, ny4, nz4, bc_sign)
    hx4, hy4, hz4 = _molecular_field(nx4, ny4, nz4, eta, U, **kwargs)
    hx4, hy4, hz4 = _project(nx4, ny4, nz4, hx4, hy4, hz4)
    k4x = (dt / gamma) * hx4; k4y = (dt / gamma) * hy4; k4z = (dt / gamma) * hz4

    # 合成
    nx_new = nx + (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
    ny_new = ny + (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
    nz_new = nz + (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0
    nx_new, ny_new, nz_new = _apply_bc(nx_new, ny_new, nz_new, bc_sign)
    return _normalize(nx_new, ny_new, nz_new)


def relax_director(nx, ny, nz, eta, U=0.0, n_steps=200, dt=1e-7,
                   method="rk4", verbose=True, check_divergence=True,
                   **kwargs):
    """
    PDE 松弛求解器: 从初值 (ansatz) 出发，沿 -∇F 方向演化至稳态。

    参数:
        nx, ny, nz: 初值指向矢场 (来自 ansatz)，形状 (L, L, Lz)
        eta: 螺距 (仿真单位)
        U: 电压 (V)
        n_steps: 松弛步数
        dt: 时间步长 (s)，默认 1e-7 (论文 Δt=1e-6 偏大易发散，此处更保守)
        method: "rk4" 或 "euler"
        check_divergence: 检测发散，若发散则提前停止并返回当前结果

    返回:
        nx, ny, nz: 松弛后的指向矢场（兜底: 若发散则返回 ansatz 原值）
    """
    if verbose:
        print(f"  PDE 松弛: η={eta}, U={U}V, method={method}, "
              f"n_steps={n_steps}, dt={dt:.1e}s")
    nx = nx.copy(); ny = ny.copy(); nz = nz.copy()
    step_fn = _rk4_step if method == "rk4" else _euler_step

    for step in range(n_steps):
        try:
            nx, ny, nz = step_fn(nx, ny, nz, eta, U, dt, **kwargs)
        except Exception as e:
            if verbose:
                print(f"  PDE 步 {step} 异常: {e}；返回当前结果 (兜底)")
            break
        if check_divergence:
            norm = np.sqrt(nx * nx + ny * ny + nz * nz)
            if not np.isfinite(norm).all() or norm.max() > 1e3:
                if verbose:
                    print(f"  PDE 步 {step} 检测到发散；返回 ansatz 兜底")
                # 兜底: 重新生成 ansatz
                from ansatz import generate_single_toron
                nx, ny, nz = generate_single_toron(eta=eta)
                return nx, ny, nz
        if verbose and (step + 1) % max(1, n_steps // 5) == 0:
            from frank_osseen import compute_free_energy
            F = compute_free_energy(nx, ny, nz, eta=eta, U=U)
            print(f"    step {step+1}/{n_steps}: F = {F:.4e} J")
    return nx, ny, nz


# ==================== 自检 ====================
if __name__ == "__main__":
    import time
    from ansatz import generate_single_toron
    from frank_osseen import compute_free_energy
    from pom_imaging import compute_pom_image

    print("=== PDE 松弛求解器自检 ===")
    for eta in [32, 40]:
        for U in [0.0, 3.5]:
            print(f"\n--- η={eta}, U={U}V ---")
            nx0, ny0, nz0 = generate_single_toron(eta=eta)
            F0 = compute_free_energy(nx0, ny0, nz0, eta=eta, U=U)
            I0 = compute_pom_image(nx0, ny0, nz0)
            print(f"  ansatz: F = {F0:.4e} J, I_mean = {I0.mean():.4f}")

            t0 = time.time()
            try:
                nx_r, ny_r, nz_r = relax_director(
                    nx0, ny0, nz0, eta=eta, U=U,
                    n_steps=50, dt=5e-8, method="euler")
                F1 = compute_free_energy(nx_r, ny_r, nz_r, eta=eta, U=U)
                I1 = compute_pom_image(nx_r, ny_r, nz_r)
                print(f"  松弛后: F = {F1:.4e} J (ΔF = {F1-F0:+.2e}), "
                      f"I_mean = {I1.mean():.4f}")
                print(f"  耗时 {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  PDE 松弛失败 ({e})；使用 ansatz 兜底")
    print("\nOK (PDE 模块为可选增强；ansatz 兜底始终可用)")
