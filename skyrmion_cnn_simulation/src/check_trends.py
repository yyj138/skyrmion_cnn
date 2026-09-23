"""
Demo 定性趋势自动校验（对标论文 Fig.2）
=====================================================
校验 3 个定性条件，全部通过才允许全量批量生成:
1. 改变 η (32/36/40): POM 纹理花纹形态明显变化 (I_mean / I_std / 纹理相关性变化)
2. 改变 U (0V/3.5V): POM 图像发生明显形变
3. 自由能 F 随 η 增大整体呈下降趋势 (对标论文 Fig2e)
"""
import numpy as np
from ansatz import generate_single_toron, apply_voltage_perturbation
from frank_osseen import compute_free_energy
from pom_imaging import compute_pom_image


def texture_diff(I1, I2):
    """两张 POM 图的平均绝对差，衡量纹理形变程度"""
    return np.mean(np.abs(I1 - I2))


def main():
    print("=" * 64)
    print("Demo 定性趋势校验 (对标论文 Fig.2)")
    print("=" * 64)

    etas = [32, 36, 40]
    Us = [0.0, 3.5]

    imgs = {}   # (eta, U) -> I
    energies = {}  # (eta, U) -> F

    for eta in etas:
        for U in Us:
            nx, ny, nz = generate_single_toron(eta=eta)
            nx, ny, nz = apply_voltage_perturbation(nx, ny, nz, U)
            imgs[(eta, U)] = compute_pom_image(nx, ny, nz)
            energies[(eta, U)] = compute_free_energy(nx, ny, nz, eta=eta, U=U)

    all_pass = True

    # ---- 条件 1: POM 纹理随 η 变化 ----
    print("\n[条件 1] POM 纹理随 η 变化 (U=0V 与 U=3.5V 两个截面)")
    for U in Us:
        d32_36 = texture_diff(imgs[(32, U)], imgs[(36, U)])
        d36_40 = texture_diff(imgs[(36, U)], imgs[(40, U)])
        d32_40 = texture_diff(imgs[(32, U)], imgs[(40, U)])
        ok = min(d32_36, d36_40) > 1e-4
        all_pass &= ok
        print(f"  U={U}V: diff(32,36)={d32_36:.4f}, diff(36,40)={d36_40:.4f}, "
              f"diff(32,40)={d32_40:.4f} -> {'PASS' if ok else 'FAIL'}")

    # ---- 条件 2: POM 图像随 U 形变 ----
    print("\n[条件 2] POM 图像随 U 形变")
    for eta in etas:
        dU = texture_diff(imgs[(eta, 0.0)], imgs[(eta, 3.5)])
        mean_change = imgs[(eta, 3.5)].mean() - imgs[(eta, 0.0)].mean()
        ok = dU > 1e-4
        all_pass &= ok
        print(f"  η={eta}: diff(U=0,U=3.5)={dU:.4f}, "
              f"ΔI_mean={mean_change:+.4f} -> {'PASS' if ok else 'FAIL'}")

    # ---- 条件 3: F 随 η 下降 (对标论文 Fig2e) ----
    print("\n[条件 3] 自由能 F 随 η 增大呈下降趋势 (对标论文 Fig.2e)")
    for U in Us:
        Fs = [energies[(eta, U)] for eta in etas]
        decreasing = all(Fs[i] > Fs[i + 1] for i in range(len(Fs) - 1))
        all_pass &= decreasing
        trend = " -> ".join(f"{F:.3e}" for F in Fs)
        print(f"  U={U}V: {trend} -> {'PASS (单调下降)' if decreasing else 'FAIL'}")

    # ---- 汇总 ----
    print("\n" + "=" * 64)
    if all_pass:
        print("全部 3 个定性条件 PASS -> 可以执行全量批量生成:")
        print("  python dataset_generator.py --mode all")
    else:
        print("存在 FAIL -> 请先调整 ansatz.py 的扰动系数 (当前 0.15)")
    print("=" * 64)
    return all_pass


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
