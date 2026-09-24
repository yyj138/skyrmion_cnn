# 数据集说明（DATASET.md）— 交付成员 B

## 1. 数据集生成方式（重要声明，报告可直接引用）

> 本数据集没有采用论文原始 Frank-Oseen PDE 松弛求解得到平衡态 toron；采用论文给出的解析 ansatz 初场作为等效平衡态指向矢场。手性螺旋、电压诱导形变使用简化经验模型。Frank-Oseen 自由能标签是对 ansatz 构型直接做体积积分的结果。pde_relaxation.py 实现了完整松弛求解器，但受计算性能限制，没有用于批量数据集生成，可作为后续拓展工作。

具体简化:
- 指向矢场: 论文 Eq.(3)-(6) 解析 ansatz + 人为注入手性扭转 q0·(z-Cz)（模拟 PDE 松弛后 twist 项驱动的螺旋特征），未运行 PDE 松弛动力学；
- 电压效应: 经验缩放模型 |nz| 变为 |nz|·(1−0.15·(U/3.5)²)，非 Frank-Oseen 介电耦合求解；
- 双 toron: 两个 ansatz 场线性叠加后归一化，不含弹性相互作用；
- 电场: 静态直流 U 扫描（0–3.5V，8 档等间隔），未实现论文的脉冲宽度调制 (PWM) 电场（幅值/频率/占空比多样性）。

## 2. 目录结构

```
outputs/dataset/
├── pom_images/          # CNN 训练用灰度图 (300×300, 单通道, 0-255 PNG)
├── labels/              # 标签 csv
│   ├── labels_single.csv   # 单 toron (88 张)
│   ├── labels_double.csv   # 双 toron (880 张)
│   ├── labels_mixed.csv    # 单+双混合 (968 张)
│   └── labels_demo.csv     # 6 张最小验证样本
└── splits/              # 已划分好的 train/val/test 表 (7 个任务)
    ├── split_pitch_classification_single.csv              # 单 toron 螺距分类（85/15）
    ├── split_voltage_regression_single.csv
    ├── split_free_energy_regression_single.csv
    ├── split_free_energy_regression_double.csv
    ├── split_free_energy_regression_mixed.csv
    ├── split_voltage_regression_mixed.csv
    └── split_pitch_classification_mixed.csv
```

另: `outputs/visualizations/presentation/` 下的 `*_pres.png` 是仿论文紫调伪彩展示图，
**仅用于报告/PPT，绝对不要送入 CNN 训练**。训练只用 `pom_images/` 的灰度图。

## 3. 图片规格

- 尺寸: 300×300 像素（仿真晶格单位，1 格子 = 0.3125 μm，对应物理 94 μm 视场）
- 通道: 单通道灰度（PNG mode=L，0–255）
- 强度: 归一化 POM 透射光强 ×255（正交偏振，λ=500nm，Δn=0.06）
- 无缩放、无对比度拉伸、无伪彩 —— 原始琼斯矩阵计算结果

## 4. csv 字段含义

| 字段 | 类型 | 含义 |
|------|------|------|
| filename | str | 图片文件名（对应 pom_images/ 下文件）|
| eta | int | 胆甾相螺距 η（仿真晶格单位，30–40；η<30 无 toron 已过滤）|
| U | float | 外加电压 (V)，0.0–3.5 |
| free_energy | float | Frank-Oseen 自由能标签 (J)。注意: ansatz 构型体积积分值，非平衡稳态自由能 |
| kind | str | "single" / "double" |
| n_torons | int | toron 数量 1 或 2 |
| separation | float | （仅双 toron csv）两 toron 中心间距（晶格单位，30–75）|
| split | str | （仅 splits/ 下 csv）"train" / "val" / "test" |

## 5. 数据集划分比例（严格按论文规定，seed=42）

| 任务 | train | val | test |
|------|-------|-----|------|
| 螺距分类（单 toron）| 85% | — | 15% |
| 电压回归（单 toron）| 80% | 15% | 5% |
| 自由能回归（单 toron）| 80% | 15% | 5% |
| 自由能回归（双 toron）| 80% | 15% | 5% |
| 混合数据集（各任务）| 76.5% | 13.5% | 10% |

> 注：成员 B 为 ① 螺距分类任务另行派生了一份**分层抽样**划分表 `split_pitch_classification_single_stratified.csv`（保证 train/val/test 均覆盖全部 11 个 η 类别）。该文件属成员 B 的产出，存放于 `cnn-inversion/data/`，不在本目录内。

划分依据: `splits/split_*.csv` 的 split 列。成员 B 直接按 csv 读取即可，
**请勿重新随机划分**，保证与成员 A 的校验结果一致、可复现。

## 6. 成员 B 使用注意

1. 训练图片只从 `pom_images/` 读取，按 `splits/split_*.csv` 的 filename 字段索引；
2. `presentation/` 的伪彩图禁止入模；
3. 在线数据增强按论文: 仅 0–180° 旋转 + 水平/垂直翻转，不保存增强图片，无缩放裁剪；
4. 分类任务标签 = eta（11 类: 30–40）；回归任务标签 = U 或 free_energy；
5. free_energy 量级 ~1e-13 J，建议训练前做标准化（如 z-score）以稳定收敛；
6. 若发现标签-图像不匹配（filename 对不上、NaN 等），立即反馈成员 A，勿自行修改 csv。

---

相关文档：[`../README.md`](../README.md)（仓库总览）、[`README.md`](README.md)（仿真方法与本子项目说明）、[`../cnn-inversion/README.md`](../cnn-inversion/README.md)（CNN 训练端）。
