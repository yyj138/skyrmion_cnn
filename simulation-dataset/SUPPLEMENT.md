# 补充测试数据集说明（SUPPLEMENT.md）— 交付成员 B

对应需求：仓库根目录 [`DATASET_SUPPLEMENT_REQUEST.md`](../DATASET_SUPPLEMENT_REQUEST.md)
生成脚本：[`src/generate_supplement.py`](src/generate_supplement.py)
生成日志：[`outputs/supplement/generation_log.txt`](outputs/supplement/generation_log.txt)

## 1. 概述

在原数据集（968 张）之外新增 **440 张模拟 POM 灰度图**，作为**测试数据**用于检验已训练模型对新图片的预测效果。

| 数据部分 | 图片数 | 说明 |
| --- | ---: | --- |
| 单 toron | 220 张 | η=30~40 共 11 类，每类 20 张；8 档电压均匀覆盖 |
| 双 toron | 220 张 | η=30~40 共 11 类，每类 20 张；8 档电压 + 10 档间距均衡覆盖 |
| 混合 | 440 行 | 仅合并标签表，不另生成图片 |

**本数据集仅为测试集**：不加入训练集、不参与任何训练/验证划分；原数据集与已训练模型完全未动（已复核原 `pom_images/` 仍为 974 张、原 labels 仍为 88/880/968 行）。

## 2. 目录结构

```
outputs/supplement/
├── pom_images/                     # 440 张灰度 PNG（与原 pom_images/ 完全独立）
├── labels/
│   ├── labels_supplement_single.csv   # 220 行
│   ├── labels_supplement_double.csv   # 220 行
│   └── labels_supplement_mixed.csv    # 440 行（单+双合并）
└── generation_log.txt              # 生成方法、参数与种子记录
```

> 注意：本目录**没有** splits/ 划分文件——全部图片都是测试数据，无需划分。

## 3. 生成方法（重要声明，报告可直接引用）

> 与原数据集同源：论文 Eq.(3)-(6) 解析 ansatz + 手性扭转 q0·(z−Cz) 注入 +
> 经验电压扰动（|nz|·(1−0.15·(U/3.5)²)），未运行 PDE 松弛；
> 自由能标签为对**新构型**做 Frank-Oseen 体积积分的结果；
> POM 为 Jones 矩阵正交偏振成像（λ=500nm，Δn=0.06）。

**与原数据集的唯一区别**：原数据集固定 ansatz 几何参数 R=14.4、B=0.5（论文给定值）；
本次为满足"新构型、非复制旧图"的要求，对 R（toron 半径）、B（扭转壁宽度）做小幅物理变异，
每张图均使用与原值不同的参数组合重新生成指向矢场，并重新计算自由能：

| 变体 | R (格子) | B | 用于 |
| --- | --- | --- | --- |
| v1 | 13.4 | 0.45 | 单 toron 8U/类 + 双 toron 全部 |
| v2 | 15.4 | 0.55 | 单 toron 8U/类 |
| v3 | 14.4 | 0.45 | 单 toron 4U/类（U=0.0/1.0/2.0/3.0） |

报告表述规范（沿用原口径）：
- 正确表述："在 ansatz 几何参数（R, B）上做小幅变异生成新构型，对每个新构型做 Frank-Oseen 体积积分作为标签"；
- 错误表述："得到 Frank-Oseen 极小化的自由能"、"复现 PDE 松弛平衡态解"。

## 4. 图片规格（与原数据集一致）

- 尺寸：300×300 像素（1 格子 = 0.3125 μm，物理视场 ≈ 94 μm）
- 通道：单通道灰度（PNG mode=L，0–255），归一化透射光强 ×255
- 无缩放、无对比度拉伸、无伪彩
- 文件名带变体后缀，与原图无任何重名：
  - 单：`single_eta{η}_U{U}_v{N}.png`（如 `single_eta35_U1p50_v1.png`）
  - 双：`double_eta{η}_U{U}_sep{sep}_v{N}.png`（如 `double_eta35_U2p00_sep50_v1.png`）

## 5. csv 字段含义

| 字段 | 类型 | 含义 |
|------|------|------|
| filename | str | 图片文件名（对应 supplement/pom_images/ 下文件）|
| eta | int | 胆甾相螺距 η（30–40，11 类，与原数据集类别一致）|
| U | float | 外加电压 (V)，0.0–3.5 共 8 档 |
| free_energy | float | Frank-Oseen 自由能标签 (J)，对该图新构型积分所得，量级 ~1e-13 |
| kind | str | "single" / "double" |
| n_torons | int | 1 或 2 |
| separation | float | （仅 double/ mixed csv）两 toron 中心间距（30–75，步长 5）|
| **R** | float | **新增列**：该图 ansatz toron 半径（格子单位），原数据集无此列 |
| **B** | float | **新增列**：该图 ansatz 扭转壁宽度参数，原数据集无此列 |

> R/B 列为补充元数据，仅用于追溯构型差异；成员 B 读取时按列名取所需字段即可，
> 多出的列不影响原有按 filename/eta/U/free_energy 索引的逻辑。

## 6. 参数覆盖明细

**单 toron（每 η 类 20 张）**：8U × v1 + 8U × v2 + 4U × v3 = 20 张；
电压 8 档（0.0/0.5/…/3.5V）在 v1、v2 中全覆盖，v3 补充在 U=0.0/1.0/2.0/3.0 四档。

**双 toron（每 η 类 20 张，全部 v1）**：10 档间距（30/35/…/75）× 2 个电压；
电压配对规则为 sep 索引 i → U[i mod 8] 与 U[(i+4) mod 8]，
因此 8 档电压中 4 档出现 3 次、4 档出现 2 次，间距每档恰好 2 张。

已复核：单/双 toron 每个 η 类恰好 20 张；双 toron 每个 η 类 U 覆盖 8 档、sep 覆盖 10 档。

## 7. 交付前校验结果（2026-10-22 实测）

- 440 张 PNG：尺寸全部 300×300，mode=L，uint8；
- 与原 968 张：**文件名零交集、全量 MD5 内容零重复**；补充集内部亦零重复；
- 所有图 (R,B) 均非原值 (14.4, 0.5)；
- 自由能范围：单 [−7.78e-14, 1.58e-13] J，双 [−7.19e-14, 1.54e-13] J，
  与原数据集量级一致；随 U 增大单调下降（物理趋势正确）；
- 原数据集未修改。

## 8. 成员 B 使用注意

1. **仅作测试**：用 `cnn-inversion/runs/` 下已训练模型直接对本目录图片推理，
   评估指标（分类 accuracy / 混淆矩阵，回归 R² / pred-vs-true），
   **不要**把本目录任何图片加入训练或验证；
2. 图片路径按 `outputs/supplement/pom_images/` 读取，labels 按 filename 字段索引；
3. free_energy 量级 ~1e-13 J，与原数据集相同，沿用训练时的 z-score 标准化参数
   （用训练集统计量，勿用本测试集重新拟合）；
4. 分类任务标签 = eta（11 类 30–40，与原模型类别一致）；回归标签 = U 或 free_energy；
5. 本数据集构型来自 (R,B) 变异，分布与原训练集略有差异——若测试指标下降，
   属预期内的"新构型泛化"检验结果，报告中如实说明变异幅度即可；
6. 可复现：参数网格为确定性生成（种子 20261022 仅作记录，无随机抽样），
   重跑 `python src/generate_supplement.py`（在 `simulation-dataset/src/` 下）可逐位复现。

---

相关文档：[`DATASET.md`](DATASET.md)（原 968 张数据集说明）、
[`../DATASET_SUPPLEMENT_REQUEST.md`](../DATASET_SUPPLEMENT_REQUEST.md)（需求清单）、
[`../cnn-inversion/README.md`](../cnn-inversion/README.md)（CNN 训练端）。
