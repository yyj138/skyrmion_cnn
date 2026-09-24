# 大创项目成员 B 最终交付说明

> 本文中出现的代码/数据路径，除特别说明外均**相对于仓库根目录**；文内可点击的 Markdown 链接使用相对本文档的路径。

## 1. 项目概述

本项目复现论文 **Terroa, Tasinkevych & Dias, _Convolutional neural network analysis of optical texture patterns in liquid-crystal skyrmions_, Scientific Reports 15:10921 (2025)**（DOI: [10.1038/s41598-025-89699-2](https://doi.org/10.1038/s41598-025-89699-2)，原文副本：[`../paper/s41598-025-89699-2.pdf`](../paper/s41598-025-89699-2.pdf)）中的图像参数反演任务。成员 B 负责 CNN 模型搭建、训练、评估和结果整理。

> **勘误（2026-09-24 仓库整理时修正）**：本节此前把复现对象写为《Deep learning-based quantitative analysis of three-dimensional topological structures in chiral nematic liquid crystals》，该标题有误，正确对象为上述 Sci. Rep. 论文。

当前实现覆盖以下 7 个任务（编号与仓库根 README 及各 `runs/` 目录保持一致）：

| # | 任务 | 结果目录 |
|---|------|----------|
| ① | 单 toron 螺距分类 | `cnn-inversion/runs/pitch_final/` |
| ② | 单 toron 电压回归 | `cnn-inversion/runs/voltage/` |
| ③ | 单 toron 自由能回归 | `cnn-inversion/runs/energy_single/` |
| ④ | 双 toron 自由能回归 | `cnn-inversion/runs/energy_double/` |
| ⑤ | 混合数据 自由能回归 | `cnn-inversion/runs/mixed_energy/` |
| ⑥ | 混合数据 电压回归 | `cnn-inversion/runs/mixed_voltage/` |
| ⑦ | 混合数据 螺距分类 | `cnn-inversion/runs/mixed_pitch_final/` |

> **口径统一说明**：本文件此前把 ⑤⑥⑦ 的顺序记为「混合螺距分类 → 混合电压回归 → 混合能量回归」，与 `runs/` 目录及根 README 的编号不一致，现已统一为上表口径。「孤子」与「toron」指同一对象，全文统一使用 toron。

数据由参数化平衡态近似生成，双 toron 采用线性叠加近似，未进行完整偏微分方程松弛计算。这属于实施方案允许的降级路线，仍保留了「参数生成图像、CNN 反演参数、定量评估」的完整技术链条，最终汇报时应如实说明。

## 2. 交付内容

项目中的主要交付物如下：

| 内容 | 路径 |
| --- | --- |
| 数据读取和增强 | `cnn-inversion/src/data.py` |
| CNN 模型定义 | `cnn-inversion/src/models.py` |
| 训练程序 | `cnn-inversion/src/train.py` |
| 评估和绘图程序 | `cnn-inversion/src/evaluate.py` |
| 快速自检程序 | `cnn-inversion/src/smoke_test.py` |
| 依赖环境 | `cnn-inversion/requirements.txt` |
| 数据集及划分 | `simulation-dataset/outputs/dataset/` |
| 模型、日志、指标和图片 | `cnn-inversion/runs/` |
| 论文结构选择说明 | `docs/PAPER_BASIS.md` |
| 训练记录 | `docs/TRAINING_RECORD.md` |

正式成果目录应采用下表所列版本：

| 任务 | 正式结果目录 | 测试结果 |
| --- | --- | --- |
| 单 toron 螺距分类 | `cnn-inversion/runs/pitch_final/` | Accuracy = 1.0000 |
| 单 toron 电压回归 | `cnn-inversion/runs/voltage/` | R² = 0.8761，MSE = 0.1549，MAE = 0.2602 |
| 单 toron 自由能回归 | `cnn-inversion/runs/energy_single/` | R² = 0.9988 |
| 双 toron 自由能回归 | `cnn-inversion/runs/energy_double/` | R² = 0.9985 |
| 混合 自由能回归 | `cnn-inversion/runs/mixed_energy/` | R² = 0.9989 |
| 混合 电压回归 | `cnn-inversion/runs/mixed_voltage/` | R² = 0.9952，MSE = 0.00714，MAE = 0.0565 |
| 混合 螺距分类 | `cnn-inversion/runs/mixed_pitch_final/` | Accuracy = 0.9897 |

每个完整运行目录包含模型、数据划分信息、训练日志、测试指标和结果图片；核心文件为 `model.keras`、`meta.json`、`training_log.csv`、`metrics.json`、`loss_curve.png`，分类任务另含混淆矩阵，回归任务另含预测值与真实值对比图。

## 3. 实现依据和训练设置

- 优化器：Adam
- 初始学习率：0.001
- 分类损失：交叉熵
- 回归损失：均方误差
- 随机种子：42
- 图像尺寸：300 × 300，灰度图
- 网络卷积块数：螺距 4 块、电压 3 块、自由能 4 块
- 单 toron 螺距最终划分：训练 64、验证 11、测试 13；训练、验证和测试均覆盖 30 至 40 的全部 11 个类别

论文正文与网络结构示意图对螺距模型卷积块数量存在不一致，本项目以结构图为主要依据采用 4 块；电压模型依据图注采用 3 块；自由能模型采用 4 块。该取舍已在 `docs/PAPER_BASIS.md` 中记录。

## 4. 可复现性和结果边界

`cnn-inversion/runs/pitch_final/` 使用当前动态确定性增强：相同种子可复现，不同训练轮次仍会改变增强方式。

`cnn-inversion/runs/mixed_pitch_final/` 使用动态确定性增强重训 45 轮。相同 seed 的独立数据流和训练曲线均已验证可重现，不同 epoch 的增强结果会变化。考虑混合图像对方向变换较敏感，每轮约 10% 样本接受旋转或翻转，其余保留原图。测试集 Accuracy 为 0.9897。

其余五个回归模型为已验证的历史冻结结果，模型加载和测试指标均已复核；历史训练采用旧版随机增强，因此模型与指标可以交付，但从头训练时不保证获得逐轮完全相同的日志。

电压单 toron 测试集仅有 4 个样本，R² 对个别样本较敏感，因此其 0.8761 结果应结合测试集规模解释。自由能标签数量级很小，汇报时以 R² 为主要指标，避免只展示接近零的 MSE 和 MAE。

## 5. 不纳入正式交付的目录

以下目录均已从最终交付中**排除**（未包含在本仓库内），列出以备追溯：

- `runs/mixed_pitch_retrained/`：固定增强复核训练，应用户要求中途停止，未生成模型。
- `runs/mixed_pitch_v2/`：固定逐样本增强得到的历史对照，已由动态增强的 `runs/mixed_pitch_final/` 替代。
- `runs/mixed_pitch/`：⑦ 任务的旧失败存档（loss 停在 ln(11)≈2.397），作为排查过程证据保留在原始工作区，未纳入本仓库。
- `runs/pitch/`：旧数据划分未覆盖全部螺距类别，已由 `runs/pitch_final/` 替代。
- `legacy_defect_experiment/`：向列相液晶拓扑缺陷实验复现项目，与本次 skyrmion CNN 主交付无关，**不属于本仓库**（本仓库内不存在该目录，相关交叉引用已清理）。

## 6. 当前验收结论

数据文件完整，7 个正式模型均可加载，保存指标已通过重新推理核对；单 toron 螺距最终模型已使用覆盖全部类别的新划分重新训练。代码还通过了小规模端到端冒烟测试。因此，成员 B 的核心技术成果已经形成，可以交给成员 A 合并。

回归任务的 1:1 参考线已经修复，5 个回归目录的图片均已重新生成；`cnn-inversion/README.md` 也已更新为两个最终分类目录。

仓库整理阶段已完成以下打包与汇报整理工作：

1. 已通过 `.gitignore` 排除 `.idea/`、`__pycache__/`、`_smoke/`、虚拟环境与编辑器临时文件；中断训练目录、历史对照目录与 `legacy_defect_experiment/` 未纳入仓库。
2. 汇报材料中需明确写出数据生成采用降级近似、论文结构冲突的处理方式，以及单 toron 电压测试集较小这一限制（已在仓库根 `README.md` 第 6、7 节写明）。

上述工作不需要重新训练模型。

## 7. 交接说明

本文件记录的是截至 2026 年 9 月 15 日的真实状态（2026-09-24 仓库整理时修正了第 1 节的论文标题与任务编号口径）。正式评估应只读取第 2 节列出的 7 个目录，不应将中断训练目录当作实验结果。除非更换数据、修改网络结构或要求补充多随机种子统计，否则目前不需要重新训练。
