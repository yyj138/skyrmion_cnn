# 液晶 skyrmion 光学纹理的 CNN 分析与定量反演

> 大学生创新创业训练计划（大创）项目 —— 复现论文 *Convolutional neural network analysis of optical texture patterns in liquid-crystal skyrmions*，并在此基础上完成一次完整的「物理仿真生成数据 → 深度学习反演参数 → 定量评估」闭环。

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Paper](https://img.shields.io/badge/Sci.%20Rep.-15%3A10921%20(2025)-1f6feb)](https://doi.org/10.1038/s41598-025-89699-2)

**论文**：Terroa, Tasinkevych & Dias, *Convolutional neural network analysis of optical texture patterns in liquid-crystal skyrmions*, **Scientific Reports 15:10921 (2025)**.
DOI: [10.1038/s41598-025-89699-2](https://doi.org/10.1038/s41598-025-89699-2) ｜ 原文副本：[`paper/s41598-025-89699-2.pdf`](paper/s41598-025-89699-2.pdf)

---

## 1. 项目简介

胆甾相液晶中的 **toron / skyrmion** 三维拓扑结构，其偏振光学显微（POM）纹理与自身螺距 η、外场电压 U、自由能之间存在非线性映射。本项目按论文技术路线，把这一「结构参数 → 光学纹理」正问题用物理仿真实现，再用卷积神经网络求解「光学纹理 → 结构参数」的反问题。

两名成员的全部代码均在本仓库内，各自目录**相互独立、无代码依赖**，仅通过一个数据接口（`simulation-dataset/outputs/dataset/` 的图片 + CSV 标签 + 划分表）衔接。

| 成员 | 职责 | 目录 | 主要产出 |
|------|------|------|----------|
| **成员 A** | 物理仿真、光学后处理、数据集工程 | [`simulation-dataset/`](simulation-dataset/) | toron 三维解析 ansatz、Frank-Oseen 自由能积分、3D 琼斯矩阵 POM 成像、968 张 300×300 灰度图数据集 + 标签 + 划分表 |
| **成员 B** | CNN 建模、训练、评估、结果整理 | [`cnn-inversion/`](cnn-inversion/) | 3 套 Keras 网络、7 个任务的可复现训练与评估管线、`runs/` 全套产物、论文依据对照文档 |

总方案文档：[`docs/大创完整实施方案（2人全部负责代码，无时间线、含分工、任务清单、预期结果、降级预案、交付物）.docx`](docs/大创完整实施方案（2人全部负责代码，无时间线、含分工、任务清单、预期结果、降级预案、交付物）.docx)

---

## 2. 目录结构

```
.
├── README.md                     # 本文件（仓库总览）
├── CITATION.cff                  # 引用信息
├── .gitignore / .gitattributes
│
├── cnn-inversion/                # ── 成员 B：CNN 反演 ──────────────────
│   ├── README.md                 #   代码逐文件说明、7 个任务的复现命令
│   ├── requirements.txt          #   锁定版本依赖（tensorflow-cpu==2.21.0 等）
│   ├── data/                     #   成员 B 派生的划分表（① 任务使用的分层抽样版本）
│   ├── src/
│   │   ├── models.py             #   3 个 Keras 模型，每行代码附论文出处
│   │   ├── data.py               #   数据加载 + per-image z-score + 确定性增强
│   │   ├── train.py              #   训练入口（Adam lr=1e-3、EarlyStopping）
│   │   ├── evaluate.py           #   测试集评估 + 论文风格绘图
│   │   └── smoke_test.py         #   端到端冒烟测试（合成数据，秒级）
│   └── runs/                     #   7 个正式训练任务的全套产物
│       ├── pitch_final/          #     ① 单 toron 螺距分类
│       ├── voltage/              #     ② 单 toron 电压回归
│       ├── energy_single/        #     ③ 单 toron 自由能回归
│       ├── energy_double/        #     ④ 双 toron 自由能回归
│       ├── mixed_energy/         #     ⑤ 混合 自由能回归
│       ├── mixed_voltage/        #     ⑥ 混合 电压回归
│       └── mixed_pitch_final/    #     ⑦ 混合 螺距分类
│
├── simulation-dataset/           # ── 成员 A：物理仿真与数据集工程 ─────
│   ├── README.md                 #   仿真方法、关键参数（论文 Table 1）、简化声明
│   ├── DATASET.md                #   数据集交付说明：字段含义、划分比例、使用禁忌
│   ├── requirements.txt          #   numpy / matplotlib / pillow / tqdm
│   ├── src/                      #   10 个脚本
│   │   ├── config.py             #     全部仿真参数（对齐论文 Table 1）
│   │   ├── ansatz.py             #     toron 三维解析 ansatz（论文 Eq.3-6）
│   │   ├── frank_osseen.py       #     Frank-Oseen 自由能积分（论文 Eq.1）
│   │   ├── pom_imaging.py        #     3D 琼斯矩阵 POM 成像（向量化）
│   │   ├── dataset_generator.py  #     批量生成 + 数据集划分
│   │   ├── visualize.py          #     数据校验可视化
│   │   ├── pde_relaxation.py     #     可选：PDE 松弛求解器（未用于批量生成）
│   │   ├── main.py               #     最小验证 demo（6 组样本）
│   │   ├── check_trends.py       #     定性物理趋势自动校验
│   │   └── check_deliverables.py #     交付物完整性校验
│   └── outputs/
│       ├── dataset/
│       │   ├── pom_images/       #     968 张 300×300 灰度 PNG（CNN 训练唯一图源）
│       │   ├── labels/           #     4 个标签 CSV
│       │   └── splits/           #     7 个 train/val/test 划分表
│       └── visualizations/       #     截面图、POM gallery、标签分布、presentation/ 展示图
│
├── docs/                         # ── 项目文档 ────────────────────────
│   ├── PAPER_BASIS.md            #   每条代码的论文依据 + 14 条 ASSUMPTION
│   ├── TRAINING_RECORD.md        #   7 个任务的训练与诊断全记录
│   ├── FINAL_DELIVERY.md         #   成员 B 最终交付说明与验收结论
│   └── 大创完整实施方案（...）.docx
│
└── paper/                        # ── 参考文献 ────────────────────────
    └── s41598-025-89699-2.pdf
```

---

## 3. 环境依赖

两个子项目依赖不重叠，请**分别**安装。推荐使用虚拟环境。

### 3.1 成员 B：CNN 训练与评估

Python **3.11.x**（`cnn-inversion/requirements.txt` 按 3.11 / 3.13 双版本验证）。

```bash
cd cnn-inversion
python -m venv .venv

# Windows PowerShell
.venv\Scripts\activate
# Git Bash / WSL
# source .venv/bin/activate

pip install -r requirements.txt
```

| 包 | 版本 | 说明 |
|---|---|---|
| `tensorflow-cpu` | 2.21.0 | 内含 Keras 3；论文原始实现同为 Keras |
| `numpy` | 2.4.6 (Py<3.12) / 2.5.1 (Py≥3.12) | 2.5.x 要求 Python ≥ 3.12 |
| `pandas` | 3.0.3 | 标签 CSV 读取 |
| `pillow` | 12.3.0 | 图像读取 |
| `matplotlib` | 3.11.1 | 曲线 / 混淆矩阵 / 散点图 |

> **GPU 说明**：TensorFlow ≥ 2.11 在 Windows 原生不再支持 CUDA，本任务为 CPU 训练（实测足够）。

### 3.2 成员 A：物理仿真与数据集生成

```bash
cd simulation-dataset
pip install -r requirements.txt
```

| 包 | 用途 |
|---|---|
| `numpy` | 三维指向矢场、自由能积分 |
| `matplotlib` | 校验可视化 |
| `pillow` | PNG 读写 |
| `tqdm` | 批量生成进度条 |

---

## 4. 快速开始

### 4.1 冒烟测试（无需真实数据集，约 10 秒）

```bash
cd cnn-inversion
python src/smoke_test.py
```

用 120 张合成图跑通「读数据 → 增强 → 训练 → 评估 → 出图」全链路，末尾打印 `SMOKE TEST PASSED`。

### 4.2 物理仿真最小验证（约 1 分钟）

```bash
cd simulation-dataset/src
python main.py          # 6 组样本跑通 ansatz → 自由能 → POM → CSV 全链路
python check_trends.py  # 3 个定性物理趋势自动校验
```

> ⚠️ **必须在 `simulation-dataset/src/` 目录下运行**：`config.py` 中的输出路径为相对当前工作目录的 `../outputs`，在仓库根目录直接执行会写到错误位置。

### 4.3 复现全部 7 个训练任务

```bash
cd cnn-inversion
DS=../simulation-dataset/outputs/dataset

# ① 单 toron 螺距分类
python src/train.py --task pitch --csv $DS/labels/labels_single.csv --images $DS/pom_images \
  --split-csv data/split_pitch_classification_single_stratified.csv \
  --out runs/pitch_final --patience 30 --epochs 250

# ② 单 toron 电压回归
python src/train.py --task voltage --csv $DS/labels/labels_single.csv --images $DS/pom_images \
  --split-csv $DS/splits/split_voltage_regression_single.csv --out runs/voltage --patience 30

# ③ 单 toron 自由能回归
python src/train.py --task energy --csv $DS/labels/labels_single.csv --images $DS/pom_images \
  --split-csv $DS/splits/split_free_energy_regression_single.csv --out runs/energy_single --patience 30

# ④ 双 toron 自由能回归
python src/train.py --task energy --csv $DS/labels/labels_double.csv --images $DS/pom_images \
  --split-csv $DS/splits/split_free_energy_regression_double.csv --out runs/energy_double --patience 30

# ⑤ 混合 自由能回归
python src/train.py --task energy --csv $DS/labels/labels_mixed.csv --images $DS/pom_images \
  --split-csv $DS/splits/split_free_energy_regression_mixed.csv --out runs/mixed_energy --patience 30

# ⑥ 混合 电压回归
python src/train.py --task voltage --csv $DS/labels/labels_mixed.csv --images $DS/pom_images \
  --split-csv $DS/splits/split_voltage_regression_mixed.csv --out runs/mixed_voltage --patience 30

# ⑦ 混合 螺距分类 —— 必须加 --det-aug，否则在线增强不可复现（见 docs/TRAINING_RECORD.md 第 7 节）
python src/train.py --task pitch --csv $DS/labels/labels_mixed.csv --images $DS/pom_images \
  --split-csv $DS/splits/split_pitch_classification_mixed.csv --out runs/mixed_pitch_final \
  --det-aug --aug-prob 0.1 --patience 80 --epochs 45
```

评估（对任意已训练目录）：

```bash
cd cnn-inversion
python src/evaluate.py --run runs/pitch_final
```

主要参数：`--task {pitch,voltage,energy}`、`--csv`、`--images`、`--split-csv`、`--out`、`--seed`（默认 42）、`--patience`、`--aug-prob`、`--det-aug`。完整说明见 [`cnn-inversion/README.md`](cnn-inversion/README.md)。

### 4.4 重新生成数据集（约 4 小时）

```bash
cd simulation-dataset/src
python dataset_generator.py --mode all --n_voltages 8
```

---

## 5. 复现结果

全部指标取自各 `runs/*/metrics.json` 的磁盘实际值（已对模型重新推理核对）。

| # | 任务 | 正式目录 | 测试集 | 实测指标 | 论文基线 |
|---|------|----------|--------|----------|----------|
| ① | 单 toron 螺距分类 | `cnn-inversion/runs/pitch_final/` | 13 | **Accuracy = 1.0000** | ≈ 0.96 |
| ② | 单 toron 电压回归 | `cnn-inversion/runs/voltage/` | **4** | **R² = 0.8761**，MSE = 0.1549，MAE = 0.2602 | ≈ 0.94 |
| ③ | 单 toron 自由能回归 | `cnn-inversion/runs/energy_single/` | **4** | **R² = 0.9988** | ≈ 0.96 |
| ④ | 双 toron 自由能回归 | `cnn-inversion/runs/energy_double/` | 44 | **R² = 0.9985** | ≈ 0.999 |
| ⑤ | 混合 自由能回归 | `cnn-inversion/runs/mixed_energy/` | 97 | **R² = 0.9989** | — |
| ⑥ | 混合 电压回归 | `cnn-inversion/runs/mixed_voltage/` | 97 | **R² = 0.9952**，MSE = 0.00714，MAE = 0.0565 | — |
| ⑦ | 混合 螺距分类 | `cnn-inversion/runs/mixed_pitch_final/` | 97 | **Accuracy = 0.9897**（错 1 张） | 无数值基准¹ |

¹ 论文未给出该任务的数值准确率，正文仅定性描述（p.6–7，Fig.5e）。可比参照为**同架构的单 toron 分类 ≈0.96**（p.3，Fig.2h）。注意 p.7 结论段的 *"coefficients of determination exceeding 0.94 in all the cases"* 指的是**回归**，不可用作分类基准。

每个任务目录包含：`model.keras`（权重）、`meta.json`（划分表 + 类别表 + 标签标准化统计量 + 停轮 epoch，可复现凭证）、`training_log.csv`（逐 epoch 日志）、`metrics.json`、`loss_curve.png`，分类任务另含 `confusion_matrix.png`，回归任务另含 `pred_vs_true.png`（1:1 参考线 + 标准差阴影，复刻论文 Fig.3d/3g 图注风格）。

---

## 6. 数据生成方法与简化声明

> **本节为学术诚信声明，撰写报告或论文时请沿用以下表述。**

本数据集**没有**采用论文原始方法（Frank-Oseen PDE 松弛求解平衡态 toron）；而是采用论文给出的**解析 ansatz 初场作为等效平衡态指向矢场**，手性螺旋与电压诱导形变使用简化经验模型。自由能标签是对 ansatz 构型直接做体积积分的结果，**不是自由能极小化求解的结果**。

具体简化：

1. 主流程不运行 PDE 松弛动力学，直接将解析 ansatz 场作为等效平衡态；
2. 手性螺旋扭转 `q0·(z-Cz)` 为人为注入，用于模拟 twist 项驱动的螺旋特征，并非自由能极小化自然产生；
3. 电压效应为经验缩放模型（`|nz| → |nz|·(1−0.15·(U/3.5)²)`），非介电耦合求解；
4. 双 toron 为两个 ansatz 场线性叠加后归一化，**不含弹性相互作用**；
5. 电场为静态直流电压扫描（0–3.5 V，8 档），**未实现论文的脉冲宽度调制（PWM）** 电场；
6. `pde_relaxation.py` 实现了完整松弛求解器（4 阶 Runge-Kutta，已验证自由能单调下降），但受算力限制未用于批量生成，保留作为后续拓展工作。

**表述规范**：

- ✅ 正确：「对解析 ansatz 构型做 Frank-Oseen 自由能体积积分，作为标签」
- ❌ 错误：「得到 Frank-Oseen 极小化的自由能」「复现 Frank-Oseen 松弛得到平衡态解」

**由此带来的结论边界**：本项目的 ansatz + 线性叠加数据比论文的 PDE 松弛数据「更简单」，需要学习的映射更平滑，因此**部分指标高于论文属预期**，不应解读为「超越原论文」。

---

## 7. 已知局限

1. **单 toron 电压 / 自由能回归的测试集仅 4 张**。R² 由 4 个点算出，方差极大；② 的 0.8761 低于论文 ≈0.94，很可能是小样本波动而非方法缺陷（论文用 100 张原图，其 5% 同样只有约 5 张）。汇报时建议以 **MAE 为主要指标**并主动披露测试集规模；如需稳健数字，请用 `--seed` 做多种子或 k-fold 的 mean±std。
2. **②–⑥ 五个回归模型为历史冻结结果**，训练时使用旧版随机在线增强，模型与指标可交付，但从头训练不保证得到逐轮完全相同的日志。①⑦ 已切换到确定性在线增强，同 seed 可复现。
3. **数据划分的类别覆盖**：论文 Fig.2h 的分类轴为 `{23,24,25,26,28,…}`，而本项目的 ansatz 在 η<30 时无 toron（退化为均匀指向矢场），有效 η 为 30–40 共 11 类，与论文不完全一致。
4. **`--patience` 与论文不同**（本项目用 30 / 80，论文为 5）：稀疏 POM 纹理学习较慢，patience=5 会在 epoch 13 提前停止（acc 0.077）。属数据驱动的实现偏差。
5. **论文正文与结构示意图对螺距模型的卷积块数存在不一致**，本项目以结构图为主要依据取 4 块，电压模型依图注取 3 块。取舍依据见 `docs/PAPER_BASIS.md`。

---

## 8. 文档索引

| 文档 | 内容 |
|------|------|
| [`cnn-inversion/README.md`](cnn-inversion/README.md) | 成员 B 代码逐文件说明、实验任务清单与结果、复现/评估命令、产物格式 |
| [`simulation-dataset/README.md`](simulation-dataset/README.md) | 成员 A 仿真方法、关键参数（对齐论文 Table 1）、数据集划分比例、简化假设表述规范 |
| [`simulation-dataset/DATASET.md`](simulation-dataset/DATASET.md) | 数据集字段含义、图片规格、使用禁忌（团队内部数据接口约定） |
| [`docs/PAPER_BASIS.md`](docs/PAPER_BASIS.md) | **每条代码的论文依据**：架构块数、FC 节点数、激活函数、损失、优化器、EarlyStopping、划分比例，全部引用论文原文与页码；附 14 条论文未写明的假设（ASSUMPTION）；含 ⑦ 任务的失败与修复全过程 |
| [`docs/TRAINING_RECORD.md`](docs/TRAINING_RECORD.md) | 7 个任务的训练与诊断全记录：多 seed 对照、patience 调优、混合螺距分类的误诊与更正、在线增强不可复现缺陷的定位与修复 |
| [`docs/FINAL_DELIVERY.md`](docs/FINAL_DELIVERY.md) | 成员 B 最终交付说明、验收结论、不纳入正式交付的历史目录说明 |

---

## 9. 引用

若本项目对你的研究有帮助，请优先引用原论文：

```bibtex
@article{Terroa2025SkyrmionCNN,
  title   = {Convolutional neural network analysis of optical texture patterns in liquid-crystal skyrmions},
  author  = {Terroa, and Tasinkevych, M. and Dias, C. S.},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {10921},
  year    = {2025},
  doi     = {10.1038/s41598-025-89699-2}
}
```

本仓库的引用信息见 [`CITATION.cff`](CITATION.cff)。

---

## 10. 致谢

本项目为大学生创新创业训练计划（大创）项目成果，两名成员全部负责代码实现并交叉评审。感谢论文作者 Terroa、Tasinkevych 与 Dias 公开完整的方法描述与参数表，使复现得以开展。
