# roleB — 液晶 skyrmion POM 光学纹理的 CNN 参数反演

> 大创项目成员 B 的工作分支：复现论文 *Convolutional neural network analysis of optical texture patterns in liquid-crystal skyrmions* 中的三类 CNN 架构，完成「POM 光学纹理 → 螺距 / 电压 / 自由能」的定量反演，共 7 个任务。

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Paper](https://img.shields.io/badge/Sci.%20Rep.-15%3A10921%20(2025)-1f6feb)](https://doi.org/10.1038/s41598-025-89699-2)

**论文**：Terroa, Tasinkevych & Dias, *Convolutional neural network analysis of optical texture patterns in liquid-crystal skyrmions*, **Scientific Reports 15:10921 (2025)**.
DOI: [10.1038/s41598-025-89699-2](https://doi.org/10.1038/s41598-025-89699-2) ｜ 原文副本：[`paper/s41598-025-89699-2.pdf`](paper/s41598-025-89699-2.pdf)

---

## 1. 本分支的定位

本仓库按成员分工使用分支管理：

| 分支 | 内容 |
|---|---|
| [`main`](https://github.com/yyj138/skyrmion_cnn/tree/main) | 汇总：成员 A + 成员 B 的完整交付（双项目结构） |
| [`roleA`](https://github.com/yyj138/skyrmion_cnn/tree/roleA) | **成员 A**：物理仿真（toron 解析 ansatz、Frank-Oseen 自由能、3D 琼斯矩阵 POM 成像）、968 张 300×300 灰度 POM 数据集 + 标签 CSV + 划分表 |
| **`roleB`（本分支）** | **成员 B**：CNN 建模、训练、评估、结果分析。**不含**成员 A 的仿真代码与数据集 |

两端仅通过一个数据接口衔接：A 的 `outputs/dataset/`（图片 + 标签 CSV + train/val/test 划分表）。字段含义与使用禁忌见 [`roleA` 分支的 `docs/dataset_readme.md`](https://github.com/yyj138/skyrmion_cnn/blob/roleA/docs/dataset_readme.md)。

### 获取 A 的数据集（训练前置步骤）

```bash
# 在本分支仓库根目录执行，把 A 的数据拉到 _roleA/（该目录已在 .gitignore 中排除）
git clone -b roleA --depth 1 https://github.com/yyj138/skyrmion_cnn.git _roleA
```

完成后数据集位于 `_roleA/outputs/dataset/`，下文命令中的 `DS` 即指向此处。

---

## 2. 目录结构

```
.
├── README.md                     # 本文件（roleB 分支说明）
├── CITATION.cff                  # 引用信息
├── .gitignore / .gitattributes
│
├── cnn-inversion/                # ── 成员 B 的 CNN 反演 ──────────────
│   ├── README.md                 #   代码逐文件说明、7 个任务的完整复现命令
│   ├── requirements.txt          #   锁定版本依赖（tensorflow-cpu==2.21.0 等）
│   ├── data/
│   │   └── split_pitch_classification_single_stratified.csv
│   │                             #   成员 B 派生的分层抽样划分表（① 任务使用）
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
├── docs/
│   ├── PAPER_BASIS.md            #   每条代码的论文依据 + 14 条 ASSUMPTION
│   ├── TRAINING_RECORD.md        #   7 个任务的训练与诊断全记录
│   ├── FINAL_DELIVERY.md         #   成员 B 最终交付说明与验收结论
│   └── 大创完整实施方案（...）.docx
│
└── paper/
    └── s41598-025-89699-2.pdf    #   原论文
```

> `_roleA/`（A 的数据克隆目录，运行时产生）不在上图中，且已被 `.gitignore` 排除。

---

## 3. 环境依赖

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

---

## 4. 快速开始

### 4.1 冒烟测试（无需数据集，约 10 秒）

```bash
cd cnn-inversion
python src/smoke_test.py
```

用 120 张合成图跑通「读数据 → 增强 → 训练 → 评估 → 出图」全链路，末尾打印 `SMOKE TEST PASSED`。

### 4.2 复现全部 7 个训练任务

前置：已按第 1 节克隆 A 的数据到 `_roleA/`。

```bash
cd cnn-inversion
DS=../_roleA/outputs/dataset

# ① 单 toron 螺距分类（使用成员 B 派生的分层划分表，train/val/test 均覆盖 11 类）
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

## 6. 已知局限

1. **单 toron 电压 / 自由能回归的测试集仅 4 张**。R² 由 4 个点算出，方差极大；② 的 0.8761 低于论文 ≈0.94，很可能是小样本波动而非方法缺陷。汇报时建议以 **MAE 为主要指标**并主动披露测试集规模；如需稳健数字，请用 `--seed` 做多种子或 k-fold 的 mean±std。
2. **②–⑥ 五个回归模型为历史冻结结果**，训练时使用旧版随机在线增强，模型与指标可交付，但从头训练不保证得到逐轮完全相同的日志。①⑦ 已切换到确定性在线增强，同 seed 可复现。
3. **数据生成采用降级路线**（解析 ansatz 替代 PDE 松弛、双 toron 线性叠加、静态直流电压替代 PWM），因此数据的映射比论文数据更平滑，**部分指标高于论文属预期**，不应解读为「超越原论文」。完整的简化声明见 [`roleA` 分支的 README](https://github.com/yyj138/skyrmion_cnn/blob/roleA/README.md)。
4. **`--patience` 与论文不同**（本项目用 30 / 80，论文为 5）：稀疏 POM 纹理学习较慢，patience=5 会在 epoch 13 提前停止（acc 0.077）。属数据驱动的实现偏差。
5. **论文正文与结构示意图对螺距模型的卷积块数存在不一致**，本项目以结构图为主要依据取 4 块，电压模型依图注取 3 块。取舍依据见 `docs/PAPER_BASIS.md`。

---

## 7. 文档索引

| 文档 | 内容 |
|------|------|
| [`cnn-inversion/README.md`](cnn-inversion/README.md) | 代码逐文件说明、实验任务清单与结果、复现/评估命令、产物格式 |
| [`docs/PAPER_BASIS.md`](docs/PAPER_BASIS.md) | **每条代码的论文依据**：架构块数、FC 节点数、激活函数、损失、优化器、EarlyStopping、划分比例，全部引用论文原文与页码；附 14 条论文未写明的假设（ASSUMPTION）；含 ⑦ 任务的失败与修复全过程 |
| [`docs/TRAINING_RECORD.md`](docs/TRAINING_RECORD.md) | 7 个任务的训练与诊断全记录：多 seed 对照、patience 调优、混合螺距分类的误诊与更正、在线增强不可复现缺陷的定位与修复 |
| [`docs/FINAL_DELIVERY.md`](docs/FINAL_DELIVERY.md) | 成员 B 最终交付说明、验收结论、不纳入正式交付的历史目录说明 |
| [`roleA` 分支](https://github.com/yyj138/skyrmion_cnn/tree/roleA) | 成员 A 的物理仿真与数据集工程（含 `docs/dataset_readme.md` 数据字段说明） |

---

## 8. 引用

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

## 9. 致谢

本项目为大学生创新创业训练计划（大创）项目成果，两名成员全部负责代码实现并交叉评审。感谢论文作者 Terroa、Tasinkevych 与 Dias 公开完整的方法描述与参数表，使复现得以开展。
