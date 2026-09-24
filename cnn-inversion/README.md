# cnn-inversion — 液晶 skyrmion CNN 参数反演（成员 B）

> 大创项目成员 B 的工作目录：基于成员 A 生成的物理仿真 POM 数据集，复现并训练论文 *Terroa et al., Sci. Rep. 15:10921 (2025)* 中的三类 CNN 架构，完成「光学纹理 → 螺距 / 电压 / 自由能」的定量反演。

**论文**：Terroa, Tasinkevych & Dias, *Convolutional neural network analysis of optical texture patterns in liquid-crystal skyrmions*, **Scientific Reports 15:10921 (2025)**.
DOI: [10.1038/s41598-025-89699-2](https://doi.org/10.1038/s41598-025-89699-2) ｜ 原文副本：[`../paper/s41598-025-89699-2.pdf`](../paper/s41598-025-89699-2.pdf)

**与成员 A 的关系**：本目录可独立运行，数据来自同仓库 [`roleA` 分支的 `outputs/dataset/`](https://github.com/yyj138/skyrmion_cnn/tree/roleA/outputs/dataset)（A 已交付：968 张 300×300 灰度 POM 图 + 4 个标签 CSV + 8 个 train/val/test 划分表）。字段含义与使用禁忌详见 [A 的 `docs/dataset_readme.md`](https://github.com/yyj138/skyrmion_cnn/tree/roleA/docs/dataset_readme.md)。

---

## 1. 文件清单

| 文件 | 作用 |
|------|------|
| `src/models.py` | 3 个 Keras 模型：螺距分类器（4 块卷积 + FC(32,32,16) + softmax 11）、电压回归器（3 块 + 4×FC(32) + 线性）、自由能回归器（4 块 + FC(32,16) + 线性）。每行代码附论文出处。 |
| `src/data.py` | 数据加载：300×300 灰度 → **per-image z-score 归一化**（稀疏 POM 纹理的关键）；rot90 + flip 增强；`augment_stateless()` 使用由样本索引派生的确定性随机序列（同 seed 跨重跑一致、逐 epoch 变化）；按 A 的 `split_*.csv` 划分。 |
| `src/train.py` | 训练入口：Adam lr=0.001、EarlyStopping、回归标签 z-score、随机种子全栈固定、动态确定性在线增强（默认启用，`--legacy-random-aug` 可回退旧路径）。 |
| `src/evaluate.py` | 测试集评估：loss 曲线、混淆矩阵（分类）、pred_vs_true 散点（带 1:1 线 + 标准差阴影，复刻 Fig.3d/3g 图注风格）；输出 `metrics.json`。 |
| `src/smoke_test.py` | 端到端冒烟测试：用合成数据跑通三个模型（不需要真实数据集，秒级完成）。 |
| `requirements.txt` | 锁定版本依赖：`tensorflow-cpu==2.21.0`、`numpy`、`pandas`、`pillow`、`matplotlib`。 |
| `data/` | 成员 B 派生的数据划分表：`split_pitch_classification_single_stratified.csv`（① 任务使用，按类别分层抽样，保证 train/val/test 均覆盖全部 11 个 η 类别）。其余划分表由成员 A 提供，见 [`roleA` 分支的 `outputs/dataset/splits/`](https://github.com/yyj138/skyrmion_cnn/tree/roleA/outputs/dataset/splits)。 |
| `runs/` | 7 个正式训练任务的输出，每个目录含 `model.keras`、`meta.json`、`metrics.json`、`training_log.csv`、`loss_curve.png`、`confusion_matrix.png`（分类）或 `pred_vs_true.png`（回归）。 |

> 代码逐行的论文依据、14 条论文未写明的实现假设（ASSUMPTION）、⑦ 任务的失败与修复全过程见 [`../docs/PAPER_BASIS.md`](../docs/PAPER_BASIS.md)；完整训练与诊断过程见 [`../docs/TRAINING_RECORD.md`](../docs/TRAINING_RECORD.md)。

---

## 2. 环境与安装

Python 3.11.x（与 `requirements.txt` 锁定的 `tensorflow-cpu==2.21.0` 配套）。

```bash
cd cnn-inversion

python -m venv .venv
# Windows PowerShell
.venv\Scripts\activate
# Git Bash / WSL
# source .venv/bin/activate

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

冒烟测试（无需数据集，确认全链路通畅）：

```bash
cd cnn-inversion
python src/smoke_test.py
```

> `src/` 是 Python 的模块搜索根：脚本内部使用 `from data import ...` 这类同级导入，因此请从 `cnn-inversion/` 目录以 `python src/xxx.py` 方式调用（`smoke_test.py` 使用 `Path(__file__).parent` 定位自身，会自动在 `src/` 下创建临时目录 `_smoke/`，该目录已在 `.gitignore` 中排除）。

---

## 3. 复现的论文任务与结果

全部指标取自各 `runs/*/metrics.json` 的磁盘实际值（已重新推理核对）。

| # | 任务 | 测试集 | 论文基线 | 实测 | 状态 |
|---|------|--------|---------|------|------|
| 1 | 螺距分类（单 toron） | 13 | acc ≈ 0.96 | **acc = 1.000** | ✅ |
| 2 | 电压回归（单 toron） | 4 | R² ≈ 0.94 | **MAE = 0.260 V, R² = 0.876** | ✅（R² 受 n=4 影响，MAE 更稳） |
| 3 | 自由能回归（单 toron） | 4 | R² ≈ 0.96 | **R² = 0.9988** | ✅ |
| 4 | 自由能回归（双 toron） | 44 | R² ≈ 0.999 | **R² = 0.9985** | ✅（基本持平） |
| 5 | 自由能回归（混合） | 97 | — | **R² = 0.9989** | ✅ |
| 6 | 电压回归（混合） | 97 | — | **R² = 0.9952** | ✅ |
| 7 | 螺距分类（混合） | 97 | 无数值基准（注） | **acc = 0.9897**（97 张错 1 张） | ✅ |

> **论文基线出处**：① ≈0.96（p.3, Fig.2h）、② ≈0.94（p.5, Fig.3d）、③ ≈0.96（p.6, Fig.3g）、④ ≈0.999（p.6, Fig.4g）。
> 论文**未给 ⑦ 的数值准确率**，仅定性 "the confusion matrix … in Fig. 5e reveals a very good agreement"（p.6–7）；可比参照为同架构的单 toron 分类 ≈0.96。
> 注：p.7 结论段的 "coefficients of determination exceeding 0.94 in all the cases" 指**回归**，不可用作分类基准。

**关于任务 7（混合螺距分类）**：历史版本曾停在约 0.09 的随机水平。排查后发现旧在线增强在多线程 `map` 中使用全局随机源，同 seed 重跑也可能得到不同增强。根因**不是数据缺陷**（双 toron 图像确实携带 η 信号），而是代码缺陷。当前正式路径改用预先生成的确定性随机序列：同 seed 跨重跑一致，同时每个 epoch 重新采样。由于混合图像对方向变换更敏感，正式重训每轮约 10% 训练样本接受确定性随机增强，其余保留原图。最终结果存放于 `runs/mixed_pitch_final/`，测试集 Accuracy = 0.9897（97 张错 1 张）。完整定位过程见 `../docs/PAPER_BASIS.md` 第 7 节。

**报告建议**：
1. 电压（单 toron）测试集仅 n=4，R² 对单点异常值敏感，建议以 **MAE 为主要汇报指标**；
2. 单次运行的波动较大，论文指标应附多种子 mean±std 或声明所用 seed（`--seed` 已支持）；
3. 本项目数据比论文数据「更简单」（解析 ansatz vs PDE 松弛），指标偏高属预期，不应解读为超越原论文。

---

## 4. 训练命令

关键参数：`--task {pitch, voltage, energy}`、`--csv <labels.csv>`、`--images <pom_images_dir>`、`--split-csv <A 的划分表>`、`--out <run_dir>`、`--seed`（默认 42）、`--patience`、`--aug-prob`。动态确定性增强默认启用；`--legacy-random-aug` 仅用于复核旧实验。

```bash
cd cnn-inversion
DS=../_roleA/outputs/dataset     # 由 A 的分支克隆而来，见下方说明

# 1) 螺距分类（单 toron）
python src/train.py --task pitch --csv $DS/labels/labels_single.csv --images $DS/pom_images --split-csv data/split_pitch_classification_single_stratified.csv --out runs/pitch_final --patience 30 --epochs 250

# 2) 电压回归（单 toron）
python src/train.py --task voltage --csv $DS/labels/labels_single.csv --images $DS/pom_images --split-csv $DS/splits/split_voltage_regression_single.csv --out runs/voltage --patience 30

# 3) 自由能回归（单 toron）
python src/train.py --task energy --csv $DS/labels/labels_single.csv --images $DS/pom_images --split-csv $DS/splits/split_free_energy_regression_single.csv --out runs/energy_single --patience 30

# 4) 自由能回归（双 toron）
python src/train.py --task energy --csv $DS/labels/labels_double.csv --images $DS/pom_images --split-csv $DS/splits/split_free_energy_regression_double.csv --out runs/energy_double --patience 30

# 5) 自由能回归（混合）
python src/train.py --task energy --csv $DS/labels/labels_mixed.csv --images $DS/pom_images --split-csv $DS/splits/split_free_energy_regression_mixed.csv --out runs/mixed_energy --patience 30

# 6) 电压回归（混合）
python src/train.py --task voltage --csv $DS/labels/labels_mixed.csv --images $DS/pom_images --split-csv $DS/splits/split_voltage_regression_mixed.csv --out runs/mixed_voltage --patience 30

# 7) 螺距分类（混合）—— 动态确定性增强正式重训
python src/train.py --task pitch --csv $DS/labels/labels_mixed.csv --images $DS/pom_images --split-csv $DS/splits/split_pitch_classification_mixed.csv --out runs/mixed_pitch_final --patience 80 --epochs 45 --det-aug --aug-prob 0.1
```

> **提示**：`--det-aug` 是默认正式路径，命令中显式写出便于审计。⑦ 的 `--aug-prob 0.1` 表示每轮约 10% 样本接受动态变换，**不是**固定选择同一批样本。
> Windows PowerShell 用户若不想用 `$DS` 变量，可直接使用完整相对路径，或先 `cd` 到 `cnn-inversion/`。

---

## 5. 评估命令与产物

```bash
cd cnn-inversion
python src/evaluate.py --run runs/pitch_final
python src/evaluate.py --run runs/voltage
python src/evaluate.py --run runs/energy_single
python src/evaluate.py --run runs/energy_double
python src/evaluate.py --run runs/mixed_energy
python src/evaluate.py --run runs/mixed_voltage
python src/evaluate.py --run runs/mixed_pitch_final
```

每个 `runs/<task>/` 目录下的产物：

| 文件 | 内容 |
|------|------|
| `metrics.json` | 分类任务输出 `accuracy`；回归任务输出 `R2`、`MSE`、`MAE` |
| `loss_curve.png` | 训练 & 验证 loss 曲线（复刻 Fig.2g/3c/3f/4f/5b） |
| `confusion_matrix.png` | 分类任务，复刻 Fig.2h/5e |
| `pred_vs_true.png` | 回归任务，预测-真值散点 + 1:1 线 + 标准差阴影，复刻 Fig.3d/3g 图注风格 |
| `model.keras` | 训练好的模型权重 |
| `training_log.csv` | 每个 epoch 的 train/val loss 与指标 |
| `meta.json` | 划分表、类别表、回归标签标准化统计量、停轮 epoch（可复现凭证） |

> 本目录只包含 7 个**正式**运行结果。历史对照目录（`mixed_pitch_v2/`）、旧划分目录（`pitch/`）与中断训练目录已从仓库中排除，其结论已被上述正式运行取代，相关复盘记录保留在 `../docs/TRAINING_RECORD.md` 与 `../docs/FINAL_DELIVERY.md` 中。

---

## 6. 相关文档

- [`../README.md`](../README.md)：仓库总览（两个子项目的分工与整体结构）
- [`../docs/PAPER_BASIS.md`](../docs/PAPER_BASIS.md)：每条代码的论文依据 + 14 条 ASSUMPTION + 缺陷记录
- [`../docs/TRAINING_RECORD.md`](../docs/TRAINING_RECORD.md)：7 个任务的训练与诊断详细记录
- [`../docs/FINAL_DELIVERY.md`](../docs/FINAL_DELIVERY.md)：成员 B 最终交付说明与验收结论
- `roleA` 分支的 `README.md`：成员 A 的物理仿真与数据集生成端
