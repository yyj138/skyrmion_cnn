# 大创 CNN 复现：训练全记录（成员B）

> 本文中出现的代码/数据路径，除特别说明外均**相对于仓库根目录**（训练命令章节除外，该节注明工作目录）。

> 论文：Terroa, Tasinkevych & Dias, *Convolutional neural network analysis of optical
> texture patterns in liquid-crystal skyrmions*, Sci. Rep. 15:10921 (2025).
> DOI: 10.1038/s41598-025-89699-2
> 记录时间：2026-09-08 ~ 2026-09-09，2026-09-14 更正 ｜ 详细论文依据见 `docs/PAPER_BASIS.md`
>
> **2026-09-14 更正摘要**：⑦ 混合螺距分类已复现成功（test acc = 0.9897）；
> 旧版第 7 节"数据局限"结论作废；第 6 节 ②③ 数字已按 `metrics.json` 磁盘实际值修正。

## 1. 任务定位

成员 B 负责：CNN 复现 → 训练 → 评估 → 绘图分析。成员 A 负责物理仿真与数据集
（已交付：968 张 300×300 灰度 POM 图 + 4 个标签 csv + 7 个任务划分表）。

## 2. 环境

- Python **3.11.16** 虚拟环境（PyCharm 项目 `PythonProject4`）
- 依赖（`cnn-inversion/requirements.txt`，清华源安装）：
  tensorflow-cpu **2.21.0** / numpy **2.4.6**（3.11 上限；2.5.x 需 ≥3.12）/
  pandas 3.0.3 / pillow 12.3.0 / matplotlib 3.11.1
- RTX 5060 Ti 未使用：TF ≥2.11 在 Windows 原生不支持 GPU，CPU 足够

## 3. 代码与论文的对应关系（要点）

- 三套 Keras 网络严格按论文 Fig.2f/3b/3e（visualkeras 架构图像素级取证）：
  分类 4 块+FC(32,32,16)+softmax11；电压 3 块+FC(32×4)+线性1；自由能 4 块+FC(32,16)+线性1
- 训练按论文：Adam(lr=0.001)、分类用 categorical cross-entropy、回归用 MSE、
  早停 patience=5（默认值）
- 论文未写明项全部标注 ASSUMPTION 并记录于 `docs/PAPER_BASIS.md` 第 3 节

## 4. 数据集成时的 5 处关键适配

| # | 适配 | 原因 |
|---|------|------|
| 1 | 分类类别从 CSV 动态读取（校验=11 类） | A 的数据 η 为 30–40 整数，与论文 Fig.2h 轴（{23,24,25,26,28,...}）不同（ansatz 在 η<30 无 toron） |
| 2 | `--split-csv` 直接读 A 的划分表 | A 要求勿重新随机划分，保证两端可复现；分类表无 val 行时从 train 确定性划 15% |
| 3 | 回归标签 z-score（训练集统计量） | free_energy ~1e-13 J，不标准化 MSE~1e-26，float32 梯度过小学不动 |
| 4 | **每图 z-score 输入标准化**（关键） | A 的稀疏纹理（99% 像素为 0）仅 /255 时 test acc=0.000；加每图 z-score 后 acc=1.000 |
| 5 | `--patience 30`（论文为 5） | A 的稀疏纹理学习慢，patience=5 在 epoch 13 停住（acc 0.077），30 才收敛（acc 1.000）。属数据驱动偏差，报告需说明 |

## 5. 随机种子问题（重要教训）

- 首次 pitch 训练 test acc=0.615，与验证环境的 1.000 不符 → 根因：旧代码未固定
  全局种子，权重初始化+增强抽样是随机抽签，88 张小数据上方差巨大
- 修复：`tf.keras.utils.set_random_seed(seed)`；三种子（42/43/44）复跑 test acc
  **全部 1.000**，跨机器可复现
- 报告启示：论文只报单次值（≈0.96）未披露种子/波动；我们应披露 seed 与波动范围
- **2026-09-14 补充：还有一层更隐蔽的非确定性。** `set_random_seed()` 只固定了
  **权重初始化**，**固定不了在线增强**——`augment()` 在多线程 map 里抢全局随机数，
  同 seed 重跑每张图拿到的增强仍不同。已确证：固定 seed=42，同一条训练路径两次
  取 batch 的**最大逐像素差 15.198**（2.70% 像素不同）；改成由样本索引派生的
  `augment_stateless()` 后差 **0.000**。新增 `--det-aug` 开关（默认关闭）。
  详见 `docs/PAPER_BASIS.md` 第 7 节"缺陷记录"。
- **因此上一条"三种子全部 1.000"需限定范围**：那说明 ① 这个任务收敛裕度大、
  恰好稳定，**不能**据此认为整条管线可复现。⑦ 就因此失败过（见第 7 节）。

## 6. 七任务结果汇总

| # | 任务 | 测试集 | 结果（`runs/*/metrics.json` 实测） | 论文基准 | 状态 |
|---|------|--------|------------------------------------|----------|------|
| ① | 单 toron 螺距分类 | 13 张 | **acc = 1.0000** | ≈0.96 | ✅ |
| ② | 单 toron 电压回归 | **4 张** | **R² = 0.8761** | ≈0.94 | ⚠️ 见下 |
| ③ | 单 toron 自由能回归 | **4 张** | **R² = 0.9988** | ≈0.96 | ✅ |
| ④ | 双 toron 自由能回归 | 44 张 | **R² = 0.9985** | ≈0.999 | ✅（基本持平） |
| ⑤ | 混合·自由能回归 | 97 张 | **R² = 0.9989** | — | ✅ |
| ⑥ | 混合·电压回归 | 97 张 | **R² = 0.9952** | — | ✅ |
| ⑦ | 混合·螺距分类 | 97 张 | **acc = 0.9897**（97 张错 1 张） | 无数值基准，见注 | ✅ |

> **数字口径（2026-09-14 更正）**：上表一律取 `runs/*/metrics.json` 的磁盘实际值。
> 旧版本记录的 ② = 0.958、③ = 0.9998 与磁盘不符，已更正为 0.8761 / 0.9988。

### ⚠️ ② 与 ③ 的测试集只有 4 张，R² 在此规模上不具统计意义

单 toron 共 88 张，按论文的 80/15/5 划分后 **test 仅 4 张**（① 走分类的 85/15
划分，故有 13 张）。R² 由 4 个点算出，方差极大：

- ② 实测 0.8761 **低于论文 ≈0.94**，但这**很可能是小样本波动，而非方法缺陷**
  —— 论文用 100 张原图，其 5% 同样只有约 5 张，存在同类问题；
- ③ 的 0.9988 同样是在 4 个点上算的，虽高也不可靠。
- **报告建议**：① 主动披露 ②③ 的测试集规模；② 若需稳健数字，改做多种子 /
  k-fold 的 mean±std（`--seed` 已支持），不要用单次值下结论。

### ⑦ 的基准说明

论文**未给出该任务的数值准确率**，正文仅定性："the confusion matrix for the
classification of the LC pitch in Fig. 5e reveals a very good agreement as well"
（p.6–7，以 Fig.5e 混淆矩阵呈现）。可比的分类数值参照为**同架构的单 toron
分类 ≈0.96**（p.3，Fig.2h）——本地 0.9897 达到并在数值上超过该参照。
注意 p.7 结论段的 "coefficients of determination exceeding 0.94 in all the cases"
指的是**回归**，不可用作分类基准。

指标高于论文属预期：A 的 ansatz + 线性叠加数据比论文 PDE 松弛数据"更简单"，
要学的函数更平滑。报告中应主动说明此点，避免评审误读为"超过原论文"。

## 7. 混合螺距分类：一次误诊及其更正（2026-09-14）

> ⚠️ 本节旧版本判定为"数据局限：双 toron 图不携带可学习的 η 信号"，
> **该结论已于 2026-09-14 撤销**。若有汇报材料引用过，请一并更正。

### 旧结论（已作废，保留存证）

曾依据三层"隔离证据"判定问题出在 A 的数据（并准备了降级预案、拟反馈 A 重新生成数据）：

1. **管线无恙**：同批双 toron 图上电压回归 R²=0.995、自由能回归 R²=0.999
2. **双 toron 是病灶**：仅用 880 张双 toron 图单独训练分类，40 epoch 后 train acc 仍 0.10
3. **信号物理缺失**：最近质心分类器 test acc——single=0.462、double=0.045
   （低于随机 0.091）、mixed=0.062

据此推断：η 经手性扭转 q0=2π/η 注入 ansatz 属**相位类信息**，双 toron
"线性叠加 + 归一化"时被冲掉。

### 为什么错了

- **证据 3 无效**：NCM 探针退化成 double=0.045 **低于随机线**——退化探针不能
  证明"信号不存在"，只能说明该探针不适用；
- **证据 2 归属错判**：该隔离实验复现出来的是同一个**代码缺陷**，不是数据缺陷。
  两者同时存在，旧判断把结果记在了数据头上；
- **直接反证**：双 toron 跨 η 的 uint8 平均差 0.12–0.34、约 1500 像素不同、
  **无一对完全相同**（与单 toron 同量级）；逻辑回归 probe test acc 0.437–0.448
  = **4.8–4.9× 随机水平** → **η 信号确实存在**。

### 真实根因

`cnn-inversion/src/data.py` 的在线增强 `augment()` 使用**全局随机源**（`tf.random.uniform` /
`tf.image.random_flip_*`），却挂在 `num_parallel_calls=AUTOTUNE` 的**多线程** map 上
——多线程抢占全局随机数使每张图每次运行拿到的增强都不同，**同 seed 重跑不可复现**。
原失败运行（48 epoch，loss 停在 ln(11)≈2.397）是撞上了坏轨迹。
完整定位过程与因子实验见 `docs/PAPER_BASIS.md` 第 7 节"缺陷记录"。

### 修复与结果

新增 `augment_stateless()`（随机量由「样本索引 + seed」派生）+ `cnn-inversion/src/train.py` 的 `--det-aug`
（**默认关闭**，故 ①–⑥ 已固化产物完全不受影响）。⑦ 以
`--det-aug --patience 80 --epochs 400`（seed=42）重跑：

- **训练**：e5 val_acc 起跳 → **e58 train_acc 首达 1.0** → **e60 起 train/val
  双双 1.0000**，早停于 e209
- **测试集 accuracy = 0.9897**（97 张仅错 1 张），**10/11 类召回率 100%**；
  唯一错误是 `double_eta30_U3p50_sep30.png` 的 **η=30 → 31 相邻类误判**
  （η=30 为数据集最小类，也是 η<30 不产生 toron 的临界值）

**结论：⑦ 达标。A 的数据无需修改，降级预案不启用，7 个任务全部达标。**
`runs/mixed_pitch`（旧失败存档）保留在原始工作区作为排查过程证据，**未纳入本仓库**。

## 8. 产出清单（`cnn-inversion/runs/` 下每任务一套）

`model.keras`（权重）、`training_log.csv`、`loss_curve.png`、
`confusion_matrix.png`（分类）/ `pred_vs_true.png`（回归）、
`metrics.json`（指标）、`meta.json`（划分+类别+标签统计，可复现凭证）

## 9. 复现命令（PowerShell，工作目录 = 仓库的 `cnn-inversion/`）

```powershell
cd cnn-inversion
$DS = "../_roleA/outputs/dataset"

# 例：单 toron 螺距分类（其余任务换 --csv/--split-csv/--out，见 docs/PAPER_BASIS.md 第 5 节）
python src/train.py --task pitch --csv "$DS/labels/labels_single.csv" --images "$DS/pom_images" `
  --split-csv "data/split_pitch_classification_single_stratified.csv" --out runs/pitch_final --patience 30
python src/evaluate.py --run runs/pitch_final
# ⑦ 混合螺距分类（**必须加 --det-aug**，否则结果不可复现，见第 7 节）
python src/train.py --task pitch --csv "$DS/labels/labels_mixed.csv" --images "$DS/pom_images" `
  --split-csv "$DS/splits/split_pitch_classification_mixed.csv" --out runs/mixed_pitch_final `
  --det-aug --patience 80 --epochs 400
python src/evaluate.py --run runs/mixed_pitch_final
```

要点：每条训练命令都带 `--split-csv`（A 的划分表）；`--patience` ①–⑥ 用 30、
⑦ 用 80；全局种子已固定（默认 42，可用 `--seed` 更换并建议多种子报告 mean±std）；
**⑦ 必须带 `--det-aug`**（修复在线增强不可复现，见第 7 节）。
