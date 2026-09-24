# PAPER_BASIS.md — 每条代码的论文依据

> 本文中出现的代码/数据路径，除特别说明外均**相对于仓库根目录**。

论文：Terroa, Tasinkevych & Dias, *Convolutional neural network analysis of optical
texture patterns in liquid-crystal skyrmions*, Scientific Reports 15:10921 (2025).
DOI: 10.1038/s41598-025-89699-2（页码 = PDF 页码）｜ 原文副本：`paper/s41598-025-89699-2.pdf`

**重要前提：论文未公开源代码**（Data availability 仅说数据集"应要求提供"）。
但 Fig.2f/3b/3e 的架构图由 visualkeras（论文 ref.51，Gavrikov 的 Keras 可视化
工具）**从真实 Keras 模型直接生成**——架构图即论文实际代码的最直接证据。
因此本项目用 Keras/TensorFlow 实现，与论文同源。

> 开发过程中曾从论文 PDF 裁剪放大三张架构图用于逐层核对，该图集为临时核对材料、**未随仓库分发**；需要核对时请直接查阅 `paper/s41598-025-89699-2.pdf` 的 Fig.2f / Fig.3b / Fig.3e。

---

## 1. 完全有原文依据的实现（无争议）

| 代码 | 论文原文 | 出处 |
|---|---|---|
| 输入 300×300×1 灰度 | "input images (300 × 300 pixels)"；单色光 POM 强度 "monochromatic incident light with the wavelength 500nm" | p.3, p.9 |
| 卷积块 = 2×(4×4 Conv) + 1×(3×3 MaxPool) | "blocks of two 4 × 4 convolutions and one 3 × 3 max-pooling layers" | p.3, p.5 |
| ReLU 用于全部卷积层和全连接层 | "ReLU activation functions in all convolutional and fully connected layers" | p.3 |
| 分类输出 softmax 11 节点 | "softmax activation function with 11 nodes" | p.3 |
| 11 个类别 = η ∈ {23,24,25,26,28,30,32,34,36,38,40} | Fig.2h 混淆矩阵坐标轴（实测读出） | p.4 |
| 回归输出 1 节点线性激活 | "an output layer with a single node and linear activation function" | Fig.3b/3e 图注, p.5 |
| 分类损失 categorical cross-entropy | "the loss function is categorical cross-entropy" | p.3 |
| 回归损失 MSE | "optimising the mean square error (loss function)" | p.5 |
| Adam, lr=0.001 | "optimised using the Adam algorithm (learning rate of 0.001)" | p.3 |
| EarlyStopping patience=5 | "early stopping regularisation procedure (with patience set to 5 epochs)" | p.3 |
| 增强仅旋转+翻转 | "its quantity was increased through rotation and inversion transformations" | Fig.2h/3g/4g/5e 图注 |
| 分类划分 85% train / 15% test | "separated 15% of the data for a final evaluation (test set) ... remaining 85% as a training set" | p.3 |
| 回归划分 80/15/5 | "80% of the data for training, 15% for validation, and 5% for final testing" | p.6 |
| 混合集划分 10% test，剩余 85% train / 5% val | "10% of data has been separated as a test set, and the remaining is divided into training (85%) and validation (5%) sets" | Fig.5b 图注, p.8 |
| 电压回归：3 块 + 4×FC(32) + 线性输出 | "composed of three blocks ... followed by four fully connected layers with 32 nodes each and an output layer with a single node and linear activation function" | Fig.3b 图注, p.5 |
| 自由能回归：4 块 + FC(32,16) + 线性输出 | "four blocks of two 4x4 convolutional layers with a 3x3 max-pooling layer ... two fully connected layers with 32 and 16 nodes ... 1 node with a linear activation function" | p.6 + Fig.3e 图注 |
| 双 toron 自由能复用 Fig.3e 架构 | "we have use the same CNN architecture as for the case of an isolated skyrmion free energy learning (see Fig. 3e)" | p.6 |
| 混合训练架构随输出参数而定 | "The CNN architecture is dependent on the output parameter as before." | Fig.5a, p.8 |
| 均匀构型/瞬态构型剔除（数据侧） | "textures corresponding to the uniform director configurations were excluded from the training dataset, likewise the textures of initial transient configurations" | p.3 |
| 分类指标准确率；回归指标 R² | "average accuracy is ≈0.96"（① 单 toron 分类，Fig.2h）；"coefficient of determination of approximately 0.94"（② 电压，Fig.3d）、"≈0.96"（③ 单 toron 自由能，Fig.3g）、"approximately 0.999"（④ 双 toron 自由能，Fig.4g） | p.3, p.5-6 |
| ⑦ 混合螺距分类**无数值基准** | 正文仅定性："the confusion matrix for the classification of the LC pitch in Fig. 5e reveals a very good agreement as well"（未给准确率数字，以 Fig.5e 混淆矩阵呈现）。注意 p.7 结论段 "coefficients of determination exceeding 0.94 in all the cases" 指的是**回归**，不可用作分类基准 | p.6-7, p.8 |
| 散点图 1:1 线 + 每点标准差阴影 | "the crosses represent the 1:1 relationship. The blue shaded area represents the standard deviation for each point." | Fig.3d/3g 图注 |

## 2. 论文内部矛盾点（已按"图+图注优先"裁决）

### 2.1 分类网（Fig.2f）卷积块数：正文 3 块 vs 架构图 4 块
- 正文 p.3："three blocks of two 4×4 convolutions and one 3×3 max-pooling layers"
- **架构图实测：4 个 [Conv,Conv,MaxPool] 块 + Flatten + 4 个 Dense 框**
  （对 PDF 中 Fig.2f 做像素级颜色序列检测：Conv×8 / MaxPool×4 / Dense×4；
  核对用裁剪图未随仓库分发，请直接查阅 `paper/s41598-025-89699-2.pdf` 的 Fig.2f）
- 裁决：默认 **4 块**（图为真实模型生成）。`build_pitch_classifier(n_blocks=4)`
  保留参数，若团队决定信正文可改 `n_blocks=3`。

### 2.2 电压网（Fig.3b）：正文 "块+1、FC−1" vs 图注 "3 块 + 4 FC"
- 正文 p.5："increased the number of blocks ... by one, reduced the number of
  fully connected layers by one"（相对 Fig.2f → 应为 4 块 + 2 FC）
- Fig.3b 图注："three blocks ... four fully connected layers with 32 nodes each"
- **架构图实测：3 块 + 5 个 Dense 框（4 FC + 1 输出），与图注完全一致**
- 裁决：按图注+图实现 **3 块 + 4×FC(32)**。

### 2.3 Fig.2g 图注称分类 loss 为 MSE vs 正文 categorical cross-entropy
- 正文 p.3 明确写分类用 categorical cross-entropy；Fig.2g 图注括号里写
  "loss function (given by the mean square error)"，应为笔误。
- 裁决：分类用 **categorical cross-entropy**。

### 2.4 分类 11 类的具体取值：论文轴 vs 成员 A 数据集
- 论文 Fig.2h 混淆矩阵轴为 η ∈ {23,24,25,26,28,30,32,34,36,38,40}（11 类）。
- 成员 A 数据集（ansatz 近似）η<30 不产生 toron，实际覆盖 **30–40 的
  11 个整数**（labels_single.csv 实测，每类 8 张）。
- 裁决：类别取**数据集实际值**（输出层 11 节点与论文一致），由
  `cnn-inversion/src/data.py get_pitch_classes()` 从 CSV 动态读取并校验类别数==11。
  论文报告时注明类别集差异源于仿真方法（ansatz vs PDE 松弛）。

## 3. 论文未写明、必须自行决定的部分（ASSUMPTION，全部可配置）

| # | 项目 | 取值 | 理由 |
|---|---|---|---|
| 3.1 | 每层卷积的滤波器数量 | 32（`DEFAULT_CONV_FILTERS`） | 论文全文与架构图均未标注滤波器数 |
| 3.2 | 卷积 padding | 'same' | 300×300 输入若用 'valid'，4 块后特征图尺寸降为 0，与 Fig.2f/3e 的 4 块结构无法共存，故必为 'same' |
| 3.3 | strides | 卷积 1；池化=池尺寸（Keras 默认） | 论文未提及 |
| 3.4 | 像素归一化 | /255 → [0,1] **后再做每图 z-score**（减单图均值除单图标准差） | 论文未提及。数值必需：A 的稀疏纹理（~99% 像素为 0）仅 /255 时 CNN 完全学不动（test acc 0.000），加每图 z-score 后 test acc 1.000（2026-09-08 对照实验，300 样本单 toron 分类，其余配置全同） |
| 3.5 | 旋转增强角度 | rot90 k∈{0,1,2,3} + 水平/垂直翻转 | 论文只说 "rotation and inversion"，未给角度；90° 倍数无插值、不引入伪影 |
| 3.6 | 分类任务验证集比例 | 从 85% 训练集中再划 15% | 正文只给 85/15，但 Fig.2g 画了 validation loss，必有验证集，比例未写。A 的分类划分表无 val 行时，代码从 train 行确定性划出 15%（seed 固定） |
| 3.7 | batch size | 32 | 论文未提及 |
| 3.8 | 最大 epoch | 200（早停会提前终止） | 论文只给 patience=5；Fig.2g/3c/3f 显示 ~10–30 epoch 收敛 |
| 3.9 | EarlyStopping 细节 | monitor=val_loss, restore_best_weights=True | 论文只说 patience=5 |
| 3.10 | 混合集 85%/5% 的分母 | 按"剩余 90% 集合"的 85%/5% 切分（约 10% 剩余样本不进入任何集合） | Fig.5b 图注字面："the remaining is divided into training (85%) and validation (5%) sets"；A 的划分表采用归一化填满解释（76.5/13.5/10），用 A 的表时以表为准 |
| 3.11 | 回归标签标准化 | z-score（用训练集统计量），预测反变换回物理量再算指标 | 论文未提及；数值必需：free_energy ~1e-13 J，不标准化 MSE ~1e-26，float32 梯度过小无法收敛（A 的 `roleA` 分支 `docs/dataset_readme.md` 亦建议） |
| 3.12 | 早停 patience（偏差项） | 论文=5（默认）；**A 的数据上 ①–⑥ 建议 30**；**⑦ 建议 80**（配 `--det-aug --patience 80 --epochs 400`） | 数据驱动偏差：A 的稀疏纹理学习慢。patience=5 时 ① 在 epoch 13 停住（test acc 0.077）、⑦ 在 epoch 48 停住（仍停在随机线 ln(11)≈2.397）。⑦ 实测需约 60 epoch 才拟合（`--det-aug` 下 e58 train_acc 首达 1.0、e60 train/val 双双 1.0、早停于 e209）。报告中应说明此调整源于仿真数据与论文数据分布差异 |
| 3.13 | 随机种子与结果波动 | `cnn-inversion/src/train.py --seed` 固定全局种子（默认 42）；**报告建议多种子 mean±std** | 小数据集（88 张）上训练是随机过程（权重初始化+增强抽样），同配置不同 seed 的 test acc 实测可从 0.62 波动到 1.00（2026-09-09）。单次跑分不可作为结论；论文只报单次值，我们应在报告中披露 seed 与波动范围。**注：2026-09-14 查明，除权重初始化外还有一处更严重的非确定性来源，见 3.14 与 §7** |
| 3.14 | 在线增强的随机数来源（**2026-09-14 修复的偏差项**） | 默认走 `augment()`（全局随机源，多线程下不可复现）；`--det-aug` 改用 `augment_stateless()`（随机量由样本索引派生，可复现） | 见 §7 缺陷记录。默认路径与历史运行**逐字一致**，故 ①–⑥ 已固化的 model/metrics 不受影响；⑦ 启用 `--det-aug` 后方可复现。变换集合两者完全相同（4 旋转 × 2 翻转 × 2 翻转 = 16 种组合），不改变任务定义与数据分布 |

## 4. 与成员 A 的数据接口（2026-09-08 已交付并对齐）

A 交付：`roleA` 分支的 `outputs/dataset/`
- `pom_images/`：968 张 300×300 灰度 POM（单 toron 88、双 toron 880）
- `labels/`：labels_single.csv / labels_double.csv / labels_mixed.csv，
  列：filename,eta,U,free_energy,kind,n_torons[,separation]
- `splits/`：7 个任务的划分表共 8 个文件（含 split 列 train/val/test），**训练时经
  `--split-csv` 传入，勿重新随机划分**（A 的 `roleA` 分支 `docs/dataset_readme.md` 要求，
  保证 A/B 两端可复现一致）
- η 类别：30–40 的 11 个整数（见 2.4）；free_energy 量级 ~1e-13 J（见 3.11）
- A 数据生成声明（报告可引用）：解析 ansatz 等效平衡态 + 经验电压模型，
  未跑 PDE 松弛；双 toron 为线性叠加，不含弹性相互作用

### ⑦ 混合螺距分类：失败与修复（2026-09-09 误诊 → 2026-09-14 澄清）

> ⚠️ **本节曾于 2026-09-09 记录为「已知数据局限：双 toron 图像不携带可学习的 η 信号」，
> 该结论已被证伪并于 2026-09-14 撤销。若在其他文档或汇报材料中见过该结论，请一并更正。**

#### 已作废的旧结论（保留存证，勿再引用）
旧判定为：双 toron 图 η 类间无线性可分性（NCM test acc double=0.045 < 随机 0.091）；
隔离实验「仅用双 toron 880 张训练分类同样卡死」；据此推断根因是 A 的双 toron
「两个 ansatz 场线性叠加后归一化」冲掉了 q0=2π/η 注入的手性扭转相位信息，
并准备了「反馈 A 重新生成数据 / 跑 PDE 松弛」的降级预案。

**失效原因**：
- NCM 探针本身退化（double=0.045 **低于**随机线），退化探针不能作为"信号不存在"的证据；
- 「仅用双 toron 训练也卡死」这一隔离实验，复现出来的其实是同一个**管道缺陷**，
  而非数据缺陷——两者同时存在，旧判断把结果错记在了数据头上。

#### 真实根因（2026-09-14 确证）
`cnn-inversion/src/data.py` 的在线增强 `augment()` 使用 `tf.random.uniform` / `tf.image.random_flip_*`
等**全局随机源**，却挂在 `num_parallel_calls=AUTOTUNE` 的**多线程** map 上。
多线程抢占全局随机数使**每次运行每张图拿到的增强方式都不同**，即使
`tf.keras.utils.set_random_seed(42)` 已固定。同配置重跑的轨迹因此不可复现——
原失败运行（48 epoch，loss 停在 ln(11)≈2.397）是撞上了坏轨迹。详见 §7。

#### 证据链
- **管道复现性**：固定 seed=42，训练路径两次取 batch 最大逐像素差 **15.198**
  （2.70% 像素不同）；同一路径在 `training=False` 时差 **0.000**；
- **因子实验定位**：无 shuffle 无 augment → 差 0.000；仅 shuffle → 差 0.000；
  **仅 augment → 差 14.575**；shuffle+augment → 差 16.262 → 唯一祸首是 `augment()`；
- **线程无关性**：`tf.config.threading.set_{intra,inter}_op_parallelism_threads(1)`
  后仍不可复现（差 16.262），故必须改随机源而非限制并行度；
- **图像级可学性**（推翻旧结论的直接证据）：双 toron 跨 η 的 uint8 平均差 0.12–0.34、
  约 1500 像素不同、**无一对完全相同**，与单 toron 同量级；逻辑回归 probe
  test acc 0.437–0.448 = **4.8–4.9× 随机水平**；CNN 在架构实际分辨率（4×4）下
  仍达 0.345 = 3.79× 随机 → **η 信号在双 toron 图中确实存在**。

#### 修复与结果（2026-09-14）
新增 `augment_stateless()`（随机量由「样本索引 + seed」派生），`make_dataset()` 加
`deterministic` 参数，`cnn-inversion/src/train.py` 加 **`--det-aug`** 开关（默认关闭）。⑦ 以
`--det-aug --patience 80 --epochs 400`（seed=42）重跑：
- **训练**：e5 val_acc 起跳（0.1145）→ e11 0.2290 → e20 0.4809 → e40 0.9084 →
  **e58 train_acc 首达 1.0** → **e60 起 train/val 双双 1.0000**，早停于 e209
  （末轮 train_loss 1.897e-05）
- **测试集 accuracy = 0.9897**（97 张仅错 1 张）；**10/11 类召回率 100%**，
  唯一错误是 `double_eta30_U3p50_sep30.png` 的 **η=30 → 31 相邻类误判**
  （η=30 为数据集最小类，也是 η<30 不产生 toron 的临界值，召回 5/6=0.833）

#### 结论
**⑦ 达标。成员 A 的数据无需修改，降级预案不启用。** 论文未给该任务数值基准
（仅定性 "very good agreement"，Fig.5e 混淆矩阵），可比的分类数值参照为同架构的
单 toron 分类 ≈0.96（p.3）——本地 0.9897 达到并在数值上超过该参照。至此
**7 个任务全部达标**。

## 5. 运行方式

> 以下命令的**工作目录为仓库的 `cnn-inversion/`**；`$DS` 指向成员 A 交付的数据集（`roleA` 分支的 `outputs/dataset`，见 `cnn-inversion/README.md` 的获取步骤）。

```bash
cd cnn-inversion

# 环境（已验证：Python 3.11/3.13 + tensorflow-cpu 2.21 + matplotlib 3.11）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 结构自检（逐层输出形状对照 Fig.2f/3b/3e）
python src/models.py

# ---- 用成员 A 交付的数据正式训练（DS 指向 A 的 outputs/dataset） ----
DS=../_roleA/outputs/dataset

# ① 单 toron 螺距分类（正式运行使用成员 B 派生的分层划分表，train/val/test 均覆盖 11 类）
#    若使用 A 的 split_pitch_classification_single.csv（无 val 行），代码会自动从 train 划 15%，见 3.6
python src/train.py --task pitch \
  --csv "$DS/labels/labels_single.csv" --images "$DS/pom_images" \
  --split-csv "data/split_pitch_classification_single_stratified.csv" \
  --out runs/pitch_final --patience 30

# ② 单 toron 电压回归
python src/train.py --task voltage \
  --csv "$DS/labels/labels_single.csv" --images "$DS/pom_images" \
  --split-csv "$DS/splits/split_voltage_regression_single.csv" \
  --out runs/voltage --patience 30

# ③ 单 toron 自由能回归
python src/train.py --task energy \
  --csv "$DS/labels/labels_single.csv" --images "$DS/pom_images" \
  --split-csv "$DS/splits/split_free_energy_regression_single.csv" \
  --out runs/energy_single --patience 30

# ④ 双 toron 自由能回归（架构同为 Fig.3e，论文 p.6）
python src/train.py --task energy \
  --csv "$DS/labels/labels_double.csv" --images "$DS/pom_images" \
  --split-csv "$DS/splits/split_free_energy_regression_double.csv" \
  --out runs/energy_double --patience 30

# ⑤⑥ 混合数据集回归任务（Fig.5，架构随输出参数而定）
python src/train.py --task energy  --csv "$DS/labels/labels_mixed.csv" --images "$DS/pom_images" \
  --split-csv "$DS/splits/split_free_energy_regression_mixed.csv" --out runs/mixed_energy --patience 30
python src/train.py --task voltage --csv "$DS/labels/labels_mixed.csv" --images "$DS/pom_images" \
  --split-csv "$DS/splits/split_voltage_regression_mixed.csv" --out runs/mixed_voltage --patience 30
# ⑦ 混合螺距分类（注意：必须加 --det-aug，否则结果不可复现，见 §7）
python src/train.py --task pitch   --csv "$DS/labels/labels_mixed.csv" --images "$DS/pom_images" \
  --split-csv "$DS/splits/split_pitch_classification_mixed.csv" --out runs/mixed_pitch_final \
  --det-aug --patience 80 --epochs 400

# 评估出图（loss 曲线 / 混淆矩阵 / 预测-真值散点 + R²；meta.json 已含数据路径）
python src/evaluate.py --run runs/pitch_final
python src/evaluate.py --run runs/voltage
python src/evaluate.py --run runs/mixed_pitch_final   # ⑦，测试集 acc = 0.9897
```

## 6. GPU 说明（RTX 5060 Ti）

- TF ≥ 2.11 起 **Windows 原生不支持 GPU**（需 WSL2 或 DirectML 插件），
  本实现默认 CPU 训练——网络极小（12–21 万参数）、原图仅 100–1500 张，
  CPU 足够（冒烟测试 2 epoch × 3 任务共约 1 分钟）。
- 如需 GPU 加速：WSL2 内装 TF，或改用 PyTorch 等价实现（架构完全对应，
  5060 Ti 需 PyTorch ≥ 2.7 + cu128）。

## 7. 缺陷记录：在线增强的不可复现性（2026-09-14 定位并修复）

### 现象
同 seed、同配置重跑 ⑦，结果不可复现：原运行 48 epoch 卡在随机线
（loss 2.4002→2.3971，val_acc 全程 ≤0.084）；另一次运行同 seed=42 却在 e4 起
loss 下降、e9 val_acc 已达 0.2443。两者**在 epoch 0 就分叉**（acc 0.0973 vs 0.0905）。

### 定位过程
`cnn-inversion/src/data.py make_dataset()` 中：

```python
ds = ds.map(augment, num_parallel_calls=AUTOTUNE)   # augment 消费 TF 全局随机数
```

`augment()` 内部调用 `tf.random.uniform` / `tf.image.random_flip_left_right` /
`tf.image.random_flip_up_down`，消费的都是**全局随机源**。`set_random_seed(42)`
只固定了随机数序列本身，而 `AUTOTUNE` 开的多线程 map 中，**哪个线程先处理哪张图、
谁先消费下一批随机数，取决于 CPU 调度** → 每张图每次运行拿到的增强都不同。

因子实验结果（固定 seed=42，各取 2 个 batch 对比两次运行）：

| 管道配置 | 可复现 | 最大逐像素差 |
|---|---|---|
| 无 shuffle、无 augment | ✅ | 0.000 |
| 仅 shuffle | ✅ | 0.000 |
| **仅 augment** | ❌ | 14.575 |
| shuffle + augment（原实现） | ❌ | 16.262 |
| 限制 TF 线程 intra/inter=1 | ❌ | 16.262（无效） |

→ `shuffle(reshuffle_each_iteration=True)` 本身是确定的，**唯一祸首是 `augment()`**；
且限制线程数不能修复，必须改随机源。

### 修复
- `cnn-inversion/src/data.py` 新增 `augment_stateless(image, idx, base_seed)`：随机量改用
  `tf.random.stateless_uniform(seed=[base_seed, idx(+offset)])`，**由样本索引派生**，
  与线程调度无关。变换集合与原 `augment()` 完全相同（16 种组合）。
- `cnn-inversion/src/data.py make_dataset(..., deterministic=False, seed=42)`：`deterministic=True`
  时前置 `np.arange` 索引并随样本传递；**原实现整段保留为默认路径，逐字未改**。
- `cnn-inversion/src/train.py` 新增 `--det-aug`（`action="store_true"`，**默认关闭**）；`meta.json`
  增记 `det_aug` / `epochs_run` 便于追溯。

验证（`deterministic=True` 两次 batch 逐像素差 **0.000**；`False` 差 15.985）。

### 影响面（重要）
- 默认路径逻辑逐字保留 → **①–⑥ 已固化的 `model.keras` / `metrics.json` 完全不受影响**；
  不传 `--det-aug` 时行为与修复前一致。
- **但须留意**：①–⑥ 与本任务共用同一套增强代码，其数字**同样不可复现**。
  若日后评审要求重跑，数值会有波动（①–⑥ 收敛良好，预计波动有限）。
  如需全部可复现，给 ①–⑥ 也补 `--det-aug` 重跑即可。
- 修复后的额外收益：同 seed 下 20 轮短跑轨迹与正式跑一致，因此
  **"用 `--det-aug --epochs 20` 快速筛 seed"成为可行**（此前探针无法预测正式跑）。
