# 液晶 skyrmion CNN 分析：物理仿真与数据集工程（成员 A）

复现论文：Terroa, Tasinkevych, Dias, "Convolutional neural network analysis of optical texture patterns in liquid-crystal skyrmions", Scientific Reports 15:10921 (2025)，论文文件见 [s41598-025-89699-2.pdf](s41598-025-89699-2.pdf)。

本项目实现大创分工中成员 A 的全部代码：物理仿真（toron 解析 ansatz、Frank-Oseen 自由能、Jones 矩阵 POM 成像）、批量数据集生成、数据集划分、可视化校验，以及一个可选的 PDE 松弛求解器。产出的数据集供成员 B 训练 CNN 使用。

## 目录结构

```
skyrmion_cnn_simulation/
├── src/                                  # 源代码
│   ├── config.py                         # 全部仿真参数 (对齐论文 Table 1)
│   ├── ansatz.py                         # Task 2-1: toron 三维解析 ansatz
│   ├── frank_osseen.py                   # Task 2-2: Frank-Oseen 自由能积分
│   ├── pom_imaging.py                    # Task 2-3: Jones 矩阵 POM 成像
│   ├── dataset_generator.py              # Task 2-4,2-5: 批量生成 + 数据集划分
│   ├── visualize.py                      # Task 2-6: 数据校验可视化
│   ├── pde_relaxation.py                 # 可选: PDE 松弛求解器
│   ├── main.py                           # 模块1: 最小验证 demo
│   ├── check_trends.py                   # 定性物理趋势自动校验
│   └── check_deliverables.py             # 交付物完整性校验
├── outputs/
│   ├── dataset/
│   │   ├── pom_images/                   # POM 灰度图 (300x300 PNG, 训练用)
│   │   ├── labels/                       # csv 标签文件
│   │   └── splits/                       # train/val/test 划分 csv
│   └── visualizations/                   # 截面图、趋势图；presentation/ 为报告展示图
├── s41598-025-89699-2.pdf                # 原论文
├── README.md                             # 本文件
└── dataset_readme.md                     # 数据集交付说明（给成员 B）
```

## 环境依赖

```bash
pip install numpy matplotlib pandas tqdm pillow
# 成员 B 的 CNN 训练另需: pip install tensorflow scikit-learn
```

## 快速开始

```bash
cd src

# 1. 最小验证 demo（6 组样本，约 1 分钟），跑通 ansatz -> 自由能 -> POM -> csv 全链路
python main.py

# 2. 三个定性物理趋势自动校验（纹理随 eta 变化、随 U 形变、F 随 eta 下降）
python check_trends.py

# 3. 各模块自检
python ansatz.py            # 指向矢场生成
python frank_osseen.py      # 自由能计算
python pom_imaging.py       # POM 成像（含均匀场暗场验证）

# 4. 数据校验可视化（截面图、POM gallery、标签分布、展示图）
python visualize.py

# 5. 完整数据集生成（单 toron 88 张 + 双 toron 880 张，约 4 小时）
python dataset_generator.py --mode all --n_voltages 8
```

## 任务清单与对应文件

| 任务 | 文件 | 说明 |
|------|------|------|
| 2-1 toron 三维 ansatz | ansatz.py | 复现论文 Eq.(3)-(6)，输出 (300,300,32) nx,ny,nz；支持单/双 toron |
| 2-2 Frank-Oseen 自由能 | frank_osseen.py | 复现论文 Eq.(1)，含 K1/K2/K3 与电场项 |
| 2-3 Jones 矩阵 POM | pom_imaging.py | 向量化加速，Δn=0.06，λ=500nm，正交偏振 |
| 2-4 批量数据集生成 | dataset_generator.py | eta 取 30-40，U 取 0-3.5V，自动丢弃 eta<30 |
| 2-5 数据集划分 | dataset_generator.py | 分类 85/15，回归 80/15/5，混合 10% 测试 |
| 2-6 数据校验可视化 | visualize.py | 截面图、POM gallery、标签分布、趋势校验 |
| 可选 PDE 松弛 | pde_relaxation.py | 4 阶 Runge-Kutta 求解器，批量生成不使用 |
| 模块1 最小验证 | main.py | 6 组小样本跑通全链路 |

## 关键参数（论文 Table 1）

| 参数 | 仿真值 | 物理值 | 说明 |
|------|--------|--------|------|
| L | 300 | 94 μm | 仿真盒子边长 |
| Lz | 32 | 10 μm | 限制面间距 |
| Δx | 1 | 0.3125 μm | 格子间距 |
| K1 | 17.2 | 17.2e-12 N | Splay 弹性常数 |
| K2 | 7.51 | 7.51e-12 N | Twist 弹性常数 |
| K3 | 17.2 | 17.2e-12 N | Bend 弹性常数 |
| γ | 162 | 0.162 Pa·s | 旋转粘度 |
| Δε | -3.7 | -3.7 | 介电各向异性 |
| Δn | 0.06 | 0.06 | 双折射 |
| λ | - | 500 nm | 单色光波长 |
| R | 14.4 | 4.5 μm | ansatz toron 半径 (0.45·Lz) |
| B | 0.5 | - | ansatz 扭转壁宽度参数 |

单位约定：长度、螺距等除特别注明外均为仿真晶格单位，1 格子 = 0.3125 μm。

## 数据集划分比例（严格按论文规定，seed=42）

| 任务 | 训练 | 验证 | 测试 |
|------|------|------|------|
| 螺距分类（单 toron） | 85% | - | 15% |
| 电压回归（单 toron） | 80% | 15% | 5% |
| 自由能回归（单 toron） | 80% | 15% | 5% |
| 自由能回归（双 toron） | 80% | 15% | 5% |
| 混合数据集（各任务） | 76.5% | 13.5% | 10% |

## 简化假设与兜底说明（写报告时沿用此表述）

本数据集没有采用论文原始 Frank-Oseen PDE 松弛求解得到平衡态 toron；采用论文给出的解析 ansatz 初场作为等效平衡态指向矢场。手性螺旋、电压诱导形变使用简化经验模型。Frank-Oseen 自由能标签是对 ansatz 构型直接做体积积分的结果。pde_relaxation.py 实现了完整松弛求解器（4 阶 Runge-Kutta，已验证自由能单调下降），但受计算性能限制，没有用于批量数据集生成，可作为后续拓展工作。

具体简化：

1. 主流程不运行 PDE 松弛动力学，直接将解析 ansatz 场作为等效平衡态指向矢场；
2. 手性螺旋扭转 q0·(z-Cz) 为人为注入，模拟 PDE 松弛后 twist 项驱动的螺旋特征，不是自由能极小化自然产生；
3. 电压效应为经验缩放模型（|nz| 随 U 减小），不是 Frank-Oseen 介电耦合求解；
4. 双 toron 为两个 ansatz 场线性叠加后归一化，不含弹性相互作用；
5. 电场为静态直流电压扫描（0-3.5V 共 8 档），未实现论文的脉冲宽度调制（PWM）电场。

报告表述规范：

- 正确表述："对解析 ansatz 构型做 Frank-Oseen 自由能体积积分，作为标签"；
- 错误表述："得到 Frank-Oseen 极小化的自由能"、"复现 Frank-Oseen 松弛得到平衡态解"。

训练图片（pom_images/）保持原始灰度；visualizations/presentation/ 下的伪彩展示图仅用于报告，不得送入 CNN。

## 数据集规模

- 单 toron：88 张（eta 取 30-40 共 11 值 × U 取 0-3.5V 共 8 档）
- 双 toron：880 张（同上 × 分离距离 10 档：30-75）
- 混合：968 张；另有 6 张 demo 小样本
- 标签与划分字段说明见 [dataset_readme.md](dataset_readme.md)
