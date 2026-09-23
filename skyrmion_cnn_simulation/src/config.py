# 仿真参数配置（严格对齐论文 Table 1 与 Methods 描述）
# 论文: Scientific Reports 15:10921 (2025) "CNN analysis of optical texture patterns
#        in liquid-crystal skyrmions" by Terroa, Tasinkevych, Dias
#
# 【单位约定，报告绘图必读】
# 除特别注明 "物理" 外，所有长度/螺距参数均为【仿真晶格单位】（1 格子 = 0.3125 μm）。
# 换算: 值_物理 = 值_仿真 × 换算系数。绘图时若标注物理单位必须乘以换算系数，
# 不能混用两套单位。
#
# 【电场简化声明】
# 论文原文使用脉冲宽度调制 (pulse-width modulated) 电场（不同幅值/频率/占空比）
# 生成训练数据；本项目配置的是【静态直流电压 U】（0-3.5 V 扫描），
# 未实现 PWM 调制，属于已知简化，报告需注明。

# === 仿真网格 ===
# L, Lz 为仿真晶格单位；括号内为对应物理尺寸（Δx=0.3125 μm 换算）
L = 300                # 仿真盒子边长（横向）= 300 × 0.3125 μm = 93.75 ≈ 94 μm
Lz = 32                # 限制面间距（z 方向）= 32 × 0.3125 μm = 10 μm
DX = 0.3125e-6         # 1 个仿真格子的物理间距 (m)
DT = 1e-6              # 1 个仿真时间步的物理时长 (s)，仅 PDE 松弛使用

# === Frank-Oseen 弹性常数（物理单位 N）===
K1 = 17.2e-12          # Splay
K2 = 7.51e-12          # Twist
K3 = 17.2e-12          # Bend
GAMMA = 0.162           # 旋转粘度 (Pa·s)，用于 PDE 松弛
EPS0 = 8.854e-12        # 真空介电常数 (F/m)
DELTA_EPS = -3.7        # 介电各向异性 (无量纲)

# === Ansatz 几何参数（论文给定值，单位: 仿真晶格格子）===
# 论文: R = 0.45 * Lz, B = 0.5, Cx = Cy = L/2, Cz = Lz/2
R_SKY = 0.45 * Lz      # = 14.4 格子 = 4.5 μm（物理）
B_WALL = 0.5            # 扭转壁宽度参数（无量纲），壁宽 w ≈ 2/B = 4 格子
CX = L / 2.0            # 150
CY = L / 2.0            # 150
CZ = Lz / 2.0           # 16

# === POM 光学参数 ===
DELTA_N = 0.06          # 双折射
WAVELENGTH = 500e-9     # 单色光波长 (m)，论文要求 500 nm
POM_SIZE = L            # 输出 POM 图像分辨率 = 300x300

# === 数据集扫描范围（论文 Section: Single-skyrmion data generation）===
PITCH_MIN = 23          # η 最小值（仿真单位）
PITCH_MAX = 40          # η 最大值（仿真单位），整数步进
PITCH_VALID_MIN = 30    # η < 30 无 toron（uniform director），需丢弃
VOLTAGE_MIN = 0.0       # U 最小值 (V)
VOLTAGE_MAX = 3.5       # U 最大值 (V)

# === 数据集划分比例（论文严格规定）===
SPLIT_PITCH_CLASSIFICATION = {"train": 0.85, "test": 0.15}                  # 螺距分类
SPLIT_REGRESSION = {"train": 0.80, "val": 0.15, "test": 0.05}              # 电压/自由能回归
SPLIT_MIXED = {"train": 0.85 * 0.90, "val": 0.15 * 0.90, "test": 0.10}     # 混合数据集 (10% 测试)

# === 默认输出目录 ===
OUTPUT_DIR = "../outputs"
DATASET_DIR = "../outputs/dataset"
POM_DIR = "../outputs/dataset/pom_images"
LABELS_DIR = "../outputs/dataset/labels"
SPLIT_DIR = "../outputs/dataset/splits"
VIZ_DIR = "../outputs/visualizations"
