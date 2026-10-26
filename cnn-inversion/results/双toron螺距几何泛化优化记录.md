# 双 toron 螺距模型的几何泛化优化记录

日期：2026-10-26。源代码基线：远端 `main` 提交 `33179be`。
运行环境：Python 3.11.2、TensorFlow 2.21.0。

## 目的与数据边界

平台旧双 toron 螺距模型在 `testbatch_20261026` 的 220 张图上正确 162 张。
该批 308 张图只用于此前一次独立测试；本轮**未将其用于训练、验证、选权重或调参，
也未用新权重在该批次重新报告指标**。

旧补充集的双 toron 图只覆盖 `(R,B)=(13.4,0.45)`。为检查几何泛化，使用原仿真
ansatz、经验电压扰动和 Jones POM 成像流程生成新的双 toron 灰度图：每组几何参数
包含 11 个 η 类别，每类 4 个电压/间距组合。所有图均为 300×300、单通道灰度。

| 用途 | 几何参数组 `(R,B)` | 张数 | 目录 |
|---|---|---:|---|
| 开发训练 | `(13.0,0.42)`、`(13.0,0.58)`、`(15.8,0.42)`、`(15.8,0.58)`、`(14.4,0.42)`、`(14.4,0.58)` | 264 | `simulation-dataset/outputs/pitch_geometry_development_v2/` 与 `pitch_geometry_extra_train_v3/` |
| 分组验证 | `(13.6,0.52)`、`(15.2,0.48)` | 88 | `simulation-dataset/outputs/pitch_geometry_development_v2/` |
| 第一版隔离评测 | `(14.2,0.44)`、`(14.6,0.56)` | 88 | `simulation-dataset/outputs/pitch_geometry_sealed_v2/` |
| 第二版隔离评测 | `(13.2,0.51)`、`(15.6,0.49)` | 88 | `simulation-dataset/outputs/pitch_geometry_sealed_v3/` |
| 固定参数组合审计 | `(13.3,0.50)`、`(15.5,0.50)` | 88 | `simulation-dataset/outputs/pitch_geometry_audit_v4/` |

开发训练还使用原数据训练侧双 toron 675 张及补充集开发侧双 toron 110 张。
原数据验证侧 118 张只用于退化检查；原测试侧、补充集封存侧和上述各隔离组均未进入训练。
生成脚本及 SHA-256 清单保存在各数据目录，训练入口为
[`../src/train_double_pitch_geometry.py`](../src/train_double_pitch_geometry.py)。
第二版的复现命令（从仓库根目录运行）：

```powershell
python simulation-dataset/src/generate_pitch_geometry_v2.py
python simulation-dataset/src/generate_pitch_geometry_v3.py
python cnn-inversion/src/train_double_pitch_geometry.py --out geometry_double_pitch_v3 --extra-dev pitch_geometry_extra_train_v3 --sealed pitch_geometry_sealed_v3 --freeze-blocks 0 --lr 0.00005 --epochs 60 --new-repeats 2 --sup-repeats 2
python simulation-dataset/src/generate_pitch_geometry_v3.py --audit-only
python cnn-inversion/src/audit_double_pitch_geometry.py
```

## 两版候选模型

| 模型 | 调整 | 分组验证正确数（旧权重 39/88） | 原数据验证 | 隔离评测旧权重 → 候选 | 决定 |
|---|---|---:|---:|---:|---|
| `geometry_double_pitch_v2` | 冻结前两层卷积块，训练 30 轮 | 53/88 | 114/118 | 77/88 → 75/88 | 未采用 |
| `geometry_double_pitch_v3` | 增加中间半径训练组，放开全部卷积块，早停于第 39 轮；第 33 轮验证损失最低 | 69/88 | 118/118 | 33/88 → 63/88 | 按开发集条件采用 |

第二版的分组验证明细：`(13.6,0.52)` 从 33/44 到 **42/44**；
`(15.2,0.48)` 从 6/44 到 **27/44**。第二版隔离组明细：
`(13.2,0.51)` 从 33/44 到 **36/44**；`(15.6,0.49)` 从 0/44 到 **27/44**。
模型选择规则在读取第二版隔离组之前已固定：分组验证正确数须超过旧权重，
且原数据验证准确率不得低于 0.98。第二版满足该条件；隔离组只评测一次，
没有据此改权重、阈值或训练参数。逐轮损失与完整计数见
[`../runs/geometry_double_pitch_v3/metrics.json`](../runs/geometry_double_pitch_v3/metrics.json)。

原开发图及第二版隔离图的电压/间距配对按 η 轮换，可能形成可利用的参数组合规律。
因此在权重固定后，另用每个 η 都采用同样四组电压/间距配对的图做了一次审计。
审计正确数为旧模型 **38/88**、新模型 **60/88**：其中 `(13.3,0.50)` 从
36/44 降至 33/44，`(15.5,0.50)` 从 2/44 升至 27/44。审计结果说明总体
提升不完全依赖轮换配对，但并非每组构型都提高。完整结果见
[`double_pitch_audit_v4.json`](double_pitch_audit_v4.json)。审计后未再调整模型。

## 平台接入与限制

`platform/predictor.py` 仅将双 toron 螺距权重切至
`cnn-inversion/runs/geometry_double_pitch_v3/model.keras`；单 toron 螺距、类型路由、
电压和自由能权重保持原状。旧双 toron 权重仍在
`cnn-inversion/runs/supplement_geometry_v1/mixed_pitch_final/model.keras`，可回退。

固定参数组合审计准确率为 **60/88（68.18%）**，说明仍有 28 张错图；这项改进不能外推成
原 308 张测试图的新准确率，也不能证明真实显微照片有效。要更新原独立测试结论，
应由数据生成方另交付新的封存批次，再做一次平台级评测。
