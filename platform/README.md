# 图像参数预测平台

本平台用于大创项目中期答辩演示：选择内置单、双 toron 模拟 POM 图，或上传一张 300×300 的 PNG/JPEG 图像，即自动调用 [`predictor.py`](predictor.py) 中的 `SkyrmionPredictor`，展示类型、螺距 η、预测电压和模拟自由能。内置示例的标注值可在结果下方展开对照。平台的双 toron 螺距模型使用几何参数扩展训练后的 `geometry_double_pitch_v3`；单 toron 螺距、类型路由和电压仍使用补训模型，自由能沿用原有模型。训练与验证记录见 [`../cnn-inversion/results/双toron螺距几何泛化优化记录.md`](../cnn-inversion/results/双toron螺距几何泛化优化记录.md)。

## 本机启动

Windows 上双击本目录的 [`start_platform.cmd`](start_platform.cmd)。脚本会检查 Python 3.11、依赖、五个模型和示例图，启动本机服务并打开浏览器；保持命令窗口开启，按 Ctrl+C 停止。首次运行若没有可用虚拟环境，会创建项目根目录下的 `.venv` 并安装依赖，需要联网。

也可以在 PowerShell 中从 `skyrmion_cnn/platform` 目录手动运行：

```powershell
py -3.11 -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8765
```

浏览器打开 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)。首次启动需加载五个 TensorFlow 模型，服务器显示 `Application startup complete` 后即可使用。模型权重由 `cnn-inversion/runs/` 读取，无需重新训练。一键脚本的 `--check-only` 参数可只运行启动前检查。

## 接口

| 路径 | 用途 |
| --- | --- |
| `GET /api/health` | 模型加载状态 |
| `GET /api/examples` | 内置模拟图清单 |
| `GET /api/examples/{id}/image` | 内置模拟图原始灰度 PNG |
| `POST /api/predict` | `multipart/form-data`，文件字段 `image`；返回类型、螺距、电压、越界标记、模拟自由能和模型适用范围 |

上传层核验 PNG/JPEG 文件内容、8 MB 大小上限和 300×300 尺寸，不自动裁剪或缩放；并拦截明显彩色、空白或亮区分布与项目模拟图明显不同的图像。基础外观检查在当前仓库的 1,414 张原始与补充模拟 PNG 上未误拒，但**不能判定任意图片是否属于模型训练分布**。服务启动时加载一次模型，请求复用同一对象；临时文件在推理后删除。内置示例来自补充模拟数据的开发侧，平台副本在 `assets/examples/`，未更改原始数据集。示例标注值与 `../simulation-dataset/outputs/supplement/labels/` 中的 CSV 行核对一致，标注值不参与推理；开发/封存划分见 [`../cnn-inversion/data/supplement_split.csv`](../cnn-inversion/data/supplement_split.csv)。

## 验证

安装 `requirements-dev.txt` 后，从 `platform` 目录运行：

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
..\.venv\Scripts\python.exe -m unittest server.test_app -v
```

页面和浏览器流程检查脚本为 [`verify_browser.py`](verify_browser.py)。它需要 Playwright 与 Chrome，并需要服务先运行；测试自动预测、示例对照、上传、异常图像拦截、手机布局和快速切换。

```powershell
..\.venv\Scripts\python.exe verify_browser.py
```

## 研究边界

模型仅在项目模拟 POM 图上验证；真实显微照片尚未独立验证。类型模型只能区分单、双 toron，不能识别无 toron 或无关图像。基础外观检查只能拒绝部分明显不合适的输入，不能代替独立的分布外检测与真实图验证。η 沿用仿真晶格单位，不可直接标成 μm；自由能是解析 ansatz 构型的模拟估计。对外开放前仍需处理真实图验证、访问限制和请求限流。
