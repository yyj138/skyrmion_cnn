"""Preflight and run the local defense demo from a Python 3.11 environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8765/"
HEALTH_URL = URL + "api/health"
LOCAL_HTTP = build_opener(ProxyHandler({}))
REQUIRED_FILES = [
    "cnn-inversion/runs/supplement_router_v1/model.keras",
    "cnn-inversion/runs/supplement_geometry_v1/pitch_final/model.keras",
    "cnn-inversion/runs/geometry_double_pitch_v3/model.keras",
    "cnn-inversion/runs/supplement_voltage_v1/mixed_voltage/model.keras",
    "cnn-inversion/runs/mixed_energy/model.keras",
    "cnn-inversion/runs/mixed_pitch_final/meta.json",
    "cnn-inversion/runs/mixed_voltage/meta.json",
    "cnn-inversion/runs/mixed_energy/meta.json",
    "platform/assets/examples/single_eta36_U1p50_v2.png",
    "platform/assets/examples/double_eta36_U1p50_sep45_v1.png",
]


def preflight() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"需要 Python 3.11，当前为 {sys.version.split()[0]}。")

    missing = [name for name in REQUIRED_FILES if not (PROJECT_ROOT / name).is_file()]
    if missing:
        raise RuntimeError("缺少模型或示例文件：\n" + "\n".join(missing))

    imports = subprocess.run(
        [sys.executable, "-c", "import fastapi, uvicorn, multipart, PIL, numpy, pandas, tensorflow"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if imports.returncode:
        lines = (imports.stderr or imports.stdout).strip().splitlines()
        detail = "\n".join(lines[-6:])
        raise RuntimeError("Python 依赖加载失败。请检查 Python 3.11 环境及平台依赖。\n" + detail)
    print("自检通过：Python、依赖、五个模型及示例图均已就绪。", flush=True)


def server_status() -> str:
    try:
        with LOCAL_HTTP.open(HEALTH_URL, timeout=2) as response:
            payload = json.load(response)
        if payload.get("status") == "ready" and payload.get("model_scope") == "synthetic_POM_300x300":
            return "ready"
        return "occupied"
    except HTTPError as exc:
        return "unavailable" if exc.code == 503 else "occupied"
    except (URLError, TimeoutError, OSError):
        return "free"
    except (ValueError, AttributeError):
        return "occupied"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="只检查环境与文件，不启动服务")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    try:
        preflight()
    except RuntimeError as exc:
        print(f"自检未通过：{exc}", file=sys.stderr)
        return 1
    if args.check_only:
        return 0

    status = server_status()
    if status == "occupied":
        print("本机 8765 端口已被其他服务占用。请关闭占用该端口的程序后重试。", file=sys.stderr)
        return 1
    if status == "unavailable":
        print("本机已有平台服务，但模型未就绪。请检查该服务窗口中的模型加载日志。", file=sys.stderr)
        return 1
    if status == "ready":
        print(f"平台已经运行：{URL}", flush=True)
        if not args.no_browser:
            webbrowser.open_new_tab(URL)
        return 0

    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8765"],
        cwd=PROJECT_ROOT / "platform",
    )
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                print("服务启动失败，请查看上方模型加载日志。", file=sys.stderr)
                return 1
            current_status = server_status()
            if current_status == "unavailable":
                print("服务已启动，但模型加载失败。请查看上方日志。", file=sys.stderr)
                return 1
            if current_status == "ready":
                print(f"平台已就绪：{URL}", flush=True)
                if not args.no_browser:
                    webbrowser.open_new_tab(URL)
                print("保持此窗口打开；按 Ctrl+C 停止平台。", flush=True)
                return process.wait()
            time.sleep(0.5)
        print("模型加载超过 90 秒，请查看上方日志。", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
