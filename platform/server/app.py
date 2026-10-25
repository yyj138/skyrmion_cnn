"""Serve the POM image prediction demo and its existing CNN predictor."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
import logging
from pathlib import Path
import tempfile
from threading import Lock
from typing import Callable, Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_IMAGES = FRONTEND_ROOT / "assets" / "examples"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
EXPECTED_SIZE = (300, 300)
LOGGER = logging.getLogger(__name__)
EXAMPLES = {
    "single": {
        "label": "单 toron 示例",
        "description": "项目模拟 POM 图 · η=36 · 开发侧",
        "filename": "single_eta36_U1p50_v2.png",
        "reference": {"kind": "single", "eta": 36, "voltage_V": 1.5, "free_energy_J": 8.23497051291545e-14},
    },
    "double": {
        "label": "双 toron 示例",
        "description": "项目模拟 POM 图 · η=36 · 开发侧",
        "filename": "double_eta36_U1p50_sep45_v1.png",
        "reference": {"kind": "double", "eta": 36, "voltage_V": 1.5, "free_energy_J": 7.891813095575058e-14},
    },
}


class Prediction(BaseModel):
    kind: Literal["single", "double"]
    eta: int
    voltage_V: float
    voltage_clipped_to_training_range: bool
    free_energy_J: float
    model_scope: Literal["synthetic_POM_300x300"]


def load_predictor():
    """Import the platform predictor without requiring a particular working directory."""
    from predictor import SkyrmionPredictor

    return SkyrmionPredictor()


def check_simulated_pom_appearance(image: Image.Image) -> str | None:
    """Reject obvious mismatches; this does not establish model-domain membership."""
    # All 1,816 existing 300×300 project PNGs pass these deliberately broad limits.
    # Their bright-pixel fractions are 0.0057–0.0211; this is not an OOD classifier.
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    colored_fraction = np.count_nonzero(np.ptp(rgb, axis=2) > 8) / gray.size
    if colored_fraction > 0.01:
        return "图像含明显彩色区域。当前仅支持项目模拟灰度 POM 图，请上传原始灰度图。"
    bright_fraction = np.count_nonzero(gray > 16) / gray.size
    if float(gray.std()) < 4 or bright_fraction < 0.001:
        return "图像纹理不足，无法按项目模拟 POM 图预测。请选择包含纹理的原始模拟图。"
    if float(np.median(gray)) > 32 or bright_fraction > 0.08:
        return "图像亮区分布与项目模拟 POM 图差异较大。请选择同源模拟图；真实显微图需先验证。"
    return None


def create_app(predictor_factory: Callable | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = None
        app.state.load_error = None
        app.state.predict_lock = Lock()
        try:
            app.state.predictor = await asyncio.to_thread(predictor_factory or load_predictor)
        except Exception as exc:
            # The page can still explain why prediction is unavailable.
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Failed to load prediction models")
        yield

    app = FastAPI(title="图像参数预测平台", lifespan=lifespan)

    @app.get("/api/health")
    async def health():
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="模型尚未就绪，请检查服务日志和模型文件。")
        return {"status": "ready", "model_scope": "synthetic_POM_300x300"}

    @app.get("/api/examples")
    async def examples():
        return [
            {
                "id": example_id,
                "label": item["label"],
                "description": item["description"],
                "image_url": f"/api/examples/{example_id}/image",
                "reference": item["reference"],
            }
            for example_id, item in EXAMPLES.items()
        ]

    @app.get("/api/examples/{example_id}/image")
    async def example_image(example_id: str):
        item = EXAMPLES.get(example_id)
        if item is None:
            raise HTTPException(status_code=404, detail="示例不存在。")
        path = EXAMPLE_IMAGES / item["filename"]
        if not path.is_file():
            raise HTTPException(status_code=503, detail="示例图像暂不可用。")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/predict", response_model=Prediction)
    async def predict(image: UploadFile = File(...)):
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="模型尚未就绪，请稍后重试。")
        if image.content_type not in {"image/png", "image/jpeg"}:
            raise HTTPException(status_code=400, detail="请选择 PNG 或 JPEG 图像。")

        data = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="图像文件不能超过 8 MB。")
        try:
            with Image.open(BytesIO(data)) as decoded:
                image_format = decoded.format
                image_size = decoded.size
                decoded.load()
                if image_format in {"PNG", "JPEG"} and image_size == EXPECTED_SIZE:
                    appearance_error = check_simulated_pom_appearance(decoded)
                else:
                    appearance_error = None
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(status_code=400, detail="无法读取图像，请选择有效的 PNG 或 JPEG 文件。") from None

        if image_format not in {"PNG", "JPEG"}:
            raise HTTPException(status_code=400, detail="图像内容必须为 PNG 或 JPEG 格式。")
        if image_size != EXPECTED_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"图像尺寸为 {image_size[0]}×{image_size[1]}，必须为 300×300 像素；平台不会自动缩放。",
            )
        if appearance_error is not None:
            raise HTTPException(status_code=400, detail=appearance_error)

        suffix = ".png" if image_format == "PNG" else ".jpg"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="skyrmion-upload-", suffix=suffix, delete=False) as temporary:
                temporary.write(data)
                temporary_path = Path(temporary.name)

            def run_inference():
                with app.state.predict_lock:
                    return app.state.predictor.predict(temporary_path)

            result = await asyncio.to_thread(run_inference)
            return Prediction.model_validate(result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="预测暂时失败，请重试或检查服务日志。") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    app.mount("/assets", StaticFiles(directory=FRONTEND_ROOT / "assets"), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(FRONTEND_ROOT / "index.html", media_type="text/html")

    return app


app = create_app()
