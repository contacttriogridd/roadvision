"""RoadVision production API (FastAPI).

Endpoints
  GET  /health            liveness + which models are loaded
  GET  /model-info        classes, calibration (alpha, thresholds), modes
  POST /predict           JSON detections + severity index for one image
  POST /predict/annotated PNG with drawn boxes
  POST /predict/batch     up to 8 images
  GET  /                  minimal web UI (api/static/index.html)
  /ui                     Gradio interface (mounted when gradio is installed — used on HF Spaces)

Input validation: content-type + magic-byte check, size limit, decodability,
min/max side, enum-validated mode, bounded confidence (pydantic/Query constraints).

    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from roadvision import __version__  # noqa: E402
from roadvision.utils.config import load_config  # noqa: E402

CFG = load_config(os.environ.get("RV_CONFIG", ROOT / "configs/default.yaml"),
                  [o for o in os.environ.get("RV_OVERRIDE", "").split(",") if o])
os.chdir(ROOT)
MAX_BYTES = int(CFG.app.max_upload_mb * 1024 * 1024)
MAGIC = {b"\xff\xd8\xff": "jpeg", b"\x89PNG": "png", b"RIFF": "webp", b"BM": "bmp"}


class Mode(str, Enum):
    hybrid = "hybrid"
    yolo = "yolo"
    rtdetr = "rtdetr"
    ensemble = "ensemble"


class DetectionOut(BaseModel):
    cls: int
    label: str = Field(examples=["D40"])
    name: str = Field(examples=["Pothole"])
    score: float = Field(ge=0, le=1)
    box: list[float] = Field(min_length=4, max_length=4, description="x1, y1, x2, y2 (pixels)")
    yolo_score: Optional[float] = None
    vit_bg_prob: Optional[float] = None


class PredictionOut(BaseModel):
    filename: str
    mode: Mode
    threshold: float
    num_detections: int
    detections: list[DetectionOut]
    counts: dict[str, int]
    severity_index: float = Field(ge=0, le=100, description="heuristic maintenance-priority score")
    condition: str = Field(examples=["Fair"])
    image_size: list[int]
    latency_ms: float


def maybe_download_weights() -> None:
    repo = os.environ.get("RV_WEIGHTS_REPO") or CFG.app.hf_weights_repo
    if not repo:
        return
    from huggingface_hub import hf_hub_download

    for key in ("detector", "rtdetr", "vit", "calibration"):
        dst = Path(CFG.app[key])
        if not dst.exists():
            try:
                p = hf_hub_download(repo, dst.name, local_dir=str(dst.parent))
                print(f"downloaded {p}")
            except Exception as exc:  # optional files (e.g. rtdetr) may be absent
                print(f"weights {dst.name} not in {repo}: {exc}")


maybe_download_weights()
from roadvision.inference.predictor import from_config  # noqa: E402

PREDICTOR = from_config(CFG)

app = FastAPI(title="RoadVision API", version=__version__,
              description="Real-time road damage detection with YOLO and Vision Transformers (RDD2022 classes D00/D10/D20/D40).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def read_image(file: UploadFile) -> np.ndarray:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(415, f"Unsupported content type {file.content_type}; use JPEG/PNG/WebP/BMP")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File larger than {CFG.app.max_upload_mb} MB")
    if not any(data.startswith(m) for m in MAGIC):
        raise HTTPException(400, "File content is not a valid image (magic bytes mismatch)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Image could not be decoded")
    h, w = img.shape[:2]
    if min(h, w) < CFG.app.min_side or max(h, w) > CFG.app.max_side:
        raise HTTPException(422, f"Image sides must be within [{CFG.app.min_side}, {CFG.app.max_side}] px; got {w}x{h}")
    return img


def run(img: np.ndarray, mode: Mode, conf: Optional[float]) -> dict:
    if mode.value not in PREDICTOR.available_modes:
        raise HTTPException(503, f"Mode '{mode.value}' not loaded. Available: {PREDICTOR.available_modes}")
    return PREDICTOR.predict(img, mode.value, conf)


ConfQ = Query(None, ge=0.01, le=0.99, description="override calibrated threshold")


@app.get("/health")
def health():
    return {"status": "ok" if PREDICTOR.available_modes else "no-models", "modes": PREDICTOR.available_modes,
            "version": __version__}


@app.get("/model-info")
def model_info():
    return {"classes": dict(zip(PREDICTOR.classes, PREDICTOR.class_names)), "modes": PREDICTOR.available_modes,
            "alpha": PREDICTOR.alpha, "thresholds": PREDICTOR.thresholds, "imgsz": PREDICTOR.imgsz}


@app.post("/predict", response_model=PredictionOut)
async def predict(file: UploadFile = File(...), mode: Mode = Mode.hybrid, conf: Optional[float] = ConfQ):
    img = await read_image(file)
    return {"filename": file.filename, **run(img, mode, conf)}


@app.post("/predict/annotated", responses={200: {"content": {"image/png": {}}}})
async def predict_annotated(file: UploadFile = File(...), mode: Mode = Mode.hybrid, conf: Optional[float] = ConfQ):
    img = await read_image(file)
    res = run(img, mode, conf)
    ok, buf = cv2.imencode(".png", PREDICTOR.draw(img, res))
    return Response(buf.tobytes(), media_type="image/png",
                    headers={"X-Severity": str(res["severity_index"]), "X-Detections": str(res["num_detections"])})


@app.post("/predict/batch", response_model=list[PredictionOut])
async def predict_batch(files: list[UploadFile] = File(...), mode: Mode = Mode.hybrid, conf: Optional[float] = ConfQ):
    if not 1 <= len(files) <= 8:
        raise HTTPException(422, "Send between 1 and 8 images")
    return [{"filename": f.filename, **run(await read_image(f), mode, conf)} for f in files]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "api/static/index.html")


try:  # Gradio UI for Hugging Face Spaces, served from the same container
    import gradio as gr

    from app.gradio_ui import build_ui

    app = gr.mount_gradio_app(app, build_ui(PREDICTOR), path="/ui")
except ImportError:
    pass
