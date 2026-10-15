"""API contract tests with a stub predictor (no weights / GPU needed)."""
import io
import sys
import types

import numpy as np
import pytest
from PIL import Image


class StubPredictor:
    classes, class_names = ["D00", "D10", "D20", "D40"], ["L", "T", "A", "P"]
    available_modes, alpha, thresholds, imgsz = ["yolo", "hybrid"], 0.5, {"yolo": .25, "hybrid": .3}, 640

    def predict(self, img, mode, conf=None):
        h, w = img.shape[:2]
        return {"mode": mode, "threshold": conf or .3, "num_detections": 1,
                "detections": [{"cls": 3, "label": "D40", "name": "P", "score": .9, "box": [1, 1, 30, 30],
                                "yolo_score": .8, "vit_bg_prob": .05}],
                "counts": {"D00": 0, "D10": 0, "D20": 0, "D40": 1}, "severity_index": 12.0, "condition": "Good",
                "image_size": [w, h], "latency_ms": 1.0}

    def draw(self, img, res):
        return img


@pytest.fixture(scope="module")
def client():
    fake = types.ModuleType("roadvision.inference.predictor")
    fake.from_config = lambda cfg: StubPredictor()
    sys.modules["roadvision.inference.predictor"] = fake
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


def _png(w=128, h=96):
    b = io.BytesIO()
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(b, "PNG")
    return b.getvalue()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict?mode=hybrid", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200 and r.json()["detections"][0]["label"] == "D40"


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_fake_png(client):
    r = client.post("/predict", files={"file": ("x.png", b"not really a png", "image/png")})
    assert r.status_code == 400


def test_rejects_tiny_image(client):
    r = client.post("/predict", files={"file": ("x.png", _png(20, 20), "image/png")})
    assert r.status_code == 422


def test_conf_bounds_and_mode_enum(client):
    f = {"file": ("x.png", _png(), "image/png")}
    assert client.post("/predict?conf=1.5", files=f).status_code == 422
    assert client.post("/predict?mode=foo", files=f).status_code == 422
    assert client.post("/predict?mode=rtdetr", files=f).status_code == 503   # not loaded


def test_annotated_png(client):
    r = client.post("/predict/annotated", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
