"""Unified inference engine used by evaluation, the FastAPI service and the Gradio UI.

Modes
  yolo      : one-stage CNN detector only (fastest)
  rtdetr    : transformer detector only
  hybrid    : YOLO proposals re-scored/re-labelled by the ViT refiner (proposed method)
  ensemble  : Weighted Boxes Fusion of YOLO + RT-DETR

Hybrid fusion for a proposal with YOLO label c_y, confidence s_y and ViT posterior p:
    p_dmg = p[:4] / sum(p[:4]),   p_bg = p[4]
    label = argmax_c  alpha * onehot(c_y)_c + (1 - alpha) * p_dmg_c
    score = s_y^alpha * ((1 - p_bg) * p_dmg[label])^(1 - alpha)
alpha and the score threshold are calibrated on the validation split (grid search).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ..data.augment import to_chw, vit_eval_tf
from ..data.crops import padded_crop
from ..models.ensemble import weighted_boxes_fusion
from ..utils.common import read_json

SEVERITY_W = {0: 1.0, 1: 1.0, 2: 2.0, 3: 3.0}   # alligator cracks and potholes are structurally worse
COLORS = [(0, 200, 255), (255, 170, 0), (180, 0, 255), (0, 0, 255)]


@dataclass
class Detection:
    cls: int
    label: str
    name: str
    score: float
    box: list[float]          # x1, y1, x2, y2 in pixels
    yolo_score: float | None = None
    vit_bg_prob: float | None = None


def severity_index(dets: list[Detection], w: int, h: int) -> tuple[float, str]:
    """Heuristic 0-100 maintenance-priority score: class weight x confidence x sqrt(relative area)."""
    s = sum(SEVERITY_W[d.cls] * d.score * np.sqrt((d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) / (w * h))
            for d in dets)
    idx = float(min(100.0, 100.0 * s / 1.5))
    level = "Good" if idx < 15 else "Fair" if idx < 40 else "Poor" if idx < 70 else "Critical"
    return round(idx, 1), level


class RoadVisionPredictor:
    def __init__(self, classes: list[str], class_names: list[str], yolo_path: str | None = None,
                 rtdetr_path: str | None = None, vit_path: str | None = None, calibration: str | None = None,
                 device: str = "cpu", imgsz: int = 640, proposal_conf: float = 0.05, iou: float = 0.6,
                 ensemble_weights=(2, 1), wbf_iou: float = 0.55):
        import torch
        from ultralytics import RTDETR, YOLO

        self.torch = torch
        self.classes, self.class_names = classes, class_names
        self.device = "cpu" if device == "cpu" or not torch.cuda.is_available() else f"cuda:{device}" if str(device).isdigit() else device
        self.imgsz, self.proposal_conf, self.iou = imgsz, proposal_conf, iou
        self.ens_w, self.wbf_iou = list(ensemble_weights), wbf_iou
        self.yolo = YOLO(yolo_path) if yolo_path and Path(yolo_path).exists() else None
        self.rtdetr = RTDETR(rtdetr_path) if rtdetr_path and Path(rtdetr_path).exists() else None
        self.vit = None
        if vit_path and Path(vit_path).exists():
            from ..models.vit_refiner import load_vit

            self.vit, ck = load_vit(vit_path, self.device)
            self.vit_tf = vit_eval_tf(ck["img_size"])
            self.pad = ck.get("context_pad", 0.15)
        cal = read_json(calibration) if calibration and Path(calibration).exists() else {}
        self.alpha = cal.get("alpha", 0.5)
        self.thresholds = cal.get("thresholds", {"yolo": 0.25, "rtdetr": 0.35, "hybrid": 0.25, "ensemble": 0.25})

    @property
    def available_modes(self) -> list[str]:
        m = []
        if self.yolo: m.append("yolo")
        if self.rtdetr: m.append("rtdetr")
        if self.yolo and self.vit: m.append("hybrid")
        if self.yolo and self.rtdetr: m.append("ensemble")
        return m

    # ---------- raw model outputs (boxes, scores, labels) ----------
    def _ultra(self, model, img_bgr: np.ndarray, conf: float):
        r = model.predict(img_bgr, conf=conf, iou=self.iou, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        b = r.boxes
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)

    def vit_probs(self, img_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, len(self.classes) + 1))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        batch = [to_chw(self.vit_tf(image=padded_crop(rgb, b, self.pad))["image"]) for b in boxes]
        x = self.torch.from_numpy(np.stack(batch)).to(self.device)
        with self.torch.no_grad():
            return self.vit(x).float().softmax(-1).cpu().numpy()

    @staticmethod
    def fuse(yb, ys, yl, probs, alpha: float):
        if len(yb) == 0:
            return yb, ys, yl, np.zeros(0)
        k = probs.shape[1] - 1
        p_dmg = probs[:, :k] / np.clip(probs[:, :k].sum(1, keepdims=True), 1e-9, None)
        p_bg = probs[:, k]
        onehot = np.eye(k)[yl]
        labels = (alpha * onehot + (1 - alpha) * p_dmg).argmax(1)
        vit_part = (1 - p_bg) * p_dmg[np.arange(len(labels)), labels]
        scores = np.power(ys, alpha) * np.power(np.clip(vit_part, 1e-9, None), 1 - alpha)
        return yb, scores, labels, p_bg

    def raw(self, img_bgr: np.ndarray, mode: str, alpha: float | None = None, low_conf: bool = False):
        """Return (boxes, scores, labels, extras) before thresholding (used for calibration/eval)."""
        c = 0.001 if low_conf else self.proposal_conf
        if mode == "yolo":
            return (*self._ultra(self.yolo, img_bgr, c), {})
        if mode == "rtdetr":
            return (*self._ultra(self.rtdetr, img_bgr, c), {})
        if mode == "hybrid":
            yb, ys, yl = self._ultra(self.yolo, img_bgr, self.proposal_conf)
            probs = self.vit_probs(img_bgr, yb)
            b, s, l, pbg = self.fuse(yb, ys, yl, probs, self.alpha if alpha is None else alpha)
            return b, s, l, {"yolo_scores": ys, "p_bg": pbg, "probs": probs, "yolo_labels": yl}
        if mode == "ensemble":
            p1 = self._ultra(self.yolo, img_bgr, c)
            p2 = self._ultra(self.rtdetr, img_bgr, c)
            return (*weighted_boxes_fusion([p1, p2], self.ens_w, self.wbf_iou), {})
        raise ValueError(f"unknown mode {mode}")

    def predict(self, img_bgr: np.ndarray, mode: str = "hybrid", conf: float | None = None) -> dict:
        if mode not in self.available_modes:
            raise ValueError(f"mode '{mode}' unavailable; loaded: {self.available_modes}")
        t0 = time.perf_counter()
        b, s, l, ex = self.raw(img_bgr, mode)
        thr = self.thresholds.get(mode, 0.25) if conf is None else conf
        keep = s >= thr
        dets = []
        for i in np.where(keep)[0]:
            dets.append(Detection(int(l[i]), self.classes[l[i]], self.class_names[l[i]], round(float(s[i]), 4),
                                  [round(float(v), 1) for v in b[i]],
                                  round(float(ex["yolo_scores"][i]), 4) if "yolo_scores" in ex else None,
                                  round(float(ex["p_bg"][i]), 4) if "p_bg" in ex else None))
        dets.sort(key=lambda d: -d.score)
        H, W = img_bgr.shape[:2]
        sev, level = severity_index(dets, W, H)
        return {"mode": mode, "threshold": thr, "num_detections": len(dets), "detections": [asdict(d) for d in dets],
                "counts": {n: sum(d.label == n for d in dets) for n in self.classes},
                "severity_index": sev, "condition": level, "image_size": [W, H],
                "latency_ms": round(1000 * (time.perf_counter() - t0), 1)}

    def draw(self, img_bgr: np.ndarray, result: dict) -> np.ndarray:
        out = img_bgr.copy()
        t = max(2, int(round(0.002 * max(out.shape[:2]))))
        for d in result["detections"]:
            x1, y1, x2, y2 = map(int, d["box"])
            col = COLORS[d["cls"] % len(COLORS)]
            cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
            txt = f"{d['label']} {d['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t / 2, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
            cv2.putText(out, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t / 2, (255, 255, 255), 1, cv2.LINE_AA)
        return out


def from_config(cfg) -> RoadVisionPredictor:
    a = cfg.app
    return RoadVisionPredictor(list(cfg.data.classes), list(cfg.data.class_names), a.detector, a.rtdetr, a.vit,
                               a.calibration, cfg.project.device, cfg.yolo.imgsz, cfg.hybrid.yolo_conf,
                               cfg.hybrid.iou, cfg.ensemble.weights, cfg.ensemble.wbf_iou)
