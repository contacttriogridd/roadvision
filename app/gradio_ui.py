"""Gradio interface (mounted at /ui by api/main.py; also runnable stand-alone)."""
from __future__ import annotations

import cv2
import gradio as gr
import pandas as pd

DESC = """**RoadVision** detects road damage in dash-cam / phone images:
D00 longitudinal crack · D10 transverse crack · D20 alligator crack · D40 pothole.
*Hybrid* = YOLO proposals re-scored by a Vision Transformer (recommended)."""


def build_ui(predictor) -> gr.Blocks:
    modes = predictor.available_modes or ["hybrid"]

    def infer(image, mode, conf, use_cal):
        if image is None:
            raise gr.Error("Please upload an image.")
        if mode not in predictor.available_modes:
            raise gr.Error(f"Mode {mode} not loaded. Available: {predictor.available_modes}")
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        res = predictor.predict(bgr, mode, None if use_cal else conf)
        vis = cv2.cvtColor(predictor.draw(bgr, res), cv2.COLOR_BGR2RGB)
        table = pd.DataFrame([{"class": d["label"], "name": d["name"], "score": d["score"],
                               "box (x1,y1,x2,y2)": ", ".join(f"{v:.0f}" for v in d["box"])} for d in res["detections"]])
        summary = (f"### Condition: **{res['condition']}**  ·  severity index {res['severity_index']}/100\n"
                   f"{res['num_detections']} detections · threshold {res['threshold']:.2f} · {res['latency_ms']} ms")
        return vis, summary, table

    with gr.Blocks(title="RoadVision") as ui:
        gr.Markdown("# RoadVision — Real-Time Road Damage Detection\n" + DESC)
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="numpy", label="Road image")
                mode = gr.Radio(modes, value=modes[0] if "hybrid" not in modes else "hybrid", label="Model")
                use_cal = gr.Checkbox(True, label="Use validation-calibrated threshold")
                conf = gr.Slider(0.05, 0.95, 0.25, step=0.05, label="Manual confidence threshold")
                btn = gr.Button("Detect damage", variant="primary")
            with gr.Column():
                out = gr.Image(label="Detections")
                summ = gr.Markdown()
        tab = gr.Dataframe(label="Detections")
        btn.click(infer, [img, mode, conf, use_cal], [out, summ, tab])
    return ui


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from roadvision.inference.predictor import from_config
    from roadvision.utils.config import load_config

    build_ui(from_config(load_config(root / "configs/default.yaml"))).launch(server_name="0.0.0.0", server_port=7860)
