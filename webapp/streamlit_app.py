"""RoadVision — Streamlit front-end (Streamlit Community Cloud entry point).

    streamlit run webapp/streamlit_app.py

Weights are read from models/; if absent they are downloaded from RV_WEIGHTS_URL
(defaults to the GitHub release of this repository).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
os.environ.setdefault("RV_WEIGHTS_URL", "https://github.com/contacttriogridd/roadvision/releases/download/v1.0")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from roadvision.inference.weights import ensure_weights  # noqa: E402
from roadvision.utils.config import load_config  # noqa: E402

COLORS = {"D00": "#38bdf8", "D10": "#a78bfa", "D20": "#f472b6", "D40": "#fb6a3c"}
EXAMPLES = ROOT / "app/examples"

st.set_page_config(page_title="RoadVision · Road Damage Intelligence", page_icon="🛣️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button {font-family:"Space Grotesk",system-ui,sans-serif}
.block-container {padding-top:2.2rem; max-width:1240px}
.rv-eyebrow {font:500 12px "JetBrains Mono",monospace; letter-spacing:.12em; text-transform:uppercase; color:#ffb020}
.rv-h1 {font-size:clamp(32px,4.4vw,54px); line-height:1.04; letter-spacing:-.035em; font-weight:700; margin:.35rem 0 .6rem}
.rv-h1 em {font-style:normal; color:#ffb020}
.rv-lede {color:#8a94a3; font-size:16.5px; max-width:640px}
.rv-chip {display:inline-flex; align-items:center; gap:7px; padding:5px 11px; margin:12px 6px 0 0; border-radius:999px;
  border:1px solid rgba(255,255,255,.08); background:#13171c; font-size:13px; color:#8a94a3}
.rv-chip i {width:8px; height:8px; border-radius:2px; display:inline-block}
.rv-kpi {background:#13171c; border:1px solid rgba(255,255,255,.07); border-radius:16px; padding:14px 18px}
.rv-kpi.hl {background:linear-gradient(160deg,rgba(255,176,32,.13),#13171c 60%); border-color:rgba(255,176,32,.3)}
.rv-kpi .k {font:500 11px "JetBrains Mono",monospace; letter-spacing:.1em; text-transform:uppercase; color:#5a6372}
.rv-kpi .v {font-size:30px; font-weight:600; letter-spacing:-.02em}
.rv-kpi .s {font-size:12.5px; color:#8a94a3}
.rv-cond {font-size:30px; font-weight:700; letter-spacing:-.02em; margin:0}
.rv-det {display:flex; align-items:center; gap:12px; padding:9px 12px; margin-bottom:6px; border-radius:11px;
  border:1px solid rgba(255,255,255,.07); background:#0f1216}
.rv-det .sw {width:34px; height:30px; border-radius:8px; display:grid; place-items:center; font:700 11px "JetBrains Mono",monospace; color:#0b0d10}
.rv-det .nm {flex:1; font-weight:500}
.rv-det .nm small {display:block; font:11.5px "JetBrains Mono",monospace; color:#5a6372}
.rv-det .sc {font:600 14px "JetBrains Mono",monospace}
.rv-note {font:12px "JetBrains Mono",monospace; color:#5a6372}
div[data-testid="stFileUploader"] section {border:1.5px dashed rgba(255,255,255,.14); border-radius:14px}
.stButton button[kind="primary"] {font-weight:700; border-radius:12px; padding:.6rem 0}
footer {visibility:hidden}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading YOLO + ViT models…")
def load_predictor():
    cfg = load_config(ROOT / "configs/default.yaml", [ROOT / "configs/fast45.yaml"])
    cfg.project["device"] = "cpu"
    ensure_weights(cfg)
    from roadvision.inference.predictor import from_config

    return from_config(cfg), cfg


@st.cache_data
def load_summary(path: str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


predictor, cfg = load_predictor()
summary = load_summary(str(Path(cfg.app.weights_dir) / "eval_summary.json"))

# ---------------- hero ----------------
left, right = st.columns([1.25, 1], gap="large")
with left:
    st.markdown('<div class="rv-eyebrow">— — Road damage intelligence</div>'
                '<div class="rv-h1">See every crack <em>before</em> it becomes a pothole.</div>'
                '<div class="rv-lede">A YOLO detector proposes damage in milliseconds; a Vision Transformer verifies '
                'every proposal and re-scores it. Upload a dash-cam or phone photo to get boxes, damage types and a '
                'maintenance-priority score.</div>'
                + "".join(f'<span class="rv-chip"><i style="background:{c}"></i>{k} {n}</span>'
                          for (k, c), n in zip(COLORS.items(), ["Longitudinal", "Transverse", "Alligator", "Pothole"])),
                unsafe_allow_html=True)
with right:
    if summary:
        y, h = summary["modes"].get("yolo", {}), summary["modes"].get("hybrid", {})
        ci = h.get("f1_ci95") or [None, None]
        tiles = [("Hybrid F1", f"{h.get('f1', 0):.3f}", f"95% CI {ci[0]:.3f}–{ci[1]:.3f}" if ci[0] else "held-out test", True),
                 ("Precision", f"{h.get('precision', 0):.3f}", f"vs {y.get('precision', 0):.3f} YOLO-only", False),
                 ("mAP@50", f"{y.get('mAP50', 0):.3f}", "YOLOv8s detector", False),
                 ("Throughput", f"{y.get('fps', 0):.0f}", "FPS · YOLO on GPU", False)]
        for row in (tiles[:2], tiles[2:]):
            cols = st.columns(2)
            for col, (k, v, s, hl) in zip(cols, row):
                col.markdown(f'<div class="rv-kpi {"hl" if hl else ""}"><div class="k">{k}</div><div class="v">{v}</div>'
                             f'<div class="s">{s}</div></div>', unsafe_allow_html=True)
            st.write("")
        st.markdown(f'<div class="rv-note">{summary["split"]}</div>', unsafe_allow_html=True)

st.divider()

# ---------------- controls ----------------
ctrl, view = st.columns([1, 2.1], gap="large")
with ctrl:
    st.markdown('<div class="rv-eyebrow">01 · Image</div>', unsafe_allow_html=True)
    up = st.file_uploader("Road photo", type=["jpg", "jpeg", "png", "webp", "bmp"], label_visibility="collapsed")
    samples = sorted(EXAMPLES.glob("*.jpg"))
    st.caption("…or try a held-out test image")
    opts = ["—"] + [p.name[:3] for p in samples]
    want = st.query_params.get("sample", "—")
    sample = st.radio("Sample", opts, index=opts.index(want) if want in opts else 0, horizontal=True,
                      label_visibility="collapsed")
    if samples:
        st.image([str(p) for p in samples], width=64, caption=[p.name[:3] for p in samples])

    st.markdown('<div class="rv-eyebrow">02 · Model</div>', unsafe_allow_html=True)
    modes = predictor.available_modes
    mode = st.segmented_control("Model", modes, default="hybrid" if "hybrid" in modes else modes[0],
                                format_func={"hybrid": "Hybrid · YOLO + ViT", "yolo": "YOLO · detector"}.get,
                                label_visibility="collapsed") or modes[0]

    st.markdown('<div class="rv-eyebrow">03 · Threshold</div>', unsafe_allow_html=True)
    cal = st.toggle(f"Calibrated threshold ({predictor.thresholds.get(mode, 0.25):.2f})", value=True)
    thr = st.slider("Manual threshold", 0.05, 0.95, float(predictor.thresholds.get(mode, 0.25)), 0.05, disabled=cal)
    min_score = st.slider("Hide boxes below score", 0.0, 0.95, 0.0, 0.05)

img_bytes, name = None, None
if up is not None:
    img_bytes, name = up.getvalue(), up.name
elif sample != "—":
    p = next(p for p in samples if p.name.startswith(sample))
    img_bytes, name = p.read_bytes(), p.name

with view:
    if img_bytes is None:
        st.info("Upload a road photo or pick a sample image on the left to run the detector.", icon="🛣️")
    else:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("This file could not be decoded as an image.")
            st.stop()
        with st.spinner(f"Analyzing {name} with {mode}…"):
            res = predictor.predict(img, mode, None if cal else thr)
        res["detections"] = [d for d in res["detections"] if d["score"] >= min_score]
        vis = cv2.cvtColor(predictor.draw(img, res), cv2.COLOR_BGR2RGB)
        st.image(vis, caption=f"{name} · {len(res['detections'])} detections · {res['latency_ms']:.0f} ms (CPU)",
                 width="stretch")
        ok, png = cv2.imencode(".png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        st.download_button("Download annotated PNG", png.tobytes(), file_name=Path(name).stem + "_roadvision.png",
                           mime="image/png")

        sev = res["severity_index"]
        col = "#34d399" if sev < 25 else "#ffb020" if sev < 50 else "#fb6a3c" if sev < 75 else "#f87171"
        a, b, c = st.columns([1, 1.2, 1.5], gap="medium")
        with a:
            st.markdown('<div class="rv-eyebrow">Condition</div>'
                        f'<p class="rv-cond" style="color:{col}">{res["condition"]}</p>', unsafe_allow_html=True)
            st.progress(min(1.0, sev / 100), text=f"Severity {sev:.0f}/100")
            st.caption(f"mode {res['mode']} · threshold {res['threshold']:.2f}")
        with b:
            st.markdown('<div class="rv-eyebrow">Damage by class</div>', unsafe_allow_html=True)
            counts = pd.Series({k: 0 for k in COLORS}) + pd.Series(
                [d["label"] for d in res["detections"]], dtype=str).value_counts()
            st.bar_chart(counts.fillna(0).astype(int), color="#ffb020", height=190)
        with c:
            st.markdown('<div class="rv-eyebrow">Detections</div>', unsafe_allow_html=True)
            if not res["detections"]:
                st.caption("No damage above the threshold.")
            for d in sorted(res["detections"], key=lambda d: -d["score"]):
                x1, y1, x2, y2 = d["box"]
                st.markdown(f'<div class="rv-det"><div class="sw" style="background:{COLORS[d["label"]]}">{d["label"]}</div>'
                            f'<div class="nm">{d["name"]}<small>{x2 - x1:.0f}×{y2 - y1:.0f} px</small></div>'
                            f'<div class="sc">{d["score"]:.3f}</div></div>', unsafe_allow_html=True)

# ---------------- evaluation ----------------
if summary:
    st.divider()
    st.markdown("### Measured, not claimed")
    rows = []
    for m, label in (("yolo", "YOLOv8s (detector only)"), ("hybrid", "Hybrid YOLO + ViT (proposed)")):
        v = summary["modes"].get(m)
        if v:
            ci = v.get("f1_ci95") or [np.nan, np.nan]
            rows.append({"Model": label, "F1": v["f1"], "F1 95% CI": f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                         "Precision": v["precision"], "Recall": v["recall"], "mAP@50": v.get("mAP50"), "FPS": v.get("fps")})
    st.dataframe(pd.DataFrame(rows).style.format({"F1": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
                                                  "mAP@50": "{:.3f}", "FPS": "{:.0f}"}),
                 hide_index=True, width="stretch")
    p = summary.get("paired_hybrid_vs_yolo", {})
    if p:
        sig = p["delta_ci95"][0] > 0
        st.caption(f"Paired bootstrap hybrid − YOLO: ΔF1 = {p['delta_f1']:+.4f}, 95% CI [{p['delta_ci95'][0]:.4f}, "
                   f"{p['delta_ci95'][1]:.4f}], one-sided p = {p['p_value_one_sided']:.3f}. "
                   + ("Statistically reliable gain." if sig else
                      "The interval includes zero, so the F1 gain is not significant on this split; the ViT mainly "
                      "raises precision.") + " Reduced-budget run (15 + 4 epochs) on the RDD2022-China subset.")

st.markdown('<div class="rv-note" style="margin-top:2rem">RoadVision · DAKSHINESH S · M.Sc. AI &amp; ML, Coimbatore '
            'Institute of Technology · 19MAM56 · <a href="https://github.com/contacttriogridd/roadvision">source</a></div>',
            unsafe_allow_html=True)
