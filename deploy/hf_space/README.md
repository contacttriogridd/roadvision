---
title: RoadVision
emoji: 🛣️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-time road damage detection with YOLO + Vision Transformers
---

# RoadVision — Real-Time Road Damage Detection

Upload a road image and the app detects **D00 longitudinal cracks, D10 transverse cracks,
D20 alligator cracks and D40 potholes** (RDD2022 classes).

* `/` — web UI · `/ui` — Gradio UI · `/docs` — interactive REST API (FastAPI / OpenAPI)
* Models: YOLO detector, RT-DETR, ViT refiner (hybrid cascade), WBF ensemble
