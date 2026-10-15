# RoadVision — Real-Time Road Damage Detection using YOLO and Vision Transformers

19MAM56 Advanced Machine Learning Lab · CAT II project · DAKSHINESH S, M.Sc. AI&ML, CIT Coimbatore

RoadVision finds **D00 longitudinal cracks, D10 transverse cracks, D20 alligator cracks and D40 potholes** in road images (RDD2022 dataset).
**How it works:** a YOLO detector proposes boxes in real time, and a Vision Transformer, fine-tuned on crops that include a background class built from mined false positives, re-labels and re-scores each one. A fusion weight α and every threshold are calibrated on the validation split. The baselines are RT-DETR (a transformer detector) and a WBF ensemble of YOLO and RT-DETR.

```
RDD2022 → cleaning + stratified split → YOLO ─┬─────────────► proposals ─┐
                                              └► crops + hard negatives → ViT ─► fusion(α) → detections
                                    RT-DETR ──► WBF ensemble (baseline)                      │
Optuna TPE + MLflow  ·  CRDDC-protocol eval + bootstrap CI  ·  FastAPI + Gradio in Docker → HF Spaces
```

## Rubric → where it is
| Rubric item (marks) | Where |
|---|---|
| Data engineering & visualisation (8) | `data/prepare.py` (cleaning, missing labels, stratified split, audit), `data/augment.py`, `data/crops.py`, PCA/t-SNE/UMAP in `explain/explainability.py`, 17 figures in `visualization/plots.py` |
| Modelling & pipeline optimisation (10) | `models/detectors.py` (YOLO, RT-DETR), `models/vit_refiner.py` (ViT transfer learning), `inference/predictor.py` (hybrid fusion), `models/ensemble.py` (WBF), `tuning/optuna_search.py` (TPE + pruning), `utils/tracking.py` (MLflow), `scripts/run_pipeline.py` |
| Benchmarking & comparative analysis (10) | `evaluation/metrics.py` (CRDDC F1, AP50, confusion matrix, error taxonomy, bootstrap), `evaluation/evaluate.py` (val-only calibration, ablation, per-country results), `evaluation/benchmark.py` (published SOTA, `configs/china_811.yaml` for a like-for-like comparison) |
| Deployment & containerisation (4 + 4) | `api/main.py` (FastAPI with validation), `app/gradio_ui.py`, `Dockerfile`, `docker-compose.yml`, `deploy/` (Hugging Face Space) |
| Documentation & report (4) | `report/RoadVision_IEEE.tex` (IEEE, Section 5 structure, auto-filled tables), this README, `tests/` |

## Run it (Colab/Kaggle GPU, about 8–12 h for the full run)
Open `notebooks/RoadVision_Colab.ipynb`, or run:
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --stages all                    # download → … → report
python scripts/run_pipeline.py --stages all --skip tune_yolo tune_vit   # faster, skips HPO
python scripts/run_pipeline.py --override configs/china_811.yaml --stages prepare train_yolo crops train_vit train_rtdetr evaluate benchmark
mlflow ui --backend-store-uri ./mlruns                         # experiment tracking
```
Quick CPU check on synthetic data: `make smoke`. Tests: `make test`.

Outputs: weights go to `models/`, metrics to `runs/metrics/`, figures to `runs/figures/` and `report/figures/`, and LaTeX tables to `report/tables/`.

## Deploy
```bash
uvicorn api.main:app --port 8000          # local: http://localhost:8000  (/docs, /ui)
docker build -t roadvision . && docker run -p 7860:7860 -v $PWD/models:/home/user/app/models roadvision
HF_USER=<your-hf-name> bash deploy/push_to_hf_space.sh   # needs `huggingface-cli login`
```

## Report
Upload the `report/` folder to Overleaf and compile `RoadVision_IEEE.tex`. Every table and number is filled in automatically by the `report` stage. After your run, re-read the lines marked `%% EDIT-AFTER-RUN` so the discussion matches your actual results.

## Results (fill in after training)
| Model | P | R | F1 | mAP50 | FPS |
|---|---|---|---|---|---|
| YOLO | | | | | |
| RT-DETR | | | | | |
| WBF ensemble | | | | | |
| **Hybrid YOLO+ViT** | | | | | |

Links: GitHub `<repo-url>` · Hugging Face Space `<space-url>`
