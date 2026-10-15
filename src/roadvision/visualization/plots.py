"""All analytical figures for the report (written to <workdir>/figures and report/figures).

  01 class distribution by country          08 F1-vs-threshold calibration curves
  02 box geometry (area / aspect / centres)  09 hybrid alpha calibration
  03 augmentation gallery                    10 error taxonomy + recall by size
  04 YOLO / RT-DETR training curves          11 F1 by country heat-map
  05 ViT loss / accuracy curves              12 benchmark vs state of the art
  06 normalised confusion matrix             (+ embeddings, attention rollout, EigenCAM,
  07 per-class precision-recall curves          Optuna history/importance from their modules)

    python -m roadvision.visualization.plots
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils.common import get_logger, read_json
from ..utils.config import load_config, workdir

log = get_logger("plots")
PALETTE = ["#2E86AB", "#F18F01", "#A23B72", "#C73E1D", "#6C757D"]
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def _save(fig, cfg, name: str) -> None:
    for d in (workdir(cfg, "figures"), Path("report/figures")):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png")
    plt.close(fig)
    log.info("figure %s", name)


def class_distribution(cfg, boxes: pd.DataFrame) -> None:
    t = boxes.pivot_table(index="country", columns="label", values="cls", aggfunc="count", fill_value=0)
    t = t.reindex(columns=list(cfg.data.classes), fill_value=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    t.plot.bar(stacked=True, ax=ax[0], color=PALETTE[:4], width=0.75)
    ax[0].set(title="Instances per class and country", xlabel="", ylabel="# boxes")
    ax[0].legend([f"{c} {n}" for c, n in zip(cfg.data.classes, cfg.data.class_names)], fontsize=8)
    ax[0].tick_params(axis="x", rotation=30)
    (t.T / t.T.sum()).T.plot.barh(stacked=True, ax=ax[1], color=PALETTE[:4], legend=False)
    ax[1].set(title="Class proportions (domain shift across countries)", xlabel="share", ylabel="")
    _save(fig, cfg, "01_class_distribution")


def box_geometry(cfg, boxes: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, c in enumerate(cfg.data.classes):
        s = boxes[boxes.label == c]
        if len(s):
            sns.kdeplot(np.log10(s.rel_area.clip(1e-6)), ax=ax[0], label=c, color=PALETTE[i], fill=True, alpha=.2,
                        warn_singular=False)
            sns.kdeplot(np.log10(s.aspect.clip(1e-2, 1e2)), ax=ax[1], label=c, color=PALETTE[i], warn_singular=False)
    ax[0].set(title="Relative box area", xlabel="log10(box area / image area)"); ax[0].legend()
    ax[1].set(title="Aspect ratio (w/h)", xlabel="log10 aspect  (>0 = wide, <0 = tall)")
    ax[1].axvline(0, color="k", lw=.8, ls="--")
    h = ax[2].hist2d(boxes.cx, boxes.cy, bins=40, cmap="magma")
    ax[2].invert_yaxis(); ax[2].set(title="Box-centre heat-map", xlabel="x / W", ylabel="y / H")
    fig.colorbar(h[3], ax=ax[2])
    _save(fig, cfg, "02_box_geometry")


def augmentation_gallery(cfg, images: pd.DataFrame) -> None:
    import cv2

    from ..data.augment import road_augment

    aug = road_augment(cfg.augmentation)
    sample = images[images.n_boxes > 0].sample(min(3, len(images)), random_state=1)
    fig, ax = plt.subplots(len(sample), 4, figsize=(13, 2.6 * len(sample)), squeeze=False)
    for r, row in enumerate(sample.itertuples()):
        img = cv2.cvtColor(cv2.imread(str(Path(cfg.data.processed_dir) / "images" / row.split / f"{row.stem}.jpg")),
                           cv2.COLOR_BGR2RGB)
        ax[r, 0].imshow(img); ax[r, 0].set_title("original" if r == 0 else "")
        for k in range(1, 4):
            ax[r, k].imshow(aug(image=img)["image"]); ax[r, k].set_title("augmented" if r == 0 else "")
        for a in ax[r]:
            a.axis("off")
    _save(fig, cfg, "03_augmentation_gallery")


def detector_curves(cfg) -> None:
    runs = [(k, Path(cfg.project.workdir) / "detect" / k / "results.csv") for k in ("yolo", "rtdetr")]
    runs = [(k, p) for k, p in runs if p.exists()]
    if not runs:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for i, (k, p) in enumerate(runs):
        d = pd.read_csv(p); d.columns = [c.strip() for c in d.columns]
        tr = [c for c in d.columns if c.startswith("train/") and "loss" in c]
        va = [c for c in d.columns if c.startswith("val/") and "loss" in c]
        ax[0].plot(d.epoch, d[tr].sum(1), color=PALETTE[i], label=f"{k} train")
        ax[0].plot(d.epoch, d[va].sum(1), color=PALETTE[i], ls="--", label=f"{k} val")
        ax[1].plot(d.epoch, d["metrics/mAP50(B)"], color=PALETTE[i], label=f"{k} mAP50")
        ax[1].plot(d.epoch, d["metrics/mAP50-95(B)"], color=PALETTE[i], ls=":", label=f"{k} mAP50-95")
        ax[2].plot(d.epoch, d["metrics/precision(B)"], color=PALETTE[i], label=f"{k} P")
        ax[2].plot(d.epoch, d["metrics/recall(B)"], color=PALETTE[i], ls="--", label=f"{k} R")
    ax[0].set(title="Total loss", xlabel="epoch"); ax[1].set(title="Validation mAP", xlabel="epoch")
    ax[2].set(title="Validation precision / recall", xlabel="epoch")
    for a in ax: a.legend(fontsize=7)
    _save(fig, cfg, "04_detector_training_curves")


def vit_curves(cfg) -> None:
    p = workdir(cfg, "metrics") / "vit_history.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(d.epoch, d.train_loss, label="train", color=PALETTE[0]); ax[0].plot(d.epoch, d.val_loss, label="val", color=PALETTE[1])
    ax[0].set(title="ViT refiner loss", xlabel="epoch"); ax[0].legend()
    ax[1].plot(d.epoch, d.val_acc, label="val accuracy", color=PALETTE[2]); ax[1].plot(d.epoch, d.val_macro_f1, label="val macro-F1", color=PALETTE[3])
    ax[1].set(title="ViT refiner accuracy", xlabel="epoch"); ax[1].legend()
    _save(fig, cfg, "05_vit_training_curves")


def eval_figures(cfg) -> None:
    mp = workdir(cfg, "metrics")
    if not (mp / "eval_results.json").exists():
        return
    res = read_json(mp / "eval_results.json")
    modes = list(res["modes"])
    names = list(cfg.data.classes) + ["BG"]
    best = "hybrid" if "hybrid" in modes else modes[0]
    # 06 confusion
    fig, axs = plt.subplots(1, min(2, len(modes)), figsize=(6 * min(2, len(modes)), 4.8), squeeze=False)
    for a, m in zip(axs[0], [best] + [x for x in ("yolo",) if x in modes and x != best]):
        M = np.array(res["modes"][m]["confusion"], float)
        Mn = M / np.maximum(1, M.sum(1, keepdims=True))
        sns.heatmap(Mn, annot=M.astype(int), fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=a,
                    cbar=False, vmin=0, vmax=1)
        a.set(title=f"Confusion ({m}), row-normalised colour", xlabel="predicted", ylabel="ground truth")
    _save(fig, cfg, "06_confusion_matrix")
    # 07 PR curves
    cache = mp / "pred_cache.pkl"
    if cache.exists():
        from ..evaluation.metrics import pr_curve

        with open(cache, "rb") as f:
            c = pickle.load(f)
        fig, axs = plt.subplots(1, len(cfg.data.classes), figsize=(4 * len(cfg.data.classes), 3.6), squeeze=False)
        for k, cls in enumerate(cfg.data.classes):
            for i, m in enumerate(modes):
                r, p = pr_curve(c["test"][m], c["gts_test"], k)
                axs[0, k].plot(r, p, color=PALETTE[i], label=f"{m} AP={res['modes'][m]['AP50_per_class'][k]:.3f}")
            axs[0, k].set(title=f"{cls} {cfg.data.class_names[k]}", xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
            axs[0, k].legend(fontsize=7)
        _save(fig, cfg, "07_pr_curves")
    # 08 F1 vs threshold
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, m in enumerate(modes):
        cur = np.array(res["f1_curves_val"][m])
        ax.plot(cur[:, 0], cur[:, 1], color=PALETTE[i], label=m)
        ax.axvline(res["calibration"]["thresholds"][m], color=PALETTE[i], ls=":", lw=1)
    ax.set(title="Validation F1 vs. confidence threshold", xlabel="threshold", ylabel="F1"); ax.legend()
    _save(fig, cfg, "08_f1_threshold_curves")
    # 09 alpha
    if "alpha_grid" in res["calibration"]:
        g = pd.DataFrame(res["calibration"]["alpha_grid"])
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(g.alpha, g.val_f1, "o-", color=PALETTE[2])
        ax.axvline(res["calibration"]["alpha"], ls="--", color="k", lw=1)
        ax.set(title=r"Hybrid fusion weight $\alpha$ (val)", xlabel=r"$\alpha$  (0 = ViT only, 1 = YOLO only)", ylabel="best F1")
        _save(fig, cfg, "09_alpha_calibration")
    # 10 errors
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    e = pd.DataFrame({m: res["modes"][m]["errors"]["fp_types"] for m in modes}).fillna(0).T
    e.plot.bar(stacked=True, ax=ax[0], color=PALETTE, rot=0); ax[0].set(title="False-positive taxonomy (test)", ylabel="# FP")
    rs = pd.DataFrame({m: res["modes"][m]["errors"]["recall_by_size"] for m in modes}).reindex(["small", "medium", "large"])
    rs.plot.bar(ax=ax[1], color=PALETTE, rot=0); ax[1].set(title="Recall by object size", ylabel="recall", ylim=(0, 1))
    _save(fig, cfg, "10_error_analysis")
    # 11 country
    cdf = pd.DataFrame({m: res["modes"][m]["f1_by_country"] for m in modes})
    fig, ax = plt.subplots(figsize=(1.6 * len(modes) + 3, 0.5 * len(cdf) + 1.5))
    sns.heatmap(cdf, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set(title="Test F1 by country (domain generalisation)")
    _save(fig, cfg, "11_f1_by_country")


def benchmark_figure(cfg) -> None:
    p = workdir(cfg, "metrics") / "benchmark.csv"
    if not p.exists():
        return
    d = pd.read_csv(p).sort_values("f1")
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(d) + 1.2))
    cols = [PALETTE[3] if o else (PALETTE[0] if c else "#B0B7BF") for o, c in zip(d.ours, d.comparable)]
    ax.barh(d.method, d.f1, color=cols)
    for y, (f, pr) in enumerate(zip(d.f1, d.protocol)):
        ax.text(f + 0.005, y, f"{f:.3f}  [{pr}]", va="center", fontsize=7)
    ax.set(xlim=(0, 1.15), xlabel="F1 (IoU > 0.5)", title="Benchmark vs. published road-damage detectors")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PALETTE[3], label="this work"), Patch(color=PALETTE[0], label="literature, same protocol"),
                       Patch(color="#B0B7BF", label="literature, hidden challenge test (indicative)")], fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    _save(fig, cfg, "12_benchmark")


def main(cfg) -> None:
    proc = Path(cfg.data.processed_dir)
    boxes, images = pd.read_csv(proc / "boxes.csv"), pd.read_csv(proc / "images.csv")
    class_distribution(cfg, boxes)
    box_geometry(cfg, boxes)
    try:
        augmentation_gallery(cfg, images)
    except ImportError:
        log.warning("albumentations missing; skipping augmentation gallery")
    detector_curves(cfg)
    vit_curves(cfg)
    eval_figures(cfg)
    benchmark_figure(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    main(load_config(a.config, a.override))
