"""Explainability and representation analysis.

  * ViT embedding projections: PCA (explained variance), t-SNE and UMAP of the
    [CLS] pre-logit features, plus silhouette score and k-NN accuracy as
    quantitative measures of class separability.
  * Attention rollout (Abnar & Zuidema, 2020) for the ViT refiner.
  * Eigen-CAM (Muhammad & Yeasin, 2020) for the YOLO backbone — gradient-free,
    so it works on detection heads without class-specific gradients.

    python -m roadvision.explain.explainability
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from ..data.augment import to_chw, vit_eval_tf
from ..models.vit_refiner import load_vit
from ..utils.common import get_logger, torch_device, write_json
from ..utils.config import load_config, workdir
from ..visualization.plots import PALETTE, _save

log = get_logger("explain")


@torch.no_grad()
def vit_embeddings(model, paths: list[str], size: int, device, bs: int = 64) -> np.ndarray:
    tf = vit_eval_tf(size)
    feats = []
    for i in range(0, len(paths), bs):
        x = torch.from_numpy(np.stack([to_chw(tf(image=cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))["image"])
                                       for p in paths[i:i + bs]])).to(device)
        f = model.forward_features(x)
        feats.append(model.forward_head(f, pre_logits=True).float().cpu().numpy())
    return np.concatenate(feats)


def embedding_projections(cfg, model, ck, device, n_max: int = 3000) -> dict:
    df = pd.read_csv(Path(cfg.data.crops_dir) / "crops.csv")
    df = df[df.split.isin(["val", "test"])]
    df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), n_max // 5), random_state=0))
    X = vit_embeddings(model, df.path.tolist(), ck["img_size"], device)
    y = df.y.to_numpy()
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    pca = PCA(n_components=min(50, Xs.shape[1], len(Xs) - 1), random_state=0).fit(Xs)
    Z50 = pca.transform(Xs)
    projections = {"PCA": Z50[:, :2],
                   "t-SNE": TSNE(2, perplexity=min(30, max(5, len(Xs) // 10)), init="pca", random_state=0).fit_transform(Z50)}
    try:
        import umap

        projections["UMAP"] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Z50)
    except ImportError:
        log.warning("umap-learn not installed; skipping UMAP")
    names = ck["classes"]
    fig, ax = plt.subplots(1, len(projections) + 1, figsize=(4.6 * (len(projections) + 1), 4.2))
    ax[0].plot(np.cumsum(pca.explained_variance_ratio_), color=PALETTE[0])
    ax[0].set(title="PCA cumulative explained variance", xlabel="# components", ylabel="variance", ylim=(0, 1.02))
    for a, (k, Z) in zip(ax[1:], projections.items()):
        for c in np.unique(y):
            a.scatter(Z[y == c, 0], Z[y == c, 1], s=5, alpha=.6, color=PALETTE[c % 5], label=names[c])
        a.set(title=f"ViT [CLS] embeddings — {k}", xticks=[], yticks=[])
    ax[-1].legend(markerscale=3, fontsize=8)
    _save(fig, cfg, "13_vit_embeddings")
    stats = {"pca_var_10": float(pca.explained_variance_ratio_[:10].sum()),
             "silhouette_pca50": float(silhouette_score(Z50, y)) if len(np.unique(y)) > 1 else None,
             "knn5_cv_acc": float(cross_val_score(KNeighborsClassifier(5), Z50, y, cv=min(5, np.bincount(y).min())).mean())
             if np.bincount(y).min() >= 2 else None}
    log.info("embedding stats %s", stats)
    return stats


def attention_rollout(model, x: torch.Tensor, discard_ratio: float = 0.9) -> np.ndarray:
    """Recursively multiply head-averaged attention (+identity for residuals)."""
    attns = []
    hooks = []
    for blk in model.blocks:
        blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: attns.append(i[0].detach())))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    result = torch.eye(attns[0].shape[-1], device=x.device)
    for a in attns:
        a = a[0].mean(0)
        flat = a.flatten()
        k = int(flat.numel() * discard_ratio)
        if k:
            idx = flat.argsort()[:k]
            idx = idx[idx != 0]
            flat[idx] = 0
        a = a + torch.eye(a.shape[-1], device=x.device)
        a = a / a.sum(-1, keepdim=True)
        result = a @ result
    mask = result[0, n_prefix:]
    g = int(mask.numel() ** 0.5)
    mask = mask.reshape(g, g).cpu().numpy()
    return mask / (mask.max() + 1e-9)


def attention_gallery(cfg, model, ck, device, n_per_class: int = 3) -> None:
    df = pd.read_csv(Path(cfg.data.crops_dir) / "crops.csv")
    df = df[(df.split == "test") & (df.label != "BG")]
    sample = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), n_per_class), random_state=0))
    if sample.empty:
        return
    tf = vit_eval_tf(ck["img_size"])
    n = len(sample)
    fig, ax = plt.subplots(2, n, figsize=(2.1 * n, 4.4), squeeze=False)
    for j, r in enumerate(sample.itertuples()):
        rgb = cv2.resize(cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2RGB), (ck["img_size"],) * 2)
        x = torch.from_numpy(to_chw(tf(image=rgb)["image"]))[None].to(device)
        m = cv2.resize(attention_rollout(model, x), rgb.shape[:2][::-1])
        pred = ck["classes"][int(model(x).argmax())]
        ax[0, j].imshow(rgb); ax[0, j].set_title(f"GT {r.label}", fontsize=8)
        ax[1, j].imshow(rgb); ax[1, j].imshow(m, cmap="jet", alpha=.45); ax[1, j].set_title(f"pred {pred}", fontsize=8)
        ax[0, j].axis("off"); ax[1, j].axis("off")
    fig.suptitle("ViT attention rollout: where the transformer looks", fontweight="bold")
    _save(fig, cfg, "14_vit_attention_rollout")


def eigencam(yolo, img_bgr: np.ndarray, layer_idx: int = -4, imgsz: int = 640) -> np.ndarray:
    net = yolo.model.eval()
    acts = []
    h = net.model[layer_idx].register_forward_hook(lambda m, i, o: acts.append(o.detach()))
    lb = cv2.resize(img_bgr, (imgsz, imgsz))
    x = torch.from_numpy(lb[:, :, ::-1].transpose(2, 0, 1).copy()).float()[None] / 255.0
    x = x.to(next(net.parameters()).device)
    with torch.no_grad():
        net(x)
    h.remove()
    A = acts[0][0].float().cpu().numpy()                    # C,H,W
    C, H, W = A.shape
    flat = A.reshape(C, -1)
    flat = flat - flat.mean(1, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    cam = vt[0].reshape(H, W)
    if cam.sum() < 0:
        cam = -cam
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-9)
    return cv2.resize(cam, img_bgr.shape[:2][::-1])


def eigencam_gallery(cfg, n: int = 4) -> None:
    from ultralytics import YOLO

    if not Path(cfg.app.detector).exists():
        return
    yolo = YOLO(cfg.app.detector)
    imgs = pd.read_csv(Path(cfg.data.processed_dir) / "images.csv")
    s = imgs[(imgs.split == "test") & (imgs.n_boxes > 0)].sample(min(n, len(imgs)), random_state=3)
    fig, ax = plt.subplots(2, len(s), figsize=(3.4 * len(s), 5.2), squeeze=False)
    for j, r in enumerate(s.itertuples()):
        img = cv2.imread(str(Path(cfg.data.processed_dir) / "images" / "test" / f"{r.stem}.jpg"))
        cam = eigencam(yolo, img, imgsz=cfg.yolo.imgsz)
        res = yolo.predict(img, conf=0.25, verbose=False, imgsz=cfg.yolo.imgsz)[0].plot()
        ax[0, j].imshow(res[:, :, ::-1]); ax[1, j].imshow(img[:, :, ::-1]); ax[1, j].imshow(cam, cmap="jet", alpha=.45)
        ax[0, j].axis("off"); ax[1, j].axis("off")
    ax[0, 0].set_title("YOLO detections", loc="left"); ax[1, 0].set_title("Eigen-CAM (backbone/neck)", loc="left")
    _save(fig, cfg, "15_yolo_eigencam")


def main(cfg) -> None:
    device = torch_device(cfg.project.device)
    stats = {}
    if Path(cfg.app.vit).exists():
        model, ck = load_vit(cfg.app.vit, device)
        stats["embeddings"] = embedding_projections(cfg, model, ck, device)
        attention_gallery(cfg, model, ck, device)
    eigencam_gallery(cfg)
    write_json(stats, workdir(cfg, "metrics") / "explainability.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    main(load_config(a.config, a.override))
