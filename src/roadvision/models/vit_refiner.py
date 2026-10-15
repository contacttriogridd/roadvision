"""Vision-Transformer refiner (transfer learning, timm).

A ViT pre-trained on ImageNet-21k is fine-tuned on context-padded damage crops to
classify {D00, D10, D20, D40, BG}. At inference it re-scores YOLO proposals: it
corrects confusions between crack types (global self-attention captures the
long, thin geometry of cracks better than local convolutions) and rejects
background false positives via the BG class.

    python -m roadvision.models.vit_refiner [--params runs/optuna/vit_best.json]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..data.augment import to_chw, vit_eval_tf, vit_train_tf
from ..utils.common import get_logger, read_json, seed_everything, torch_device, write_json
from ..utils.config import load_config, workdir
from ..utils.tracking import Tracker

log = get_logger("vit")


class CropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.paths, self.y, self.tf = df.path.tolist(), df.y.to_numpy(), tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(to_chw(self.tf(image=img)["image"])), int(self.y[i])


def build_vit(backbone: str, num_classes: int, drop_path: float = 0.1, pretrained: bool = True) -> nn.Module:
    import timm

    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path)


def _loaders(cfg, p: dict):
    df = pd.read_csv(Path(cfg.data.crops_dir) / "crops.csv")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    size = cfg.vit.img_size
    ds_tr, ds_va = CropDataset(tr, vit_train_tf(cfg.augmentation, size)), CropDataset(va, vit_eval_tf(size))
    sampler = None
    if cfg.vit.balanced_sampler:
        freq = np.bincount(tr.y, minlength=len(cfg.data.classes) + 1).astype(float)
        w = 1.0 / np.maximum(freq[tr.y.to_numpy()], 1)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(tr), replacement=True)
    kw = dict(batch_size=p["batch"], num_workers=cfg.vit.num_workers, pin_memory=torch.cuda.is_available())
    return DataLoader(ds_tr, sampler=sampler, shuffle=sampler is None, drop_last=len(tr) > p["batch"], **kw), \
        DataLoader(ds_va, shuffle=False, **kw)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        probs.append(logits.float().softmax(-1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def train_vit(cfg, params: dict | None = None, epochs: int | None = None, trial=None,
              tracker: Tracker | None = None, save_path: str | None = None) -> dict:
    seed_everything(cfg.project.seed)
    p = {k: cfg.vit[k] for k in ("lr", "backbone_lr_mult", "weight_decay", "label_smoothing", "drop_path",
                                  "batch", "warmup_epochs")}
    p.update(params or {})
    epochs = epochs or cfg.vit.epochs
    device = torch_device(cfg.project.device)
    n_cls = len(cfg.data.classes) + 1
    model = build_vit(cfg.vit.backbone, n_cls, p["drop_path"]).to(device)
    tr_dl, va_dl = _loaders(cfg, p)
    head = [q for n, q in model.named_parameters() if n.startswith("head")]
    body = [q for n, q in model.named_parameters() if not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": body, "lr": p["lr"] * p["backbone_lr_mult"]},
                             {"params": head, "lr": p["lr"]}], weight_decay=p["weight_decay"])
    steps, warm = epochs * max(1, len(tr_dl)), p["warmup_epochs"] * max(1, len(tr_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=p["label_smoothing"])
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best = {"macro_f1": -1.0}
    history = []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        probs, ys = predict_loader(model, va_dl, device)
        val_loss = float(nn.functional.nll_loss(torch.log(torch.as_tensor(probs) + 1e-9), torch.as_tensor(ys)))
        pred = probs.argmax(1)
        rec = {"epoch": ep, "train_loss": tot / max(1, n), "val_loss": val_loss,
               "val_acc": accuracy_score(ys, pred), "val_macro_f1": f1_score(ys, pred, average="macro")}
        history.append(rec)
        log.info("ViT ep %d | loss %.4f | val_loss %.4f | acc %.4f | macroF1 %.4f", ep, rec["train_loss"],
                 val_loss, rec["val_acc"], rec["val_macro_f1"])
        if tracker:
            tracker.metrics({k: v for k, v in rec.items() if k != "epoch"}, step=ep)
        if rec["val_macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": rec["val_macro_f1"], "acc": rec["val_acc"], "epoch": ep}
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "backbone": cfg.vit.backbone,
                            "classes": list(cfg.data.classes) + ["BG"], "img_size": cfg.vit.img_size, "context_pad": cfg.vit.context_pad,
                            "params": p}, save_path)
        if trial is not None:
            import optuna

            trial.report(rec["val_macro_f1"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    best["history"] = history
    best["params"] = p
    return best


def load_vit(path: str, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_vit(ck["backbone"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def main(cfg, params_path: str | None) -> None:
    params = read_json(params_path) if params_path and Path(params_path).exists() else None
    tracker = Tracker(cfg.tracking.mlflow_uri, cfg.tracking.experiment, workdir(cfg, "tracking"))
    out = str(Path(cfg.app.weights_dir) / "vit_best.pt")
    with tracker.run("vit_final", tags={"stage": "train", "model": "vit"}):
        tracker.params({"backbone": cfg.vit.backbone, **(params or {})})
        res = train_vit(cfg, params, tracker=tracker, save_path=out)
    pd.DataFrame(res["history"]).to_csv(workdir(cfg, "metrics") / "vit_history.csv", index=False)
    write_json({k: v for k, v in res.items() if k != "history"}, workdir(cfg, "metrics") / "vit_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--params", default=None)
    a = ap.parse_args()
    main(load_config(a.config, a.override), a.params)
