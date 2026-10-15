"""Bayesian hyper-parameter optimisation (Optuna TPE + median pruning) for the YOLO
detector and the ViT refiner. Every trial is a nested MLflow run; the study is
persisted in SQLite so it can be resumed, and the best parameters are written to
runs/optuna/<model>_best.json for the final training stage.

    python -m roadvision.tuning.optuna_search --model yolo
    python -m roadvision.tuning.optuna_search --model vit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

from ..utils.common import get_logger, write_json
from ..utils.config import load_config, workdir
from ..utils.tracking import Tracker
from ..visualization.plots import _save

log = get_logger("optuna")


def yolo_space(trial: optuna.Trial) -> dict:
    return {
        "lr0": trial.suggest_float("lr0", 1e-4, 2e-2, log=True),
        "momentum": trial.suggest_float("momentum", 0.85, 0.98),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["SGD", "AdamW"]),
        "mosaic": trial.suggest_float("mosaic", 0.5, 1.0),
        "mixup": trial.suggest_float("mixup", 0.0, 0.3),
        "hsv_v": trial.suggest_float("hsv_v", 0.2, 0.6),
        "degrees": trial.suggest_float("degrees", 0.0, 10.0),
        "box": trial.suggest_float("box", 5.0, 10.0),
        "cls": trial.suggest_float("cls", 0.3, 1.5),
    }


def vit_space(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "backbone_lr_mult": trial.suggest_float("backbone_lr_mult", 0.03, 0.5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 0.1, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "drop_path": trial.suggest_float("drop_path", 0.0, 0.3),
    }


def make_study(cfg, name: str) -> optuna.Study:
    Path(cfg.project.workdir).mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=cfg.project.seed, multivariate=True) if cfg.optuna.sampler == "tpe" \
        else optuna.samplers.RandomSampler(seed=cfg.project.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2) if cfg.optuna.pruner == "median" \
        else optuna.pruners.NopPruner()
    return optuna.create_study(study_name=f"roadvision_{name}", direction="maximize", sampler=sampler, pruner=pruner,
                               storage=cfg.optuna.storage, load_if_exists=True)


def tune(cfg, model: str) -> dict:
    tracker = Tracker(cfg.tracking.mlflow_uri, cfg.tracking.experiment, workdir(cfg, "tracking"))
    study = make_study(cfg, model)

    def objective(trial: optuna.Trial) -> float:
        with tracker.run(f"optuna_{model}_t{trial.number}", tags={"stage": "tune", "model": model}):
            if model == "yolo":
                from ..models.detectors import train_detector

                params = yolo_space(trial)
                params.update(epochs=cfg.optuna.yolo_epochs_per_trial, patience=100, close_mosaic=0)

                def report(trainer):
                    ep = trainer.epoch
                    val = float(trainer.metrics.get("metrics/mAP50(B)", 0.0))
                    trial.report(val, ep)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                tracker.params(params)
                out = train_detector(cfg, "yolo", params, name=f"optuna_t{trial.number}",
                                     fraction=cfg.optuna.yolo_data_fraction, callbacks={"on_fit_epoch_end": report})
                score = out["val"].get("metrics/mAP50(B)", 0.0)
            else:
                from ..models.vit_refiner import train_vit

                params = vit_space(trial)
                tracker.params(params)
                score = train_vit(cfg, params, epochs=cfg.optuna.vit_epochs_per_trial, trial=trial)["macro_f1"]
            tracker.metrics({"objective": score})
            return score

    n = cfg.optuna.yolo_trials if model == "yolo" else cfg.optuna.vit_trials
    study.optimize(objective, n_trials=n, catch=(RuntimeError,), gc_after_trial=True)
    best = dict(study.best_params)
    out = workdir(cfg, "optuna")
    write_json(best, out / f"{model}_best.json")
    study.trials_dataframe().to_csv(out / f"{model}_trials.csv", index=False)
    log.info("Best %s value %.4f params %s", model, study.best_value, best)
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

        ax = plot_optimization_history(study); _save(ax.figure, cfg, f"16_optuna_history_{model}")
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 3:
            ax = plot_param_importances(study); _save(ax.figure, cfg, f"17_optuna_importance_{model}")
    except Exception as exc:  # plotting must never kill tuning
        log.warning("optuna plots skipped: %s", exc)
        plt.close("all")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--model", choices=["yolo", "vit"], required=True)
    a = ap.parse_args()
    tune(load_config(a.config, a.override), a.model)
