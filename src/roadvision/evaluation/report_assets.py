"""Turn metrics JSON into LaTeX tables + macros consumed by report/RoadVision_IEEE.tex,
so the report always reflects the latest run (no hand-copied numbers).

    python -m roadvision.evaluation.report_assets
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..utils.common import get_logger, read_json
from ..utils.config import load_config, workdir

log = get_logger("report")
OUT = Path("report/tables")
NICE = {"yolo": "YOLO", "rtdetr": "RT-DETR", "ensemble": "WBF ens.", "hybrid": "Hybrid (ours)"}


def _f(x, d=3):
    return "--" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def _esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body, encoding="utf-8")


def main(cfg) -> None:
    m = workdir(cfg, "metrics")
    proc = Path(cfg.data.processed_dir)
    macros = []
    if (proc / "cleaning_report.json").exists():
        cr = read_json(proc / "cleaning_report.json")
        sp = cr.get("split_sizes", {})
        n = lambda v: f"{v:,}" if isinstance(v, int) else "--"
        macros += [rf"\renewcommand{{\NImages}}{{{n(cr.get('images_kept'))}}}",
                   rf"\renewcommand{{\NBoxes}}{{{n(cr.get('boxes_kept'))}}}",
                   rf"\renewcommand{{\NTrain}}{{{n(sp.get('train'))}}}",
                   rf"\renewcommand{{\NVal}}{{{n(sp.get('val'))}}}",
                   rf"\renewcommand{{\NTest}}{{{n(sp.get('test'))}}}"]
        rows = "\n".join(rf"{_esc(k)} & {v:,} \\" for k, v in cr.items() if isinstance(v, int))
        write("cleaning.tex", rf"""\begin{{table}}[t]\centering\caption{{Data-cleaning audit (counts)}}\label{{tab:clean}}
\begin{{tabular}}{{lr}}\toprule Step & Count \\ \midrule
{rows}
\bottomrule\end{{tabular}}\end{{table}}
""")
    if (m / "eval_results.json").exists():
        r = read_json(m / "eval_results.json")
        cls = list(cfg.data.classes)
        lines = []
        for k, v in r["modes"].items():
            ci = v["bootstrap"]["ci95"]
            lines.append(rf"{NICE.get(k, k)} & {_f(v['precision'])} & {_f(v['recall'])} & {_f(v['f1'])} "
                         rf"\scriptsize[{_f(ci[0])},{_f(ci[1])}] & {_f(v['mAP50'])} & {_f(v['fps'], 1)} \\")
        write("main_results.tex", rf"""\begin{{table}}[t]\centering\caption{{Test-set results (CRDDC protocol, IoU$>$0.5). F1 with 95\% bootstrap CI; thresholds chosen on validation.}}\label{{tab:main}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{lccccc}}\toprule Model & P & R & F1 [95\% CI] & mAP50 & FPS \\ \midrule
{chr(10).join(lines)}
\bottomrule\end{{tabular}}}}\end{{table}}
""")
        pc = "\n".join(rf"{NICE.get(k, k)} & " + " & ".join(_f(x) for x in v["AP50_per_class"]) + r" \\"
                       for k, v in r["modes"].items())
        write("per_class.tex", rf"""\begin{{table}}[t]\centering\caption{{Per-class AP50 on the test split}}\label{{tab:perclass}}
\begin{{tabular}}{{l{'c' * len(cls)}}}\toprule Model & {' & '.join(cls)} \\ \midrule
{pc}
\bottomrule\end{{tabular}}\end{{table}}
""")
        if "ablation" in r:
            ab = "\n".join(rf"{_esc(k)} & {_f(v['precision'])} & {_f(v['recall'])} & {_f(v['f1'])} \\" for k, v in r["ablation"].items())
            write("ablation.tex", rf"""\begin{{table}}[t]\centering\caption{{Ablation of the hybrid YOLO+ViT cascade (test)}}\label{{tab:ablation}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{lccc}}\toprule Variant & P & R & F1 \\ \midrule
{ab}
\bottomrule\end{{tabular}}}}\end{{table}}
""")
        if "hybrid" in r["modes"]:
            h, y = r["modes"]["hybrid"], r["modes"].get("yolo", {})
            macros += [rf"\renewcommand{{\HybF}}{{{_f(h['f1'])}}}", rf"\renewcommand{{\HybMAP}}{{{_f(h['mAP50'])}}}",
                       rf"\renewcommand{{\HybFPS}}{{{_f(h['fps'], 1)}}}", rf"\renewcommand{{\Alpha}}{{{_f(r['calibration']['alpha'], 2)}}}"]
            if y:
                macros += [rf"\renewcommand{{\YoloF}}{{{_f(y['f1'])}}}", rf"\renewcommand{{\YoloFPS}}{{{_f(y['fps'], 1)}}}"]
            if "paired_hybrid_vs_yolo" in r:
                p = r["paired_hybrid_vs_yolo"]
                macros += [rf"\renewcommand{{\DeltaF}}{{{p['delta_f1']:+.3f}}}",
                           rf"\renewcommand{{\DeltaCI}}{{[{_f(p['delta_ci95'][0])}, {_f(p['delta_ci95'][1])}]}}",
                           rf"\renewcommand{{\PVal}}{{{p['p_value_one_sided']:.3f}}}"]
            e = h["errors"]["fp_types"]
            tot = max(1, sum(e.values()))
            macros += [rf"\renewcommand{{\FPbg}}{{{100 * e.get('background', 0) / tot:.0f}\%}}",
                       rf"\renewcommand{{\FPcls}}{{{100 * e.get('classification', 0) / tot:.0f}\%}}"]
    if (m / "benchmark.csv").exists():
        b = pd.read_csv(m / "benchmark.csv")
        def _name(x):
            return r"\textbf{" + _esc(x.method) + "}" if x.ours else _esc(x.method)

        rows = "\n".join(rf"{_name(x)} & {_esc(x.protocol)} & {_f(x.f1)} & "
                         rf"{_f(x.mAP50 if x.mAP50 == x.mAP50 else None)} \\" for x in b.itertuples())
        write("benchmark.tex", rf"""\begin{{table*}}[t]\centering\caption{{Comparison with published road-damage detectors. Rows on ``RDD2022-China 8:1:1'' share our protocol (run \texttt{{configs/china\_811.yaml}}); challenge rows use hidden test sets and are indicative.}}\label{{tab:bench}}
\begin{{tabular}}{{llcc}}\toprule Method & Protocol & F1 & mAP50 \\ \midrule
{rows}
\bottomrule\end{{tabular}}\end{{table*}}
""")
    write("macros.tex", "\n".join(macros) + "\n")
    log.info("LaTeX assets written to %s (%d macros)", OUT, len(macros))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    main(load_config(a.config, a.override))
