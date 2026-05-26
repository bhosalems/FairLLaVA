#!/usr/bin/env python3
"""Plot equity-scaled (ES) metrics from `summary_es_metrics.csv` files.

Input CSV format (produced by `scripts/summarize_gap_es.py`):
  run,attribute,metric,es,es_ci_l,es_ci_h

This script creates a 1x3 figure for attributes (race, age, gender) and, within
each subplot, plots grouped bars for ES-{BLEU-1,BLEU-4,F1-RadGraph,GreenScore}
for each method.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ATTR_ORDER = ["race", "age", "gender"]
ATTR_TITLE = {"race": "Race", "age": "Age Group", "gender": "Gender"}
METRIC_ORDER = ["BLEU-1", "BLEU-4", "F1-RadGraph", "GreenScore"]


@dataclass(frozen=True)
class EsVal:
    es: float
    ci_l: Optional[float]
    ci_h: Optional[float]


def _read_es_csv(path: Path) -> Dict[Tuple[str, str], EsVal]:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    out: Dict[Tuple[str, str], EsVal] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            attr = (row.get("attribute") or "").strip()
            metric = (row.get("metric") or "").strip()
            es_s = (row.get("es") or "").strip()
            if not attr or not metric or not es_s:
                continue
            try:
                es = float(es_s)
            except Exception:
                continue

            ci_l_s = (row.get("es_ci_l") or "").strip()
            ci_h_s = (row.get("es_ci_h") or "").strip()
            ci_l = None
            ci_h = None
            try:
                if ci_l_s:
                    ci_l = float(ci_l_s)
                if ci_h_s:
                    ci_h = float(ci_h_s)
            except Exception:
                ci_l = None
                ci_h = None

            out[(attr, metric)] = EsVal(es=es, ci_l=ci_l, ci_h=ci_h)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--fair_csv", required=True)
    ap.add_argument("--baseline_name", default="LLaVA-Rad")
    ap.add_argument("--fair_name", default="FairLLaVA")
    ap.add_argument("--out", default="results/es_from_summaries.png")
    ap.add_argument("--title", default="Equity-scaled (ES) comparison")
    ap.add_argument("--dpi", type=int, default=200)

    args = ap.parse_args()

    baseline = _read_es_csv(Path(args.baseline_csv))
    fair = _read_es_csv(Path(args.fair_csv))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), sharey=False)

    colors = {"baseline": "#4C78A8", "fair": "#F58518"}
    width = 0.36

    for ax, attr in zip(axes, ATTR_ORDER):
        xs = list(range(len(METRIC_ORDER)))

        base_vals: List[Optional[EsVal]] = [baseline.get((attr, m)) for m in METRIC_ORDER]
        fair_vals: List[Optional[EsVal]] = [fair.get((attr, m)) for m in METRIC_ORDER]

        base_y = [v.es if v is not None else float("nan") for v in base_vals]
        fair_y = [v.es if v is not None else float("nan") for v in fair_vals]

        base_err = None
        fair_err = None
        if any(v is not None and v.ci_l is not None and v.ci_h is not None for v in base_vals):
            base_err = [
                [abs(v.es - v.ci_l) if (v and v.ci_l is not None) else 0.0 for v in base_vals],
                [abs(v.ci_h - v.es) if (v and v.ci_h is not None) else 0.0 for v in base_vals],
            ]
        if any(v is not None and v.ci_l is not None and v.ci_h is not None for v in fair_vals):
            fair_err = [
                [abs(v.es - v.ci_l) if (v and v.ci_l is not None) else 0.0 for v in fair_vals],
                [abs(v.ci_h - v.es) if (v and v.ci_h is not None) else 0.0 for v in fair_vals],
            ]

        ax.bar([x - width / 2 for x in xs], base_y, width=width, color=colors["baseline"], label=args.baseline_name)
        ax.bar([x + width / 2 for x in xs], fair_y, width=width, color=colors["fair"], label=args.fair_name)

        if base_err is not None:
            ax.errorbar(
                [x - width / 2 for x in xs],
                base_y,
                yerr=base_err,
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=3,
                capthick=1,
                alpha=0.9,
            )
        if fair_err is not None:
            ax.errorbar(
                [x + width / 2 for x in xs],
                fair_y,
                yerr=fair_err,
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=3,
                capthick=1,
                alpha=0.9,
            )

        ax.set_title(ATTR_TITLE.get(attr, attr))
        ax.set_xticks(xs)
        ax.set_xticklabels([m.replace("F1-RadGraph", "F1-RG") for m in METRIC_ORDER], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("ES")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))

    fig.suptitle(args.title, y=1.12)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
