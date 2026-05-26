#!/usr/bin/env python3
"""Plot ablations2: x=ce loss weight, y=overall score only.

Reads each run directory's `summary_metrics.json` under `--root`.
Plots overall (median) vs ce_loss_weight for:
- F1-RadGraph
- GreenScore

Output is written to `--out_dir` as `ablations_ce_loss_weight_overall_only.png`.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

METRICS = ("F1-RadGraph", "GreenScore")

# Avoid word boundaries (underscores are word chars). Accept dot or underscore decimals.
_WEIGHT_RE = re.compile(r"ce_loss_weight_([0-9]+(?:[\._][0-9]+)?)")


def _extract_weight(run_name: str) -> Optional[float]:
    m = _WEIGHT_RE.search(run_name)
    if not m:
        return None
    return float(m.group(1).replace("_", "."))


def _read_summary(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _get_overall_median(summary: dict, metric: str) -> Optional[float]:
    val = (
        summary.get("metrics_by_scope", {})
        .get("overall", {})
        .get(metric, {})
        .get("median", None)
    )
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def collect(root: Path) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {m: [] for m in METRICS}
    matched_any = False

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        w = _extract_weight(run_dir.name)
        if w is None:
            continue
        matched_any = True
        summary_path = run_dir / "summary_metrics.json"
        if not summary_path.exists():
            continue
        summary = _read_summary(summary_path)
        if not isinstance(summary, dict):
            continue

        for metric in METRICS:
            ov = _get_overall_median(summary, metric)
            if ov is not None:
                out[metric].append((w, ov))

    if not matched_any:
        print(f"[WARN] No runs matched ce_loss_weight pattern under: {root}")

    return out


def plot(root: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(root)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.8))
    color_map = {
        "F1-RadGraph": "tab:orange",
        "GreenScore": "tab:green",
    }

    any_points = False
    for metric, pts in data.items():
        pts_s = sorted(pts, key=lambda t: t[0])
        if not pts_s:
            continue
        any_points = True
        xs = [p[0] for p in pts_s]
        ys = [p[1] for p in pts_s]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            color=color_map.get(metric, "C0"),
            label=metric,
        )

    ax.set_xlabel("ce loss weight")
    ax.set_ylabel("overall score")
    ax.grid(True, alpha=0.3)
    if any_points:
        ax.legend(frameon=False)

    out_path = out_dir / "ablations_ce_loss_weight_overall_only.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    print(f"Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="results/ablations2",
        help="Folder containing ablations2 run subdirectories",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results/ablations2/plots2",
        help="Output directory for plots",
    )
    args = ap.parse_args()

    plot(Path(args.root), Path(args.out_dir))


if __name__ == "__main__":
    main()
