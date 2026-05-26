#!/usr/bin/env python3
"""Plot ablations2: x=ce_loss_weight, y-left=fairness gap (or ES), y-right=overall.

Reads per-run data from each run directory's `summary_metrics.json` under `--root`.
For each demographic attribute (gender, age, race), produces one subplot with
solid ES curves and dashed overall curves for:
- F1-RadGraph
- GreenScore

Output is written to `--out_dir`.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PRIMARY_ATTRS = ("gender", "age", "race")
METRICS = ("F1-RadGraph", "GreenScore")

_WEIGHT_RE = re.compile(r"(?:^|_)ce_loss_weight_([0-9]+(?:[\._][0-9]+)?)")


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


def _get_es(summary: dict, attribute: str, metric: str) -> Optional[float]:
    for row in summary.get("equity_scaled", []):
        if str(row.get("attribute")) == attribute and str(row.get("metric")) == metric:
            try:
                return float(row.get("es"))
            except Exception:
                return None
    return None


def _get_gap(summary: dict, attribute: str, metric: str) -> Optional[float]:
    for row in summary.get("fairness_gap", []):
        if str(row.get("attribute")) == attribute and str(row.get("metric")) == metric:
            try:
                return float(row.get("gap"))
            except Exception:
                return None
    return None


def collect(root: Path, left: str) -> Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]:
    """Returns: attribute -> metric -> {'left':[(w,v)], 'overall':[(w,v)]}."""
    out: Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]] = {
        attr: {metric: {"left": [], "overall": []} for metric in METRICS} for attr in PRIMARY_ATTRS
    }

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        w = _extract_weight(run_dir.name)
        if w is None:
            continue
        summary_path = run_dir / "summary_metrics.json"
        if not summary_path.exists():
            continue
        summary = _read_summary(summary_path)
        if not isinstance(summary, dict):
            continue

        for attr in PRIMARY_ATTRS:
            for metric in METRICS:
                if left == "gap":
                    val = _get_gap(summary, attr, metric)
                else:
                    val = _get_es(summary, attr, metric)
                if val is not None:
                    out[attr][metric]["left"].append((w, val))

        for metric in METRICS:
            ov = _get_overall_median(summary, metric)
            if ov is not None:
                for attr in PRIMARY_ATTRS:
                    out[attr][metric]["overall"].append((w, ov))

    return out


def plot(root: Path, out_dir: Path, left: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(root, left=left)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    color_map = {
        "F1-RadGraph": "tab:orange",
        "GreenScore": "tab:green",
    }

    for i, attr in enumerate(PRIMARY_ATTRS):
        ax = axes[i]
        ax2 = ax.twinx()

        for metric in METRICS:
            color = color_map.get(metric, f"C{i}")

            pts_left = sorted(data[attr][metric]["left"], key=lambda t: t[0])
            if pts_left:
                xs = [p[0] for p in pts_left]
                ys = [p[1] for p in pts_left]
                left_prefix = "gap" if left == "gap" else "ES"
                ax.plot(xs, ys, marker="o", linewidth=2, color=color, linestyle="-", label=f"{left_prefix}-{metric}")

            pts_ov = sorted(data[attr][metric]["overall"], key=lambda t: t[0])
            if pts_ov:
                xs2 = [p[0] for p in pts_ov]
                ys2 = [p[1] for p in pts_ov]
                ax2.plot(
                    xs2,
                    ys2,
                    marker="s",
                    linewidth=2,
                    color=color,
                    linestyle="--",
                    label=f"overall-{metric}",
                )

        ax.set_title(attr)
        ax.set_xlabel("weight")
        ax.set_ylabel("fairness gap" if left == "gap" else "ES")
        ax2.set_ylabel("overall")
        ax.grid(True, alpha=0.3)

    legend_lines: List[Line2D] = []
    legend_labels: List[str] = []
    for metric in METRICS:
        color = color_map.get(metric, "C0")
        legend_lines.append(Line2D([0], [0], color=color, marker="o", linestyle="-", linewidth=2))
        legend_labels.append(("gap" if left == "gap" else "ES") + f"-{metric}")
    for metric in METRICS:
        color = color_map.get(metric, "C0")
        legend_lines.append(Line2D([0], [0], color=color, marker="s", linestyle="--", linewidth=2))
        legend_labels.append(f"overall-{metric}")

    fig.legend(
        legend_lines,
        legend_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.005),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])

    out_name = "ablations_ce_loss_weight_gap_simple_3x2.png" if left == "gap" else "ablations_ce_loss_weight_es_simple_3x2.png"
    out_path = out_dir / out_name
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
    ap.add_argument(
        "--left",
        type=str,
        default="gap",
        choices=["gap", "es"],
        help="What to plot on the left axis: fairness gap ('gap') or equity scaled score ('es').",
    )
    args = ap.parse_args()

    plot(Path(args.root), Path(args.out_dir), left=args.left)


if __name__ == "__main__":
    main()
