#!/usr/bin/env python3
"""Plot age-lambda ablations: ES vs lambda for multiple metrics.

This matches the ablations-1 layout under results/ablations where run folders are named like:
    ...-age_lmda_0_1-<timestamp>__<query_tag>/summary_metrics.json

Figure layout:
    - single plot with 4 lines, one per metric (ES-BLEU-1, ES-BLEU-4, ES-F1-RadGraph, ES-GreenScore)
    - x-axis is a fixed set of age lambda values (default: 0.1, 0.2, 0.6, 2.0, 5.0)
    - y-axis is ES for the age attribute for that metric

If some lambdas are missing in the sweep, this script linearly interpolates (and
linearly extrapolates beyond the observed range) to provide y-values on the fixed grid.

Example:
    python scripts/plot_ablations_dem_lambda.py \
        --root results/ablations \
        --out results/ablations/plots/age_lambda_es_4metrics.png
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PRIMARY_ATTRS = ("age", "gender", "race")


@dataclass(frozen=True)
class AgeRun:
    run_dir: Path
    run_name: str
    lam: float
    es_by_metric: Dict[str, float]


_LMDA_RE = re.compile(r"\b(age|gender|race)_lmda_([0-9_]+)\b")


def _parse_lam_token(tok: str) -> float:
    # encodings we see in directory names:
    # - '0_1' -> 0.1
    # - '0_6' -> 0.6
    # - '2'   -> 2.0
    # - '4'   -> 4.0
    if tok.isdigit():
        return float(tok)
    if "_" in tok:
        left, right = tok.split("_", 1)
        right = right.replace("_", "")
        if left.isdigit() and right.isdigit():
            return float(f"{left}.{right}")
    # fallback: last resort
    return float(tok.replace("_", "."))


def _extract_primary_attr_and_lam(run_name: str) -> Optional[Tuple[str, float]]:
    m = _LMDA_RE.search(run_name)
    if not m:
        return None
    attr = m.group(1)
    lam = _parse_lam_token(m.group(2))
    return attr, lam


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_age_es_by_metric(summary: dict, metrics: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in summary.get("equity_scaled", []):
        if str(row.get("attribute")) != "age":
            continue
        m = str(row.get("metric"))
        if m in metrics:
            out[m] = float(row["es"])
    missing = [m for m in metrics if m not in out]
    if missing:
        raise KeyError(f"Missing age equity_scaled entries for metrics={missing}")
    return out


def collect_age_runs(root: Path, metrics: Sequence[str]) -> List[AgeRun]:
    runs: List[AgeRun] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name

        parsed = _extract_primary_attr_and_lam(run_name)
        if not parsed:
            continue
        primary_attr, lam = parsed
        if primary_attr != "age":
            continue

        summary_path = run_dir / "summary_metrics.json"
        if not summary_path.exists():
            continue

        summary = _load_json(summary_path)
        es_by_metric = _get_age_es_by_metric(summary, metrics)
        runs.append(AgeRun(run_dir=run_dir, run_name=run_name, lam=lam, es_by_metric=es_by_metric))
    return runs


def _sorted_unique(xs: Sequence[float]) -> List[float]:
    return sorted(set(xs))


def _interp_extrap(xs: List[float], ys: List[float], x_targets: List[float]) -> List[float]:
    # Linear interpolation inside range; linear extrapolation outside.
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have same length")
    if not xs:
        raise ValueError("no points")

    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    xs_s = [p[0] for p in pairs]
    ys_s = [p[1] for p in pairs]

    if len(xs_s) == 1:
        return [ys_s[0] for _ in x_targets]

    def lerp(x0, y0, x1, y1, x):
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    out = []
    for x in x_targets:
        # exact hit
        for xx, yy in pairs:
            if abs(xx - x) < 1e-12:
                out.append(yy)
                break
        else:
            if x < xs_s[0]:
                out.append(lerp(xs_s[0], ys_s[0], xs_s[1], ys_s[1], x))
            elif x > xs_s[-1]:
                out.append(lerp(xs_s[-2], ys_s[-2], xs_s[-1], ys_s[-1], x))
            else:
                # find bracket
                for i in range(len(xs_s) - 1):
                    if xs_s[i] <= x <= xs_s[i + 1]:
                        out.append(lerp(xs_s[i], ys_s[i], xs_s[i + 1], ys_s[i + 1], x))
                        break
                else:
                    # should not happen
                    out.append(ys_s[-1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/ablations", help="Folder containing ablation run subdirectories")
    ap.add_argument("--out", type=str, required=True, help="Output path (.png/.pdf)")
    ap.add_argument(
        "--lambdas",
        type=str,
        default="0.1,0.2,0.6,2.0,4.0",
        help="Comma-separated fixed x-axis lambda values. Missing values are interpolated.",
    )
    ap.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Metric to plot (e.g., BLEU-1, BLEU-4, F1-RadGraph, GreenScore)",
    )

    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_targets = [float(x.strip()) for x in str(args.lambdas).split(",") if x.strip()]
    metric = str(args.metric)
    if not x_targets:
        raise SystemExit("--lambdas produced an empty list")
    # Collect all runs for all attributes
    runs_by_attr = {attr: [] for attr in PRIMARY_ATTRS}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        parsed = _extract_primary_attr_and_lam(run_name)
        if not parsed:
            continue
        attr, lam = parsed
        if attr not in PRIMARY_ATTRS:
            continue
        summary_path = run_dir / "summary_metrics.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        # Find ES for this metric and attribute
        es = None
        for row in summary.get("equity_scaled", []):
            if str(row.get("attribute")) == attr and str(row.get("metric")) == metric:
                es = float(row["es"])
                break
        if es is not None:
            runs_by_attr[attr].append((lam, es))

    # Defer matplotlib import until after parsing.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
    x_pos = list(range(len(x_targets)))
    for attr in PRIMARY_ATTRS:
        pts = sorted(runs_by_attr[attr], key=lambda t: t[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        y_interp = _interp_extrap(xs, ys, x_targets)
        ax.plot(x_pos, y_interp, marker="o", linewidth=2, label=attr)
    ax.set_xlabel("lambda")
    ax.set_ylabel(f"ES-{metric}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(x) for x in x_targets])
    ax.grid(True, alpha=0.3)
    ax.legend(title="Attribute")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
