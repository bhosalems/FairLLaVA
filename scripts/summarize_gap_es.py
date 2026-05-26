#!/usr/bin/env python3
"""Summarize fairness gap + ES metrics for an existing prediction directory.

This is intended to be run after `scripts/infer_eval.py` has produced:
  - merged_preds.jsonl
  - subgroup jsonls via `scripts/stratify_results.py` (age_*, gender_*, race_*)

For CXR datasets (mimic-cxr / padchest), we compute per-scope metrics using
`llava/eval/rrg_eval/run.py` and then compute:
  - fairness gap: max(group) - min(group)
  - equity-scaled: ES = M_all / (1 + gap)

IMPORTANT: We keep reported metrics unchanged, but for *gap/ES computation*
we scale certain metrics by 100 (percent-like scale):
  - GreenScore
  - F1-RadGraph
  - ROUGE-L

Outputs (written into pred_dir):
  - metrics/<scope>/main.csv
  - summary_metrics.json
  - summary_fairness_gap.csv
  - summary_es_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_REPORT_METRICS = ["BLEU-1", "BLEU-4", "F1-RadGraph", "GreenScore", "ROUGE-L"]


def _discover_group_scopes(pred_dir: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"age": [], "gender": [], "race": []}
    for p in pred_dir.glob("*.jsonl"):
        stem = p.stem
        if stem in {"merged_preds", "merged_demographics"}:
            continue
        if stem.startswith("test_"):
            continue
        if "_" not in stem:
            continue
        attr = stem.split("_", 1)[0]
        if attr in out:
            out[attr].append(stem)

    for k in out:
        out[k] = sorted(set(out[k]))
    return out


@dataclass(frozen=True)
class MetricCI:
    median: float
    ci_l: Optional[float] = None
    ci_h: Optional[float] = None

    def approx_se(self) -> Optional[float]:
        if self.ci_l is None or self.ci_h is None:
            return None
        width = self.ci_h - self.ci_l
        if not math.isfinite(width) or width <= 0:
            return None
        return width / (2.0 * 1.96)


def _metric_needs_x100_for_gap_es(metric: str) -> bool:
    m = (metric or "").strip().lower()
    m = m.replace(" ", "").replace("_", "").replace("–", "-").replace("—", "-")

    if m == "greenscore" or "greenscore" in m:
        return True
    if "radgraph" in m and "f1" in m:
        return True
    if m in {"rougel", "rouge-l", "rougelsum", "rouge-lsum"}:
        return True
    return False


def _scale_ci_for_gap_es(metric: str, ci: MetricCI) -> MetricCI:
    if not _metric_needs_x100_for_gap_es(metric):
        return ci
    return MetricCI(
        median=ci.median * 100.0,
        ci_l=(ci.ci_l * 100.0) if ci.ci_l is not None else None,
        ci_h=(ci.ci_h * 100.0) if ci.ci_h is not None else None,
    )


def _read_main_csv(path: Path) -> Dict[str, MetricCI]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or len(rows) < 2:
        raise ValueError(f"Unexpected CSV content (too short): {path}")

    header = rows[0]
    metric_names = [h.strip() for h in header[1:]]

    def row_to_map(row: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, cell in zip(metric_names, row[1:]):
            name = name.strip()
            if not name or name == "greenscore":
                continue
            cell = (cell or "").strip()
            if cell == "":
                continue
            try:
                out[name] = float(cell)
            except Exception:
                continue
        return out

    idx_to_row = {(r[0] or "").strip(): r for r in rows[1:] if r}
    if "median" not in idx_to_row:
        raise ValueError(f"Expected 'median' row in {path}")

    median_map = row_to_map(idx_to_row["median"])
    ci_l_map = row_to_map(idx_to_row.get("ci_l", [""] * len(header)))
    ci_h_map = row_to_map(idx_to_row.get("ci_h", [""] * len(header)))

    out_ci: Dict[str, MetricCI] = {}
    for metric, median in median_map.items():
        out_ci[metric] = MetricCI(median=median, ci_l=ci_l_map.get(metric), ci_h=ci_h_map.get(metric))
    return out_ci


def _build_scopes(pred_dir: Path) -> Dict[str, Path]:
    scopes: Dict[str, Path] = {"overall": pred_dir / "merged_preds.jsonl"}
    groups = _discover_group_scopes(pred_dir)
    for stems in groups.values():
        for stem in stems:
            scopes[stem] = pred_dir / f"{stem}.jsonl"
    return scopes


def _compute_group_metrics(
    pred_dir: Path,
    *,
    scopes: Dict[str, Path],
    scorers: List[str],
    bootstrap_ci: bool,
) -> Dict[str, Dict[str, MetricCI]]:
    from llava.eval.rrg_eval import run as rrg_run

    results: Dict[str, Dict[str, MetricCI]] = {}
    for scope, jsonl_path in scopes.items():
        if not jsonl_path.exists():
            continue
        out_dir = pred_dir / "metrics" / scope
        out_dir.mkdir(parents=True, exist_ok=True)

        rrg_run.main(
            filepath=str(jsonl_path),
            scorers=scorers,
            bootstrap_ci=bootstrap_ci,
            output_dir=str(out_dir),
            run_name=f"{pred_dir.name}-{scope}",
        )

        results[scope] = _read_main_csv(out_dir / "main.csv")

    return results


def _mc_gap_and_es(
    group_vals: Dict[str, MetricCI],
    *,
    rng_seed: int,
    n_samples: int,
) -> Tuple[float, Optional[Tuple[float, float]], float, Optional[Tuple[float, float]]]:
    import random

    medians = [v.median for v in group_vals.values()]
    gap = max(medians) - min(medians)
    m_all = sum(medians) / max(1, len(medians))
    es = m_all / (1.0 + gap)

    ses = [v.approx_se() for v in group_vals.values()]
    if any(se is None for se in ses):
        return gap, None, es, None

    r = random.Random(rng_seed)

    draws: Dict[str, List[float]] = {}
    for name, v in group_vals.items():
        se = v.approx_se()
        assert se is not None
        draws[name] = [r.gauss(v.median, se) for _ in range(n_samples)]

    gap_draws: List[float] = []
    es_draws: List[float] = []
    names = list(group_vals.keys())
    for i in range(n_samples):
        vals = [draws[n][i] for n in names]
        g = max(vals) - min(vals)
        m = sum(vals) / len(vals)
        gap_draws.append(g)
        es_draws.append(m / (1.0 + g))

    gap_draws.sort()
    es_draws.sort()

    def pct(xs: List[float], p: float) -> float:
        if not xs:
            return float("nan")
        k = int(round((p / 100.0) * (len(xs) - 1)))
        k = max(0, min(len(xs) - 1, k))
        return xs[k]

    return gap, (pct(gap_draws, 2.5), pct(gap_draws, 97.5)), es, (pct(es_draws, 2.5), pct(es_draws, 97.5))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--report_metrics", default=",".join(DEFAULT_REPORT_METRICS))
    ap.add_argument("--bootstrap_ci", action="store_true", default=True)
    ap.add_argument("--mc_samples", type=int, default=20000)
    ap.add_argument("--mc_seed", type=int, default=0)

    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    if not (pred_dir / "merged_preds.jsonl").exists():
        raise SystemExit(f"Missing: {pred_dir/'merged_preds.jsonl'}")

    report_metrics = [m.strip() for m in args.report_metrics.split(",") if m.strip()]

    # Ensure stratification exists
    existing = _discover_group_scopes(pred_dir)
    if not any(existing.values()):
        cmd = [sys.executable, str(repo_root / "scripts" / "stratify_results.py"), "--pred_dir", str(pred_dir), "--dataset", args.dataset]
        rc = os.system(" ".join(cmd))
        if rc != 0:
            raise SystemExit(f"Failed stratify_results.py (exit={rc})")

    scopes = _build_scopes(pred_dir)
    metrics_by_scope = _compute_group_metrics(pred_dir, scopes=scopes, scorers=report_metrics, bootstrap_ci=args.bootstrap_ci)

    discovered = _discover_group_scopes(pred_dir)
    per_attr: Dict[str, Dict[str, Dict[str, MetricCI]]] = {}
    for attr, stems in discovered.items():
        g: Dict[str, Dict[str, MetricCI]] = {}
        for stem in stems:
            if stem in metrics_by_scope:
                g[stem] = metrics_by_scope[stem]
        per_attr[attr] = g

    gap_rows: List[Dict[str, object]] = []
    es_rows: List[Dict[str, object]] = []

    for attr, groups in per_attr.items():
        for metric in report_metrics:
            group_metric_ci: Dict[str, MetricCI] = {}
            for group_name, group_metrics in groups.items():
                if metric in group_metrics:
                    group_metric_ci[group_name] = _scale_ci_for_gap_es(metric, group_metrics[metric])

            if len(group_metric_ci) < 2:
                continue

            gap, gap_ci, es, es_ci = _mc_gap_and_es(
                group_metric_ci,
                rng_seed=args.mc_seed + hash((pred_dir.name, attr, metric)) % 100000,
                n_samples=args.mc_samples,
            )

            gap_rows.append(
                {
                    "run": pred_dir.name,
                    "attribute": attr,
                    "metric": metric,
                    "gap": gap,
                    "gap_ci_l": gap_ci[0] if gap_ci else "",
                    "gap_ci_h": gap_ci[1] if gap_ci else "",
                }
            )
            es_rows.append(
                {
                    "run": pred_dir.name,
                    "attribute": attr,
                    "metric": metric,
                    "es": es,
                    "es_ci_l": es_ci[0] if es_ci else "",
                    "es_ci_h": es_ci[1] if es_ci else "",
                }
            )

    summary_obj: Dict[str, object] = {
        "pred_dir": str(pred_dir),
        "dataset": args.dataset,
        "report_metrics": report_metrics,
        "bootstrap_ci": args.bootstrap_ci,
        "metrics_by_scope": {scope: {m: vars(ci) for m, ci in metrics.items()} for scope, metrics in metrics_by_scope.items()},
        "fairness_gap": gap_rows,
        "equity_scaled": es_rows,
    }

    (pred_dir / "summary_metrics.json").write_text(json.dumps(summary_obj, indent=2) + "\n", encoding="utf-8")

    gap_csv = pred_dir / "summary_fairness_gap.csv"
    with gap_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(gap_rows[0].keys()) if gap_rows else ["run", "attribute", "metric", "gap", "gap_ci_l", "gap_ci_h"],
        )
        w.writeheader()
        for r in gap_rows:
            w.writerow(r)

    es_csv = pred_dir / "summary_es_metrics.csv"
    with es_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(es_rows[0].keys()) if es_rows else ["run", "attribute", "metric", "es", "es_ci_l", "es_ci_h"],
        )
        w.writeheader()
        for r in es_rows:
            w.writerow(r)

    print(f"Wrote: {gap_csv}")
    print(f"Wrote: {es_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
