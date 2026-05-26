#!/usr/bin/env python3
"""Batch inference + eval for MI-lambda ablation checkpoints.

This script is designed to be *non-magical* and robust:
- Runs `scripts/infer_eval.py` (optional) per checkpoint to generate predictions.
- Ensures demographic stratification files exist via `scripts/stratify_results.py`.
- Computes metrics + bootstrap CIs for:
    - overall: merged_preds.jsonl
    - per-group: race_*.jsonl, age_*.jsonl, gender_*.jsonl
  writing each group's metrics into a dedicated folder so nothing gets overwritten.
- Computes:
    - fairness gap: max(group) - min(group)
    - equity-scaled metric: ES-M_a = M_all / (1 + gap)
  with *approximate* 95% CIs using Monte Carlo propagation from per-group CIs.

Outputs (per prediction_dir):
- metrics/<scope>/main.csv  (raw medians + CI for that scope)
- summary_metrics.json      (all parsed metrics)
- summary_fairness_gap.csv  (gap + CI per attribute per metric)
- summary_es_metrics.csv    (ES + CI per attribute per metric)

Also writes a combined CSV across runs if `--combined-out` is set.

Assumptions (consistent with existing repo scripts):
- `scripts/infer_eval.py` writes to `prediction_dir` and creates `merged_preds.jsonl`.
- `scripts/stratify_results.py` creates subgroup jsonl files in the same folder.
- Group files are JSONL with keys: {prediction, reference, ...}.

This does NOT rely on W&B for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Metrics we report in the tables (matches your screenshots)
DEFAULT_REPORT_METRICS = ["BLEU-1", "BLEU-4", "F1-RadGraph", "GreenScore"]


def _jsonl_has_key(path: Path, key: str) -> bool:
    """Return True if the first non-empty JSONL record contains `key`."""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    return False
                return key in rec
    except FileNotFoundError:
        return False
    return False


def _read_jsonl_floats(path: Path, key: str) -> List[float]:
    vals: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSONL in {path} at line {line_no}: {e}") from e
            if key not in rec:
                raise KeyError(f"Missing '{key}' in {path} at line {line_no}")
            try:
                vals.append(float(rec[key]))
            except Exception as e:
                raise ValueError(f"Non-numeric '{key}' in {path} at line {line_no}: {rec[key]!r}") from e
    return vals


def _bootstrap_mean_ci(vals: List[float], *, n_resamples: int = 500, seed: int = 3) -> Tuple[float, float, float]:
    """Return (median_of_bootstrap_means, ci_l, ci_h) using percentile bootstrap.

    Matches the intent of GREEN's aggregator path (SciPy bootstrap on mean), but avoids
    instantiating the heavy GREEN model.
    """
    import numpy as np

    if len(vals) == 0:
        nan = float("nan")
        return nan, nan, nan
    if len(vals) < 2:
        m = float(np.mean(vals))
        return m, m, m

    x = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    # Sample indices with replacement: [n_resamples, n]
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = x[idx].mean(axis=1)
    med = float(np.median(means))
    ci_l = float(np.percentile(means, 2.5))
    ci_h = float(np.percentile(means, 97.5))
    return med, ci_l, ci_h


def _ensure_metric_in_main_csv(
    main_csv: Path,
    *,
    metric_name: str,
    median: float,
    ci_l: Optional[float],
    ci_h: Optional[float],
    extra_row_name: Optional[str] = None,
    extra_row_value: Optional[str] = None,
) -> None:
    """Insert/overwrite a metric column in a run.py-style main.csv."""

    rows: List[List[str]] = []
    if main_csv.exists():
        with main_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

    if not rows:
        rows = [["", metric_name], ["median", ""], ["ci_l", ""], ["ci_h", ""]]

    header = rows[0]
    if metric_name not in header:
        header.append(metric_name)
        for r in rows[1:]:
            r.append("")

    col = header.index(metric_name)

    idx = { (r[0] or "").strip(): r for r in rows[1:] if r }
    for key in ("median", "ci_l", "ci_h"):
        if key not in idx:
            new_r = [key] + [""] * (len(header) - 1)
            rows.append(new_r)
            idx[key] = new_r

    idx["median"][col] = str(median)
    idx["ci_l"][col] = "" if ci_l is None else str(ci_l)
    idx["ci_h"][col] = "" if ci_h is None else str(ci_h)

    if extra_row_name is not None:
        if extra_row_name not in idx:
            new_r = [extra_row_name] + [""] * (len(header) - 1)
            rows.append(new_r)
            idx[extra_row_name] = new_r
        idx[extra_row_name][col] = "" if extra_row_value is None else extra_row_value

    with main_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _metric_needs_x100_for_gap_es(metric: str) -> bool:
    """Return True if metric should be scaled by 100 for gap/ES computations.

    We keep *reported* metrics unchanged, but compute fairness gaps and ES
    on a percent-like scale for certain metrics that are naturally in [0, 1].
    """
    m = (metric or "").strip().lower()
    m = m.replace(" ", "").replace("_", "").replace("–", "-").replace("—", "-")

    # GreenScore
    if m == "greenscore" or "greenscore" in m:
        return True

    # RadGraph F1
    if "radgraph" in m and "f1" in m:
        return True

    # ROUGE-L (some pipelines name it rougeL / ROUGE-Lsum)
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


def _slugify(s: str) -> str:
    """Make a filesystem-friendly tag."""
    s = (s or "").strip()
    if not s:
        return "untagged"
    out = []
    last_underscore = False
    for ch in s:
        ok = ch.isalnum() or ch in {"-", "_"}
        if ok:
            out.append(ch)
            last_underscore = False
        else:
            if not last_underscore:
                out.append("_")
                last_underscore = True
    tag = "".join(out).strip("_")
    return tag or "untagged"

def _discover_group_scopes(pred_dir: Path) -> Dict[str, List[str]]:
    """Discover subgroup scopes produced by scripts/stratify_results.py.

    Expected filename pattern: <attr>_<group>.jsonl where attr in {age, gender, race}.
    Examples:
      - age_0_44.jsonl
      - gender_m.jsonl
      - race_white.jsonl

    Returns mapping: attr -> list of scope names (stem without .jsonl).
    """
    out: Dict[str, List[str]] = {"age": [], "gender": [], "race": []}
    for p in pred_dir.glob("*.jsonl"):
        stem = p.stem
        if stem in {"merged_preds", "merged_demographics"}:
            continue
        if stem.startswith("test_"):
            # inference shards
            continue
        # subgroup files are named like: age_0_44, gender_m, race_white
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
        # 95% CI ≈ mean ± 1.96*SE
        return width / (2.0 * 1.96)


def _run(cmd: List[str], *, cwd: Path, env: Optional[Dict[str, str]] = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=sys.stdout, stderr=sys.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def _read_main_csv(path: Path) -> Dict[str, MetricCI]:
    """Parse run.py's main.csv (median/ci_l/ci_h rows) into dict[metric]->MetricCI."""
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or len(rows) < 2:
        raise ValueError(f"Unexpected CSV content (too short): {path}")

    header = rows[0]
    # First column is the row index (blank header)
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
                # Some cells (like the greenscore list row) are non-numeric.
                continue
        return out

    # Find median/ci rows
    idx_to_row = { (r[0] or "").strip(): r for r in rows[1:] if r }
    if "median" not in idx_to_row:
        raise ValueError(f"Expected 'median' row in {path}")

    median_map = row_to_map(idx_to_row["median"])
    ci_l_map = row_to_map(idx_to_row.get("ci_l", [""] * len(header)))
    ci_h_map = row_to_map(idx_to_row.get("ci_h", [""] * len(header)))

    out_ci: Dict[str, MetricCI] = {}
    for metric, median in median_map.items():
        out_ci[metric] = MetricCI(
            median=median,
            ci_l=ci_l_map.get(metric),
            ci_h=ci_h_map.get(metric),
        )
    return out_ci


def _ensure_stratified(pred_dir: Path, *, dataset: str, repo_root: Path) -> None:
    required = [pred_dir / "merged_preds.jsonl"]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required predictions file: {p}")

    # If subgroup files exist, stratification likely ran before.
    # However, in this repo the per-sample `greenscore` can be injected later into
    # merged_preds.jsonl by the evaluator. If subgroup files were generated *before*
    # that injection, they will be missing `greenscore` and downstream eval will
    # unnecessarily recompute GREEN per subgroup. Detect and repair that.
    existing = _discover_group_scopes(pred_dir)
    if any(existing.values()):
        merged_has_green = _jsonl_has_key(pred_dir / "merged_preds.jsonl", "greenscore")
        if not merged_has_green:
            return

        # If any subgroup file is missing greenscore, rerun stratification.
        needs_regen = False
        for stems in existing.values():
            for stem in stems:
                p = pred_dir / f"{stem}.jsonl"
                if p.exists() and not _jsonl_has_key(p, "greenscore"):
                    needs_regen = True
                    break
            if needs_regen:
                break
        if not needs_regen:
            return

        print("[stratify] subgroup files missing greenscore; regenerating stratified JSONLs")

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "stratify_results.py"),
            "--pred_dir",
            str(pred_dir),
            "--dataset",
            dataset,
        ],
        cwd=repo_root,
    )


def _ensure_merged_greenscore(
    pred_dir: Path,
    *,
    repo_root: Path,
    bootstrap_ci: bool,
) -> None:
    """Ensure merged_preds.jsonl contains per-sample `greenscore`.

    GREEN is very expensive. If merged_preds.jsonl doesn't already have cached
    per-sample scores, compute them *once* on the merged file so stratified
    subgroup files inherit the cached field and subgroup evaluation can reuse it.
    """
    merged = pred_dir / "merged_preds.jsonl"
    if not merged.exists():
        raise FileNotFoundError(f"Missing required predictions file: {merged}")

    if _jsonl_has_key(merged, "greenscore"):
        return

    # Import here so this script stays importable in environments without eval deps.
    from llava.eval.rrg_eval import run as rrg_run

    cache_dir = pred_dir / "metrics" / "_greenscore_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("[greenscore] merged_preds.jsonl missing greenscore; computing once to cache per-sample scores")

    # This will (a) compute GREEN and (b) inject `greenscore` into merged_preds.jsonl.
    # We intentionally write outputs to a dedicated cache folder so reuse_existing_metrics
    # for scope='overall' doesn't accidentally read an incomplete main.csv.
    rrg_run.main(
        filepath=str(merged),
        scorers=["GreenScore"],
        bootstrap_ci=bootstrap_ci,
        output_dir=str(cache_dir),
        run_name=f"{pred_dir.name}-greenscore-cache",
    )


def _compute_group_metrics(
    pred_dir: Path,
    *,
    repo_root: Path,
    scopes: Dict[str, Path],
    scorers: List[str],
    bootstrap_ci: bool,
    reuse_existing: bool,
) -> Dict[str, Dict[str, MetricCI]]:
    """Compute metrics for each scope by calling run.py's evaluator directly."""
    # Import here to keep script importable even if deps aren't installed in some contexts.
    from llava.eval.rrg_eval import run as rrg_run

    results: Dict[str, Dict[str, MetricCI]] = {}

    for scope, jsonl_path in scopes.items():
        if not jsonl_path.exists():
            continue

        out_dir = pred_dir / "metrics" / scope
        out_dir.mkdir(parents=True, exist_ok=True)

        main_csv = out_dir / "main.csv"
        if reuse_existing and main_csv.exists():
            print(f"[reuse] metrics already exist: {main_csv}")
            results[scope] = _read_main_csv(main_csv)
            continue

        # If per-sample greenscore already exists, avoid recomputing GREEN for every subgroup.
        wants_green = any((s or "").strip() == "GreenScore" for s in scorers)
        has_green = wants_green and _jsonl_has_key(jsonl_path, "greenscore")
        scorers_to_run = [s for s in scorers if s != "GreenScore"] if has_green else list(scorers)

        # Call run.py's main() directly (avoids Fire parsing issues)
        if scorers_to_run:
            print(f"[run] scoring {scope}: {jsonl_path} -> {out_dir}")
            rrg_run.main(
                filepath=str(jsonl_path),
                scorers=scorers_to_run,
                bootstrap_ci=bootstrap_ci,
                output_dir=str(out_dir),
                run_name=f"{pred_dir.name}-{scope}",
            )
        else:
            # Ensure output exists even if we skip the evaluator entirely.
            out_dir.mkdir(parents=True, exist_ok=True)

        if has_green:
            gs = _read_jsonl_floats(jsonl_path, "greenscore")
            med, ci_l, ci_h = _bootstrap_mean_ci(gs) if bootstrap_ci else (float(sum(gs) / max(1, len(gs))), None, None)
            # Preserve run.py's extra row naming convention.
            _ensure_metric_in_main_csv(
                main_csv,
                metric_name="GreenScore",
                median=med,
                ci_l=ci_l,
                ci_h=ci_h,
                extra_row_name="greenscore",
                extra_row_value=json.dumps(gs),
            )

        results[scope] = _read_main_csv(main_csv)

    return results


def _mc_gap_and_es(
    group_vals: Dict[str, MetricCI],
    *,
    rng_seed: int,
    n_samples: int,
) -> Tuple[float, Optional[Tuple[float, float]], float, Optional[Tuple[float, float]]]:
    """Return (gap, gap_ci, es, es_ci) using MC propagation.

    Uses Normal(mean=median, se=approx_se) per group when CI is available.
    If CIs are missing, returns CI=None.
    """
    import random

    medians = [v.median for v in group_vals.values()]
    gap = max(medians) - min(medians)
    m_all = sum(medians) / max(1, len(medians))
    es = m_all / (1.0 + gap)

    ses = [v.approx_se() for v in group_vals.values()]
    if any(se is None for se in ses):
        return gap, None, es, None

    r = random.Random(rng_seed)

    # Sample per group
    draws: Dict[str, List[float]] = {}
    for name, v in group_vals.items():
        se = v.approx_se()
        assert se is not None
        # Box-Muller via random.gauss
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

    gap_ci = (pct(gap_draws, 2.5), pct(gap_draws, 97.5))
    es_ci = (pct(es_draws, 2.5), pct(es_draws, 97.5))
    return gap, gap_ci, es, es_ci


def _build_scopes(pred_dir: Path) -> Dict[str, Path]:
    scopes: Dict[str, Path] = {"overall": pred_dir / "merged_preds.jsonl"}
    groups = _discover_group_scopes(pred_dir)
    for stems in groups.values():
        for stem in stems:
            scopes[stem] = pred_dir / f"{stem}.jsonl"
    return scopes


def _summarize(
    pred_dir: Path,
    *,
    dataset: str,
    query_tag: str,
    repo_root: Path,
    report_metrics: List[str],
    scorers: List[str],
    bootstrap_ci: bool,
    mc_samples: int,
    mc_seed: int,
    reuse_existing_metrics: bool,
) -> Tuple[Path, Dict[str, object]]:
    """Compute metrics, fairness gaps, ES metrics, and write summary files."""
    # Ensure cached GREEN exists *before* stratification so subgroup JSONLs include greenscore.
    if any((s or "").strip() == "GreenScore" for s in scorers):
        _ensure_merged_greenscore(pred_dir, repo_root=repo_root, bootstrap_ci=bootstrap_ci)

    _ensure_stratified(pred_dir, dataset=dataset, repo_root=repo_root)

    scopes = _build_scopes(pred_dir)
    metrics_by_scope = _compute_group_metrics(
        pred_dir,
        repo_root=repo_root,
        scopes=scopes,
        scorers=scorers,
        bootstrap_ci=bootstrap_ci,
        reuse_existing=reuse_existing_metrics,
    )

    # Extract per-attribute groups (based on discovered jsonl stems)
    discovered = _discover_group_scopes(pred_dir)
    per_attr: Dict[str, Dict[str, Dict[str, MetricCI]]] = {}
    for attr, stems in discovered.items():
        g: Dict[str, Dict[str, MetricCI]] = {}
        for stem in stems:
            if stem in metrics_by_scope:
                g[stem] = metrics_by_scope[stem]
        per_attr[attr] = g

    # Compute fairness gaps and ES
    gap_rows: List[Dict[str, object]] = []
    es_rows: List[Dict[str, object]] = []

    for attr, groups in per_attr.items():
        # For each metric, collect MetricCI per group
        for metric in report_metrics:
            group_metric_ci: Dict[str, MetricCI] = {}
            for group_name, group_metrics in groups.items():
                if metric in group_metrics:
                    group_metric_ci[group_name] = _scale_ci_for_gap_es(metric, group_metrics[metric])

            if len(group_metric_ci) < 2:
                continue

            gap, gap_ci, es, es_ci = _mc_gap_and_es(
                group_metric_ci,
                rng_seed=mc_seed + hash((pred_dir.name, attr, metric)) % 100000,
                n_samples=mc_samples,
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

    # Write summary artifacts
    summary_obj: Dict[str, object] = {
        "pred_dir": str(pred_dir),
        "dataset": dataset,
        "query_tag": query_tag,
        "report_metrics": report_metrics,
        "scorers": scorers,
        "bootstrap_ci": bootstrap_ci,
        "metrics_by_scope": {
            scope: {m: vars(ci) for m, ci in metrics.items()}
            for scope, metrics in metrics_by_scope.items()
        },
        "fairness_gap": gap_rows,
        "equity_scaled": es_rows,
    }

    (pred_dir / "summary_metrics.json").write_text(
        json.dumps(summary_obj, indent=2) + "\n",
        encoding="utf-8",
    )

    gap_csv = pred_dir / "summary_fairness_gap.csv"
    with gap_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gap_rows[0].keys()) if gap_rows else ["run", "attribute", "metric", "gap", "gap_ci_l", "gap_ci_h"])
        w.writeheader()
        for r in gap_rows:
            w.writerow(r)

    es_csv = pred_dir / "summary_es_metrics.csv"
    with es_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(es_rows[0].keys()) if es_rows else ["run", "attribute", "metric", "es", "es_ci_l", "es_ci_h"])
        w.writeheader()
        for r in es_rows:
            w.writerow(r)

    return pred_dir, summary_obj


def _iter_runs_from_summary(summary_json: Path) -> Iterable[Tuple[str, Path]]:
    """Yield (run_tag, checkpoint_path) from driver SUMMARY.json."""
    d = json.loads(summary_json.read_text(encoding="utf-8"))
    runs = d.get("runs") or []
    for r in runs:
        run_tag = None
        env = r.get("env_overrides") or {}
        run_tag = env.get("RUN_TAG")
        ckpt = r.get("checkpoint_dir_guess")
        if run_tag and ckpt:
            yield run_tag, Path(ckpt)


def _filter_and_slice_runs(
    runs: List[Tuple[str, Path]],
    *,
    include_substrings: List[str],
    exclude_substrings: List[str],
    run_tag_regex: Optional[str],
    start: int,
    max_runs: Optional[int],
) -> List[Tuple[str, Path]]:
    include_substrings = [s for s in (include_substrings or []) if s]
    exclude_substrings = [s for s in (exclude_substrings or []) if s]

    rx = re.compile(run_tag_regex) if run_tag_regex else None

    filtered: List[Tuple[str, Path]] = []
    for run_tag, ckpt_dir in runs:
        if include_substrings and not any(s in run_tag for s in include_substrings):
            continue
        if exclude_substrings and any(s in run_tag for s in exclude_substrings):
            continue
        if rx and not rx.search(run_tag):
            continue
        filtered.append((run_tag, ckpt_dir))

    if start and start > 0:
        filtered = filtered[start:]
    if max_runs is not None:
        if max_runs < 0:
            raise ValueError("--max-runs must be >= 0")
        filtered = filtered[:max_runs]
    return filtered


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-summary", default=None, help="Path to mi_lambda_ablations/<ts>/SUMMARY.json from run_mi_lambda_ablations.py")
    ap.add_argument("--checkpoint-root", default=None, help="If set, scan this dir for checkpoint folders matching '*_lmda_*'.")
    ap.add_argument(
        "--checkpoint-root-all",
        action="store_true",
        help="When scanning --checkpoint-root, include all subdirectories (default: only names containing '_lmda_' or '-lmda-').",
    )
    ap.add_argument("--prefix-base", default="ablations", help="Prefix base used for results folders (default: ablations)")

    ap.add_argument(
        "--include-run-tag",
        action="append",
        default=[],
        help="Only evaluate runs whose run_tag contains this substring (repeatable).",
    )
    ap.add_argument(
        "--exclude-run-tag",
        action="append",
        default=[],
        help="Skip runs whose run_tag contains this substring (repeatable).",
    )
    ap.add_argument(
        "--run-tag-regex",
        default=None,
        help="Only evaluate runs whose run_tag matches this regex (applied after include/exclude filters).",
    )
    ap.add_argument("--start", type=int, default=0, help="Skip the first N runs after filtering (default: 0)")
    ap.add_argument("--max-runs", type=int, default=None, help="Evaluate at most N runs after filtering")
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next run if inference/summarization fails (e.g., CUDA OOM).",
    )

    ap.add_argument("--model_base", default="lmsys/vicuna-7b-v1.5")
    ap.add_argument("--query_file", required=True)
    ap.add_argument(
        "--query_tag",
        default=None,
        help=(
            "Tag appended to output folder names so the same checkpoint can be run on multiple test sets. "
            "Default: derived from --query_file basename (stem)."
        ),
    )
    ap.add_argument("--loader", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch_size", type=int, default=8)

    ap.add_argument("--results_root", default=str(repo_root / "results" / "ablations"), help="Root directory for prediction outputs")

    ap.add_argument("--skip_infer", action="store_true", help="Skip running scripts/infer_eval.py (assume predictions already exist)")
    ap.add_argument(
        "--force_infer",
        action="store_true",
        help="Force re-running inference even if merged_preds.jsonl already exists.",
    )
    ap.add_argument(
        "--infer_post_eval",
        action="store_true",
        help="Let infer_eval.py run its own post-eval (default: skip post-eval to avoid duplicate work).",
    )
    ap.add_argument("--dry_run", action="store_true", help="Print planned commands and exit")

    ap.add_argument("--report_metrics", default=",".join(DEFAULT_REPORT_METRICS), help="Comma-separated metrics to report")
    ap.add_argument("--bootstrap_ci", action="store_true", default=True)
    ap.add_argument("--mc_samples", type=int, default=20000)
    ap.add_argument("--mc_seed", type=int, default=0)

    ap.add_argument(
        "--reuse_existing_metrics",
        action="store_true",
        default=True,
        help="Reuse existing metrics/<scope>/main.csv when present (default: True).",
    )
    ap.add_argument(
        "--force_summaries",
        action="store_true",
        help="Force recomputing and overwriting summary_*.{json,csv} files even if they exist.",
    )

    ap.add_argument("--combined_out", default=None, help="Optional combined CSV path for all runs (wide format)")

    args = ap.parse_args()

    report_metrics = [m.strip() for m in args.report_metrics.split(",") if m.strip()]

    # scorers passed to run.py evaluator: use only what we need
    scorers = list(report_metrics)

    query_tag = _slugify(args.query_tag or Path(args.query_file).stem)

    runs: List[Tuple[str, Path]] = []
    if args.runs_summary:
        runs = list(_iter_runs_from_summary(Path(args.runs_summary)))
    elif args.checkpoint_root:
        root = Path(args.checkpoint_root)
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            name = p.name
            if args.checkpoint_root_all:
                runs.append((name, p))
            else:
                if "_lmda_" in name or "-lmda-" in name:
                    runs.append((name, p))
    else:
        raise SystemExit("Provide either --runs-summary or --checkpoint-root")

    runs = _filter_and_slice_runs(
        runs,
        include_substrings=args.include_run_tag,
        exclude_substrings=args.exclude_run_tag,
        run_tag_regex=args.run_tag_regex,
        start=args.start,
        max_runs=args.max_runs,
    )

    if not runs:
        raise SystemExit("No runs found")

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    combined_rows: List[Dict[str, object]] = []

    for run_tag, ckpt_dir in runs:
        # Output naming includes query_tag so you can evaluate multiple test files per checkpoint.
        # Example: ablations_age_lmda_0.1__chat_test_MIMIC_CXR_all_dem_clean_2K
        safe_run_tag = _slugify(run_tag)
        prefix = f"{args.prefix_base}_{safe_run_tag}__{query_tag}"
        pred_dir = results_root / prefix

        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "infer_eval.py"),
            "--model_base",
            args.model_base,
            "--prefix",
            prefix,
            "--query_file",
            args.query_file,
            "--model_path",
            str(ckpt_dir),
            "--fairness_finetune",
            "False",
            "--batch_size",
            str(args.batch_size),
            "--prediction_dir",
            str(pred_dir),
            "--loader",
            args.loader,
            "--image_folder",
            args.image_folder,
            "--dataset",
            args.dataset,
        ]

        # This batch script computes metrics/gap/ES itself; avoid doing it twice.
        if not args.infer_post_eval:
            cmd.append("--skip_post_eval")
            cmd.append("--skip_gap_es")

        if args.dry_run:
            print("Planned infer_eval:")
            print(" ".join(cmd))
            print(f"Planned summarize: {pred_dir}")
            continue

        merged_preds = pred_dir / "merged_preds.jsonl"
        need_infer = not merged_preds.exists() or args.force_infer
        if args.skip_infer:
            need_infer = False
        if need_infer:
            if merged_preds.exists() and args.force_infer:
                print(f"[run] forcing inference even though {merged_preds} exists")
            else:
                print(f"[run] inference -> {pred_dir}")
            try:
                _run(cmd, cwd=repo_root)
            except Exception as e:
                if args.continue_on_error:
                    print(f"[error] inference failed for {run_tag}: {e}")
                    combined_rows.append(
                        {
                            "run": prefix,
                            "checkpoint": str(ckpt_dir),
                            "query_file": args.query_file,
                            "query_tag": query_tag,
                            "dataset": args.dataset,
                            "error": f"inference_failed: {e}",
                        }
                    )
                    continue
                raise
        elif not merged_preds.exists():
            raise FileNotFoundError(
                f"Missing {merged_preds}; cannot skip inference. Remove --skip_infer or pass --force_infer."
            )
        else:
            print(f"[skip] inference (found {merged_preds})")

        summary_json = pred_dir / "summary_metrics.json"
        gap_csv = pred_dir / "summary_fairness_gap.csv"
        es_csv = pred_dir / "summary_es_metrics.csv"
        try:
            if (
                not args.force_summaries
                and summary_json.exists()
                and gap_csv.exists()
                and es_csv.exists()
            ):
                print(f"[skip] summaries already exist: {summary_json}")
                summary_obj = json.loads(summary_json.read_text(encoding="utf-8"))
            else:
                if args.force_summaries and summary_json.exists():
                    print(f"[run] forcing summary recompute: {pred_dir}")
                else:
                    print(f"[run] summarize metrics/gap/es: {pred_dir}")
                _, summary_obj = _summarize(
                    pred_dir,
                    dataset=args.dataset,
                    query_tag=query_tag,
                    repo_root=repo_root,
                    report_metrics=report_metrics,
                    scorers=scorers,
                    bootstrap_ci=args.bootstrap_ci,
                    mc_samples=args.mc_samples,
                    mc_seed=args.mc_seed,
                    reuse_existing_metrics=args.reuse_existing_metrics,
                )
        except Exception as e:
            if args.continue_on_error:
                print(f"[error] summarize failed for {run_tag}: {e}")
                combined_rows.append(
                    {
                        "run": prefix,
                        "checkpoint": str(ckpt_dir),
                        "query_file": args.query_file,
                        "query_tag": query_tag,
                        "dataset": args.dataset,
                        "error": f"summarize_failed: {e}",
                    }
                )
                continue
            raise

        # Build a wide row similar to your table screenshots
        row: Dict[str, object] = {
            "run": prefix,
            "checkpoint": str(ckpt_dir),
            "query_file": args.query_file,
            "query_tag": query_tag,
            "dataset": args.dataset,
        }

        # Overall metrics
        overall = (summary_obj.get("metrics_by_scope") or {}).get("overall") or {}
        for metric in report_metrics:
            o = overall.get(metric) or {}
            if o:
                row[f"Overall.{metric}"] = o.get("median")
                row[f"Overall.{metric}.ci_l"] = o.get("ci_l")
                row[f"Overall.{metric}.ci_h"] = o.get("ci_h")

        # Gaps + ES
        for r in summary_obj.get("fairness_gap", []):
            attr = r["attribute"]
            metric = r["metric"]
            row[f"{attr}.gap.{metric}"] = r["gap"]
            row[f"{attr}.gap.{metric}.ci_l"] = r.get("gap_ci_l", "")
            row[f"{attr}.gap.{metric}.ci_h"] = r.get("gap_ci_h", "")

        for r in summary_obj.get("equity_scaled", []):
            attr = r["attribute"]
            metric = r["metric"]
            row[f"{attr}.es.{metric}"] = r["es"]
            row[f"{attr}.es.{metric}.ci_l"] = r.get("es_ci_l", "")
            row[f"{attr}.es.{metric}.ci_h"] = r.get("es_ci_h", "")

        combined_rows.append(row)

    if args.dry_run:
        return 0

    if args.combined_out and combined_rows:
        combined_out = Path(args.combined_out)
        combined_out.parent.mkdir(parents=True, exist_ok=True)
        # Union of keys
        keys: List[str] = []
        seen = set()
        for r in combined_rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)

        with combined_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in combined_rows:
                w.writerow(r)

        print(f"Wrote combined summary: {combined_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
