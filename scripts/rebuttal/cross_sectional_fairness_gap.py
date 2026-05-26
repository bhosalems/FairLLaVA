#!/usr/bin/env python3
"""Cross-sectional fairness gap analysis on merged demographics JSONL.

Reads a JSONL where each line is one sample with demographics and a scalar metric
(e.g., `greenscore`).

Two common modes:

1) Cross-section means by (age_bin, race) and compute gaps within each age_bin.
2) Stratified gaps: compute race gaps within each (age_bin, gender) stratum,
     then aggregate (weighted or unweighted) across strata.

This is intentionally lightweight: stdlib only.

Example (stratified race gaps within age×gender bins):
    python scripts/rebuttal/cross_sectional_fairness_gap.py \
        --input results/mi_age_clean/merged_demographics.jsonl \
        --metric greenscore \
        --strata_fields age_group,gender \
        --gap_field race_major \
        --min_group_n 20 \
        --outdir results/mi_age_clean/fairness_gap__age_x_gender__gap_race

Age binning fallback: if a row's `age_group` is missing/empty, it falls back to
binning `anchor_age` using `--age_edges/--age_labels`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_AGE_EDGES = [0.0, 44.0, 65.0, 200.0]  # [-inf,44), [44,65), [65,inf)
DEFAULT_AGE_LABELS = ["<44", "44-64", "65+"]


@dataclass
class GroupStats:
    n: int
    mean: float
    std: float
    sem: float


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    try:
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None


def _normalize_str(x: Any) -> str:
    if x is None:
        return "UNKNOWN"
    s = str(x).strip()
    if not s:
        return "UNKNOWN"
    return s


def _bin_age(anchor_age: Any, edges: Sequence[float], labels: Sequence[str]) -> str:
    age = _safe_float(anchor_age)
    if age is None:
        return "UNKNOWN"
    # edges like [0,44,65,200] and labels length 3 for 3 intervals.
    for i in range(len(labels)):
        lo = edges[i]
        hi = edges[i + 1]
        if lo <= age < hi:
            return labels[i]
    # If it doesn't fit (e.g., negative age or > last edge), bucket.
    if age < edges[0]:
        return f"<{edges[0]:g}"
    return f">={edges[-1]:g}"


def _compute_stats(values: List[float]) -> GroupStats:
    n = len(values)
    mean = sum(values) / float(n)
    if n <= 1:
        std = 0.0
        sem = 0.0
    else:
        var = sum((v - mean) ** 2 for v in values) / float(n - 1)
        std = math.sqrt(max(var, 0.0))
        sem = std / math.sqrt(float(n))
    return GroupStats(n=n, mean=mean, std=std, sem=sem)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    yield obj
                else:
                    # Skip non-dict JSON (rare but possible)
                    continue
            except Exception:
                # Skip malformed lines (e.g. stray tokens)
                continue


def _is_radgraph_f1_metric(metric_field: str) -> bool:
    m = str(metric_field).strip().lower().replace("_", "-")
    # Accept a few common variants.
    return m in {"radgraph-f1", "radgraphf1", "f1-radgraph", "radgraph-f1-score", "radgraph"} or (
        "radgraph" in m and "f1" in m
    )


def _get_radgraph_f1_scorer(reward_level: str, batch_size: int):
    # Lazy import because this pulls in heavy deps (radgraph, numpy, scipy, hf hub).
    from llava.eval.rrg_eval.rrg_eval.f1radgraph import F1RadGraphv2

    # NOTE: The upstream radgraph model does not support multi-document minibatching
    # in the configuration used here (it raises NotImplementedError). We therefore
    # force batch_size=1 and score one (ref,hyp) pair at a time.
    return F1RadGraphv2(reward_level=reward_level, batch_size=1)


def _load_radgraph_cache(path: str) -> Dict[str, float]:
    cache: Dict[str, float] = {}
    if not path or (not os.path.exists(path)):
        return cache
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = str(row.get("id", "")).strip()
            if not rid:
                continue
            v = _safe_float(row.get("radgraph_f1"))
            if v is None:
                continue
            cache[rid] = float(v)
    return cache


def _write_radgraph_cache(path: str, cache: Dict[str, float]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "radgraph_f1"])
        w.writeheader()
        for rid in sorted(cache.keys()):
            w.writerow({"id": rid, "radgraph_f1": cache[rid]})


def parse_edges(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    edges = [float(p) for p in parts]
    if len(edges) < 2:
        raise ValueError("Need at least 2 edges")
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError("Edges must be strictly increasing")
    return edges


def parse_labels(s: str) -> List[str]:
    parts = [p.strip() for p in s.split(",")]
    labels = [p for p in parts if p != ""]
    if not labels:
        raise ValueError("Need at least 1 label")
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="Path to merged_demographics.jsonl",
    )
    ap.add_argument(
        "--metric",
        default="greenscore",
        help="Scalar metric field to analyze (default: greenscore)",
    )
    ap.add_argument(
        "--metric_scale",
        choices=["auto", "none", "100"],
        default="auto",
        help=(
            "Scale metric values before computing means/gaps. "
            "'auto' scales by 100 if values appear to be in [0,1] (recommended), "
            "'100' always scales by 100, 'none' disables scaling."
        ),
    )
    ap.add_argument(
        "--reference_field",
        default="reference",
        help="Reference text field (used for RadGraph-F1 computation when needed)",
    )
    ap.add_argument(
        "--prediction_field",
        default="prediction",
        help="Prediction/generated text field (used for RadGraph-F1 computation when needed)",
    )
    ap.add_argument(
        "--id_field",
        default="id",
        help="Row id field (used for RadGraph-F1 caching; default: id)",
    )
    ap.add_argument(
        "--radgraph_cache_csv",
        default=None,
        help=(
            "Optional CSV cache with columns id,radgraph_f1. "
            "Used only when --metric is a RadGraph-F1 variant and the field is missing in JSONL."
        ),
    )
    ap.add_argument(
        "--radgraph_reward_level",
        default="partial",
        choices=["simple", "partial", "complete", "all"],
        help="RadGraph reward level (default: partial)",
    )
    ap.add_argument(
        "--radgraph_batch_size",
        type=int,
        default=1,
        help="Batch size for RadGraph scoring (forced to 1 due to upstream limitation; default: 1)",
    )
    ap.add_argument(
        "--strata_fields",
        default=None,
        help=(
            "Comma-separated fields defining strata (e.g. age_group,gender). "
            "Within each stratum, we compute gaps over --gap_field. "
            "If omitted, defaults to age_bin_field only (age stratification)."
        ),
    )
    ap.add_argument(
        "--gap_field",
        default=None,
        help=(
            "Field to compute gaps across within each stratum (e.g. race_major). "
            "If omitted, defaults to --race_field."
        ),
    )
    ap.add_argument(
        "--age_bin_field",
        default="age_group",
        help="If present and non-empty per row, use this categorical age bin field.",
    )
    ap.add_argument(
        "--age_numeric_field",
        default="anchor_age",
        help="Numeric age field used to bin when age_bin_field missing/empty.",
    )
    ap.add_argument(
        "--age_edges",
        default=",".join(str(x) for x in DEFAULT_AGE_EDGES),
        help='Comma-separated bin edges for numeric age binning (default: "0,44,65,200").',
    )
    ap.add_argument(
        "--age_labels",
        default=",".join(DEFAULT_AGE_LABELS),
        help='Comma-separated labels for numeric age bins (default: "<44,44-64,65+").',
    )
    ap.add_argument(
        "--race_field",
        default="race_major",
        help="Race field (default: race_major)",
    )
    ap.add_argument(
        "--min_group_n",
        type=int,
        default=30,
        help="Minimum n required per (age,race) group to be included in gap computation (default: 30).",
    )
    ap.add_argument(
        "--drop_unknown",
        action="store_true",
        help="Drop UNKNOWN age/race bins.",
    )
    ap.add_argument(
        "--baseline_race",
        default="White",
        help="(Unused) Kept for backwards-compatibility; pairwise outputs are no longer written.",
    )
    ap.add_argument(
        "--weighting",
        choices=["stratum_n", "equal"],
        default="stratum_n",
        help=(
            "How to aggregate gaps across strata: "
            "'stratum_n' weights by stratum sample size (recommended), "
            "'equal' averages gaps equally across eligible strata."
        ),
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: alongside input under fairness_gap__age_x_race",
    )

    args = ap.parse_args()

    in_path = args.input
    metric_field = args.metric
    age_bin_field = args.age_bin_field
    age_numeric_field = args.age_numeric_field
    race_field = args.race_field
    strata_fields_raw = args.strata_fields
    if strata_fields_raw is None or str(strata_fields_raw).strip() == "":
        strata_fields: List[str] = [age_bin_field]
    else:
        strata_fields = [p.strip() for p in str(strata_fields_raw).split(",") if p.strip()]
        if not strata_fields:
            strata_fields = [age_bin_field]
    gap_field = args.gap_field if args.gap_field is not None else race_field

    edges = parse_edges(args.age_edges)
    labels = parse_labels(args.age_labels)
    if len(labels) != (len(edges) - 1):
        raise ValueError(
            f"age_labels must have len(age_edges)-1 labels (got labels={len(labels)} edges={len(edges)})"
        )

    outdir = args.outdir
    if outdir is None:
        base = os.path.dirname(os.path.abspath(in_path))
        outdir = os.path.join(base, "fairness_gap__age_x_race")
    os.makedirs(outdir, exist_ok=True)

    # Collect metric values by (stratum, gap_group) intersection.
    # stratum is a tuple aligned with strata_fields.
    values_by_group: Dict[Tuple[Tuple[str, ...], str], List[float]] = defaultdict(list)
    # Also track total rows per stratum for coverage.
    stratum_total_n: Dict[Tuple[str, ...], int] = defaultdict(int)
    total_rows_seen = 0
    used_rows = 0
    metric_missing_in_input = 0
    metric_failed_or_unusable = 0
    max_metric_seen: Optional[float] = None

    need_radgraph = _is_radgraph_f1_metric(metric_field)
    radgraph_cache: Dict[str, float] = {}
    pending_radgraph: List[Tuple[str, str, str, Tuple[str, ...], str]] = []
    # (row_id, reference, prediction, stratum_key, gap_group)
    radgraph_loaded_from_cache = 0
    radgraph_scored_now = 0
    if need_radgraph and args.radgraph_cache_csv:
        radgraph_cache = _load_radgraph_cache(args.radgraph_cache_csv)

    for row in _iter_jsonl(in_path):
        total_rows_seen += 1
        metric = _safe_float(row.get(metric_field))

        stratum_vals: List[str] = []
        drop_row = False
        for field in strata_fields:
            if field == age_bin_field:
                v = _normalize_str(row.get(field))
                if v == "UNKNOWN":
                    v = _bin_age(row.get(age_numeric_field), edges, labels)
            else:
                v = _normalize_str(row.get(field))
            if args.drop_unknown and v == "UNKNOWN":
                drop_row = True
                break
            stratum_vals.append(v)
        if drop_row:
            continue

        gap_group = _normalize_str(row.get(gap_field))
        if args.drop_unknown and gap_group == "UNKNOWN":
            continue

        stratum_key = tuple(stratum_vals)

        # If the requested metric is RadGraph-F1 and it isn't present, compute it.
        if metric is None and need_radgraph:
            metric_missing_in_input += 1
            row_id = _normalize_str(row.get(args.id_field))
            ref = str(row.get(args.reference_field) or "")
            hyp = str(row.get(args.prediction_field) or "")
            cached = radgraph_cache.get(row_id) if row_id != "UNKNOWN" else None
            if cached is not None:
                metric = float(cached)
                radgraph_loaded_from_cache += 1
            else:
                pending_radgraph.append((row_id, ref, hyp, stratum_key, gap_group))
        if metric is None:
            # Missing metric and not computable, or RadGraph scoring deferred to pending list.
            if not need_radgraph:
                metric_missing_in_input += 1
            metric_failed_or_unusable += 1
            continue

        metric_f = float(metric)
        if max_metric_seen is None or metric_f > max_metric_seen:
            max_metric_seen = metric_f

        stratum_total_n[stratum_key] += 1
        values_by_group[(stratum_key, gap_group)].append(metric_f)
        used_rows += 1

    if pending_radgraph:
        try:
            scorer = _get_radgraph_f1_scorer(
                reward_level=args.radgraph_reward_level,
                batch_size=max(1, int(args.radgraph_batch_size)),
            )
        except Exception as e:
            raise RuntimeError(
                "Requested RadGraph-F1 metric but could not initialize RadGraph scorer. "
                "Ensure required dependencies are installed (radgraph, numpy, scipy, huggingface_hub) "
                "and that the RadGraph weights can be downloaded or are already cached."
            ) from e

        for (row_id, ref, hyp, stratum_key, gap_group) in pending_radgraph:
            out = scorer(hyps=[hyp], refs=[ref])
            reward_list = out[1]
            try:
                f1 = float(reward_list[0][0])
            except Exception:
                f1 = float("nan")
            if not (math.isnan(f1) or math.isinf(f1)):
                radgraph_scored_now += 1
                if row_id != "UNKNOWN":
                    radgraph_cache[row_id] = f1
                stratum_total_n[stratum_key] += 1
                values_by_group[(stratum_key, gap_group)].append(f1)
                if max_metric_seen is None or f1 > max_metric_seen:
                    max_metric_seen = f1
                used_rows += 1
            else:
                metric_failed_or_unusable += 1

        if args.radgraph_cache_csv:
            _write_radgraph_cache(args.radgraph_cache_csv, radgraph_cache)

    # Decide scaling and apply to all values before computing stats/gaps.
    scale_factor = 1.0
    if args.metric_scale == "none":
        scale_factor = 1.0
    elif args.metric_scale == "100":
        scale_factor = 100.0
    else:
        # Heuristic: if max is <= ~1.0, treat as [0,1] metric and scale to percent.
        if max_metric_seen is not None and max_metric_seen <= 1.0001:
            scale_factor = 100.0

    if scale_factor != 1.0:
        for k in list(values_by_group.keys()):
            values_by_group[k] = [v * scale_factor for v in values_by_group[k]]

    # Compute stats for each (stratum, gap_group).
    stats_rows: List[Dict[str, Any]] = []
    stats_by_group: Dict[Tuple[Tuple[str, ...], str], GroupStats] = {}
    for (stratum_key, gap_group), values in sorted(values_by_group.items(), key=lambda x: (x[0][0], x[0][1])):
        st = _compute_stats(values)
        stats_by_group[(stratum_key, gap_group)] = st
        row_out: Dict[str, Any] = {}
        for i, field in enumerate(strata_fields):
            row_out[field] = stratum_key[i] if i < len(stratum_key) else ""
        row_out.update(
            {
                gap_field: gap_group,
                "n": st.n,
                "mean": st.mean,
                "std": st.std,
                "sem": st.sem,
            }
        )
        stats_rows.append(row_out)

    # Fairness gaps within each stratum: max(mean)-min(mean) across gap groups.
    groups_by_stratum: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for (stratum_key, gap_group) in stats_by_group.keys():
        groups_by_stratum[stratum_key].append(gap_group)

    gaps_by_stratum: List[Dict[str, Any]] = []

    eligible_strata_gaps: List[Tuple[Tuple[str, ...], float, int]] = []  # (stratum, gap, weight_n)
    total_metric_rows = sum(stratum_total_n.values())
    covered_rows = 0

    for stratum_key in sorted(groups_by_stratum.keys()):
        candidates: List[Tuple[str, GroupStats]] = []
        for grp in sorted(set(groups_by_stratum[stratum_key])):
            st = stats_by_group.get((stratum_key, grp))
            if st is None:
                continue
            if st.n < args.min_group_n:
                continue
            candidates.append((grp, st))

        stratum_n_all = int(stratum_total_n.get(stratum_key, 0))
        stratum_n_eligible = int(sum(st.n for _, st in candidates))

        out_row: Dict[str, Any] = {}
        for i, field in enumerate(strata_fields):
            out_row[field] = stratum_key[i] if i < len(stratum_key) else ""

        if len(candidates) < 2:
            out_row.update(
                {
                    "eligible_groups": len(candidates),
                    "gap_max_minus_min": "",
                    "argmax_group": "",
                    "argmin_group": "",
                    "mean_max": "",
                    "mean_min": "",
                    "stratum_n": stratum_n_all,
                    "eligible_n": stratum_n_eligible,
                    "min_group_n": args.min_group_n,
                }
            )
            gaps_by_stratum.append(out_row)
            continue

        candidates_sorted = sorted(candidates, key=lambda x: x[1].mean)
        (min_grp, min_st) = candidates_sorted[0]
        (max_grp, max_st) = candidates_sorted[-1]
        gap = max_st.mean - min_st.mean

        out_row.update(
            {
                "eligible_groups": len(candidates_sorted),
                "gap_max_minus_min": gap,
                "argmax_group": max_grp,
                "argmin_group": min_grp,
                "mean_max": max_st.mean,
                "mean_min": min_st.mean,
                "stratum_n": stratum_n_all,
                "eligible_n": stratum_n_eligible,
                "min_group_n": args.min_group_n,
            }
        )
        gaps_by_stratum.append(out_row)
        eligible_strata_gaps.append((stratum_key, gap, stratum_n_all))
        covered_rows += stratum_n_all

    # Aggregate across strata.
    delta_strat = None
    if eligible_strata_gaps:
        if args.weighting == "equal":
            delta_strat = sum(g for _, g, _ in eligible_strata_gaps) / float(len(eligible_strata_gaps))
        else:
            denom = float(sum(w for _, _, w in eligible_strata_gaps))
            if denom > 0:
                delta_strat = sum(g * float(w) for _, g, w in eligible_strata_gaps) / denom

    # Write outputs.
    strata_tag = "_x_".join(strata_fields)
    group_tag = f"gap_{gap_field}"
    gap_csv = os.path.join(outdir, f"fairness_gap_by_{strata_tag}__over_{gap_field}.csv")
    with open(gap_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = (
            list(strata_fields)
            + [
                "eligible_groups",
                "gap_max_minus_min",
                "argmax_group",
                "argmin_group",
                "mean_max",
                "mean_min",
                "stratum_n",
                "eligible_n",
                "min_group_n",
            ]
        )
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in gaps_by_stratum:
            w.writerow(r)

    summary = {
        "input": os.path.abspath(in_path),
        "metric": metric_field,
        "strata_fields": strata_fields,
        "gap_field": gap_field,
        "age_bin_field": age_bin_field,
        "age_numeric_field": age_numeric_field,
        "age_edges": edges,
        "age_labels": labels,
        "race_field": race_field,
        "min_group_n": args.min_group_n,
        "drop_unknown": bool(args.drop_unknown),
        "total_json_objects_seen": total_rows_seen,
        "rows_used": used_rows,
        "rows_metric_missing_in_input": metric_missing_in_input,
        "rows_metric_failed_or_unusable": metric_failed_or_unusable,
        "num_intersection_groups": len(stats_rows),
        "num_strata_total": len(stratum_total_n),
        "num_strata_eligible": len(eligible_strata_gaps),
        "weighting": args.weighting,
        "delta_strat": delta_strat,
        "metric_scale": {
            "mode": args.metric_scale,
            "scale_factor": scale_factor,
            "units": "percent" if scale_factor == 100.0 else "raw",
        },
        "radgraph": {
            "computed_missing": bool(need_radgraph),
            "cache_csv": os.path.abspath(args.radgraph_cache_csv) if args.radgraph_cache_csv else None,
            "reward_level": args.radgraph_reward_level,
            "batch_size": int(args.radgraph_batch_size),
            "loaded_from_cache": int(radgraph_loaded_from_cache),
            "scored_now": int(radgraph_scored_now),
        },
        "coverage": {
            "rows_with_metric_after_filters": total_metric_rows,
            "rows_in_eligible_strata": covered_rows,
            "coverage_fraction": (float(covered_rows) / float(total_metric_rows)) if total_metric_rows else None,
        },
        "outputs": {
            "fairness_gap_by_strata": os.path.abspath(gap_csv),
        },
    }

    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:")
    print(" -", gap_csv)
    print(" -", summary_path)
    if delta_strat is not None:
        print("Delta_strat (aggregated across eligible strata):", delta_strat)


if __name__ == "__main__":
    main()
