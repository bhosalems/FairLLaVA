#!/usr/bin/env python3
"""Bootstrap 95% CI and p-values for fairness gaps or equity-scaled metrics.

Given multiple `merged_demographics.jsonl` files (different methods) on the same set of ids,
this script:

1) Computes per-method fairness gap for each demographic attribute as:
    gap(attribute, metric) = max_g mean(metric | group=g) - min_g mean(metric | group=g)

2) Optionally computes equity-scaled (ES) for each attribute as:
    ES(attribute, metric) = overall_mean(metric) / (1 + gap(attribute, metric))

3) Computes 95% bootstrap percentile confidence intervals for each method's statistic.

4) Computes paired bootstrap p-values comparing each baseline method against a
    target method (default: fairllava) for each attribute, using the bootstrap
    distribution of (stat_baseline - stat_target).

5) Produces bar plots with error bars and p-value brackets.

Notes:
- GreenScore is read from the JSONL `greenscore` field.
- F1-RadGraph is computed from `reference`/`prediction` using the in-repo RadGraph scorer
  if it is not already present; results are cached on disk.

Outputs:
- stats CSV and JSON summaries
- two figures (one per metric) with 3 subplots (gender, age_group, race_major)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


ATTRS = [
    ("gender", "gender"),
    ("age", "age_group"),
    ("race", "race_major"),
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    path: Path


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _as_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return float((ys[mid - 1] + ys[mid]) / 2.0)


def _get_radgraph_scorer(reward_level: str):
    from llava.eval.rrg_eval.rrg_eval.f1radgraph import F1RadGraphv2

    # IMPORTANT: batch_size must be 1 due to upstream RadGraph limitation.
    return F1RadGraphv2(reward_level=reward_level, batch_size=1)


def _radgraph_f1_single(scorer, ref: str, pred: str) -> float:
    if not ref or not pred:
        return 0.0
    # scorer(...) returns (mean_reward, reward_list, ...)
    _, reward_list, *_ = scorer(refs=[ref], hyps=[pred])
    if not reward_list:
        return 0.0
    try:
        return float(reward_list[0][0])
    except Exception:
        return 0.0


def _ensure_radgraph_f1(
    rows: List[dict],
    cache_path: Path,
    reward_level: str,
    id_key: str = "id",
    ref_key: str = "reference",
    pred_key: str = "prediction",
    out_key: str = "radgraph_f1",
) -> None:
    """Populate rows[*][out_key] by computing RadGraph F1 with an on-disk cache."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: Dict[Tuple[str, str], float] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    cache[(str(d["id"]), str(d["pair_hash"]))] = float(d["radgraph_f1"])
                except Exception:
                    continue

    scorer = None
    to_compute: List[Tuple[int, str, str, str, str]] = []
    for i, r in enumerate(rows):
        if out_key in r and _as_float(r.get(out_key)) is not None:
            continue
        rid = str(r.get(id_key, ""))
        ref = str(r.get(ref_key, ""))
        pred = str(r.get(pred_key, ""))
        pair_hash = _sha1_text(ref + "\n" + pred)
        key = (rid, pair_hash)
        if key in cache:
            r[out_key] = float(cache[key])
            continue
        to_compute.append((i, rid, pair_hash, ref, pred))

    if not to_compute:
        return

    scorer = _get_radgraph_scorer(reward_level=reward_level)

    # Append-only cache writes so we can resume if interrupted.
    with open(cache_path, "a") as cache_f:
        for j, (i, rid, pair_hash, ref, pred) in enumerate(to_compute, start=1):
            score = _radgraph_f1_single(scorer, ref=ref, pred=pred)
            rows[i][out_key] = float(score)
            cache_f.write(json.dumps({"id": rid, "pair_hash": pair_hash, "radgraph_f1": float(score)}) + "\n")
            if j % 50 == 0 or j == len(to_compute):
                print(f"  [radgraph] computed {j}/{len(to_compute)}")


def _index_by_id(rows: List[dict], id_key: str = "id") -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for r in rows:
        rid = r.get(id_key)
        if rid is None:
            continue
        out[str(rid)] = r
    return out


def _gap_from_sample(
    ids: Sequence[str],
    by_id: Dict[str, dict],
    attr_key: str,
    metric_key: str,
    min_group_n: int,
    scale: float,
    agg: str,
) -> Optional[float]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for rid in ids:
        r = by_id.get(rid)
        if not r:
            continue
        g = r.get(attr_key)
        v = _as_float(r.get(metric_key))
        if g is None or v is None:
            continue
        groups[str(g)].append(v * scale)

    group_stats = []
    for _, vals in groups.items():
        if len(vals) < min_group_n:
            continue
        if agg == "median":
            group_stats.append(_median(vals))
        else:
            group_stats.append(sum(vals) / len(vals))

    if len(group_stats) < 2:
        return None
    return max(group_stats) - min(group_stats)


def _mean_metric_from_sample(
    ids: Sequence[str],
    by_id: Dict[str, dict],
    metric_key: str,
    scale: float,
    agg: str,
) -> Optional[float]:
    vals: List[float] = []
    for rid in ids:
        r = by_id.get(rid)
        if not r:
            continue
        v = _as_float(r.get(metric_key))
        if v is None:
            continue
        vals.append(v * scale)
    if not vals:
        return None
    if agg == "median":
        return _median(vals)
    return sum(vals) / len(vals)


def _equity_scaled_from_sample(
    ids: Sequence[str],
    by_id: Dict[str, dict],
    attr_key: str,
    metric_key: str,
    min_group_n: int,
    scale: float,
    agg: str,
) -> Optional[float]:
    # Compute on *unscaled* gap, then apply display scaling to ES at the end.
    # If scale!=1, we scale the metric values feeding both overall and group means
    # and also scale the final ES so the y-axis matches the chosen scale.
    overall = _mean_metric_from_sample(ids, by_id, metric_key=metric_key, scale=scale, agg=agg)
    if overall is None:
        return None
    gap = _gap_from_sample(ids, by_id, attr_key, metric_key, min_group_n=min_group_n, scale=scale, agg=agg)
    if gap is None:
        return None
    # ES definition used in this repo: ES = overall / (1 + gap)
    return float(overall) / (1.0 + float(gap))


def _bootstrap_ids(rng: random.Random, ids: Sequence[str], n: int) -> List[str]:
    return [ids[rng.randrange(0, n)] for _ in range(n)]


def _bootstrap_ids_stratified(rng: random.Random, group_to_ids: Dict[str, List[str]]) -> List[str]:
    """Stratified bootstrap: resample within each group preserving group sizes."""
    out: List[str] = []
    for ids_g in group_to_ids.values():
        if not ids_g:
            continue
        n = len(ids_g)
        out.extend(ids_g[rng.randrange(0, n)] for _ in range(n))
    return out


def _make_group_index(ids: Sequence[str], by_id: Dict[str, dict], attr_key: str) -> Dict[str, List[str]]:
    group_to_ids: Dict[str, List[str]] = defaultdict(list)
    for rid in ids:
        r = by_id.get(rid)
        if not r:
            continue
        g = r.get(attr_key)
        group_to_ids[str(g) if g is not None else "__MISSING__"].append(rid)
    return dict(group_to_ids)


def _bootstrap_ci(
    ids: Sequence[str],
    by_id: Dict[str, dict],
    attr_key: str,
    metric_key: str,
    min_group_n: int,
    scale: float,
    n_boot: int,
    seed: int,
    stat: str,
    bootstrap: str,
    group_to_ids: Optional[Dict[str, List[str]]],
    agg: str,
) -> Tuple[float, float, float, List[float]]:
    """Returns (obs, ci_l, ci_h, boot_samples) for the selected statistic."""

    if stat == "gap":
        obs = _gap_from_sample(ids, by_id, attr_key, metric_key, min_group_n=min_group_n, scale=scale, agg=agg)
    else:
        obs = _equity_scaled_from_sample(ids, by_id, attr_key, metric_key, min_group_n=min_group_n, scale=scale, agg=agg)
    if obs is None:
        return float("nan"), float("nan"), float("nan"), []

    rng = random.Random(seed)
    boots: List[float] = []
    for _ in range(n_boot):
        if bootstrap == "stratified":
            if not group_to_ids:
                b_ids = list(ids)
            else:
                b_ids = _bootstrap_ids_stratified(rng, group_to_ids)
        else:
            n = len(ids)
            b_ids = _bootstrap_ids(rng, ids, n)
        if stat == "gap":
            v = _gap_from_sample(b_ids, by_id, attr_key, metric_key, min_group_n=min_group_n, scale=scale, agg=agg)
        else:
            v = _equity_scaled_from_sample(b_ids, by_id, attr_key, metric_key, min_group_n=min_group_n, scale=scale, agg=agg)
        if v is None:
            continue
        boots.append(float(v))

    if not boots:
        return float(obs), float("nan"), float("nan"), []

    boots_s = sorted(boots)
    lo = boots_s[int(0.025 * (len(boots_s) - 1))]
    hi = boots_s[int(0.975 * (len(boots_s) - 1))]
    return float(obs), float(lo), float(hi), boots


def _paired_bootstrap_pvalue(
    ids: Sequence[str],
    by_id_a: Dict[str, dict],
    by_id_b: Dict[str, dict],
    attr_key: str,
    metric_key: str,
    min_group_n: int,
    scale: float,
    n_boot: int,
    seed: int,
    stat: str,
    p_mode: str,
    bootstrap: str,
    group_to_ids: Optional[Dict[str, List[str]]],
    agg: str,
) -> float:
    """Paired bootstrap p-value for diff = stat(a) - stat(b).

    We form diffs = stat(baseline) - stat(fairllava).

    Directional hypotheses:
    - stat='gap' (lower is better): H1 gap_fairllava < gap_baseline  => diff > 0
    - stat='es'  (higher is better): H1 es_fairllava  > es_baseline   => diff < 0
    """

    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(n_boot):
        if bootstrap == "stratified":
            if not group_to_ids:
                b_ids = list(ids)
            else:
                b_ids = _bootstrap_ids_stratified(rng, group_to_ids)
        else:
            n = len(ids)
            b_ids = _bootstrap_ids(rng, ids, n)
        if stat == "gap":
            ga = _gap_from_sample(b_ids, by_id_a, attr_key, metric_key, min_group_n=min_group_n, scale=scale, agg=agg)
            gb = _gap_from_sample(b_ids, by_id_b, attr_key, metric_key, min_group_n=min_group_n, scale=scale, agg=agg)
        else:
            ga = _equity_scaled_from_sample(b_ids, by_id_a, attr_key, metric_key, min_group_n=min_group_n, scale=scale, agg=agg)
            gb = _equity_scaled_from_sample(b_ids, by_id_b, attr_key, metric_key, min_group_n=min_group_n, scale=scale, agg=agg)
        if ga is None or gb is None:
            continue
        diffs.append(float(ga - gb))

    if not diffs:
        return float("nan")

    # Empirical probabilities
    le0 = sum(1 for d in diffs if d <= 0.0) / len(diffs)
    ge0 = sum(1 for d in diffs if d >= 0.0) / len(diffs)

    if p_mode == "two-sided":
        p = 2.0 * min(le0, ge0)
        return max(min(p, 1.0), 0.0)

    # Directional (one-sided) per the hypotheses above.
    if stat == "gap":
        # Want diff > 0; p = P(diff <= 0)
        return max(min(le0, 1.0), 0.0)
    else:
        # Want diff < 0; p = P(diff >= 0)
        return max(min(ge0, 1.0), 0.0)


def _fmt_p(p: float) -> str:
    if p is None or math.isnan(p):
        return "p=NA"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def _p_alt_label(stat: str) -> str:
    # Alternative in terms of diff = baseline - fairllava
    if stat == "gap":
        return ">0"
    return "<0"


def _add_p_bracket(ax, x1: float, x2: float, y: float, text: str):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1)
    ax.text((x1 + x2) / 2, y + h * 1.3, text, ha="center", va="bottom", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fairllava", type=str, required=True)
    ap.add_argument("--llavarad", type=str, required=True)
    ap.add_argument("--reweighting", type=str, required=True)
    ap.add_argument("--resampling", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/gap_bootstrap")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--min_group_n", type=int, default=20)
    ap.add_argument(
        "--stat",
        type=str,
        default="gap",
        choices=["gap", "es"],
        help="Statistic to bootstrap: fairness gap ('gap') or equity-scaled ('es') = overall/(1+gap).",
    )
    ap.add_argument(
        "--p_mode",
        type=str,
        default="directional",
        choices=["directional", "two-sided"],
        help=(
            "P-value mode for baseline vs fairllava paired bootstrap over diff=baseline-fairllava. "
            "'directional' uses one-sided hypothesis (gap: diff>0, es: diff<0)."
        ),
    )
    ap.add_argument(
        "--baselines",
        type=str,
        default="llavarad,reweighting,resampling",
        help="Comma-separated baselines to compare vs fairllava (e.g., 'llavarad').",
    )
    ap.add_argument(
        "--bootstrap",
        type=str,
        default="iid",
        choices=["iid", "stratified"],
        help="Bootstrap mode: iid resampling of ids, or stratified resampling within each group for the current attribute.",
    )
    ap.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation for overall and per-group metrics: mean or median.",
    )
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply metrics by this factor (e.g., 100 for percentage points)")
    ap.add_argument("--radgraph_reward_level", type=str, default="partial", choices=["simple", "partial", "complete", "all"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_methods = {
        "llavarad": MethodSpec("llavarad", Path(args.llavarad)),
        "reweighting": MethodSpec("reweighting", Path(args.reweighting)),
        "resampling": MethodSpec("resampling", Path(args.resampling)),
        "fairllava": MethodSpec("fairllava", Path(args.fairllava)),
    }
    baseline_names = [b.strip() for b in args.baselines.split(",") if b.strip()]
    baseline_names = [b for b in baseline_names if b != "fairllava"]
    for b in baseline_names:
        if b not in all_methods:
            raise ValueError(f"Unknown baseline '{b}'. Allowed: {sorted(all_methods.keys())}")
    if not baseline_names:
        raise ValueError("--baselines must include at least one baseline (e.g., 'llavarad')")

    methods = [all_methods[b] for b in baseline_names] + [all_methods["fairllava"]]

    print("Loading JSONLs...")
    rows_by_method: Dict[str, List[dict]] = {m.name: _load_jsonl(m.path) for m in methods}

    # Compute RadGraph F1 (cached) if missing
    for m in methods:
        cache_path = out_dir / f"radgraph_cache__{m.name}.jsonl"
        print(f"Ensuring RadGraph F1 for {m.name} (cache: {cache_path.name})")
        _ensure_radgraph_f1(
            rows_by_method[m.name],
            cache_path=cache_path,
            reward_level=args.radgraph_reward_level,
            out_key="radgraph_f1",
        )

    by_id: Dict[str, Dict[str, dict]] = {m.name: _index_by_id(rows_by_method[m.name]) for m in methods}

    # Use intersection of ids across all methods for paired comparisons
    id_sets = [set(by_id[m.name].keys()) for m in methods]
    ids_all = sorted(set.intersection(*id_sets))
    print("id counts:", {m.name: len(by_id[m.name]) for m in methods}, "intersection:", len(ids_all))

    # Precompute stratification indices from fairllava demographics (shared across methods).
    strat_index: Dict[str, Dict[str, List[str]]] = {}
    if args.bootstrap == "stratified":
        ref_by_id = by_id["fairllava"]
        for _attr_label, attr_key in ATTRS:
            strat_index[attr_key] = _make_group_index(ids_all, ref_by_id, attr_key=attr_key)

    metrics = [
        ("F1-RadGraph", "radgraph_f1"),
        ("GreenScore", "greenscore"),
    ]

    # Collect stats
    stats_rows = []
    pval_rows = []

    for metric_label, metric_key in metrics:
        for attr_label, attr_key in ATTRS:
            # CI per method
            for m in methods:
                obs, lo, hi, _ = _bootstrap_ci(
                    ids_all,
                    by_id[m.name],
                    attr_key=attr_key,
                    metric_key=metric_key,
                    min_group_n=args.min_group_n,
                    scale=args.scale,
                    n_boot=args.n_boot,
                    seed=args.seed,
                    stat=args.stat,
                    bootstrap=args.bootstrap,
                    group_to_ids=strat_index.get(attr_key),
                    agg=args.agg,
                )
                stats_rows.append(
                    {
                        "metric": metric_label,
                        "attribute": attr_label,
                        "method": m.name,
                        "stat": args.stat,
                        "value": obs,
                        "ci_l": lo,
                        "ci_h": hi,
                        "n": len(ids_all),
                        "min_group_n": args.min_group_n,
                        "scale": args.scale,
                    }
                )

            # p-values vs fairllava
            for baseline in baseline_names:
                p = _paired_bootstrap_pvalue(
                    ids_all,
                    by_id_a=by_id[baseline],
                    by_id_b=by_id["fairllava"],
                    attr_key=attr_key,
                    metric_key=metric_key,
                    min_group_n=args.min_group_n,
                    scale=args.scale,
                    n_boot=args.n_boot,
                    seed=args.seed + 17,
                    stat=args.stat,
                    p_mode=args.p_mode,
                    bootstrap=args.bootstrap,
                    group_to_ids=strat_index.get(attr_key),
                    agg=args.agg,
                )
                # Also compute two-sided for reference (useful to report alongside).
                p_two = _paired_bootstrap_pvalue(
                    ids_all,
                    by_id_a=by_id[baseline],
                    by_id_b=by_id["fairllava"],
                    attr_key=attr_key,
                    metric_key=metric_key,
                    min_group_n=args.min_group_n,
                    scale=args.scale,
                    n_boot=args.n_boot,
                    seed=args.seed + 17,
                    stat=args.stat,
                    p_mode="two-sided",
                    bootstrap=args.bootstrap,
                    group_to_ids=strat_index.get(attr_key),
                    agg=args.agg,
                )
                pval_rows.append(
                    {
                        "metric": metric_label,
                        "attribute": attr_label,
                        "stat": args.stat,
                        "baseline": baseline,
                        "target": "fairllava",
                        "p": p,
                        "p_two_sided": p_two,
                        "p_mode": args.p_mode,
                        "diff_alt": _p_alt_label(args.stat),
                        "n": len(ids_all),
                        "n_boot": args.n_boot,
                    }
                )

    # Write tables
    suffix = f"__{'_vs_'.join(baseline_names + ['fairllava'])}__{args.bootstrap}__{args.agg}"
    stats_csv = out_dir / f"{args.stat}_bootstrap_ci{suffix}.csv"
    with open(stats_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)

    p_csv = out_dir / f"{args.stat}_pvalues_vs_fairllava{suffix}.csv"
    with open(p_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pval_rows[0].keys()))
        w.writeheader()
        w.writerows(pval_rows)

    with open(out_dir / f"{args.stat}_bootstrap_summary{suffix}.json", "w") as f:
        json.dump({"stats": stats_rows, "pvalues": pval_rows}, f, indent=2)

    print(f"Wrote: {stats_csv}")
    print(f"Wrote: {p_csv}")

    # Plot per metric: 3 subplots (attributes)
    method_order = baseline_names + ["fairllava"]
    colors = {
        "llavarad": "#4C78A8",
        "reweighting": "#F58518",
        "resampling": "#54A24B",
        "fairllava": "#E45756",
    }

    for metric_label, _metric_key in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
        for i, (attr_label, _attr_key) in enumerate(ATTRS):
            ax = axes[i]
            rows = [r for r in stats_rows if r["metric"] == metric_label and r["attribute"] == attr_label]
            by_method = {r["method"]: r for r in rows}
            xs = list(range(len(method_order)))
            ys = [by_method[m]["value"] for m in method_order]
            yerr_lo = [by_method[m]["value"] - by_method[m]["ci_l"] for m in method_order]
            yerr_hi = [by_method[m]["ci_h"] - by_method[m]["value"] for m in method_order]
            ax.bar(xs, ys, color=[colors[m] for m in method_order], alpha=0.85)
            ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=4, linewidth=1)

            ax.set_title(attr_label)
            ax.set_xticks(xs)
            ax.set_xticklabels(method_order, rotation=0)
            ax.grid(True, axis="y", alpha=0.25)
            ylab = "fairness gap" if args.stat == "gap" else "equity scaled"
            ax.set_ylabel(ylab + (" (scaled)" if args.scale != 1.0 else ""))

            # p-value brackets baseline -> fairllava
            fair_x = method_order.index("fairllava")
            p_rows = [p for p in pval_rows if p["metric"] == metric_label and p["attribute"] == attr_label]
            p_by_base = {p["baseline"]: p for p in p_rows}
            ymax = max([v for v in ys if not math.isnan(v)] + [0.0])
            step = max(0.02 * max(1e-9, ymax), 0.01)
            base_y = ymax + step
            k = 0
            for base in baseline_names:
                if base not in p_by_base:
                    continue
                bx = method_order.index(base)
                _add_p_bracket(ax, bx, fair_x, base_y + k * step, _fmt_p(p_by_base[base]["p"]))
                k += 1

        supt = "Fairness gap" if args.stat == "gap" else "Equity scaled"
        fig.suptitle(f"{supt} ({args.agg}) + 95% CI — {metric_label}")
        fig.tight_layout(rect=[0, 0.02, 1, 0.92])
        out_path = out_dir / f"{args.stat}_bootstrap_ci_pvals__{metric_label.replace(' ', '_').replace('/', '_')}{suffix}.png"
        fig.savefig(out_path, dpi=400, bbox_inches="tight")
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
