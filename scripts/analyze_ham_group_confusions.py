#!/usr/bin/env python3
"""Analyze HAM fairness runs using group_confusions.json.

This script is intentionally dependency-free.

Typical usage:
  python scripts/analyze_ham_group_confusions.py --root results/ham --top_k 8

It prints:
  1) Per-run, per-sex (and per-age) Accuracy/BalancedAcc/MacroF1 computed from confusion matrices.
  2) Largest MacroF1-vs-BalAcc divergences and per-class precision/recall/F1 drivers.
  3) Sex gaps from summary_metrics.json (if present).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import isnan
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PerClass:
    support: int
    tp: int
    precision: float
    recall: float
    f1: float


@dataclass
class Stats:
    acc: float
    balanced_acc: float
    macro_f1: float
    total: int
    per_class: Dict[str, PerClass]


def _safe_div(num: float, den: float) -> float:
    return float("nan") if den == 0 else num / den


def stats_from_confusion(labels_true: List[str], labels_pred: List[str], mat: List[List[int]]) -> Stats:
    idx_pred = {lab: i for i, lab in enumerate(labels_pred)}

    col_sums = [0] * len(labels_pred)
    row_sums = [0] * len(labels_true)

    for i in range(len(labels_true)):
        row = mat[i]
        row_sums[i] = sum(row)
        for j, v in enumerate(row):
            col_sums[j] += v

    per: Dict[str, PerClass] = {}
    recalls: List[float] = []
    f1s: List[float] = []

    for i, lab in enumerate(labels_true):
        j = idx_pred.get(lab)
        tp = mat[i][j] if j is not None else 0
        actual = row_sums[i]
        pred_total = col_sums[j] if j is not None else 0

        rec = _safe_div(tp, actual) if actual else float("nan")
        prec = _safe_div(tp, pred_total) if pred_total else float("nan")

        if actual and (prec == prec) and (rec == rec) and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        elif actual:
            f1 = 0.0
        else:
            f1 = float("nan")

        per[lab] = PerClass(
            support=actual,
            tp=tp,
            precision=prec,
            recall=rec,
            f1=f1,
        )

        if actual:
            recalls.append(rec)
            f1s.append(f1)

    balanced_acc = sum(recalls) / len(recalls) if recalls else float("nan")
    macro_f1 = sum(f1s) / len(f1s) if f1s else float("nan")

    total = sum(row_sums)
    correct = 0
    for i, lab in enumerate(labels_true):
        j = idx_pred.get(lab)
        if j is not None:
            correct += mat[i][j]
    acc = _safe_div(correct, total) if total else float("nan")

    return Stats(acc=acc, balanced_acc=balanced_acc, macro_f1=macro_f1, total=total, per_class=per)


def _fmt_pct(x: float) -> str:
    if x != x:
        return "NA"
    return f"{100 * x:5.1f}"


def _iter_runs(root: Path) -> List[Tuple[str, dict]]:
    out: List[Tuple[str, dict]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        p = d / "group_confusions.json"
        if not p.exists():
            continue
        out.append((d.name, json.loads(p.read_text())))
    return out


def _sort_key_gid(gid: str):
    try:
        return int(gid)
    except Exception:
        return gid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/ham")
    ap.add_argument("--top_k", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    runs = _iter_runs(root)
    print(f"Found {len(runs)} runs with group_confusions.json under {root}")

    # 1) Per-sex table
    print("\nPer-sex divergence (values in %, computed from confusion):")
    print("run\tsex\tn\tAcc\tBalAcc\tMacroF1\t(MacroF1-BalAcc)")

    divergence_rank = []

    for name, gc in runs:
        sex_groups = gc["sex"]["groups"]
        for sex in sorted(sex_groups.keys(), key=_sort_key_gid):
            g = sex_groups[sex]
            c = g["confusion"]
            stats = stats_from_confusion(c["labels_true"], c["labels_pred"], c["matrix"])
            diff = stats.macro_f1 - stats.balanced_acc
            divergence_rank.append((abs(diff), name, "sex", sex, stats, g))
            print(
                f"{name}\t{sex}\t{g['n']}\t{_fmt_pct(stats.acc)}\t{_fmt_pct(stats.balanced_acc)}\t{_fmt_pct(stats.macro_f1)}\t{_fmt_pct(diff)}"
            )

    # 2) Per-age table
    print("\nPer-age divergence (values in %, computed from confusion):")
    print("run\tage\tn\tAcc\tBalAcc\tMacroF1\t(MacroF1-BalAcc)")

    for name, gc in runs:
        age_groups = gc["age"]["groups"]
        for age in sorted(age_groups.keys(), key=_sort_key_gid):
            g = age_groups[age]
            c = g["confusion"]
            stats = stats_from_confusion(c["labels_true"], c["labels_pred"], c["matrix"])
            diff = stats.macro_f1 - stats.balanced_acc
            divergence_rank.append((abs(diff), name, "age", age, stats, g))
            print(
                f"{name}\t{age}\t{g['n']}\t{_fmt_pct(stats.acc)}\t{_fmt_pct(stats.balanced_acc)}\t{_fmt_pct(stats.macro_f1)}\t{_fmt_pct(diff)}"
            )

    # 3) Detailed drivers for largest divergences
    print("\nTop divergences: class drivers (F1 - recall).")
    for _, name, attr, gid, stats, g in sorted(divergence_rank, reverse=True)[: args.top_k]:
        deltas = []
        for lab, s in stats.per_class.items():
            if s.support == 0:
                continue
            if s.precision != s.precision or s.recall != s.recall:
                continue
            deltas.append((s.f1 - s.recall, lab, s))
        deltas.sort()  # most negative first

        print(
            f"\n{name} {attr}={gid} n={g['n']}  Acc={_fmt_pct(stats.acc)} BalAcc={_fmt_pct(stats.balanced_acc)} MacroF1={_fmt_pct(stats.macro_f1)}"
        )
        print("Worst (F1 - recall):")
        for d, lab, s in deltas[:3]:
            print(
                f"  {lab}: support={s.support:3d}  P={_fmt_pct(s.precision)} R={_fmt_pct(s.recall)} F1={_fmt_pct(s.f1)}  (F1-R)={_fmt_pct(d)}"
            )
        print("Best (F1 - recall):")
        for d, lab, s in deltas[-3:][::-1]:
            print(
                f"  {lab}: support={s.support:3d}  P={_fmt_pct(s.precision)} R={_fmt_pct(s.recall)} F1={_fmt_pct(s.f1)}  (F1-R)={_fmt_pct(d)}"
            )

    # 4) Sex gaps from summary_metrics.json
    print("\nSex gaps from summary_metrics.json (if present):")
    print("run\tgap_Acc\tgap_BalAcc\tgap_MacroF1")
    for name, _ in runs:
        sm = root / name / "summary_metrics.json"
        if not sm.exists():
            continue
        d = json.loads(sm.read_text())
        gaps = {(r["attribute"], r["metric"]): r["gap"] for r in d.get("fairness_gap", [])}
        ga = gaps.get(("sex", "Accuracy"))
        gb = gaps.get(("sex", "BalancedAcc"))
        gm = gaps.get(("sex", "Macro-F1"))

        def pf(x):
            return "NA" if x is None else f"{x:0.3f}"

        print(f"{name}\t{pf(ga)}\t{pf(gb)}\t{pf(gm)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
