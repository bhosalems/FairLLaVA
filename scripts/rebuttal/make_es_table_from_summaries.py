#!/usr/bin/env python3
"""Build a table of Equity-Scaled (ES) metrics from per-run summary CSVs.

Reads `summary_es_metrics.csv` files produced by `scripts/summarize_gap_es.py`
and formats a compact table like the paper screenshot:
  - columns grouped by attribute (race / age / gender)
  - within each attribute: ES-BLEU-1, ES-BLEU-4, ES-F1-RadGraph, ES-GreenScore

This does not recompute metrics; it only summarizes existing outputs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ATTR_ORDER = ["race", "age", "gender"]
METRIC_ORDER = ["BLEU-1", "BLEU-4", "F1-RadGraph", "GreenScore"]


@dataclass(frozen=True)
class RunSummary:
    method: str
    by_attr_metric: Dict[Tuple[str, str], float]


def _read_summary_es_csv(path: Path) -> Dict[Tuple[str, str], float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    out: Dict[Tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attr = (row.get("attribute") or "").strip()
            metric = (row.get("metric") or "").strip()
            es_s = (row.get("es") or "").strip()
            if not attr or not metric or not es_s:
                continue
            try:
                es = float(es_s)
            except Exception:
                continue
            out[(attr, metric)] = es
    return out


def _fmt(x: Optional[float], *, decimals: int) -> str:
    if x is None:
        return ""
    if x != x:  # NaN
        return ""
    return f"{x:.{decimals}f}"


def _markdown_table(
    rows: List[RunSummary],
    *,
    decimals: int,
    bold_best: bool,
) -> str:
    # Precompute best per column.
    best: Dict[Tuple[str, str], float] = {}
    if bold_best:
        for attr in ATTR_ORDER:
            for metric in METRIC_ORDER:
                vals = [r.by_attr_metric.get((attr, metric)) for r in rows]
                vals_f = [v for v in vals if v is not None]
                if vals_f:
                    best[(attr, metric)] = max(vals_f)

    header1 = ["Method"]
    header2 = [""]
    for attr in ATTR_ORDER:
        header1.extend([attr.capitalize()] * len(METRIC_ORDER))
        header2.extend([f"ES-{m}" for m in METRIC_ORDER])

    lines: List[str] = []
    lines.append("| " + " | ".join(header1) + " |")
    lines.append("|" + "---|" * len(header1))
    lines.append("| " + " | ".join(header2) + " |")

    for r in rows:
        cells: List[str] = [r.method]
        for attr in ATTR_ORDER:
            for metric in METRIC_ORDER:
                v = r.by_attr_metric.get((attr, metric))
                s = _fmt(v, decimals=decimals)
                if bold_best and v is not None:
                    b = best.get((attr, metric))
                    if b is not None and abs(v - b) <= 1e-12:
                        s = f"**{s}**" if s else s
                cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def _write_csv(out_path: Path, rows: List[RunSummary]) -> None:
    cols: List[str] = ["method"]
    for attr in ATTR_ORDER:
        for metric in METRIC_ORDER:
            cols.append(f"{attr}__{metric}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row: Dict[str, object] = {"method": r.method}
            for attr in ATTR_ORDER:
                for metric in METRIC_ORDER:
                    row[f"{attr}__{metric}"] = r.by_attr_metric.get((attr, metric), "")
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Input in the form method_name=path/to/summary_es_metrics.csv OR method_name=path/to/pred_dir. "
            "Repeat for multiple methods."
        ),
    )
    ap.add_argument("--out_md", default="results/es_table.md")
    ap.add_argument("--out_csv", default="results/es_table.csv")
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument(
        "--bold_best",
        action="store_true",
        default=False,
        help="Bold the best value in each column.",
    )

    args = ap.parse_args()

    runs: List[RunSummary] = []
    for item in args.input:
        if "=" not in item:
            raise SystemExit(f"Bad --input (expected method=path): {item}")
        method, path_s = item.split("=", 1)
        method = method.strip()
        path = Path(path_s.strip())
        if path.is_dir():
            path = path / "summary_es_metrics.csv"
        runs.append(RunSummary(method=method, by_attr_metric=_read_summary_es_csv(path)))

    md = _markdown_table(runs, decimals=int(args.decimals), bold_best=bool(args.bold_best))

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")

    _write_csv(Path(args.out_csv), runs)

    print(f"Wrote: {out_md}")
    print(f"Wrote: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
