#!/usr/bin/env python3
"""Verify CheXpert labeler outputs against stored (GT) CheXpert labels.

Use-case:
  You have a JSON/JSONL file where each example contains:
    - a reference report text field
    - stored CheXpert labels (ground-truth or previously-computed labels)

This script re-runs the official CheXpert rule-based labeler
(stanfordmlgroup/chexpert-labeler/label.py) on the reference text and compares
its outputs to the stored labels.

It is designed to work in a lightweight "chexpert-label" environment:
  - does NOT import llava/torch
  - shells out to label.py

Outputs (in --outdir):
  - verify_per_label.csv
  - verify_summary.json
  - mismatches.jsonl (optional, limited)

Notes:
  - Stored labels may be:
      * dict[label_name -> value]
      * list aligned with CHEXPERT_CONDITIONS
  - Values can be in {-1, 0, 1, None/blank}
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


CHEXPERT_CONDITIONS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "x"


def _map_uncertainty_to_binary(x: Optional[float], u_policy: str = "u->0") -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    if x == 1:
        return 1
    if x == 0:
        return 0
    if x == -1:
        return 1 if u_policy == "u->1" else 0
    return 0


def _canonical_u_policy(u_policy: str) -> str:
    s = str(u_policy).strip()
    aliases = {
        "u0": "u->0",
        "u1": "u->1",
        "u_as_0": "u->0",
        "u_as_1": "u->1",
        "u_as_neg": "u->0",
        "u_as_pos": "u->1",
        "u->0": "u->0",
        "u->1": "u->1",
    }
    if s not in aliases:
        raise ValueError(f"Unknown u_policy: {u_policy!r}. Use one of: {sorted(set(aliases))}")
    return aliases[s]


def _tail_text_file(path: str, max_lines: int = 200) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[-max_lines:])
    except Exception as e:
        return f"<failed to read log file {path!r}: {e}>"


class CheXpertLabeler:
    def __init__(
        self,
        chexpert_root: str,
        python_exe: str,
        extra_pythonpath: Optional[List[str]] = None,
        heartbeat_secs: float = 0.0,
        output_mode: str = "file",
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
    ):
        self.chexpert_root = os.path.abspath(chexpert_root)
        self.python_exe = python_exe
        self.extra_pythonpath = [p for p in (extra_pythonpath or []) if str(p).strip()]
        self.heartbeat_secs = float(heartbeat_secs or 0.0)
        self.output_mode = str(output_mode or "file").strip()
        self.log_dir = os.path.abspath(log_dir) if log_dir else None
        self.log_file = os.path.abspath(log_file) if log_file else None

        self.label_py = os.path.join(self.chexpert_root, "label.py")
        if not os.path.isfile(self.label_py):
            raise FileNotFoundError(f"Could not find label.py at {self.label_py}")
        if self.output_mode not in {"file", "inherit", "capture"}:
            raise ValueError("output_mode must be one of: file, inherit, capture")

    def forward(self, reports: List[str]) -> Dict[str, List[Optional[int]]]:
        with tempfile.TemporaryDirectory() as td:
            in_csv = os.path.join(td, "reports.csv")
            out_csv = os.path.join(td, "labeled_reports.csv")

            with open(in_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for r in reports:
                    w.writerow([r if r is not None else ""])

            cmd = [
                self.python_exe,
                self.label_py,
                "--reports_path",
                in_csv,
                "--output_path",
                out_csv,
            ]

            popen_kwargs: Dict[str, Any] = {"cwd": self.chexpert_root}
            if self.extra_pythonpath:
                env = os.environ.copy()
                existing = env.get("PYTHONPATH", "")
                prefix = os.pathsep.join(self.extra_pythonpath)
                env["PYTHONPATH"] = prefix + (os.pathsep + existing if existing else "")
                popen_kwargs["env"] = env
            if sys.version_info >= (3, 7):
                popen_kwargs["text"] = True
            else:
                popen_kwargs["universal_newlines"] = True

            log_path = None
            log_fh = None
            if self.output_mode == "inherit":
                popen_kwargs["stdout"] = None
                popen_kwargs["stderr"] = None
            elif self.output_mode == "capture":
                popen_kwargs["stdout"] = subprocess.PIPE
                popen_kwargs["stderr"] = subprocess.PIPE
            else:
                log_dir = self.log_dir or os.getcwd()
                os.makedirs(log_dir, exist_ok=True)
                log_path = self.log_file or os.path.join(
                    log_dir, f"chexpert_verify_{os.getpid()}_{int(time.time())}.log"
                )
                log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
                popen_kwargs["stdout"] = log_fh
                popen_kwargs["stderr"] = subprocess.STDOUT

            proc = subprocess.Popen(cmd, **popen_kwargs)
            start_t = time.time()
            last_t = start_t
            while proc.poll() is None:
                if self.heartbeat_secs > 0 and (time.time() - last_t) >= self.heartbeat_secs:
                    elapsed = int(time.time() - start_t)
                    print(f"[chexpert] labeler still running... {elapsed}s elapsed")
                    last_t = time.time()
                time.sleep(1.0)

            stdout = None
            stderr = None
            if self.output_mode == "capture":
                stdout, stderr = proc.communicate()
            else:
                proc.wait()

            if log_fh is not None:
                try:
                    log_fh.flush()
                finally:
                    log_fh.close()

            if proc.returncode != 0 or (not os.path.exists(out_csv)):
                extra = ""
                if self.output_mode == "capture":
                    extra = f"\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                elif log_path is not None:
                    extra = f"\nLog (tail):\n{_tail_text_file(log_path)}"
                raise RuntimeError("CheXpert labeler failed.\n" f"Return code: {proc.returncode}\n" + extra)

            df = pd.read_csv(out_csv)
            missing = [c for c in CHEXPERT_CONDITIONS if c not in df.columns]
            if missing:
                raise ValueError(f"CheXpert output missing expected columns: {missing}")

            out: Dict[str, List[Optional[int]]] = {}
            for lab in CHEXPERT_CONDITIONS:
                out[lab] = [None if pd.isna(v) else int(v) for v in df[lab].astype("float").tolist()]
            return out


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    path = os.path.abspath(path)
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        # common patterns: {"data": [...]} or {"annotations": [...]}
        for k in ("data", "annotations", "items", "examples"):
            v = obj.get(k)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
        # fallback: dict-of-dicts
        vals = [v for v in obj.values() if isinstance(v, dict)]
        if vals:
            return vals
    raise ValueError(f"Unrecognized JSON structure in {path}")


def _get_first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _extract_text(row: Dict[str, Any], text_field: Optional[str]) -> str:
    if text_field:
        return str(row.get(text_field) or "")
    v = _get_first_present(row, ["reference", "ref", "gt", "report", "text", "finding", "findings"])
    return str(v or "")


def _extract_gt_labels(row: Dict[str, Any], labels_field: Optional[str]) -> Optional[Any]:
    if labels_field:
        return row.get(labels_field)
    return _get_first_present(
        row,
        [
            "chexpert_labels",
            "chexpert",
            "chexpert_label",
            "labels_chexpert",
            "labels",
            "gt_labels",
        ],
    )


def _parse_gt_labels(gt_obj: Any) -> Optional[Dict[str, Optional[int]]]:
    if gt_obj is None:
        return None

    if isinstance(gt_obj, dict):
        out: Dict[str, Optional[int]] = {}
        for lab in CHEXPERT_CONDITIONS:
            v = gt_obj.get(lab)
            if v is None:
                out[lab] = None
            elif isinstance(v, bool):
                out[lab] = int(v)
            else:
                try:
                    out[lab] = int(v)
                except Exception:
                    out[lab] = None
        return out

    if isinstance(gt_obj, list):
        if len(gt_obj) < len(CHEXPERT_CONDITIONS):
            return None
        out2: Dict[str, Optional[int]] = {}
        for i, lab in enumerate(CHEXPERT_CONDITIONS):
            v = gt_obj[i]
            if v is None:
                out2[lab] = None
            elif isinstance(v, bool):
                out2[lab] = int(v)
            else:
                try:
                    out2[lab] = int(v)
                except Exception:
                    out2[lab] = None
        return out2

    return None


def _merge_label_outputs(chunks: List[Dict[str, List[Optional[int]]]]) -> Dict[str, List[Optional[int]]]:
    merged: Dict[str, List[Optional[int]]] = {lab: [] for lab in CHEXPERT_CONDITIONS}
    for out in chunks:
        for lab in CHEXPERT_CONDITIONS:
            merged[lab].extend(out[lab])
    return merged


def _label_batched(
    labeler: CheXpertLabeler,
    reports: List[str],
    batch_size: int,
    num_workers: int,
    show_progress: bool,
    desc: str,
) -> Dict[str, List[Optional[int]]]:
    if batch_size <= 0 or batch_size >= len(reports):
        if show_progress:
            print(f"[chexpert] labeling {len(reports)} reports: {desc} (single batch)")
        return labeler.forward(reports)

    n = len(reports)
    n_batches = int((n + batch_size - 1) // batch_size)
    num_workers = max(1, int(num_workers))

    if show_progress:
        msg = f"[chexpert] labeling {n} reports in {n_batches} batches"
        if num_workers > 1:
            msg += f" with {num_workers} workers"
        msg += f": {desc}"
        print(msg)

    def _run_one(bi: int) -> Dict[str, List[Optional[int]]]:
        start = bi * batch_size
        end = min(n, (bi + 1) * batch_size)
        return labeler.forward(reports[start:end])

    outs_by_idx: List[Optional[Dict[str, List[Optional[int]]]]] = [None] * n_batches

    def _mk_pbar(total: int):
        if show_progress and tqdm is not None:
            return tqdm(total=total, desc=f"chexpert: {desc}")
        return None

    pbar = _mk_pbar(n_batches)

    if num_workers == 1:
        for bi in range(n_batches):
            outs_by_idx[bi] = _run_one(bi)
            if pbar is not None:
                pbar.update(1)
            elif show_progress and (bi == 0 or (bi + 1) % 10 == 0 or (bi + 1) == n_batches):
                end = min(n, (bi + 1) * batch_size)
                print(f"[chexpert] {desc}: batch {bi+1}/{n_batches} ({end}/{n})")
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            fut_to_bi = {ex.submit(_run_one, bi): bi for bi in range(n_batches)}
            done = 0
            for fut in as_completed(fut_to_bi):
                bi = fut_to_bi[fut]
                outs_by_idx[bi] = fut.result()
                done += 1
                if pbar is not None:
                    pbar.update(1)
                elif show_progress and (done == 1 or done % 10 == 0 or done == n_batches):
                    print(f"[chexpert] {desc}: completed {done}/{n_batches} batches")

    if pbar is not None:
        pbar.close()

    outs: List[Dict[str, List[Optional[int]]]] = []
    for bi in range(n_batches):
        out = outs_by_idx[bi]
        if out is None:
            raise RuntimeError(f"Internal error: missing output for batch {bi}")
        outs.append(out)
    return _merge_label_outputs(outs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify CheXpert labeler outputs against stored labels.")
    ap.add_argument("--input", required=True, help="Path to JSON or JSONL")
    ap.add_argument("--text_field", default=None, help="Field name for reference report text (default: auto)")
    ap.add_argument("--gt_labels_field", default=None, help="Field name for stored CheXpert labels (default: auto)")

    ap.add_argument("--chexpert_root", required=True, help="Path to stanfordmlgroup/chexpert-labeler")
    ap.add_argument("--python_exe", default=sys.executable, help="Python executable to run label.py")
    ap.add_argument("--labeler_pythonpath", nargs="*", default=[], help="Extra PYTHONPATH entries for label.py")
    ap.add_argument("--labeler_output", default="file", choices=["file", "inherit", "capture"])
    ap.add_argument("--labeler_logdir", default=None)
    ap.add_argument(
        "--labeler_logfile",
        default=None,
        help=(
            "Explicit log file path for chexpert-labeler output when --labeler_output=file. "
            "If omitted, a timestamped log is created in --labeler_logdir."
        ),
    )
    ap.add_argument("--heartbeat_secs", type=float, default=30.0)

    ap.add_argument("--u_policy", default="u0", choices=["u0", "u1", "u->0", "u->1"])
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max_mismatches", type=int, default=200)

    args = ap.parse_args()
    args.u_policy = _canonical_u_policy(args.u_policy)

    if args.outdir is None:
        base = os.path.dirname(os.path.abspath(args.input))
        tag = _slugify(os.path.splitext(os.path.basename(args.input))[0])
        args.outdir = os.path.join(base, f"chexpert_verify__{tag}")
    _ensure_dir(args.outdir)
    if args.labeler_logdir is None:
        args.labeler_logdir = args.outdir

    # Common user mistake: passing a filename to --labeler_output.
    if str(args.labeler_output).endswith(".log"):
        raise SystemExit(
            "ERROR: --labeler_output must be one of {file, inherit, capture}. "
            "It looks like you passed a log filename. Use --labeler_logfile <path> instead."
        )

    rows = _read_json_or_jsonl(args.input)
    if not rows:
        raise SystemExit("ERROR: no rows found")

    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    if args.shuffle:
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(rows)

    texts: List[str] = []
    gt_list: List[Optional[Dict[str, Optional[int]]]] = []
    used_idx: List[int] = []

    for i, row in enumerate(rows):
        text = _extract_text(row, args.text_field)
        gt_obj = _extract_gt_labels(row, args.gt_labels_field)
        gt_parsed = _parse_gt_labels(gt_obj)
        if gt_parsed is None:
            continue
        texts.append(str(text or ""))
        gt_list.append(gt_parsed)
        used_idx.append(i)

    if not texts:
        raise SystemExit(
            "ERROR: Could not find any rows with parseable GT labels. "
            "Try passing --gt_labels_field and/or --text_field."
        )

    labeler = CheXpertLabeler(
        chexpert_root=args.chexpert_root,
        python_exe=args.python_exe,
        extra_pythonpath=list(args.labeler_pythonpath or []),
        heartbeat_secs=float(args.heartbeat_secs),
        output_mode=str(args.labeler_output),
        log_dir=str(args.labeler_logdir) if args.labeler_logdir else None,
        log_file=str(args.labeler_logfile) if args.labeler_logfile else None,
    )

    pred = _label_batched(
        labeler,
        texts,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        show_progress=True,
        desc="reference",
    )

    # Build per-label agreement tables.
    records: List[Dict[str, Any]] = []
    mismatches_path = os.path.join(args.outdir, "mismatches.jsonl")
    mismatches_written = 0

    for lab in CHEXPERT_CONDITIONS:
        gt_vals: List[Optional[int]] = [g[lab] if g is not None else None for g in gt_list]
        pred_vals: List[Optional[int]] = pred[lab]

        # Exact match (treat None==None as match).
        exact_match = 0
        n = len(gt_vals)
        gt_counts = {"pos": 0, "neg": 0, "unc": 0, "blank": 0}
        pr_counts = {"pos": 0, "neg": 0, "unc": 0, "blank": 0}
        for a, b in zip(gt_vals, pred_vals):
            if a is None:
                gt_counts["blank"] += 1
            elif a == 1:
                gt_counts["pos"] += 1
            elif a == 0:
                gt_counts["neg"] += 1
            elif a == -1:
                gt_counts["unc"] += 1
            else:
                gt_counts["blank"] += 1

            if b is None:
                pr_counts["blank"] += 1
            elif b == 1:
                pr_counts["pos"] += 1
            elif b == 0:
                pr_counts["neg"] += 1
            elif b == -1:
                pr_counts["unc"] += 1
            else:
                pr_counts["blank"] += 1

            if (a is None and b is None) or (a is not None and b is not None and int(a) == int(b)):
                exact_match += 1

        exact_acc = float(exact_match / n) if n else float("nan")

        # Binary match (map -1 according to u_policy; map blank to 0)
        gt_bin = [_map_uncertainty_to_binary(v, u_policy=args.u_policy) for v in gt_vals]
        pr_bin = [_map_uncertainty_to_binary(v, u_policy=args.u_policy) for v in pred_vals]
        bin_acc = float(np.mean([1 if a == b else 0 for a, b in zip(gt_bin, pr_bin)])) if n else float("nan")

        records.append(
            {
                "label": lab,
                "N": int(n),
                "exact_acc": exact_acc,
                "binary_acc": bin_acc,
                "gt_pos": gt_counts["pos"],
                "gt_neg": gt_counts["neg"],
                "gt_unc": gt_counts["unc"],
                "gt_blank": gt_counts["blank"],
                "pred_pos": pr_counts["pos"],
                "pred_neg": pr_counts["neg"],
                "pred_unc": pr_counts["unc"],
                "pred_blank": pr_counts["blank"],
            }
        )

        # Write a few mismatches for this label
        if mismatches_written < int(args.max_mismatches):
            with open(mismatches_path, "a", encoding="utf-8") as mf:
                for j, (a, b) in enumerate(zip(gt_vals, pred_vals)):
                    if mismatches_written >= int(args.max_mismatches):
                        break
                    ok = (a is None and b is None) or (a is not None and b is not None and int(a) == int(b))
                    if ok:
                        continue
                    mf.write(
                        json.dumps(
                            {
                                "row_index": int(used_idx[j]),
                                "label": lab,
                                "gt": a,
                                "pred": b,
                                "u_policy": args.u_policy,
                                "gt_bin": int(_map_uncertainty_to_binary(a, u_policy=args.u_policy)),
                                "pred_bin": int(_map_uncertainty_to_binary(b, u_policy=args.u_policy)),
                                "text_preview": (texts[j][:400] if texts[j] else ""),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    mismatches_written += 1

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(args.outdir, "verify_per_label.csv"), index=False)

    summary = {
        "input": os.path.abspath(args.input),
        "N_used": int(len(texts)),
        "text_field": args.text_field,
        "gt_labels_field": args.gt_labels_field,
        "u_policy": args.u_policy,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "labeler_output": str(args.labeler_output),
        "outdir": os.path.abspath(args.outdir),
        "mean_exact_acc": float(df["exact_acc"].mean()),
        "mean_binary_acc": float(df["binary_acc"].mean()),
        "median_exact_acc": float(df["exact_acc"].median()),
        "median_binary_acc": float(df["binary_acc"].median()),
    }

    with open(os.path.join(args.outdir, "verify_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] wrote:")
    print("-", os.path.join(args.outdir, "verify_summary.json"))
    print("-", os.path.join(args.outdir, "verify_per_label.csv"))
    if os.path.exists(mismatches_path):
        print("-", mismatches_path)


if __name__ == "__main__":
    main()
