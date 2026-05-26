#!/usr/bin/env python3
"""Compute CheXpert label prevalence for reference vs prediction reports in a JSONL.

This is meant for sanity-checking whether label prevalence (a proxy for
clinical content) is preserved between ground-truth and generated reports.

Input JSONL must contain text fields for:
  - reference (ground truth)
  - prediction (model output)

It runs the official CheXpert rule-based labeler (stanfordmlgroup/chexpert-labeler)
via the wrapper already vendored in this repo under:
  llava/eval/rrg_eval/rrg_eval/chexpert.py

Outputs (in --outdir):
  - prevalence.json   : per-label prevalence + raw counts for refs/preds
  - prevalence.csv    : flat table version

Notes:
  - CheXpert labeler outputs values in {1, 0, -1, blank}.
  - You can map uncertainty (-1) to positive or negative via --u_policy.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import re
from typing import Any, Dict, Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# ----------------------------
# Minimal CheXpert labeler glue
# ----------------------------
# We intentionally DO NOT import `llava` here because `llava/__init__.py` pulls
# in `torch`, which is not available in lightweight labeling environments.
# Instead, we shell out to stanfordmlgroup/chexpert-labeler/label.py directly.

# CheXpert's 14 observations (order consistent with the labeler CSV)
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

# The common 5-label subset used in many papers
CONDITIONS_5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def _map_uncertainty_to_binary(
    x: Optional[float],
    u_policy: str = "u->0",
    blank_policy: str = "blank->0",
) -> int:
    """Map CheXpert outputs {1, 0, -1, blank} to {0,1}."""

    if x is None or (isinstance(x, float) and np.isnan(x)):
        if blank_policy == "blank->0":
            return 0
        # We don't currently support blank->ignore in this script.
        return 0
    if x == 1:
        return 1
    if x == 0:
        return 0
    if x == -1:
        return 1 if u_policy == "u->1" else 0
    return 0


class CheXpertLabeler:
    """Thin wrapper around stanfordmlgroup/chexpert-labeler/label.py."""

    def __init__(
        self,
        chexpert_root: str,
        python_exe: str = "python",
        extra_pythonpath: Optional[List[str]] = None,
        heartbeat_secs: float = 0.0,
        output_mode: str = "file",
        log_dir: Optional[str] = None,
    ):
        self.chexpert_root = os.path.abspath(chexpert_root)
        self.python_exe = python_exe
        self.extra_pythonpath = [p for p in (extra_pythonpath or []) if str(p).strip()]
        self.heartbeat_secs = float(heartbeat_secs or 0.0)
        self.output_mode = str(output_mode or "file").strip()
        self.log_dir = os.path.abspath(log_dir) if log_dir else None
        self.label_py = os.path.join(self.chexpert_root, "label.py")
        if not os.path.isfile(self.label_py):
            raise FileNotFoundError(f"Could not find label.py at {self.label_py}")

        if self.output_mode not in {"file", "inherit", "capture"}:
            raise ValueError("output_mode must be one of: file, inherit, capture")

    def forward(self, reports: List[str]) -> Dict[str, List[Optional[int]]]:
        with tempfile.TemporaryDirectory() as td:
            in_csv = os.path.join(td, "reports.csv")
            out_csv = os.path.join(td, "labeled_reports.csv")

            # CheXpert labeler expects a headerless, single-column CSV.
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

            # Python 3.6 compatibility: subprocess text mode differs.
            popen_kwargs = {"cwd": self.chexpert_root}
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

            # IMPORTANT: Avoid stdout/stderr=PIPE unless we also actively drain it.
            # The CheXpert labeler can be verbose; filling the pipe buffer can deadlock.
            log_path = None
            log_fh = None
            if self.output_mode == "inherit":
                popen_kwargs["stdout"] = None
                popen_kwargs["stderr"] = None
            elif self.output_mode == "capture":
                popen_kwargs["stdout"] = subprocess.PIPE
                popen_kwargs["stderr"] = subprocess.PIPE
            else:  # file
                log_dir = self.log_dir or os.getcwd()
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(
                    log_dir,
                    f"chexpert_labeler_{os.getpid()}_{int(time.time())}.log",
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
                raise RuntimeError(
                    "CheXpert labeler failed.\n" f"Return code: {proc.returncode}\n" + extra
                )

            df = pd.read_csv(out_csv)
            missing = [c for c in CHEXPERT_CONDITIONS if c not in df.columns]
            if missing:
                raise ValueError(f"CheXpert output missing expected columns: {missing}")

            out: Dict[str, List[Optional[int]]] = {}
            for lab in CHEXPERT_CONDITIONS:
                out[lab] = [None if pd.isna(v) else int(v) for v in df[lab].astype("float").tolist()]
            return out


def _tail_text_file(path: str, max_lines: int = 200) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[-max_lines:])
    except Exception as e:
        return f"<failed to read log file {path!r}: {e}>"


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

    # Progress bar: track completed batches (works for both sequential & parallel).
    def _progress_iter(total: int):
        if show_progress and tqdm is not None:
            return tqdm(total=total, desc=f"chexpert: {desc}")
        return None

    pbar = _progress_iter(n_batches)

    if num_workers == 1:
        for bi in range(n_batches):
            out = _run_one(bi)
            outs_by_idx[bi] = out
            if pbar is not None:
                pbar.update(1)
            elif show_progress and (bi == 0 or (bi + 1) % 10 == 0 or (bi + 1) == n_batches):
                end = min(n, (bi + 1) * batch_size)
                print(f"[chexpert] {desc}: batch {bi+1}/{n_batches} ({end}/{n})")
    else:
        # Each batch triggers an external process; threads are fine for coordinating I/O.
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


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _counts_from_raw(values: List[Optional[int]]) -> Dict[str, int]:
    # raw values are typically in {-1, 0, 1, None}
    c = {"pos": 0, "neg": 0, "unc": 0, "blank": 0}
    for v in values:
        if v is None:
            c["blank"] += 1
        elif v == 1:
            c["pos"] += 1
        elif v == 0:
            c["neg"] += 1
        elif v == -1:
            c["unc"] += 1
        else:
            # treat anything unexpected as blank
            c["blank"] += 1
    return c


def _prevalence_from_raw(values: List[Optional[int]], mode: str, u_policy: str) -> float:
    """Compute prevalence from raw CheXpert outputs.

    mode:
      - pos_only: count only definite positives (value==1). Everything else
        (neg=0, uncertain=-1, blank=None) contributes 0.
      - binary: convert raw values to binary using u_policy for uncertainty.
    """
    if len(values) == 0:
        return float("nan")

    if mode == "pos_only":
        ys = [1 if v == 1 else 0 for v in values]
        return float(np.mean(ys))

    if mode == "binary":
        ys = [_map_uncertainty_to_binary(v, u_policy=u_policy, blank_policy="blank->0") for v in values]
        return float(np.mean(ys))

    raise ValueError(f"Unknown prevalence mode: {mode!r}")


def _canonical_u_policy(u_policy: str) -> str:
    """Return the canonical u_policy string expected by chexpert utils.

    We accept shell-safe aliases to avoid needing quotes around `>`.
    """
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


def _detect_default_groupby_columns(columns: Iterable[str]) -> List[str]:
    """Pick a sensible default set of demographic columns if present."""
    candidates = ["gender", "age_group", "race", "ethnicity"]
    cols = set(columns)
    chosen = [c for c in candidates if c in cols]
    return chosen


def _subset_list(values: List[Any], idx: np.ndarray) -> List[Any]:
    # idx is an array of integer indices
    return [values[int(i)] for i in idx]


def _slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "x"


def _groupby_tag(groupby: List[str]) -> str:
    if not groupby:
        return "nogroup"
    return "__".join(_slugify(g) for g in groupby)


def main() -> None:
    if sys.version_info < (3, 6):
        raise SystemExit(
            "ERROR: This script requires Python >= 3.6. "
            "Your current Python is too old for required stdlib features."
        )

    ap = argparse.ArgumentParser(description="CheXpert label prevalence for refs vs preds from JSONL.")
    ap.add_argument("--jsonl", required=True, help="JSONL path with reference/prediction text fields")
    ap.add_argument("--ref_field", default="reference", help="Field name for ground-truth report text")
    ap.add_argument("--pred_field", default="prediction", help="Field name for generated report text")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N rows (useful for debugging speed/deps).",
    )

    ap.add_argument(
        "--groupby",
        nargs="*",
        default=None,
        help=(
            "Column(s) to stratify by (e.g., gender age_group). "
            "Default: auto-detect common demographic columns if present. "
            "Use --groupby (with no args) to disable stratification."
        ),
    )
    ap.add_argument(
        "--include_counts",
        action="store_true",
        help="Include raw pos/neg/unc/blank counts in groupwise CSV/JSON.",
    )

    ap.add_argument(
        "--prevalence_mode",
        default="pos_only",
        choices=["pos_only", "binary"],
        help=(
            "How to compute prevalence from CheXpert outputs. "
            "pos_only counts only definite positives (1). "
            "binary converts to 0/1 using --u_policy for uncertain (-1)."
        ),
    )

    ap.add_argument(
        "--chexpert_root",
        default=os.getenv("CHEXPERT_ROOT", ""),
        help="Path to stanfordmlgroup/chexpert-labeler clone (or set $CHEXPERT_ROOT)",
    )
    ap.add_argument(
        "--python_exe",
        default=sys.executable,
        help=(
            "Python interpreter to invoke chexpert label.py. "
            "Default: the current interpreter (recommended so dependencies match)."
        ),
    )
    ap.add_argument(
        "--labeler_pythonpath",
        nargs="*",
        default=[],
        help=(
            "Extra path(s) to prepend to PYTHONPATH only for running chexpert-labeler/label.py. "
            "Use this to point at a NegBio checkout if NegBio isn't pip-installed."
        ),
    )

    ap.add_argument(
        "--heartbeat_secs",
        type=float,
        default=30.0,
        help=(
            "Print a periodic heartbeat while waiting for chexpert-labeler to finish (helps when it runs silently). "
            "Set to 0 to disable."
        ),
    )

    ap.add_argument(
        "--labeler_output",
        default="file",
        choices=["file", "inherit", "capture"],
        help=(
            "How to handle chexpert-labeler stdout/stderr. "
            "file (default) writes combined logs to --labeler_logdir and avoids PIPE deadlocks; "
            "inherit prints labeler output directly to your terminal; "
            "capture stores output in-memory (can deadlock if very verbose)."
        ),
    )
    ap.add_argument(
        "--labeler_logdir",
        default=None,
        help=(
            "Directory to write chexpert-labeler logs when --labeler_output=file. "
            "Default: --outdir."
        ),
    )
    ap.add_argument(
        "--u_policy",
        default="u0",
        choices=["u0", "u1", "u_as_0", "u_as_1", "u_as_neg", "u_as_pos", "u->0", "u->1"],
        help=(
            "How to treat CheXpert 'uncertain' labels (-1) when converting to a binary prevalence. "
            "u0/u_as_neg maps -1 to 0; u1/u_as_pos maps -1 to 1. "
            "(Legacy values u->0/u->1 also accepted.)"
        ),
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help=(
            "If >0, run CheXpert labeler in batches (enables progress reporting, but may be slower due to repeated labeler startup). "
            "0 means run all reports in one labeler call."
        ),
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=(
            "Parallelism over batches. Only applies when --batch_size > 0. "
            "Each worker runs a separate chexpert-labeler subprocess, which can be CPU/RAM heavy."
        ),
    )
    ap.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress output (prints/tqdm).",
    )

    ap.add_argument(
        "--separate_labeling",
        action="store_true",
        help=(
            "Label references and predictions in separate chexpert-labeler runs. "
            "Default behavior labels them together in one run (faster)."
        ),
    )

    ap.add_argument("--outdir", default=None, help="Output directory (default: dirname(jsonl)/chexpert_prevalence)")
    args = ap.parse_args()

    if not args.chexpert_root:
        raise SystemExit("ERROR: --chexpert_root not provided and $CHEXPERT_ROOT is empty")

    try:
        # Only relevant when prevalence_mode=binary, but canonicalize regardless.
        args.u_policy = _canonical_u_policy(args.u_policy)
    except ValueError as e:
        raise SystemExit(f"ERROR: {e}")

    if args.outdir is None:
        # Include groupby in default output folder to avoid accidental overwrites.
        tag = _groupby_tag(list(args.groupby or []))
        args.outdir = os.path.join(
            os.path.dirname(os.path.abspath(args.jsonl)),
            f"chexpert_prevalence__{tag}",
        )
    _ensure_dir(args.outdir)

    if args.labeler_logdir is None:
        args.labeler_logdir = args.outdir

    rows = read_jsonl(args.jsonl)
    if len(rows) == 0:
        raise SystemExit("ERROR: empty JSONL")

    df_in = pd.DataFrame(rows)
    for k in (args.ref_field, args.pred_field):
        if k not in df_in.columns:
            raise SystemExit(f"ERROR: missing field in JSONL: {k}")

    if int(args.limit) > 0:
        df_in = df_in.head(int(args.limit)).copy()

    if args.groupby is None:
        args.groupby = _detect_default_groupby_columns(df_in.columns)
    # Special case: user passed `--groupby` with no args => disable stratification.
    # argparse sets args.groupby=[] in that case.

    if args.groupby:
        missing_g = [g for g in args.groupby if g not in df_in.columns]
        if missing_g:
            raise SystemExit(f"ERROR: --groupby columns not found in JSONL: {missing_g}")

    refs = [str(x or "") for x in df_in[args.ref_field].tolist()]
    preds = [str(x or "") for x in df_in[args.pred_field].tolist()]

    labeler = CheXpertLabeler(
        chexpert_root=args.chexpert_root,
        python_exe=args.python_exe,
        extra_pythonpath=list(args.labeler_pythonpath or []),
        heartbeat_secs=(0.0 if bool(args.no_progress) else float(args.heartbeat_secs)),
        output_mode=str(args.labeler_output),
        log_dir=str(args.labeler_logdir) if args.labeler_logdir else None,
    )

    show_progress = not bool(args.no_progress)

    # Raw outputs: dict[label] -> list[int|None], aligned with input row order
    if args.separate_labeling:
        raw_ref = _label_batched(
            labeler,
            refs,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            show_progress=show_progress,
            desc="reference",
        )
        raw_pred = _label_batched(
            labeler,
            preds,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            show_progress=show_progress,
            desc="prediction",
        )
    else:
        combined = refs + preds
        raw_combined = _label_batched(
            labeler,
            combined,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            show_progress=show_progress,
            desc="reference+prediction",
        )
        n_ref = len(refs)
        raw_ref = {lab: raw_combined[lab][:n_ref] for lab in CHEXPERT_CONDITIONS}
        raw_pred = {lab: raw_combined[lab][n_ref:] for lab in CHEXPERT_CONDITIONS}

    # Overall (unstratified)
    overall_records: List[Dict[str, Any]] = []
    for lab in CHEXPERT_CONDITIONS:
        ref_vals = raw_ref[lab]
        pred_vals = raw_pred[lab]
        ref_prev = _prevalence_from_raw(ref_vals, mode=args.prevalence_mode, u_policy=args.u_policy)
        pred_prev = _prevalence_from_raw(pred_vals, mode=args.prevalence_mode, u_policy=args.u_policy)
        rec: Dict[str, Any] = {
            "label": lab,
            "ref_prevalence": ref_prev,
            "pred_prevalence": pred_prev,
            "delta_pred_minus_ref": pred_prev - ref_prev,
            "N": int(len(rows)),
        }
        if args.include_counts:
            ref_counts = _counts_from_raw(ref_vals)
            pred_counts = _counts_from_raw(pred_vals)
            rec.update(
                {
                    "ref_pos": ref_counts["pos"],
                    "ref_neg": ref_counts["neg"],
                    "ref_unc": ref_counts["unc"],
                    "ref_blank": ref_counts["blank"],
                    "pred_pos": pred_counts["pos"],
                    "pred_neg": pred_counts["neg"],
                    "pred_unc": pred_counts["unc"],
                    "pred_blank": pred_counts["blank"],
                }
            )
        overall_records.append(rec)

    df_overall = pd.DataFrame.from_records(overall_records)

    # Aggregated (over-label) summaries.
    # Interpretation:
    #   - sum_*_prevalence: expected #positive labels per report (0..14)
    #   - mean_*_prevalence: average prevalence across labels (0..1)
    df_overall_summary = pd.DataFrame(
        [
            {
                "ref_prevalence_sum_over_labels": float(df_overall["ref_prevalence"].sum()),
                "pred_prevalence_sum_over_labels": float(df_overall["pred_prevalence"].sum()),
                "delta_sum_over_labels": float(df_overall["delta_pred_minus_ref"].sum()),
                "ref_prevalence_mean_over_labels": float(df_overall["ref_prevalence"].mean()),
                "pred_prevalence_mean_over_labels": float(df_overall["pred_prevalence"].mean()),
                "delta_mean_over_labels": float(df_overall["delta_pred_minus_ref"].mean()),
                "N": int(len(df_in)),
                "num_labels": int(len(CHEXPERT_CONDITIONS)),
            }
        ]
    )

    # Stratified by demographics
    group_records: List[Dict[str, Any]] = []
    if args.groupby:
        # Build stable group order
        group_df = df_in[args.groupby].copy()
        group_df = group_df.fillna("NA")
        keys_df = group_df.drop_duplicates()

        for _, key_row in keys_df.iterrows():
            mask = np.ones(len(df_in), dtype=bool)
            for g in args.groupby:
                mask &= group_df[g].astype(str).values == str(key_row[g])
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue

            for lab in CHEXPERT_CONDITIONS:
                ref_vals = _subset_list(raw_ref[lab], idx)
                pred_vals = _subset_list(raw_pred[lab], idx)
                ref_prev = _prevalence_from_raw(ref_vals, mode=args.prevalence_mode, u_policy=args.u_policy)
                pred_prev = _prevalence_from_raw(pred_vals, mode=args.prevalence_mode, u_policy=args.u_policy)

                rec = {g: str(key_row[g]) for g in args.groupby}
                rec.update(
                    {
                        "label": lab,
                        "ref_prevalence": ref_prev,
                        "pred_prevalence": pred_prev,
                        "delta_pred_minus_ref": pred_prev - ref_prev,
                        "N": int(idx.size),
                    }
                )
                if args.include_counts:
                    ref_counts = _counts_from_raw(ref_vals)
                    pred_counts = _counts_from_raw(pred_vals)
                    rec.update(
                        {
                            "ref_pos": ref_counts["pos"],
                            "ref_neg": ref_counts["neg"],
                            "ref_unc": ref_counts["unc"],
                            "ref_blank": ref_counts["blank"],
                            "pred_pos": pred_counts["pos"],
                            "pred_neg": pred_counts["neg"],
                            "pred_unc": pred_counts["unc"],
                            "pred_blank": pred_counts["blank"],
                        }
                    )
                group_records.append(rec)

    df_group = pd.DataFrame.from_records(group_records) if group_records else None

    df_group_summary = None
    if df_group is not None and args.groupby:
        df_group_summary = (
            df_group.groupby(args.groupby)
            .agg(
                {
                    "ref_prevalence": ["sum", "mean"],
                    "pred_prevalence": ["sum", "mean"],
                    "delta_pred_minus_ref": ["sum", "mean"],
                    "N": "max",
                }
            )
            .reset_index()
        )
        df_group_summary.columns = [
            "_".join([c for c in col if c]) if isinstance(col, tuple) else str(col) for col in df_group_summary.columns
        ]
        df_group_summary = df_group_summary.rename(
            columns={
                "ref_prevalence_sum": "ref_prevalence_sum_over_labels",
                "ref_prevalence_mean": "ref_prevalence_mean_over_labels",
                "pred_prevalence_sum": "pred_prevalence_sum_over_labels",
                "pred_prevalence_mean": "pred_prevalence_mean_over_labels",
                "delta_pred_minus_ref_sum": "delta_sum_over_labels",
                "delta_pred_minus_ref_mean": "delta_mean_over_labels",
                "N_max": "N",
            }
        )
        df_group_summary["num_labels"] = int(len(CHEXPERT_CONDITIONS))

    out = {
        "jsonl": os.path.abspath(args.jsonl),
        "ref_field": args.ref_field,
        "pred_field": args.pred_field,
        "prevalence_mode": args.prevalence_mode,
        "u_policy": args.u_policy,
        "groupby": args.groupby,
        "include_counts": bool(args.include_counts),
        "N": int(len(rows)),
        "overall": overall_records,
        "by_group": group_records,
        "overall_summary": df_overall_summary.to_dict(orient="records")[0],
        "by_group_summary": df_group_summary.to_dict(orient="records") if df_group_summary is not None else [],
        "subset_5": CONDITIONS_5,
    }

    with open(os.path.join(args.outdir, "prevalence.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    df_overall.to_csv(os.path.join(args.outdir, "prevalence.csv"), index=False)
    df_overall_summary.to_csv(os.path.join(args.outdir, "prevalence_summary.csv"), index=False)

    group_tag = _groupby_tag(list(args.groupby or []))
    if df_group is not None:
        df_group.to_csv(
            os.path.join(args.outdir, f"prevalence_by_group__{group_tag}.csv"),
            index=False,
        )
    if df_group_summary is not None:
        df_group_summary.to_csv(
            os.path.join(args.outdir, f"prevalence_by_group_summary__{group_tag}.csv"),
            index=False,
        )

    print("[done] wrote:")
    print("-", os.path.join(args.outdir, "prevalence.json"))
    print("-", os.path.join(args.outdir, "prevalence.csv"))
    print("-", os.path.join(args.outdir, "prevalence_summary.csv"))
    if df_group is not None:
        print("-", os.path.join(args.outdir, f"prevalence_by_group__{group_tag}.csv"))
    if df_group_summary is not None:
        print("-", os.path.join(args.outdir, f"prevalence_by_group_summary__{group_tag}.csv"))


if __name__ == "__main__":
    main()
