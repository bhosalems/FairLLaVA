#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheXpert evaluation from JSONL of predictions & references.

Input: newline-delimited JSON objects with at least:
  - "prediction": generated report string
  - "reference" : reference report string
(Other keys like id/query/greenscore are ignored.)

This script:
  1) Reads all predictions & references
  2) Runs the official CheXpert labeler (stanfordmlgroup/chexpert-labeler/label.py)
  3) Maps uncertainty using --u_policy (u->0 or u->1)
  4) Computes macro/micro Precision/Recall/F1 for CheXpert-14 and CheXpert-5
  5) Prints a compact JSON of results (optionally writes to --out)

Requirements:
  - git clone https://github.com/stanfordmlgroup/chexpert-labeler.git
  - Its dependencies installed (NegBio, NLTK, BLLIP etc. per their README)
"""

import os
import sys
import csv
import json
import argparse
import tempfile
import subprocess
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

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

CONDITIONS_5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def read_jsonl_reports(path: str) -> Tuple[List[str], List[str]]:
    preds, refs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            preds.append(obj["prediction"])
            refs.append(obj["reference"])
    assert len(preds) == len(refs) and len(preds) > 0, "No samples or mismatched preds/refs"
    return preds, refs


def run_chexpert_labeler(reports: List[str], chexpert_root: str, python_exe: str = "python") -> pd.DataFrame:
    """Run stanfordmlgroup/chexpert-labeler/label.py on a list of raw reports and return its CSV as DataFrame."""
    label_py = os.path.join(os.path.abspath(chexpert_root), "label.py")
    if not os.path.isfile(label_py):
        raise FileNotFoundError(f"label.py not found at: {label_py}")

    with tempfile.TemporaryDirectory() as td:
        in_csv = os.path.join(td, "reports.csv")
        out_csv = os.path.join(td, "labeled.csv")

        # The CLI expects a single-column CSV with no header (one report per row).
        with open(in_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in reports:
                w.writerow([r if r is not None else ""])

        cmd = [python_exe, label_py, "--reports_path", in_csv, "--output_path", out_csv]
        proc = subprocess.run(cmd, cwd=chexpert_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0 or not os.path.exists(out_csv):
            raise RuntimeError(
                f"CheXpert labeler failed.\nReturn code: {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        df = pd.read_csv(out_csv)
    return df


def map_uncertainty_to_binary(x: Optional[float], u_policy: str = "u->0", blank_policy: str = "blank->0") -> int:
    """
    CheXpert values: 1 (positive), 0 (negative), -1 (uncertain), NaN/None (blank)
    Map to binary {0,1}.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0 if blank_policy == "blank->0" else 0
    if x == 1:
        return 1
    if x == 0:
        return 0
    if x == -1:
        return 1 if u_policy == "u->1" else 0
    return 0


def extract_label_matrix(df: pd.DataFrame, u_policy: str) -> np.ndarray:
    missing = [c for c in CHEXPERT_CONDITIONS if c not in df.columns]
    if missing:
        raise ValueError(f"CheXpert CSV missing columns: {missing}")

    cols = []
    for lab in CHEXPERT_CONDITIONS:
        col = df[lab].astype("float").tolist()
        cols.append([map_uncertainty_to_binary(v, u_policy=u_policy) for v in col])
    # shape [N, L]
    return np.array(cols, dtype=np.int32).T


def prf_report(Yp: np.ndarray, Yg: np.ndarray, label_names: List[str]) -> Dict:
    """Return micro/macro P/R/F1 + (optionally) per-class (omitted by default for brevity)."""
    assert Yp.shape == Yg.shape
    # micro over flattened (treat each label instance equally)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true=Yg.reshape(-1), y_pred=Yp.reshape(-1), average="micro", zero_division=0
    )
    # macro over labels (mean of per-label scores)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true=Yg, y_pred=Yp, average="macro", zero_division=0
    )
    return {
        "micro": {"precision": float(micro_p), "recall": float(micro_r), "f1": float(micro_f1)},
        "macro": {"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f1)},
    }


def main():
    ap = argparse.ArgumentParser(description="CheXpert eval (Precision/Recall/F1) from JSONL predictions & references.")
    ap.add_argument("--jsonl", required=True, help="Path to JSONL with fields: prediction, reference")
    ap.add_argument("--chexpert_root", required=False, default=os.getenv("CHEXPERT_ROOT", ""),
                    help="Path to stanfordmlgroup/chexpert-labeler clone (or set $CHEXPERT_ROOT)")
    ap.add_argument("--python_exe", default="python", help="Python to run label.py")
    ap.add_argument("--u_policy", default="u->0", choices=["u->0", "u->1"], help="Uncertainty mapping")
    ap.add_argument("--out", default="", help="Optional path to write JSON results")
    args = ap.parse_args()

    if not args.chexpert_root:
        print("ERROR: --chexpert_root not provided and $CHEXPERT_ROOT is empty.", file=sys.stderr)
        sys.exit(1)

    preds, refs = read_jsonl_reports(args.jsonl)

    # Label predictions & references
    df_pred = run_chexpert_labeler(preds, args.chexpert_root, args.python_exe)
    df_ref  = run_chexpert_labeler(refs,  args.chexpert_root, args.python_exe)

    Yp14 = extract_label_matrix(df_pred, args.u_policy)  # [N,14]
    Yg14 = extract_label_matrix(df_ref,  args.u_policy)

    # Full 14-label report
    rep14 = prf_report(Yp14, Yg14, CHEXPERT_CONDITIONS)

    # 5-label subset
    idx5 = [CHEXPERT_CONDITIONS.index(l) for l in CONDITIONS_5]
    rep5 = prf_report(Yp14[:, idx5], Yg14[:, idx5], CONDITIONS_5)

    results = {
        "u_policy": args.u_policy,
        "N": int(Yp14.shape[0]),
        "CheXpert14": rep14,
        "CheXpert5": rep5,
    }

    txt = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()