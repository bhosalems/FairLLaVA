# chexpert_eval.py
# Drop-in analogue of your CheXbert evaluator, but for the CheXpert rule-based labeler.
# It shells out to stanfordmlgroup/chexpert-labeler/label.py and parses the output.
# (c) You. MIT license, same spirit as upstream repos.

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import os
import csv
import tempfile
import subprocess
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

# ----------------------------
# Label set and small helpers
# ----------------------------

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

def _map_uncertainty_to_binary(x: Optional[float], u_policy: str = "u->0", blank_policy: str = "blank->0") -> int:
    """
    Map CheXpert outputs {1, 0, -1, blank} to {0,1}.

    u_policy   : "u->0" (default) or "u->1"
    blank_policy: "blank->0" (default) or "blank->ignore" (ignored instances are dropped later)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        if blank_policy == "blank->0":
            return 0
        else:
            # We'll handle 'ignore' by filtering paired arrays before scoring
            return 0  # placeholder; caller should filter if using ignore
    if x == 1:
        return 1
    if x == 0:
        return 0
    if x == -1:
        return 1 if u_policy == "u->1" else 0
    # Any other stray value → negative
    return 0

def _classification_report(y_pred: List[int], y_true: List[int], target_name: str) -> Dict:
    """Return a compact dict for one label (precision/recall/f1/support)."""
    p, r, f1, s = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average="binary", zero_division=0
    )
    return {
        "label": target_name,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "support": int(s),
    }

def _aggregate_report(Y_pred: np.ndarray, Y_true: np.ndarray, label_names: List[str]) -> Dict:
    """
    Multi-label aggregation returning:
      - per-class PRF/support
      - micro/macro averages (F1, precision, recall)
    Shapes: (N, L)
    """
    assert Y_pred.shape == Y_true.shape
    N, L = Y_pred.shape
    per_class = []
    for j in range(L):
        per_class.append(_classification_report(Y_pred[:, j].tolist(), Y_true[:, j].tolist(), label_names[j]))

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true=Y_true.reshape(-1), y_pred=Y_pred.reshape(-1), average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true=Y_true, y_pred=Y_pred, average="macro", zero_division=0
    )
    return {
        "per_class": per_class,
        "micro avg": {"precision": float(micro_p), "recall": float(micro_r), "f1-score": float(micro_f1)},
        "macro avg": {"precision": float(macro_p), "recall": float(macro_r), "f1-score": float(macro_f1)},
    }

# --------------------------------
# CheXpert labeler wrapper (CLI)
# --------------------------------

class CheXpertLabeler:
    """
    Wrapper around the official CheXpert rule-based labeler (stanfordmlgroup/chexpert-labeler).
    It writes reports to a temp CSV, runs label.py, and loads the resulting CSV.

    Requirements (see upstream README):
      - NegBio cloned and added to PYTHONPATH
      - NLTK data downloaded
      - BLLIP parser model fetched (GENIA+PubMed)
    """

    def __init__(self, chexpert_root: str, python_exe: str = "python"):
        """
        chexpert_root: path to the cloned chexpert-labeler repo (contains label.py)
        python_exe   : python interpreter to use when calling label.py
        """
        self.chexpert_root = os.path.abspath(chexpert_root)
        self.python_exe = python_exe
        self.label_py = os.path.join(self.chexpert_root, "label.py")
        if not os.path.isfile(self.label_py):
            raise FileNotFoundError(f"Could not find label.py at {self.label_py}")

    def _run_labeler(self, reports: List[str]) -> pd.DataFrame:
        """
        Run CheXpert labeler on a list of raw report strings and return the labeled DataFrame.
        """
        with tempfile.TemporaryDirectory() as td:
            in_csv = os.path.join(td, "reports.csv")
            out_csv = os.path.join(td, "labeled_reports.csv")

            # CheXpert labeler expects a headerless, single-column CSV (quoted if multiline/commas)
            with open(in_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for r in reports:
                    # Protect commas/newlines by quoting; csv.writer handles it
                    w.writerow([r if r is not None else ""])

            cmd = [
                self.python_exe,
                self.label_py,
                "--reports_path", in_csv,
                "--output_path", out_csv
            ]
            # Let user have their environment; labeler relies on PYTHONPATH (NegBio), NLTK, BLLIP
            proc = subprocess.run(cmd, cwd=self.chexpert_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0 or (not os.path.exists(out_csv)):
                raise RuntimeError(
                    f"CheXpert labeler failed.\nReturn code: {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )

            df = pd.read_csv(out_csv)
            return df

    def forward(self, reports: List[str]) -> Dict[str, List[Optional[int]]]:
        """
        Return a dict: {label_name -> list of values in {-1,0,1, nan}} for each report.
        """
        df = self._run_labeler(reports)
        # Labeler column names are the observation titles:
        # Keep only the 14 official columns and preserve order.
        missing = [c for c in CHEXPERT_CONDITIONS if c not in df.columns]
        if missing:
            raise ValueError(f"CheXpert output missing expected columns: {missing}")

        out = {}
        for lab in CHEXPERT_CONDITIONS:
            # Cast to float to allow NaN, then to Python scalars
            out[lab] = [None if pd.isna(v) else int(v) for v in df[lab].astype("float").tolist()]
        return out

# ------------------------------
# Bootstrap helpers (optional)
# ------------------------------

def _compute_statistic(preds: List[int], refs: List[int], average: str):
    def _fn(indices):
        p = [preds[i] for i in indices]
        r = [refs[i] for i in indices]
        _, _, f1, _ = precision_recall_fscore_support(y_pred=p, y_true=r, average=average, zero_division=0)
        return f1
    return _fn

def bootstrap_ci(preds: List[int], refs: List[int], average: str = "micro", n_resamples: int = 500, seed: int = 3):
    rng = np.random.default_rng(seed)
    N = len(preds)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, N, size=N)
        samples.append(_compute_statistic(preds, refs, average)(idx))
    lo, med, hi = np.percentile(samples, [2.5, 50, 97.5])
    return {"median": float(med), "ci_l": float(lo), "ci_h": float(hi)}

# -----------------------------------------------------
# Public API: evaluate CheXpert on preds vs references
# -----------------------------------------------------

def evaluate_chexpert(
    preds: List[str],
    refs: List[str],
    chexpert_root: str,
    python_exe: str = "python",
    u_policies: Tuple[str, str] = ("u->0", "u->1"),   # two standard variants
    include_breakdown: bool = False,
    compute_bootstrap_ci: bool = False,
    bootstrap_resamples: int = 500,
) -> Dict[str, Dict]:
    """
    Label both predictions and references with the CheXpert labeler and compute:
      - CheXpert-14 micro/macro F1
      - CheXpert-5 micro/macro F1
      - For two uncertainty mappings: u->0 (default) and u->1 (CheXpert+ style)

    Returns a dict:
      {
        "14_u0": {...}, "5_u0": {...},
        "14_u1": {...}, "5_u1": {...}
      }
    """
    assert len(preds) == len(refs), "preds and refs must be the same length"

    labeler = CheXpertLabeler(chexpert_root=chexpert_root, python_exe=python_exe)
    # Label both sets with one call each
    out_pred = labeler.forward(preds)
    out_ref  = labeler.forward(refs)

    results = {}

    for tag, u_policy in zip(("u0", "u1"), u_policies):
        # Build Y_pred/Y_true matrices with chosen uncertainty mapping
        mat_pred = []
        mat_true = []
        for lab in CHEXPERT_CONDITIONS:
            p_col = [_map_uncertainty_to_binary(v, u_policy=u_policy, blank_policy="blank->0") for v in out_pred[lab]]
            g_col = [_map_uncertainty_to_binary(v, u_policy=u_policy, blank_policy="blank->0") for v in out_ref[lab]]
            mat_pred.append(p_col)
            mat_true.append(g_col)

        # [N, L]
        Yp = np.array(mat_pred, dtype=np.int32).T
        Yg = np.array(mat_true, dtype=np.int32).T

        # Full 14-class report
        rep14 = _aggregate_report(Yp, Yg, CHEXPERT_CONDITIONS)

        # 5-class subset
        idx5 = [CHEXPERT_CONDITIONS.index(l) for l in CONDITIONS_5]
        rep5  = _aggregate_report(Yp[:, idx5], Yg[:, idx5], CONDITIONS_5)

        # Optional bootstrap CIs on micro/macro F1 (14 labels, flattened)
        if compute_bootstrap_ci:
            micro_ci = bootstrap_ci(Yp.reshape(-1).tolist(), Yg.reshape(-1).tolist(), average="binary", n_resamples=bootstrap_resamples)
            rep14["micro avg"].update(micro_ci)

        results[f"14_{tag}"] = rep14
        results[f"5_{tag}"]  = rep5

        if include_breakdown:
            # Add quick per-class breakdown array (already in rep14["per_class"])
            pass

    return results

# --------------------
# Convenience wrapper
# --------------------

def evaluate2(preds: List[str], refs: List[str], chexpert_root: str) -> Dict[str, Dict]:
    """
    Minimal wrapper to mirror your CheXbert evaluate2 signature.
    Returns the dict of reports for 14/5 labels and u->0/u->1 mappings.
    """
    return evaluate_chexpert(preds, refs, chexpert_root=chexpert_root)

# --------------------
# Demo
# --------------------
if __name__ == "__main__":
    hyps = [
        "No pleural effusion. Normal heart size.",
        "Normal heart size.",
        "Increased mild pulmonary edema and left basal atelectasis.",
        "Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.",
        "Elevated left hemidiaphragm and blunting of the left costophrenic angle although no definite evidence of pleural effusion seen on the lateral view.",
    ]
    refs = [
        "No pleural effusions.",
        "Enlarged heart.",
        "No evidence of pneumonia. Stable cardiomegaly.",
        "Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.",
        "No acute cardiopulmonary process.",
    ]

    # Point this to your cloned chexpert-labeler repo
    CHEXPERT_ROOT = "/path/to/chexpert-labeler"

    res = evaluate2(hyps, refs, chexpert_root=CHEXPERT_ROOT)
    # Pretty-print a couple of top-level numbers
    for k in ("14_u0", "14_u1", "5_u0", "5_u1"):
        micro_f1 = res[k]["micro avg"]["f1-score"]
        macro_f1 = res[k]["macro avg"]["f1-score"]
        print(k, {"micro_f1": micro_f1, "macro_f1": macro_f1})