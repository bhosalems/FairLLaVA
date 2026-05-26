#!/usr/bin/env python3
"""Evaluate HAM10000 LLaVA-style predictions + fairness gaps.

Designed for files like:
  - query JSON: HAM10000_round3_qa_llava_test.json
  - prediction JSONL: merged_preds.jsonl (from scripts/infer_eval.py)

We treat this as a *classification* task over HAM labels:
  AKIEC, BCC, BKL, DF, MEL, NV, VASC

Metrics (overall + bootstrapped 95% CI), reported as percentages (0–100):
    - Accuracy
    - Macro-F1 (over present classes)
    - BalancedAcc (mean recall over present classes)

Fairness gaps (max-min) with bootstrapped 95% CI by:
  - SexEncoded (0/1) if available else gender (M/F/male/female)
  - anchor_age_group (0/1/2) if available

The script is robust to:
  - blank lines in JSONL
  - empty subgroup bins (they are ignored for gap computation)
  - predictions that don't contain a recognizable label (counted as incorrect)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from sacrebleu.metrics import BLEU as SacreBLEU
except Exception:
    SacreBLEU = None

try:
    from rouge_score import rouge_scorer as rouge_scorer_lib
except Exception:
    rouge_scorer_lib = None


HAM_LABELS: List[str] = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]


TEXT_METRICS: List[str] = ["BLEU-1", "BLEU-4", "ROUGE-L"]


TEXT_METRIC_BACKENDS = ("mimic", "simple")

# Simple mapping from common words to labels.
_TEXT_TO_LABEL: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bAKIEC\b", re.I), "AKIEC"),
    (re.compile(r"\bBCC\b", re.I), "BCC"),
    (re.compile(r"\bBKL\b", re.I), "BKL"),
    (re.compile(r"\bDF\b", re.I), "DF"),
    (re.compile(r"\bMEL\b", re.I), "MEL"),
    (re.compile(r"\bNV\b", re.I), "NV"),
    (re.compile(r"\bVASC\b", re.I), "VASC"),

    # Full-name-ish fallbacks
    (re.compile(r"melanoma", re.I), "MEL"),
    (re.compile(r"basal\s+cell\s+carcinoma", re.I), "BCC"),
    (re.compile(r"actinic\s+keratosis|bowen", re.I), "AKIEC"),
    (re.compile(r"benign\s+keratosis|seborrheic\s+keratosis", re.I), "BKL"),
    (re.compile(r"dermatofibroma", re.I), "DF"),
    (re.compile(r"nevus|naevus", re.I), "NV"),
    (re.compile(r"vascular", re.I), "VASC"),
]


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSONL: {path} line {line_no}: {e}") from e


def _load_query_index(query_json: Path) -> Dict[str, dict]:
    data = json.loads(query_json.read_text(encoding="utf-8"))
    out: Dict[str, dict] = {}
    for d in data:
        key = d.get("image_id") or d.get("id") or d.get("ImageID") or d.get("image")
        if key is None:
            continue
        out[str(key)] = d
    return out


def _parse_label_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    # pick the earliest match in text among all patterns
    best: Optional[Tuple[int, str]] = None
    for pat, lab in _TEXT_TO_LABEL:
        m = pat.search(text)
        if not m:
            continue
        pos = m.start()
        if best is None or pos < best[0]:
            best = (pos, lab)
    return best[1] if best else None


_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize_text(s: str) -> List[str]:
    if not s:
        return []
    return _TOK_RE.findall(str(s).lower())


def _ngram_counts(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    out: Dict[Tuple[str, ...], int] = {}
    if n <= 0:
        return out
    if len(tokens) < n:
        return out
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        out[ng] = out.get(ng, 0) + 1
    return out


def _sentence_bleu_n(pred: str, ref: str, *, n: int) -> float:
    """Sentence-level BLEU-n with simple add-one smoothing.

    This is intentionally lightweight and dependency-free; for HAM labels/text
    (short strings) it is sufficient for relative comparisons.
    """
    if n <= 0:
        return float("nan")

    pred_toks = _tokenize_text(pred)
    ref_toks = _tokenize_text(ref)

    c = len(pred_toks)
    r = len(ref_toks)
    if c == 0 or r == 0:
        return 0.0

    # Modified precisions with add-one smoothing
    log_p_sum = 0.0
    for k in range(1, n + 1):
        pred_counts = _ngram_counts(pred_toks, k)
        ref_counts = _ngram_counts(ref_toks, k)
        pred_total = sum(pred_counts.values())
        if pred_total == 0:
            # If prediction shorter than k, treat precision as ~0 with smoothing.
            match = 0
            p_k = (match + 1.0) / (0.0 + 1.0)
        else:
            match = 0
            for ng, cnt in pred_counts.items():
                match += min(cnt, ref_counts.get(ng, 0))
            p_k = (match + 1.0) / (pred_total + 1.0)
        log_p_sum += (1.0 / n) * math.log(max(p_k, 1e-12))

    # Brevity penalty
    if c < r:
        bp = math.exp(1.0 - (r / c))
    else:
        bp = 1.0

    return float(bp * math.exp(log_p_sum))


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    """LCS length (DP). Tokens are short; this is fine for one-pass precompute."""
    if not a or not b:
        return 0
    # Ensure b is the shorter dimension for less memory.
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0]
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(pred: str, ref: str) -> float:
    pred_toks = _tokenize_text(pred)
    ref_toks = _tokenize_text(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = _lcs_len(pred_toks, ref_toks)
    p = lcs / len(pred_toks)
    r = lcs / len(ref_toks)
    if p + r == 0:
        return 0.0
    return float((2.0 * p * r) / (p + r))


def _mimic_sentence_bleu(pred: str, ref: str, *, max_order: int) -> float:
    """Sentence BLEU using sacrebleu (same library used by rrg_eval/run.py).

    Returns a percentage score in [0, 100].
    """
    if SacreBLEU is None:
        raise RuntimeError("sacrebleu is not available (needed for text_metric_backend='mimic')")
    metric = SacreBLEU(max_ngram_order=max_order) if max_order != 4 else SacreBLEU()
    # sacrebleu expects references as a list-of-lists. For a single reference string:
    # references=[ [ref] ] in corpus mode; in sentence mode it takes references=[ref].
    return float(metric.sentence_score(hypothesis=str(pred or ""), references=[str(ref or "")]).score)


def _mimic_rouge_l_f1(pred: str, ref: str) -> float:
    """ROUGE-L F1 using rouge_score (same library used by rrg_eval/rrg_eval/rouge.py).

    Returns a percentage score in [0, 100].
    """
    if rouge_scorer_lib is None:
        raise RuntimeError("rouge_score is not available (needed for text_metric_backend='mimic')")
    scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=False)
    # rouge_score expects (target/reference, prediction)
    score = scorer.score(str(ref or ""), str(pred or ""))["rougeL"].fmeasure
    return float(100.0 * score)


def _normalize_sex(val) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if int(val) in (0, 1):
            return int(val)
    s = str(val).strip().lower()
    if s in {"0", "m", "male"}:
        return 0
    if s in {"1", "f", "female"}:
        return 1
    return None


def _normalize_age_group(val) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        iv = int(val)
        if iv in (0, 1, 2):
            return iv
    s = str(val).strip()
    try:
        iv = int(float(s))
        return iv if iv in (0, 1, 2) else None
    except Exception:
        return None


@dataclass(frozen=True)
class MetricCI:
    median: float
    ci_l: float
    ci_h: float


def _nan_ci() -> MetricCI:
    nan = float("nan")
    return MetricCI(nan, nan, nan)


def _confusion_counts(y_true: Sequence[str], y_pred: Sequence[Optional[str]], labels: Sequence[str]) -> Tuple[List[int], List[int], List[int]]:
    # returns (tp, fp, fn) per label (skips/ignores predictions not in labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    tp = [0] * len(labels)
    fp = [0] * len(labels)
    fn = [0] * len(labels)

    for t, p in zip(y_true, y_pred):
        if t not in idx:
            continue
        ti = idx[t]
        if p == t:
            tp[ti] += 1
        else:
            fn[ti] += 1
            if p in idx:
                fp[idx[p]] += 1

    return tp, fp, fn


def _accuracy(y_true: Sequence[str], y_pred: Sequence[Optional[str]]) -> float:
    if not y_true:
        return float("nan")
    correct = 0
    for t, p in zip(y_true, y_pred):
        if p == t:
            correct += 1
    return 100.0 * (correct / len(y_true))


def _macro_f1(y_true: Sequence[str], y_pred: Sequence[Optional[str]], labels: Sequence[str]) -> float:
    if not y_true:
        return float("nan")
    tp, fp, fn = _confusion_counts(y_true, y_pred, labels)
    f1s: List[float] = []
    for i, lab in enumerate(labels):
        support = tp[i] + fn[i]
        if support == 0:
            continue
        denom = (2 * tp[i] + fp[i] + fn[i])
        f1 = (2 * tp[i] / denom) if denom else 0.0
        f1s.append(f1)
    return 100.0 * (sum(f1s) / len(f1s)) if f1s else float("nan")


def _balanced_acc(y_true: Sequence[str], y_pred: Sequence[Optional[str]], labels: Sequence[str]) -> float:
    if not y_true:
        return float("nan")
    tp, _, fn = _confusion_counts(y_true, y_pred, labels)
    recalls: List[float] = []
    for i, lab in enumerate(labels):
        support = tp[i] + fn[i]
        if support == 0:
            continue
        recalls.append(tp[i] / support)
    return 100.0 * (sum(recalls) / len(recalls)) if recalls else float("nan")


def _confusion_matrix_with_unknown(
    y_true: Sequence[str],
    y_pred: Sequence[Optional[str]],
    labels: Sequence[str],
    *,
    unknown_label: str = "UNK",
) -> Dict[str, object]:
    """Return a full confusion matrix including an UNK column for unparsed predictions.

    Rows are true labels (always within `labels` for HAM), columns are predicted labels
    in `labels + [UNK]`.
    """
    true_idx = {lab: i for i, lab in enumerate(labels)}
    pred_labels = list(labels) + [unknown_label]
    pred_idx = {lab: i for i, lab in enumerate(pred_labels)}

    mat: List[List[int]] = [[0 for _ in pred_labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t not in true_idx:
            continue
        ti = true_idx[t]
        pi = pred_idx.get(p, pred_idx[unknown_label])
        mat[ti][pi] += 1

    # Per-class recall is row-normalized diagonal.
    per_class_recall: Dict[str, float] = {}
    supports: Dict[str, int] = {}
    for i, lab in enumerate(labels):
        row_sum = int(sum(mat[i]))
        supports[lab] = row_sum
        per_class_recall[lab] = (mat[i][i] / row_sum) if row_sum else float("nan")

    return {
        "labels_true": list(labels),
        "labels_pred": pred_labels,
        "matrix": mat,
        "supports": supports,
        "per_class_recall": per_class_recall,
    }


def _dump_group_confusions(
    out_path: Path,
    *,
    y_true: List[str],
    y_pred: List[Optional[str]],
    sex: List[Optional[int]],
    ageg: List[Optional[int]],
) -> None:
    def group_block(groups: List[Optional[int]]) -> Dict[str, object]:
        by_g: Dict[int, List[int]] = {}
        for i, g in enumerate(groups):
            if g is None:
                continue
            by_g.setdefault(int(g), []).append(i)

        out: Dict[str, object] = {}
        for g, idxs in sorted(by_g.items(), key=lambda kv: kv[0]):
            yt = [y_true[i] for i in idxs]
            yp = [y_pred[i] for i in idxs]
            out[str(g)] = {
                "n": len(idxs),
                "overall": _compute_metrics(yt, yp),
                "confusion": _confusion_matrix_with_unknown(yt, yp, HAM_LABELS),
            }
        return out

    payload = {
        "task": "ham10000",
        "labels": HAM_LABELS,
        "sex": {
            "group_field": "SexEncoded|gender",
            "groups": group_block(sex),
        },
        "age": {
            "group_field": "anchor_age_group",
            "groups": group_block(ageg),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[Optional[str]],
    *,
    extra: Optional[Dict[str, Sequence[float]]] = None,
) -> Dict[str, float]:
    out = {
        "Accuracy": _accuracy(y_true, y_pred),
        "Macro-F1": _macro_f1(y_true, y_pred, HAM_LABELS),
        "BalancedAcc": _balanced_acc(y_true, y_pred, HAM_LABELS),
    }
    if extra:
        for k, xs in extra.items():
            out[k] = _mean(xs)
    return out


def _bootstrap_ci(
    y_true: List[str],
    y_pred: List[Optional[str]],
    extra: Optional[Dict[str, List[float]]],
    *,
    n_resamples: int,
    seed: int,
) -> Dict[str, MetricCI]:
    import random

    n = len(y_true)
    if n == 0:
        keys = ["Accuracy", "Macro-F1", "BalancedAcc"] + list((extra or {}).keys())
        return {k: _nan_ci() for k in keys}
    if n == 1:
        m = _compute_metrics(y_true, y_pred, extra=extra)
        return {k: MetricCI(v, v, v) for k, v in m.items()}

    r = random.Random(seed)
    metric_keys = ["Accuracy", "Macro-F1", "BalancedAcc"] + list((extra or {}).keys())
    draws: Dict[str, List[float]] = {k: [] for k in metric_keys}

    idxs = list(range(n))
    for _ in range(n_resamples):
        sample = [r.choice(idxs) for _ in range(n)]
        yt = [y_true[i] for i in sample]
        yp = [y_pred[i] for i in sample]
        ex_s: Optional[Dict[str, List[float]]] = None
        if extra:
            ex_s = {k: [extra[k][i] for i in sample] for k in extra.keys()}
        m = _compute_metrics(yt, yp, extra=ex_s)
        for k, v in m.items():
            draws[k].append(float(v))

    out: Dict[str, MetricCI] = {}
    for k, xs in draws.items():
        xs.sort()
        lo = xs[int(0.025 * (len(xs) - 1))]
        hi = xs[int(0.975 * (len(xs) - 1))]
        med = _compute_metrics(y_true, y_pred, extra=extra)[k]
        out[k] = MetricCI(med, lo, hi)
    return out


def _fairness_gap_bootstrap(
    y_true: List[str],
    y_pred: List[Optional[str]],
    groups: List[Optional[int]],
    extra: Optional[Dict[str, List[float]]],
    *,
    n_resamples: int,
    seed: int,
) -> Dict[str, MetricCI]:
    import random

    n = len(y_true)
    if n == 0:
        keys = ["Accuracy", "Macro-F1", "BalancedAcc"] + list((extra or {}).keys())
        return {k: _nan_ci() for k in keys}

    # Point estimate
    metric_keys = ["Accuracy", "Macro-F1", "BalancedAcc"] + list((extra or {}).keys())

    def gap_for_sample(
        yt: List[str],
        yp: List[Optional[str]],
        gg: List[Optional[int]],
        ex: Optional[Dict[str, List[float]]],
    ) -> Dict[str, float]:
        by_g: Dict[int, List[int]] = {}
        for i, g in enumerate(gg):
            if g is None:
                continue
            by_g.setdefault(int(g), []).append(i)
        out: Dict[str, float] = {}
        for metric in metric_keys:
            vals: List[float] = []
            for g, inds in by_g.items():
                if not inds:
                    continue
                if metric in {"Accuracy", "Macro-F1", "BalancedAcc"}:
                    m = _compute_metrics([yt[i] for i in inds], [yp[i] for i in inds])[metric]
                else:
                    assert ex is not None
                    m = _mean([ex[metric][i] for i in inds])
                if not math.isfinite(m):
                    continue
                vals.append(float(m))
            out[metric] = (max(vals) - min(vals)) if len(vals) >= 2 else float("nan")
        return out

    point = gap_for_sample(y_true, y_pred, groups, extra)

    if n < 2:
        return {k: MetricCI(point[k], point[k], point[k]) for k in point}

    r = random.Random(seed)
    idxs = list(range(n))
    draws: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for _ in range(n_resamples):
        sample = [r.choice(idxs) for _ in range(n)]
        yt = [y_true[i] for i in sample]
        yp = [y_pred[i] for i in sample]
        gg = [groups[i] for i in sample]
        ex_s: Optional[Dict[str, List[float]]] = None
        if extra:
            ex_s = {k: [extra[k][i] for i in sample] for k in extra.keys()}
        g = gap_for_sample(yt, yp, gg, ex_s)
        for k, v in g.items():
            draws[k].append(float(v))

    out: Dict[str, MetricCI] = {}
    for k, xs in draws.items():
        xs.sort()
        lo = xs[int(0.025 * (len(xs) - 1))]
        hi = xs[int(0.975 * (len(xs) - 1))]
        out[k] = MetricCI(point[k], lo, hi)
    return out


def _equity_scaled_bootstrap(
    y_true: List[str],
    y_pred: List[Optional[str]],
    groups: List[Optional[int]],
    extra: Optional[Dict[str, List[float]]],
    *,
    n_resamples: int,
    seed: int,
) -> Dict[str, MetricCI]:
    """Compute ES = M_all / (1 + gap) with bootstrap CI.

    Gap is computed as max(group metric) - min(group metric), ignoring empty groups.
    """
    import random

    n = len(y_true)
    if n == 0:
        keys = ["Accuracy", "Macro-F1", "BalancedAcc"] + list((extra or {}).keys())
        return {k: _nan_ci() for k in keys}
    if n == 1:
        base = _compute_metrics(y_true, y_pred, extra=extra)
        # gap is always 0 with one sample (or single observed group)
        out = {}
        for k, v in base.items():
            es = float(v) / (1.0 + 0.0)
            out[k] = MetricCI(es, es, es)
        return out

    metric_keys = ["Accuracy", "Macro-F1", "BalancedAcc"] + list((extra or {}).keys())

    def gap_for_sample(
        yt: List[str],
        yp: List[Optional[str]],
        gg: List[Optional[int]],
        ex: Optional[Dict[str, List[float]]],
    ) -> Dict[str, float]:
        by_g: Dict[int, List[int]] = {}
        for i, g in enumerate(gg):
            if g is None:
                continue
            by_g.setdefault(int(g), []).append(i)

        gaps: Dict[str, float] = {k: 0.0 for k in metric_keys}
        for metric in metric_keys:
            vals: List[float] = []
            for idxs in by_g.values():
                if not idxs:
                    continue
                if metric in {"Accuracy", "Macro-F1", "BalancedAcc"}:
                    mt = _compute_metrics([yt[i] for i in idxs], [yp[i] for i in idxs])[metric]
                else:
                    assert ex is not None
                    mt = _mean([ex[metric][i] for i in idxs])
                if math.isfinite(mt):
                    vals.append(float(mt))
            if len(vals) >= 2:
                gaps[metric] = max(vals) - min(vals)
            else:
                gaps[metric] = 0.0
        return gaps

    # Point estimate
    overall = _compute_metrics(y_true, y_pred, extra=extra)
    gap_pt = gap_for_sample(y_true, y_pred, groups, extra)
    es_pt: Dict[str, float] = {}
    for k in overall.keys():
        g = float(gap_pt.get(k, 0.0))
        es_pt[k] = float(overall[k]) / (1.0 + g)

    r = random.Random(seed)
    idxs = list(range(n))
    draws: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for _ in range(n_resamples):
        sample = [r.choice(idxs) for _ in range(n)]
        yt = [y_true[i] for i in sample]
        yp = [y_pred[i] for i in sample]
        gg = [groups[i] for i in sample]
        ex_s: Optional[Dict[str, List[float]]] = None
        if extra:
            ex_s = {k: [extra[k][i] for i in sample] for k in extra.keys()}
        ov = _compute_metrics(yt, yp, extra=ex_s)
        gp = gap_for_sample(yt, yp, gg, ex_s)
        for k in metric_keys:
            g = float(gp.get(k, 0.0))
            draws[k].append(float(ov[k]) / (1.0 + g))

    out: Dict[str, MetricCI] = {}
    for k, xs in draws.items():
        xs.sort()
        lo = xs[int(0.025 * (len(xs) - 1))]
        hi = xs[int(0.975 * (len(xs) - 1))]
        out[k] = MetricCI(es_pt[k], lo, hi)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, required=True, help="Directory containing merged_preds.jsonl")
    ap.add_argument("--query_file", type=str, default=None, help="HAM10000 QA JSON (used if label/demographics not present in preds)")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: pred_dir)")
    ap.add_argument("--n_resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument(
        "--dump_group_confusions",
        action="store_true",
        help=(
            "Write per-group confusion matrices + per-class recall for classification metrics. "
            "Outputs: group_confusions.json in out_dir."
        ),
    )
    ap.add_argument(
        "--group_confusions_only",
        action="store_true",
        help=(
            "Only write group_confusions.json and exit (no bootstrap summaries). "
            "Useful to avoid overwriting existing summary files."
        ),
    )
    ap.add_argument(
        "--text_metric_backend",
        type=str,
        default="mimic",
        choices=list(TEXT_METRIC_BACKENDS),
        help=(
            "How to compute BLEU/ROUGE-L text metrics: "
            "'mimic' uses sacrebleu+rouge_score (same libs as MIMIC-CXR rrg_eval); "
            "'simple' uses a lightweight dependency-free approximation."
        ),
    )

    args = ap.parse_args()

    group_confusions_only = bool(args.group_confusions_only)

    text_backend = str(args.text_metric_backend).strip().lower()
    if text_backend == "mimic":
        # If deps are missing, fall back to the lightweight implementation.
        if SacreBLEU is None or rouge_scorer_lib is None:
            print(
                "[warn] text_metric_backend='mimic' requested but dependencies are missing; "
                "falling back to 'simple'."
            )
            text_backend = "simple"

    pred_dir = Path(args.pred_dir)
    pred_path = pred_dir / "merged_preds.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing: {pred_path}")

    out_dir = Path(args.out_dir) if args.out_dir else pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    q_index: Dict[str, dict] = {}
    if args.query_file:
        q_index = _load_query_index(Path(args.query_file))

    y_true: List[str] = []
    y_pred: List[Optional[str]] = []
    sex: List[Optional[int]] = []
    ageg: List[Optional[int]] = []

    # Precomputed per-example text similarity metrics (fast to bootstrap), stored as percentages.
    extra: Dict[str, List[float]] = {"BLEU-1": [], "BLEU-4": [], "ROUGE-L": []}

    for rec in _iter_jsonl(pred_path):
        rid = str(rec.get("id") or rec.get("image_id") or rec.get("ImageID") or rec.get("image") or "")
        qrec = q_index.get(rid) if q_index else None

        true_lab = rec.get("label") or (qrec.get("label") if qrec else None)
        if true_lab is None:
            # fallback: parse from reference text if it contains a code
            true_lab = _parse_label_from_text(str(rec.get("reference") or ""))
        if true_lab is None:
            # cannot score this example
            continue
        true_lab = str(true_lab).strip().upper()

        pred_lab = _parse_label_from_text(str(rec.get("prediction") or ""))
        pred_lab = pred_lab.upper() if pred_lab else None

        if not group_confusions_only:
            pred_text = str(rec.get("prediction") or "")
            ref_text = str(rec.get("reference") or "")
            if text_backend == "mimic":
                extra["BLEU-1"].append(_mimic_sentence_bleu(pred_text, ref_text, max_order=1))
                extra["BLEU-4"].append(_mimic_sentence_bleu(pred_text, ref_text, max_order=4))
                extra["ROUGE-L"].append(_mimic_rouge_l_f1(pred_text, ref_text))
            else:
                extra["BLEU-1"].append(100.0 * _sentence_bleu_n(pred_text, ref_text, n=1))
                extra["BLEU-4"].append(100.0 * _sentence_bleu_n(pred_text, ref_text, n=4))
                extra["ROUGE-L"].append(100.0 * _rouge_l_f1(pred_text, ref_text))

        sex_val = rec.get("SexEncoded")
        if sex_val is None and qrec is not None:
            sex_val = qrec.get("SexEncoded")
        if sex_val is None:
            sex_val = rec.get("gender") or (qrec.get("gender") if qrec else None)
        sex_n = _normalize_sex(sex_val)

        ag_val = rec.get("anchor_age_group")
        if ag_val is None and qrec is not None:
            ag_val = qrec.get("anchor_age_group")
        age_n = _normalize_age_group(ag_val)

        y_true.append(true_lab)
        y_pred.append(pred_lab)
        sex.append(sex_n)
        ageg.append(age_n)

    if group_confusions_only:
        out_path = out_dir / "group_confusions.json"
        _dump_group_confusions(out_path, y_true=y_true, y_pred=y_pred, sex=sex, ageg=ageg)
        print(f"Wrote: {out_path}")
        print(f"Scored n={len(y_true)} examples")
        return 0

    # Overall metrics
    overall_ci = _bootstrap_ci(y_true, y_pred, extra, n_resamples=args.n_resamples, seed=args.seed)

    # Write main.csv (similar shape to rrg_eval/run.py)
    metrics = ["Accuracy", "Macro-F1", "BalancedAcc"] + TEXT_METRICS
    main_csv = out_dir / "main.csv"
    lines: List[str] = []
    lines.append("," + ",".join(metrics))
    lines.append("median," + ",".join(str(overall_ci[m].median) for m in metrics))
    lines.append("ci_l," + ",".join(str(overall_ci[m].ci_l) for m in metrics))
    lines.append("ci_h," + ",".join(str(overall_ci[m].ci_h) for m in metrics))
    main_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Fairness gaps
    gap_sex = _fairness_gap_bootstrap(y_true, y_pred, sex, extra, n_resamples=args.n_resamples, seed=args.seed + 11)
    gap_age = _fairness_gap_bootstrap(y_true, y_pred, ageg, extra, n_resamples=args.n_resamples, seed=args.seed + 29)

    # Equity-scaled (ES) metrics
    es_sex = _equity_scaled_bootstrap(y_true, y_pred, sex, extra, n_resamples=args.n_resamples, seed=args.seed + 41)
    es_age = _equity_scaled_bootstrap(y_true, y_pred, ageg, extra, n_resamples=args.n_resamples, seed=args.seed + 59)

    gap_rows = []
    for attr, gci in [("sex", gap_sex), ("age", gap_age)]:
        for m in metrics:
            gap_rows.append(
                {
                    "attribute": attr,
                    "metric": m,
                    "gap": gci[m].median,
                    "gap_ci_l": gci[m].ci_l,
                    "gap_ci_h": gci[m].ci_h,
                }
            )

    gap_path = out_dir / "summary_fairness_gap.csv"
    header = ["attribute", "metric", "gap", "gap_ci_l", "gap_ci_h"]
    out_lines = [",".join(header)]
    for r in gap_rows:
        out_lines.append(",".join(str(r[h]) for h in header))
    gap_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    es_rows = []
    for attr, eci in [("sex", es_sex), ("age", es_age)]:
        for m in metrics:
            es_rows.append(
                {
                    "attribute": attr,
                    "metric": m,
                    "es": eci[m].median,
                    "es_ci_l": eci[m].ci_l,
                    "es_ci_h": eci[m].ci_h,
                }
            )

    es_path = out_dir / "summary_es_metrics.csv"
    header = ["attribute", "metric", "es", "es_ci_l", "es_ci_h"]
    out_lines = [",".join(header)]
    for r in es_rows:
        out_lines.append(",".join(str(r[h]) for h in header))
    es_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # Small JSON summary (useful for batch scripts)
    summary = {
        "n_scored": len(y_true),
        "overall": {m: overall_ci[m].__dict__ for m in metrics},
        "fairness_gap": gap_rows,
        "equity_scaled": es_rows,
    }
    (out_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.dump_group_confusions:
        out_path = out_dir / "group_confusions.json"
        _dump_group_confusions(out_path, y_true=y_true, y_pred=y_pred, sex=sex, ageg=ageg)
        print(f"Wrote: {out_path}")

    print(f"Scored n={len(y_true)} examples")
    print(f"Wrote: {main_csv}")
    print(f"Wrote: {gap_path}")
    print(f"Wrote: {es_path}")
    print(f"Wrote: {out_dir / 'summary_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
