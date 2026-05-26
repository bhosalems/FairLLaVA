#!/usr/bin/env python3
"""Convert HAM10000 QA JSON + HAM10000_metadata CSV into LLaVA-style training JSON.

Input:
- QA JSON like: /a2il/data/mbhosale/MrFair/HAM10000/round-1-QA_gen_HAM10000.json
  with entries containing: img_path, image_id, new_QA[{Q,A}, ...]
- Metadata CSV like: /a2il/data/mbhosale/MrFair/HAM10000/HAM10000_metadata
  with columns including: image_id, age, sex

Output:
- JSON list of samples, each with:
  - image: relative path under your chosen image_folder
  - conversations: [{from:"human", value:"<image>\n..."}, {from:"gpt", value:"..."}]
  - gender: "male"/"female" (or None)
  - anchor_age_group: 0/1/2 (or None)

This matches the expectation of llava/train/train.py DataCollatorForSupervisedDataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def age_to_group(age: Optional[float], cutoffs: Tuple[float, float] = (45.0, 66.0)) -> Optional[int]:
    """Map age to 3 buckets: 0=young, 1=middle, 2=old.

    This repo uses age_num=3 in several fairness configs, so we bucket ages
    into 3 groups. By default, we match PadChest-style bins: 0-44, 45-65, 66+.

    Args:
        age: raw age in years.
        cutoffs: (c0, c1) meaning:
            group 0 if age < c0
            group 1 if c0 <= age < c1
            group 2 if age >= c1
    """
    if age is None:
        return None
    c0, c1 = cutoffs
    if age < c0:
        return 0
    if age < c1:
        return 1
    return 2


def normalize_gender(sex: Any) -> Optional[str]:
    if sex is None:
        return None
    s = str(sex).strip().lower()
    if s in {"m", "male", "man"}:
        return "male"
    if s in {"f", "female", "woman", "w"}:
        return "female"
    return None


def gender_to_sex_encoded(gender: Optional[str]) -> Optional[int]:
    """PadChest-style SexEncoded: 0=male, 1=female."""
    if gender == "male":
        return 0
    if gender == "female":
        return 1
    return None


def strip_ham_prefix(img_path: str) -> str:
    p = img_path.strip()
    return p[len("HAM10000/") :] if p.startswith("HAM10000/") else p


def load_metadata_index(meta_csv: Path) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "image_id" not in r.fieldnames:
            raise ValueError(f"metadata missing image_id column: {r.fieldnames}")
        for row in r:
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                continue
            # Keep first occurrence; multiple images can share lesion_id, but image_id should be unique.
            if image_id not in idx:
                idx[image_id] = row
    return idx


def iter_qa_samples(qa_json: Path, flatten_all: bool) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    data = json.loads(qa_json.read_text(encoding="utf-8"))
    for ex in data:
        qas = ex.get("new_QA") or []
        if not isinstance(qas, list) or not qas:
            continue
        if flatten_all:
            for qa in qas:
                yield ex, qa
        else:
            yield ex, qas[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--qa-json",
        required=True,
        help="Path to round-1-QA_gen_HAM10000.json",
    )
    ap.add_argument(
        "--metadata-csv",
        required=True,
        help="Path to HAM10000_metadata (CSV)",
    )
    ap.add_argument(
        "--out-json",
        required=True,
        help="Where to write the converted LLaVA JSON",
    )
    ap.add_argument(
        "--write-split-files",
        action="store_true",
        help=(
            "If set, also write 3 separate JSON files for split=train/val/test. "
            "Defaults to using --out-json as a base name in the same directory."
        ),
    )
    ap.add_argument(
        "--out-train-json",
        default=None,
        help="Optional explicit path for the train split JSON.",
    )
    ap.add_argument(
        "--out-val-json",
        default=None,
        help="Optional explicit path for the val split JSON.",
    )
    ap.add_argument(
        "--out-test-json",
        default=None,
        help="Optional explicit path for the test split JSON.",
    )
    ap.add_argument(
        "--flatten-all-qa",
        action="store_true",
        help="If set, create one training row per QA pair (recommended).",
    )
    ap.add_argument(
        "--age-cutoffs",
        default="45,66",
        help="Two comma-separated age cutoffs for anchor_age_group (default: 45,66).",
    )
    ap.add_argument(
        "--keep-missing-demographics",
        action="store_true",
        help="If set, keep samples even if age/sex missing (anchor_age_group/gender will be None).",
    )
    ap.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Number of random entries to mark as split='test' (default: 1000).",
    )
    ap.add_argument(
        "--val-size",
        type=int,
        default=100,
        help="Number of random entries to mark as split='val' (default: 100).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for train/val/test split (default: 0).",
    )

    args = ap.parse_args()

    try:
        c0s, c1s = str(args.age_cutoffs).split(",")
        age_cutoffs = (float(c0s), float(c1s))
    except Exception:
        raise ValueError("--age-cutoffs must be like '45,66'")

    qa_json = Path(args.qa_json)
    meta_csv = Path(args.metadata_csv)
    out_json = Path(args.out_json)

    meta = load_metadata_index(meta_csv)

    out: List[Dict[str, Any]] = []
    missing_meta = 0
    missing_demo = 0

    for ex, qa in iter_qa_samples(qa_json, flatten_all=args.flatten_all_qa):
        image_id = (ex.get("image_id") or "").strip()
        img_path = (ex.get("img_path") or "").strip()
        if not image_id or not img_path:
            continue

        m = meta.get(image_id)
        if m is None:
            missing_meta += 1
            continue

        age = _safe_float(m.get("age"))
        gender = normalize_gender(m.get("sex"))
        anchor_age_group = age_to_group(age, cutoffs=age_cutoffs)
        sex_encoded = gender_to_sex_encoded(gender)

        if (gender is None or anchor_age_group is None) and not args.keep_missing_demographics:
            missing_demo += 1
            continue

        q = (qa.get("Q") or "").strip()
        a = (qa.get("A") or "").strip()
        if not q or not a:
            continue

        # LLaVA expects the image path relative to --image_folder.
        # Your QA file includes 'HAM10000/...', so strip that prefix.
        image_rel = strip_ham_prefix(img_path)

        out.append(
            {
                "image_id": image_id,
                "label": ex.get("label"),
                "image": image_rel,
                # PadChest-style demographic keys (some training code expects these names)
                "Age": age,
                "SexEncoded": sex_encoded,
                "gender": gender,
                "anchor_age_group": anchor_age_group,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{q}\n"},
                    {"from": "gpt", "value": a},
                ],
            }
        )

    # Assign split per entry, similar to PadChest JSONs.
    # We shuffle indices for reproducible random selection.
    rng = random.Random(args.seed)
    indices = list(range(len(out)))
    rng.shuffle(indices)

    test_n = max(0, min(int(args.test_size), len(indices)))
    val_n = max(0, min(int(args.val_size), max(0, len(indices) - test_n)))

    test_idx = set(indices[:test_n])
    val_idx = set(indices[test_n : test_n + val_n])

    for i, row in enumerate(out):
        if i in test_idx:
            row["split"] = "test"
        elif i in val_idx:
            row["split"] = "val"
        else:
            row["split"] = "train"

    # Optionally write separate files per split.
    if args.write_split_files or args.out_train_json or args.out_val_json or args.out_test_json:
        by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for row in out:
            s = row.get("split")
            if s in by_split:
                by_split[s].append(row)

        if args.out_train_json or args.out_val_json or args.out_test_json:
            train_path = Path(args.out_train_json) if args.out_train_json else None
            val_path = Path(args.out_val_json) if args.out_val_json else None
            test_path = Path(args.out_test_json) if args.out_test_json else None
        else:
            # Derive base name from --out-json.
            # If it ends with '_train'/'_val'/'_test'/'_all', strip that suffix.
            stem = out_json.stem
            for suffix in ("_train", "_val", "_test", "_all"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            base = out_json.with_name(stem)
            train_path = base.with_name(base.name + "_train.json")
            val_path = base.with_name(base.name + "_val.json")
            test_path = base.with_name(base.name + "_test.json")

        # Avoid overwriting the combined out_json (common if user names it '*_train.json').
        if train_path is not None and train_path.resolve() == out_json.resolve():
            train_path = out_json.with_name(out_json.stem + "_train_split.json")
        if val_path is not None and val_path.resolve() == out_json.resolve():
            val_path = out_json.with_name(out_json.stem + "_val_split.json")
        if test_path is not None and test_path.resolve() == out_json.resolve():
            test_path = out_json.with_name(out_json.stem + "_test_split.json")

        for split_name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
            if path is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(by_split[split_name], indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {len(by_split[split_name])} {split_name} samples to {path}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(out)} samples to {out_json}")
    print(f"Split counts: train={len(out) - test_n - val_n} val={val_n} test={test_n} (seed={args.seed})")
    if args.write_split_files and out_json.stem.endswith("_train"):
        print(
            "Note: --out-json name ends with '_train' but contains all splits; "
            "prefer naming it '*_all.json' to avoid confusion."
        )
    if missing_meta:
        print(f"Skipped {missing_meta} QA rows: no metadata match")
    if missing_demo and not args.keep_missing_demographics:
        print(f"Skipped {missing_demo} QA rows: missing age/sex")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
