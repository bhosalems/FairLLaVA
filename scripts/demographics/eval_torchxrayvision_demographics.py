#!/usr/bin/env python3
"""Evaluate TorchXRayVision demographic models on MIMIC-CXR and PadChest.

Runs *pretrained* demographic predictors from torchxrayvision:
- Sex:   xrv.baseline_models.mira.SexModel
- Age:   xrv.baseline_models.riken.AgeModel
- Race:  xrv.baseline_models.emory_hiti.RaceModel (targets: ["Asian","Black","White"])

This script reports:
- In-domain performance on a labeled MIMIC-CXR JSON
- Transfer performance on a labeled PadChest JSON

Input JSONs are the same schema you already use in this repo (see llava/utils.py loaders):
- MIMIC: `image` starts with "mimic/"; `gender`; `anchor_age`; `anchor_age_group`; `race_major`
- PadChest: often has both `ImageID` (.png) and `image` (.jpg); this script tries both and uses the first that exists.
    Other fields: `gender`; `Age`; `anchor_age_group`

Notes
- TorchXRayVision models expect *single-channel* images and a specific normalization:
  `xrv.datasets.normalize(img, 255)` then mean across channels.
- Race evaluation is restricted to the intersection of dataset labels and the model targets.

Example:
python scripts/demographics/eval_torchxrayvision_demographics.py \
  --mimic_json /a2il/data/.../chat_test_MIMIC_CXR_all_dem_clean_2K.json \
  --mimic_image_root /data_local1/mbhosale/MrFair/mimc-cxr-jpeg/files \
  --padchest_json /a2il/data/.../Padchest/test_findings_100.json \
  --padchest_image_root /a2il/data/mbhosale/MrFair/Padchest/images \
  --out_json results/demographics/torchxrayvision_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _read_json_any(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _normalize_gender_to_idx(val: Any) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in {"m", "male", "0"}:
        return 1  # male as positive class
    if s in {"f", "female", "1"}:
        return 0
    return None


def _age_years_from_record(rec: Dict[str, Any]) -> Optional[float]:
    for k in ("anchor_age", "Age", "age"):
        if k in rec and rec[k] is not None:
            try:
                a = float(rec[k])
            except Exception:
                continue
            if math.isnan(a) or a < 0:
                continue
            return a
    return None


def _age_group_from_age_years(age: Optional[float]) -> Optional[int]:
    if age is None:
        return None
    if age <= 44:
        return 0
    if age <= 65:
        return 1
    return 2


def _mimic_image_rel(rec: Dict[str, Any]) -> Optional[str]:
    s = rec.get("image")
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    if s.startswith("mimic/"):
        s = s[len("mimic/"):]
    return s


def _padchest_image_rel(rec: Dict[str, Any]) -> Optional[str]:
    # Return a candidate relative image path. Some exports store ImageID as .png but
    # the actual available files might be .jpg (or vice versa). The caller should
    # prefer an existing path when multiple candidates are available.
    for k in ("ImageID", "image"):
        s = rec.get(k)
        if isinstance(s, str) and s.strip():
            return s.strip()
    return None


def _resolve_image_path(rec: Dict[str, Any], image_root: Path, is_mimic: bool) -> Optional[Path]:
    if is_mimic:
        rel = _mimic_image_rel(rec)
        if rel is None:
            return None
        p = (image_root / rel).resolve()
        return p if p.exists() else None

    # PadChest: try both ImageID and image, choose the first existing.
    candidates: List[str] = []
    for k in ("ImageID", "image"):
        s = rec.get(k)
        if isinstance(s, str) and s.strip():
            candidates.append(s.strip())

    for rel in candidates:
        p = (image_root / rel).resolve()
        if p.exists():
            return p
    return None


def _race_to_emory_targets(rec: Dict[str, Any]) -> Optional[int]:
    # Emory model targets are fixed to ["Asian","Black","White"].
    race = rec.get("race_major")
    if race is None:
        return None
    s = str(race).strip().lower()
    if not s:
        return None
    if s.startswith("asian"):
        return 0
    if s.startswith("black"):
        return 1
    if s.startswith("white"):
        return 2
    return None


def _load_and_preprocess_cxr(image_path: Path, xrv) -> torch.Tensor:
    # Use PIL -> numpy to avoid torchvision.io issues.
    from PIL import Image

    img = Image.open(image_path)
    img = img.convert("RGB")
    arr = np.asarray(img)

    # Normalize to [-1024, 1024] range, then convert to single channel.
    arr = xrv.datasets.normalize(arr, 255)
    if arr.ndim == 3:
        arr = arr.mean(2)
    # shape (1,H,W)
    arr = arr[None, ...]

    transform = xrv.datasets.XRayCenterCrop()
    arr = transform(arr)
    arr = xrv.datasets.XRayResizer(224)(arr)

    t = torch.from_numpy(arr).float()
    return t


@torch.no_grad()
def _predict_sex(model, x: torch.Tensor) -> Tuple[int, float]:
    # returns (pred_idx where 1=male, 0=female), prob_male
    out = model(x[None, ...])
    out = out.detach().cpu().float().squeeze(0)

    if out.numel() == 1:
        logit = float(out.item())
        prob_male = 1.0 / (1.0 + math.exp(-logit))
        pred = 1 if prob_male >= 0.5 else 0
        return pred, prob_male

    # multiclass
    logits = out.view(-1)
    probs = torch.softmax(logits, dim=0).numpy()

    # Map model.targets to male prob if possible
    targets = getattr(model, "targets", None)
    if isinstance(targets, (list, tuple)) and len(targets) == len(probs):
        tl = [str(t).strip().lower() for t in targets]
        if "male" in tl:
            prob_male = float(probs[tl.index("male")])
        elif "m" in tl:
            prob_male = float(probs[tl.index("m")])
        else:
            prob_male = float(probs[-1])
    else:
        prob_male = float(probs[-1])

    pred = int(np.argmax(probs))
    # Convert pred to our convention if targets available
    if isinstance(targets, (list, tuple)) and len(targets) == len(probs):
        tl = [str(t).strip().lower() for t in targets]
        if tl[pred] in {"male", "m"}:
            pred_idx = 1
        elif tl[pred] in {"female", "f"}:
            pred_idx = 0
        else:
            pred_idx = 1 if prob_male >= 0.5 else 0
    else:
        pred_idx = 1 if prob_male >= 0.5 else 0

    return pred_idx, prob_male


@torch.no_grad()
def _predict_age_years(model, x: torch.Tensor) -> Optional[float]:
    out = model(x[None, ...])
    out = out.detach().cpu().float().squeeze(0)
    if out.numel() == 1:
        return float(out.item())
    # If model outputs a distribution, return expected value over class indices
    v = out.view(-1)
    probs = torch.softmax(v, dim=0)
    idx = torch.arange(len(probs), dtype=torch.float32)
    return float((probs * idx).sum().item())


@torch.no_grad()
def _predict_race_probs(model, x: torch.Tensor) -> np.ndarray:
    out = model(x[None, ...])
    out = out.detach().cpu().float().squeeze(0)
    probs = torch.softmax(out.view(-1), dim=0).numpy()
    return probs


def _eval_split(
    name: str,
    records: Sequence[Dict[str, Any]],
    image_root: Path,
    is_mimic: bool,
    tasks: Sequence[str],
    device: torch.device,
) -> Dict[str, Any]:
    import torchxrayvision as xrv

    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None

    metrics: Dict[str, Any] = {"split": name, "n_total": len(records)}

    sex_model = None
    age_model = None
    race_model = None

    if "sex" in tasks:
        print(f"[{name}] Loading TorchXRayVision SexModel (may download weights)...")
        sex_model = xrv.baseline_models.mira.SexModel().to(device).eval()
        metrics["sex_model_targets"] = getattr(sex_model, "targets", None)

    if "age" in tasks:
        print(f"[{name}] Loading TorchXRayVision AgeModel (may download weights)...")
        age_model = xrv.baseline_models.riken.AgeModel().to(device).eval()

    if "race" in tasks:
        print(f"[{name}] Loading TorchXRayVision RaceModel (may download weights)...")
        race_model = xrv.baseline_models.emory_hiti.RaceModel().to(device).eval()
        metrics["race_model_targets"] = getattr(race_model, "targets", None)

    y_sex: List[int] = []
    p_sex: List[float] = []
    y_age: List[float] = []
    p_age: List[float] = []
    y_age_group: List[int] = []
    p_age_group: List[int] = []
    y_race: List[int] = []
    p_race: List[np.ndarray] = []

    n_missing_img = 0
    n_missing_labels = {"sex": 0, "age": 0, "race": 0}

    it = records
    if tqdm is not None:
        task_str = ",".join(tasks)
        it = tqdm(records, desc=f"eval:{name} [{task_str}]", unit="img", dynamic_ncols=True)

    for i, rec in enumerate(it):
        img_path = _resolve_image_path(rec, image_root=image_root, is_mimic=is_mimic)
        if img_path is None:
            n_missing_img += 1
            continue

        x = _load_and_preprocess_cxr(img_path, xrv).to(device)

        if sex_model is not None:
            y = _normalize_gender_to_idx(rec.get("gender"))
            if y is None:
                n_missing_labels["sex"] += 1
            else:
                pred, prob_male = _predict_sex(sex_model, x)
                y_sex.append(int(y))
                p_sex.append(float(prob_male))

        if age_model is not None:
            yv = _age_years_from_record(rec)
            if yv is None:
                n_missing_labels["age"] += 1
            else:
                pv = _predict_age_years(age_model, x)
                if pv is not None:
                    y_age.append(float(yv))
                    p_age.append(float(pv))
                    yg = _age_group_from_age_years(yv)
                    pg = _age_group_from_age_years(pv)
                    if yg is not None and pg is not None:
                        y_age_group.append(int(yg))
                        p_age_group.append(int(pg))

        if race_model is not None:
            y = _race_to_emory_targets(rec) if is_mimic else None
            if y is None:
                n_missing_labels["race"] += 1
            else:
                probs = _predict_race_probs(race_model, x)
                y_race.append(int(y))
                p_race.append(probs)

        if tqdm is not None and (i % 50 == 0):
            it.set_postfix(
                missing_img=n_missing_img,
                sex=len(y_sex),
                age=len(y_age),
                race=len(y_race),
                refresh=False,
            )

    metrics["n_missing_img"] = n_missing_img
    metrics["n_missing_label_sex"] = n_missing_labels["sex"]
    metrics["n_missing_label_age"] = n_missing_labels["age"]
    metrics["n_missing_label_race"] = n_missing_labels["race"]

    # Sex metrics
    if y_sex:
        y_arr = np.asarray(y_sex)
        p_arr = np.asarray(p_sex)
        pred = (p_arr >= 0.5).astype(int)
        metrics["sex_n"] = int(len(y_arr))
        metrics["sex_acc"] = float((pred == y_arr).mean())
        try:
            from sklearn.metrics import roc_auc_score

            metrics["sex_auc"] = float(roc_auc_score(y_arr, p_arr))
        except Exception:
            pass

    # Age metrics
    if y_age:
        y_arr = np.asarray(y_age)
        p_arr = np.asarray(p_age)
        metrics["age_n"] = int(len(y_arr))
        metrics["age_mae"] = float(np.mean(np.abs(p_arr - y_arr)))
        # Bin accuracy
        if y_age_group:
            yg = np.asarray(y_age_group)
            pg = np.asarray(p_age_group)
            metrics["age_group_n"] = int(len(yg))
            metrics["age_group_acc"] = float((pg == yg).mean())

    # Race metrics (only for MIMIC)
    if y_race:
        y_arr = np.asarray(y_race)
        p_mat = np.stack(p_race, axis=0)
        metrics["race_n"] = int(len(y_arr))
        pred = np.argmax(p_mat, axis=1)
        metrics["race_acc"] = float((pred == y_arr).mean())
        try:
            from sklearn.metrics import balanced_accuracy_score, roc_auc_score

            metrics["race_bal_acc"] = float(balanced_accuracy_score(y_arr, pred))
            metrics["race_auc_ovr"] = float(roc_auc_score(y_arr, p_mat, multi_class="ovr"))
        except Exception:
            pass

    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--mimic_json", type=str, required=True)
    p.add_argument("--mimic_image_root", type=str, required=True)

    p.add_argument("--padchest_json", type=str, required=True)
    p.add_argument("--padchest_image_root", type=str, required=True)

    p.add_argument(
        "--tasks",
        type=str,
        default="sex,age,race",
        help="Comma-separated tasks to evaluate from {sex,age,race}. Race eval only runs when labels are present (MIMIC).",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional cap per split for quick smoke tests")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--out_json", type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    tasks = [t.strip().lower() for t in str(args.tasks).split(",") if t.strip()]
    for t in tasks:
        if t not in {"sex", "age", "race"}:
            raise ValueError(f"Unknown task: {t}")

    device = torch.device(args.device)

    mimic_records = _read_json_any(Path(args.mimic_json))
    pad_records = _read_json_any(Path(args.padchest_json))

    if args.limit is not None:
        mimic_records = mimic_records[: args.limit]
        pad_records = pad_records[: args.limit]

    out: Dict[str, Any] = {
        "mimic_json": args.mimic_json,
        "mimic_image_root": args.mimic_image_root,
        "padchest_json": args.padchest_json,
        "padchest_image_root": args.padchest_image_root,
        "tasks": tasks,
        "device": str(device),
    }

    out["mimic"] = _eval_split(
        name="mimic",
        records=mimic_records,
        image_root=Path(args.mimic_image_root),
        is_mimic=True,
        tasks=tasks,
        device=device,
    )
    out["padchest"] = _eval_split(
        name="padchest",
        records=pad_records,
        image_root=Path(args.padchest_image_root),
        is_mimic=False,
        tasks=tasks,
        device=device,
    )

    print(json.dumps(out, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
