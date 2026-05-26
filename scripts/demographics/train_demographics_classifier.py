#!/usr/bin/env python3
"""Train a standalone demographic classifier (age/gender/race) for CXR images.

This is intentionally decoupled from the LLaVA-Rad training pipeline.

Expected JSON format (list of dicts), matching the existing loader schema used by
`llava/utils.py`:
  - image: relative image path under an `--image_root` folder
  - gender: e.g. "M"/"F" or "male"/"female" (optional)
  - anchor_age_group: 0/1/2 (optional)
  - anchor_age: numeric age (optional)
  - race_major: coarse race bucket string (optional)

For PadChest, the JSON used in this repo often has:
  - image or ImageID
  - gender
  - Age (numeric)

The script trains on a limited labeled subset (default 50K) and evaluates on a
limited test subset (default 2K), then optionally runs transfer evaluation on
PadChest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


RACE_MAJOR_LABELS_DEFAULT: Tuple[str, ...] = (
    "White",
    "Black or African American",
    "Asian",
    "Hispanic or Latino",
    "Other",
)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_json_any(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    # jsonl
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _normalize_gender(val: Any) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in {"m", "male", "0"}:
        return 0
    if s in {"f", "female", "1"}:
        return 1
    return None


def _age_to_group_from_age(age: Any) -> Optional[int]:
    if age is None:
        return None
    try:
        a = float(age)
    except Exception:
        return None
    if math.isnan(a) or a < 0:
        return None
    # Match bins used in scripts/stratify_results.py
    if a <= 44:
        return 0
    if a <= 65:
        return 1
    return 2


def _age_group_from_record(rec: Dict[str, Any]) -> Optional[int]:
    if "anchor_age_group" in rec and rec["anchor_age_group"] is not None:
        try:
            g = int(rec["anchor_age_group"])
        except Exception:
            g = None
        if g in {0, 1, 2}:
            return g

    # fallback numeric fields
    for k in ("anchor_age", "Age", "age"):
        if k in rec and rec[k] is not None:
            return _age_to_group_from_age(rec[k])
    return None


def _race_major_from_record(rec: Dict[str, Any]) -> Optional[str]:
    v = rec.get("race_major")
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    return s


def _image_relpath_from_record(rec: Dict[str, Any]) -> Optional[str]:
    # Prefer PadChest's ImageID if present (often the true filename incl .png).
    if isinstance(rec.get("ImageID"), str) and rec["ImageID"].strip():
        s = rec["ImageID"].strip()
        return s

    # Most LLaVA-style query JSONs use `image`.
    if isinstance(rec.get("image"), str) and rec["image"].strip():
        s = rec["image"].strip()
        # Match existing loader behavior in llava/utils.py
        if s.startswith("mimic/"):
            s = s[len("mimic/"):]
        if s.startswith("HAM10000/"):
            s = s[len("HAM10000/"):]
        return s
    if isinstance(rec.get("id"), str) and rec["id"].strip():
        # last resort: treat id as filename
        return rec["id"].strip()
    return None


@dataclass(frozen=True)
class TaskSpec:
    use_gender: bool = True
    use_age: bool = True
    use_race: bool = True


class DemographicsDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        image_root: Path,
        race_labels: Sequence[str],
        task_spec: TaskSpec,
        require_all_labels: bool,
        seed: int,
        limit: Optional[int],
        transform=None,
    ) -> None:
        self.image_root = image_root
        self.race_labels = list(race_labels)
        self.race_to_idx = {r: i for i, r in enumerate(self.race_labels)}
        self.task_spec = task_spec

        filtered: List[Dict[str, Any]] = []
        for rec in records:
            img_rel = _image_relpath_from_record(rec)
            if img_rel is None:
                continue

            gender = _normalize_gender(rec.get("gender")) if task_spec.use_gender else None
            age_g = _age_group_from_record(rec) if task_spec.use_age else None
            race = _race_major_from_record(rec) if task_spec.use_race else None

            if require_all_labels:
                if task_spec.use_gender and gender is None:
                    continue
                if task_spec.use_age and age_g is None:
                    continue
                if task_spec.use_race and (race is None or race not in self.race_to_idx):
                    continue

            rec2 = dict(rec)
            rec2["_image_rel"] = img_rel
            rec2["_gender"] = gender
            rec2["_age_group"] = age_g
            rec2["_race"] = race
            filtered.append(rec2)

        rng = np.random.default_rng(seed)
        if limit is not None and len(filtered) > limit:
            idx = rng.choice(len(filtered), size=limit, replace=False)
            filtered = [filtered[i] for i in idx.tolist()]

        self.records = filtered

        if transform is None:
            # Local import to avoid hard dependency for non-training uses.
            from torchvision import transforms

            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_path = (self.image_root / rec["_image_rel"]).resolve()
        image = Image.open(img_path).convert("RGB")
        x = self.transform(image)

        # Use -1 for missing labels so we can mask in loss/metrics.
        y_gender = rec.get("_gender")
        y_age = rec.get("_age_group")
        y_race = rec.get("_race")

        out: Dict[str, Any] = {
            "x": x,
            "gender": -1 if y_gender is None else int(y_gender),
            "age": -1 if y_age is None else int(y_age),
            "race": -1 if (y_race is None or y_race not in self.race_to_idx) else int(self.race_to_idx[y_race]),
        }
        return out


class _TimmVisionEncoder(nn.Module):
    def __init__(self, backbone: str) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        feat_dim = getattr(self.model, "num_features", None)
        if feat_dim is None:
            raise ValueError(f"Backbone {backbone} does not expose num_features")
        self.feat_dim = int(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _OpenCLIPVisionEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: str) -> None:
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model
        feat_dim = getattr(model, "embed_dim", None)
        if feat_dim is None and hasattr(model, "visual"):
            feat_dim = getattr(model.visual, "output_dim", None)
        if feat_dim is None:
            raise ValueError("Could not infer OpenCLIP feature dim (embed_dim/output_dim)")
        self.feat_dim = int(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # open_clip models expose encode_image
        return self.model.encode_image(x)


def _build_transform_for_backbone(backbone: str):
    if backbone.startswith("open_clip:"):
        import open_clip

        # open_clip uses a specific preprocess; return it
        # backbone format: open_clip:MODEL:PRETRAINED
        parts = backbone.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "OpenCLIP backbone format must be 'open_clip:MODEL:PRETRAINED'. "
                "Example: open_clip:ViT-B-16:hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
        _, model_name, pretrained = parts
        _, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        return preprocess

    # Default torchvision resize/normalize for timm backbones
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _build_encoder(backbone: str) -> nn.Module:
    if backbone.startswith("open_clip:"):
        parts = backbone.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "OpenCLIP backbone format must be 'open_clip:MODEL:PRETRAINED'. "
                "Example: open_clip:ViT-B-16:hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
        _, model_name, pretrained = parts
        return _OpenCLIPVisionEncoder(model_name=model_name, pretrained=pretrained)
    return _TimmVisionEncoder(backbone)


class MultiHeadDemographicModel(nn.Module):
    def __init__(self, backbone: str, num_race: int, task_spec: TaskSpec) -> None:
        super().__init__()
        self.task_spec = task_spec

        self.encoder = _build_encoder(backbone)
        feat_dim = getattr(self.encoder, "feat_dim", None)
        if feat_dim is None:
            raise ValueError("Encoder does not expose feat_dim")

        self.gender_head = nn.Linear(feat_dim, 1) if task_spec.use_gender else None
        self.age_head = nn.Linear(feat_dim, 3) if task_spec.use_age else None
        self.race_head = nn.Linear(feat_dim, num_race) if task_spec.use_race else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        out: Dict[str, torch.Tensor] = {}
        if self.gender_head is not None:
            out["gender_logits"] = self.gender_head(feat).squeeze(-1)
        if self.age_head is not None:
            out["age_logits"] = self.age_head(feat)
        if self.race_head is not None:
            out["race_logits"] = self.race_head(feat)
        return out


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_spec: TaskSpec,
) -> Dict[str, float]:
    model.eval()

    # Accuracies on available labels
    n_gender = 0
    n_gender_ok = 0
    n_age = 0
    n_age_ok = 0
    n_race = 0
    n_race_ok = 0

    for batch in loader:
        x = batch["x"].to(device)
        out = model(x)

        if task_spec.use_gender and "gender_logits" in out:
            y = batch["gender"].to(device)
            m = y >= 0
            if m.any():
                pred = (out["gender_logits"][m] >= 0).long()
                n_gender_ok += (pred == y[m].long()).sum().item()
                n_gender += m.sum().item()

        if task_spec.use_age and "age_logits" in out:
            y = batch["age"].to(device)
            m = y >= 0
            if m.any():
                pred = out["age_logits"][m].argmax(dim=-1)
                n_age_ok += (pred == y[m].long()).sum().item()
                n_age += m.sum().item()

        if task_spec.use_race and "race_logits" in out:
            y = batch["race"].to(device)
            m = y >= 0
            if m.any():
                pred = out["race_logits"][m].argmax(dim=-1)
                n_race_ok += (pred == y[m].long()).sum().item()
                n_race += m.sum().item()

    metrics: Dict[str, float] = {}
    if task_spec.use_gender:
        metrics["gender_acc"] = float(n_gender_ok / max(1, n_gender))
        metrics["gender_n"] = float(n_gender)
    if task_spec.use_age:
        metrics["age_acc"] = float(n_age_ok / max(1, n_age))
        metrics["age_n"] = float(n_age)
    if task_spec.use_race:
        metrics["race_acc"] = float(n_race_ok / max(1, n_race))
        metrics["race_n"] = float(n_race)

    # One scalar for quick comparisons
    accs = []
    if task_spec.use_gender and n_gender > 0:
        accs.append(metrics["gender_acc"])
    if task_spec.use_age and n_age > 0:
        accs.append(metrics["age_acc"])
    if task_spec.use_race and n_race > 0:
        accs.append(metrics["race_acc"])
    metrics["mean_task_acc"] = float(np.mean(accs)) if accs else 0.0

    return metrics


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    task_spec: TaskSpec,
    gender_weight: float,
    age_weight: float,
    race_weight: float,
) -> Dict[str, float]:
    model.train()

    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        out = model(x)

        loss = 0.0
        denom = 0.0

        if task_spec.use_gender and "gender_logits" in out:
            y = batch["gender"].to(device)
            m = y >= 0
            if m.any():
                l = bce(out["gender_logits"][m], y[m].float()).mean()
                loss = loss + gender_weight * l
                denom += gender_weight

        if task_spec.use_age and "age_logits" in out:
            y = batch["age"].to(device)
            m = y >= 0
            if m.any():
                l = ce(out["age_logits"][m], y[m].long()).mean()
                loss = loss + age_weight * l
                denom += age_weight

        if task_spec.use_race and "race_logits" in out:
            y = batch["race"].to(device)
            m = y >= 0
            if m.any():
                l = ce(out["race_logits"][m], y[m].long()).mean()
                loss = loss + race_weight * l
                denom += race_weight

        if denom == 0:
            continue

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1

    return {
        "train_loss": float(total_loss / max(1, n_batches)),
        "train_batches": float(n_batches),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--mimic_train_json", type=str, required=True)
    p.add_argument("--mimic_test_json", type=str, required=True)
    p.add_argument("--mimic_image_root", type=str, required=True)

    p.add_argument("--padchest_test_json", type=str, default=None)
    p.add_argument("--padchest_image_root", type=str, default=None)

    # Sampling
    p.add_argument("--train_limit", type=int, default=50_000)
    p.add_argument("--test_limit", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=42)

    # Tasks
    p.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "gender", "age", "race"],
        help="Which classifier to train. Use 'gender'/'age'/'race' for separate single-task classifiers.",
    )
    p.add_argument("--no_gender", action="store_true", help="Advanced: disable gender head (ignored if --task is set to a single task)")
    p.add_argument("--no_age", action="store_true", help="Advanced: disable age head (ignored if --task is set to a single task)")
    p.add_argument("--no_race", action="store_true", help="Advanced: disable race head (ignored if --task is set to a single task)")
    p.add_argument("--require_all_labels", action="store_true", help="Drop samples missing any enabled task label")

    p.add_argument(
        "--race_labels",
        type=str,
        default=",".join(RACE_MAJOR_LABELS_DEFAULT),
        help="Comma-separated race_major label set used for classification.",
    )

    # Model/training
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the vision backbone and train only classification head(s) (often better for cross-domain transfer).",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--gender_loss_weight", type=float, default=1.0)
    p.add_argument("--age_loss_weight", type=float, default=1.0)
    p.add_argument("--race_loss_weight", type=float, default=1.0)

    # Output
    p.add_argument("--output_dir", type=str, required=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "gender":
        task_spec = TaskSpec(use_gender=True, use_age=False, use_race=False)
    elif args.task == "age":
        task_spec = TaskSpec(use_gender=False, use_age=True, use_race=False)
    elif args.task == "race":
        task_spec = TaskSpec(use_gender=False, use_age=False, use_race=True)
    else:
        task_spec = TaskSpec(
            use_gender=not args.no_gender,
            use_age=not args.no_age,
            use_race=not args.no_race,
        )

    race_labels = [s.strip() for s in str(args.race_labels).split(",") if s.strip()]
    if task_spec.use_race and len(race_labels) < 2:
        raise ValueError("Need >=2 race labels when --no_race is not set")

    mimic_train = _read_json_any(Path(args.mimic_train_json))
    mimic_test = _read_json_any(Path(args.mimic_test_json))

    transform = _build_transform_for_backbone(args.backbone)

    train_ds = DemographicsDataset(
        mimic_train,
        image_root=Path(args.mimic_image_root),
        race_labels=race_labels,
        task_spec=task_spec,
        require_all_labels=args.require_all_labels,
        seed=args.seed,
        limit=args.train_limit,
        transform=transform,
    )
    test_ds = DemographicsDataset(
        mimic_test,
        image_root=Path(args.mimic_image_root),
        race_labels=race_labels,
        task_spec=task_spec,
        require_all_labels=False,
        seed=args.seed,
        limit=args.test_limit,
        transform=transform,
    )

    model = MultiHeadDemographicModel(
        backbone=args.backbone,
        num_race=len(race_labels),
        task_spec=task_spec,
    ).to(device)

    if args.freeze_backbone:
        for p in model.encoder.parameters():
            p.requires_grad = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    best = {"mean_task_acc": -1.0}

    run_cfg = {
        "seed": args.seed,
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
        "task_spec": task_spec.__dict__,
        "race_labels": race_labels,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    (out_dir / "config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optim,
            device,
            task_spec,
            gender_weight=args.gender_loss_weight,
            age_weight=args.age_loss_weight,
            race_weight=args.race_loss_weight,
        )
        test_metrics = _evaluate(model, test_loader, device, task_spec)

        metrics = {"epoch": epoch, **train_metrics, **{f"mimic_test_{k}": v for k, v in test_metrics.items()}}
        (out_dir / f"metrics_epoch_{epoch}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if test_metrics.get("mean_task_acc", 0.0) > best.get("mean_task_acc", -1.0):
            best = {"epoch": epoch, **test_metrics}
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "best": best,
                "config": run_cfg,
            }
            torch.save(ckpt, out_dir / "best.pt")

        print(
            f"[epoch {epoch}] train_loss={train_metrics['train_loss']:.4f} | "
            f"mimic_test_mean_task_acc={test_metrics['mean_task_acc']:.4f}"
        )

    (out_dir / "best_metrics_mimic_test.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    # Optional PadChest transfer eval
    if args.padchest_test_json and args.padchest_image_root:
        pad_ds = DemographicsDataset(
            _read_json_any(Path(args.padchest_test_json)),
            image_root=Path(args.padchest_image_root),
            race_labels=race_labels,
            task_spec=task_spec,
            require_all_labels=False,
            seed=args.seed,
            limit=None,
            transform=transform,
        )
        pad_loader = DataLoader(pad_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        pad_metrics = _evaluate(model, pad_loader, device, task_spec)
        (out_dir / "transfer_padchest_metrics.json").write_text(json.dumps(pad_metrics, indent=2), encoding="utf-8")
        print(f"[transfer padchest] mean_task_acc={pad_metrics['mean_task_acc']:.4f}")


if __name__ == "__main__":
    main()
