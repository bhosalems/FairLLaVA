#!/usr/bin/env python3
"""
Counterfactual matching analysis for LLaVA-Rad report generation.

- Joins:
  (A) generated reports JSONL (merged_demographics.jsonl) containing:
      id, subject_id, (optionally study_id), reference, prediction, greenscore, gender, etc.
  (B) meta test JSON (chat_test_*.json) containing:
      id, image (e.g., "mimic/p16/...jpg"), view, chexpert_labels dict, conversations

- Resolves image paths for MIMIC-CXR-JPEG "files/" layout:
    image_root=/.../mimic-cxr-jpeg/files
    meta image="mimic/p16/p165.../s566.../xxx.jpg"
  => /.../files/p16/p165.../s566.../xxx.jpg

- Loads BiomedCLIP-CXR encoder using the BiomedCLIP config JSON shipped with LLaVA-Rad:
  config keys: embed_dim, vision_cfg.timm_model_name, text_cfg.hf_model_name, image_size=518
  (NOT ViT-B-16). :contentReference[oaicite:1]{index=1}

- Encodes images -> embeddings
- Forms morphology-matched F vs M pairs within CheXpert constraint buckets (and optionally same view)
- Computes per-pair:
    abs/signed delta greenscore
    abs/signed delta RadGraph-F1 (computed from reference vs prediction texts)

Outputs:
  - merged_joined.csv
  - per_sample_metrics.csv
  - image_embeddings.npy
  - matched_pairs.csv
  - summary.json
"""

import os
import re
import json
import math
import hashlib
import argparse
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as T


# -------------------------
# IO
# -------------------------
def read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# -------------------------
# Join helpers
# -------------------------
def parse_subject_study_from_id(id_str):
    # "16508811_56646773" -> (16508811, 56646773)
    try:
        a, b = str(id_str).split("_")
        return int(a), int(b)
    except Exception:
        return (None, None)


def resolve_image_path(img_rel, image_root, prefix_to_strip="mimic/"):
    """
    Robust resolver for MIMIC-CXR-JPEG files layout.
    Tries:
      1) absolute path if exists
      2) join(image_root, img_rel)
      3) if img_rel startswith prefix_to_strip, join(image_root, img_rel[len(prefix_to_strip):])
    """
    if not isinstance(img_rel, str) or len(img_rel) == 0:
        return None

    if os.path.isabs(img_rel) and os.path.exists(img_rel):
        return img_rel

    # Try direct
    p1 = os.path.join(image_root, img_rel)
    if os.path.exists(p1):
        return p1

    # Try stripping "mimic/"
    if prefix_to_strip and img_rel.startswith(prefix_to_strip):
        stripped = img_rel[len(prefix_to_strip):]
        p2 = os.path.join(image_root, stripped)
        if os.path.exists(p2):
            return p2

    return p1  # best guess; will be filtered later


# -------------------------
# CheXpert utils
# -------------------------
CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices"
]


def flatten_chexpert_dict(d):
    out = {}
    if not isinstance(d, dict):
        d = {}
    for k in CHEXPERT_LABELS:
        out[k] = d.get(k, None)
    return out


def chexpert_state(v):
    """Map label value -> {pos,neg,unc,na}."""
    if v is None:
        return "na"
    if isinstance(v, str) and v.strip() == "":
        return "na"
    try:
        fv = float(v)
    except Exception:
        return "na"
    if fv == 1.0:
        return "pos"
    if fv == 0.0:
        return "neg"
    if fv == -1.0:
        return "unc"
    return "na"


def build_bucket_keys(row, constraint_labels, match_view=False, view_col="view",
                      na_policy="match", allowed_states=("pos", "neg", "unc", "na")):
    """Return one or more bucket keys for a row.

    na_policy:
      - match: NA must match exactly (NA is treated as a real bucket value)
      - drop: drop rows that have any NA in constraints
      - wildcard: treat NA as "unknown" and DO NOT constrain on it.
                 Implementation: expand NA into all non-NA allowed states so the
                 sample can match any bucket on that label.
    """
    allowed_states = [s for s in allowed_states]
    allowed_states_set = set(allowed_states)

    # In wildcard mode, NA expands into all non-NA allowed states.
    wildcard_states = [s for s in allowed_states if s != "na"]

    choices = []
    for lb in constraint_labels:
        st = chexpert_state(row.get(lb, None))
        if st not in allowed_states_set:
            return []
        if na_policy == "drop" and st == "na":
            return []
        if na_policy == "wildcard" and st == "na":
            if len(wildcard_states) == 0:
                return []
            choices.append(wildcard_states)
        else:
            choices.append([st])

    keys = []
    for states in itertools.product(*choices) if choices else [()]:
        key = tuple(states)
        if match_view:
            key = key + (str(row.get(view_col, "")),)
        keys.append(key)
    return keys


def _states_vector(row, constraint_labels, allowed_states_set: set[str]) -> list[str] | None:
    states = []
    for lb in constraint_labels:
        st = chexpert_state(row.get(lb, None))
        if st not in allowed_states_set:
            return None
        states.append(st)
    return states


def _encode_states(states: list[str]) -> np.ndarray:
    # pos=0, neg=1, unc=2, na=3
    m = {"pos": 0, "neg": 1, "unc": 2, "na": 3}
    return np.array([m.get(s, 3) for s in states], dtype=np.int8)


def format_constraint_state_for_pair(ra, rb, constraint_labels, mode: str = "full") -> str:
    """Format a human-readable constraint_state string for a matched pair.

    mode:
      - full: include all constraint labels and emit NA explicitly, e.g. pos|pos|na|na
      - omit_na: omit labels where either side is NA, e.g. pos|pos
    """
    parts = []
    for lb in constraint_labels:
        sa = chexpert_state(ra.get(lb, None))
        sb = chexpert_state(rb.get(lb, None))
        if mode == "omit_na" and (sa == "na" or sb == "na"):
            continue
        # If they differ (shouldn't happen under current bucket logic), make it explicit.
        if sa != sb:
            parts.append(f"{sa}/{sb}")
        else:
            parts.append(sa)
    out = "|".join(parts)
    return out if out else "none"


# -------------------------
# Text fields
# -------------------------
def extract_text_from_row(row, field):
    """
    field can be:
      - direct key: "reference" / "prediction" / "impression"
      - "conversations:gpt" -> last conversation item with from=="gpt"
    """
    if field.startswith("conversations:"):
        role = field.split(":", 1)[1]
        convs = row.get("conversations", None)
        if not isinstance(convs, list):
            return ""
        for item in reversed(convs):
            if isinstance(item, dict) and item.get("from", "") == role:
                return str(item.get("value", "") or "")
        return ""
    return str(row.get(field, "") or "")


# -------------------------
# RadGraph scorer
# -------------------------
def build_radgraph_scorer(reward_level="partial"):
    """
    Uses `radgraph` package (StanfordAIMI/RRG_scorers download).
    Returns per-sample f1 via reward_list[i][0].
    """
    from radgraph import F1RadGraph
    from radgraph.radgraph import CACHE_DIR
    from huggingface_hub import hf_hub_download

    class F1RadGraphv2(F1RadGraph):
        def __init__(self, reward_level, **kwargs):
            self._download_radgraph()
            super().__init__(reward_level, **kwargs)
            assert reward_level in ["simple", "partial", "complete"]

        def _download_radgraph(self):
            tar = os.path.join(CACHE_DIR, "radgraph.tar.gz")
            if not os.path.exists(tar):
                os.makedirs(CACHE_DIR, exist_ok=True)
                hf_hub_download(
                    repo_id="StanfordAIMI/RRG_scorers",
                    filename="radgraph.tar.gz",
                    revision="d97745aa136e5beb927da7e768e99de6ae807902",
                    local_dir=CACHE_DIR,
                )

        def forward(self, refs, hyps):
            if isinstance(hyps, str):
                hyps = [hyps]
            if isinstance(refs, str):
                refs = [refs]
            assert len(refs) == len(hyps)

            n = len(hyps)
            empty = [i for i in range(n) if (len(hyps[i]) == 0) or (len(refs[i]) == 0)]
            non_empty = n - len(empty)

            report_list = (
                [hyps[i] for i in range(n) if i not in empty] +
                [refs[i] for i in range(n) if i not in empty]
            )
            inference = self.radgraph(report_list)

            reward_list = []
            ne = 0
            for i in range(n):
                if i in empty:
                    reward_list.append((0., 0., 0.))
                    continue
                hyp_ann = inference[str(ne)]
                ref_ann = inference[str(ne + non_empty)]
                reward_list.append(compute_reward(hyp_ann, ref_ann, self.reward_level))
                ne += 1

            mean = np.mean(reward_list, axis=0)
            return {"f1-radgraph": float(mean[0])}, reward_list

    def exact_entity_token_match_reward(hyp_ann, ref_ann):
        candidates = []
        for ann in [hyp_ann, ref_ann]:
            cand = []
            for ent in ann["entities"].values():
                cand.append((ent["tokens"], ent["label"]))
            candidates.append(set(cand))
        hyp_set, ref_set = candidates
        prec = (sum(1 for x in hyp_set if x in ref_set) / len(hyp_set)) if len(hyp_set) else 0.0
        rec = (sum(1 for x in ref_set if x in hyp_set) / len(ref_set)) if len(ref_set) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return (f1, prec, rec)

    def exact_entity_token_if_rel_exists_reward(hyp_ann, ref_ann):
        candidates = []
        for ann in [hyp_ann, ref_ann]:
            cand = []
            for ent in ann["entities"].values():
                if not ent["relations"]:
                    cand.append((ent["tokens"], ent["label"]))
                else:
                    cand.append((ent["tokens"], ent["label"], True))
            candidates.append(set(cand))
        hyp_set, ref_set = candidates
        prec = (sum(1 for x in hyp_set if x in ref_set) / len(hyp_set)) if len(hyp_set) else 0.0
        rec = (sum(1 for x in ref_set if x in hyp_set) / len(ref_set)) if len(ref_set) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return (f1, prec, rec)

    def exact_entity_token_if_all_match_reward(hyp_ann, ref_ann):
        candidates = []
        for ann in [hyp_ann, ref_ann]:
            cand = []
            for ent in ann["entities"].values():
                if not ent["relations"]:
                    cand.append((ent["tokens"], ent["label"]))
                else:
                    cand.extend([
                        (ent["tokens"].lower(), ent["label"], r[0],
                         ann["entities"][r[1]]["tokens"].lower())
                        for r in ent["relations"]
                    ])
            candidates.append(set(cand))
        hyp_set, ref_set = candidates
        prec = (sum(1 for x in hyp_set if x in ref_set) / len(hyp_set)) if len(hyp_set) else 0.0
        rec = (sum(1 for x in ref_set if x in hyp_set) / len(ref_set)) if len(ref_set) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return (f1, prec, rec)

    def compute_reward(hyp_ann, ref_ann, reward_level):
        if len(hyp_ann.get("entities", {})) == 0 or len(ref_ann.get("entities", {})) == 0:
            return (0., 0., 0.)
        if reward_level == "simple":
            return exact_entity_token_match_reward(hyp_ann, ref_ann)
        if reward_level == "partial":
            return exact_entity_token_if_rel_exists_reward(hyp_ann, ref_ann)
        if reward_level == "complete":
            return exact_entity_token_if_all_match_reward(hyp_ann, ref_ann)
        raise ValueError("reward_level must be simple|partial|complete")

    return F1RadGraphv2(reward_level=reward_level)


# -------------------------
# BiomedCLIP-CXR encoder (matches llava-rad biomedclipcxr_518.json) :contentReference[oaicite:2]{index=2}
# -------------------------
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_biomedclip_cxr_from_cfg(cfg_path, ckpt_path, device="cuda", fp16=False):
    """
    Build an OpenCLIP CLIP model from a BiomedCLIP-style config dict:
      { embed_dim, vision_cfg: {timm_model_name,...}, text_cfg: {hf_model_name,...} }

    This matches the llava-rad biomedclipcxr_518.json structure. :contentReference[oaicite:3]{index=3}
    """
    # NOTE: open_clip>=3.0.0 removed VisionCfg/TextCfg and constructing open_clip.model.CLIP
    # directly is not compatible with HFTextEncoder configs (it will error on token_embedding).
    # The correct way is to use the factory create_model_from_pretrained, which this repo
    # already relies on via open_clip_encoder.utils.

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # Register this config under the same name used throughout llava-rad.
    model_name = "biomedclip_cxr_518"

    from llava.model.multimodal_encoder.open_clip_encoder.utils import (
        from_pretrained,
        remove_transformer_pooler_weights,
    )

    ckpt_path = remove_transformer_pooler_weights(ckpt_path)
    model, preprocess, _tokenizer = from_pretrained(
        model_name=model_name,
        config=cfg,
        checkpoint_path=ckpt_path,
    )

    model = model.to(device).eval()
    if fp16:
        model = model.half()

    image_size = int(cfg.get("vision_cfg", {}).get("image_size", 518))
    embed_dim = int(cfg.get("embed_dim", -1))
    print(f"[encoder] cfg={os.path.basename(cfg_path)} image_size={image_size} embed_dim={embed_dim} model_name={model_name}")
    return model, preprocess


@torch.no_grad()
def encode_images(model, preprocess, image_paths, device="cuda", batch_size=32, fp16=False):
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="encoding"):
        batch = image_paths[i:i + batch_size]
        imgs = []
        for p in batch:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        x = torch.stack(imgs, dim=0).to(device)
        if fp16:
            x = x.half()
        f = model.encode_image(x)
        f = f.float()
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
        feats.append(f.cpu().numpy())

    if len(feats) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    out = np.concatenate(feats, axis=0)
    out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return out


# -------------------------
# Matching
# -------------------------
def cosine_best_match(q, cand_mat):
    sims = cand_mat @ q
    j = int(np.argmax(sims))
    return j, float(sims[j])


def make_pairs(df, emb,
               group_col="gender", a_val="F", b_val="M",
               constraint_labels=None, match_view=False, view_col="view",
               one_to_one=True, na_policy="match", allowed_states=("pos", "neg", "unc", "na"),
               min_cosine_sim: float = 0.0,
               min_shared_known_constraints: int = 0):
    constraint_labels = constraint_labels or []

    # Special handling for wildcard: NA should not form its own matching bucket.
    # Instead, treat any label where either side is NA as "don't constrain".
    if na_policy == "wildcard" and len(constraint_labels) > 0:
        allowed_states_set = set(allowed_states)
        NA_CODE = 3

        A_idx = []
        B_idx = []
        A_states = []
        B_states = []

        for i in range(len(df)):
            r = df.iloc[i]
            gv = r.get(group_col, None)
            if gv == a_val:
                side = "A"
            elif gv == b_val:
                side = "B"
            else:
                continue

            st = _states_vector(r, constraint_labels, allowed_states_set)
            if st is None:
                continue
            if match_view:
                # incorporate view into compatibility by splitting by view later
                pass
            if side == "A":
                A_idx.append(i)
                A_states.append(_encode_states(st))
            else:
                B_idx.append(i)
                B_states.append(_encode_states(st))

        if len(A_idx) == 0 or len(B_idx) == 0:
            return []

        A_states = np.stack(A_states, axis=0)  # (nA, m)
        B_states = np.stack(B_states, axis=0)  # (nB, m)
        B_emb_full = emb[B_idx]
        used_B = set()

        pairs = []
        for a_pos, i in enumerate(A_idx):
            q = emb[i]

            # compatibility: no label where both known and different
            a_vec = A_states[a_pos]  # (m,)
            # conflicts per B: any k where a!=NA and b!=NA and b!=a
            conflicts = np.zeros((len(B_idx),), dtype=bool)
            shared_known = np.zeros((len(B_idx),), dtype=np.int8)
            for k in range(a_vec.shape[0]):
                av = a_vec[k]
                bv = B_states[:, k]
                a_known = av != NA_CODE
                b_known = bv != NA_CODE
                if a_known:
                    conflicts |= (b_known & (bv != av))
                shared_known += (a_known & b_known).astype(np.int8)

            ok = ~conflicts
            if min_shared_known_constraints and int(min_shared_known_constraints) > 0:
                ok &= (shared_known >= int(min_shared_known_constraints))

            if match_view:
                a_view = str(df.iloc[i].get(view_col, ""))
                b_views = df.iloc[B_idx][view_col].astype(str).to_numpy()
                ok &= (b_views == a_view)

            if one_to_one:
                for bpos, bi in enumerate(B_idx):
                    if bi in used_B:
                        ok[bpos] = False

            if not np.any(ok):
                continue

            sims = B_emb_full @ q
            sims = sims.astype(np.float32)
            sims[~ok] = -1e9
            j_local = int(np.argmax(sims))
            sim = float(sims[j_local])
            if (min_cosine_sim is not None) and (sim < float(min_cosine_sim)):
                continue
            j = B_idx[j_local]
            if one_to_one:
                used_B.add(j)
            # key is not meaningful in wildcard mode (reporting uses format_constraint_state_for_pair)
            pairs.append((i, j, sim, ()))

        return pairs

    buckets = defaultdict(lambda: {"A": [], "B": []})
    for i in range(len(df)):
        r = df.iloc[i]
        if r.get(group_col, None) == a_val:
            side = "A"
        elif r.get(group_col, None) == b_val:
            side = "B"
        else:
            continue

        keys = build_bucket_keys(
            r, constraint_labels,
            match_view=match_view, view_col=view_col,
            na_policy=na_policy, allowed_states=allowed_states
        )
        for key in keys:
            buckets[key][side].append(i)

    pairs = []
    for key, sides in buckets.items():
        A_idx = sides["A"]
        B_idx = sides["B"]
        if len(A_idx) == 0 or len(B_idx) == 0:
            continue

        B_mat_full = emb[B_idx]
        used_B = set()

        for i in A_idx:
            q = emb[i]
            if one_to_one:
                keep = [k for k, bi in enumerate(B_idx) if bi not in used_B]
                if len(keep) == 0:
                    break
                cand_idx = [B_idx[k] for k in keep]
                cand_mat = B_mat_full[keep]
            else:
                cand_idx = B_idx
                cand_mat = B_mat_full

            j_local, sim = cosine_best_match(q, cand_mat)
            if (min_cosine_sim is not None) and (sim < float(min_cosine_sim)):
                continue
            j = cand_idx[j_local]
            if one_to_one:
                used_B.add(j)

            pairs.append((i, j, sim, key))

    return pairs


def summarize_series(x: pd.Series):
    x = x.dropna()
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p25": float(x.quantile(0.25)),
        "p75": float(x.quantile(0.75)),
    }


def _normalize_group_value(v):
    """Normalize group labels so minor formatting differences still match.

    In particular, MIMIC age bins often use an en-dash (e.g., "0–44").
    This maps common dash variants to a plain hyphen.
    """
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return v
    s = str(v).strip()
    for dash in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"]:
        s = s.replace(dash, "-")
    s = re.sub(r"\s+", " ", s)

    # Case-insensitive matching for categorical columns.
    sf = s.casefold()

    # Common aliases (helps avoid needing exact capitalization/phrasing).
    if sf in {"m", "male"}:
        return "m"
    if sf in {"f", "female"}:
        return "f"

    # Race/ethnicity-like values (MIMIC often uses long phrases).
    if "hispanic" in sf or "latino" in sf:
        return "hispanic"
    if sf == "other":
        return "other"
    if sf.startswith("white"):
        return "white"
    if sf.startswith("asian"):
        return "asian"
    if sf.startswith("black") or "african" in sf:
        return "black"
    if "american indian" in sf or "alaska native" in sf:
        return "aian"
    if "declined" in sf:
        return "declined"
    if "unknown" in sf:
        return "unknown"

    return sf


def _parse_sweep_values(s: str | None) -> list[float] | None:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _sanitize_tag(s: str) -> str:
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum() or ch in ["_", "-", "."]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _abbr_constraint(label: str) -> str:
    label = str(label)
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", label) if p]
    if len(parts) >= 2:
        abbr = "".join(p[0] for p in parts)
        return _sanitize_tag(abbr[:6])
    return _sanitize_tag(label[:6])


def _constraints_tag(constraint_labels: list[str]) -> str:
    if not constraint_labels:
        return "c0"
    joined = ",".join([str(x) for x in constraint_labels])
    h = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]
    abbr = "".join(_abbr_constraint(x) for x in constraint_labels)
    abbr = _sanitize_tag(abbr[:16])
    return f"c{len(constraint_labels)}_{abbr}_{h}"


def _auto_outdir(gen_jsonl: str, vision_tower_config: str | None, group_col: str, a_val: str, b_val: str,
                 min_cosine_sim: float, match_view: bool, one_to_one: bool,
                 constraint_labels: list[str] | None = None,
                 outdir_base: str | None = None,
                 outdir_tag: str | None = None,
                 na_policy: str | None = None) -> str:
    base_dir = outdir_base or os.path.dirname(os.path.abspath(gen_jsonl))
    enc = "encoder"
    if vision_tower_config:
        enc = os.path.splitext(os.path.basename(vision_tower_config))[0]
    enc = _sanitize_tag(enc)
    grp = _sanitize_tag(group_col)
    a = _sanitize_tag(a_val)
    b = _sanitize_tag(b_val)
    sim = f"{float(min_cosine_sim):.2f}"
    ctag = _constraints_tag(constraint_labels or [])
    extras = []
    if na_policy:
        extras.append(f"na_{_sanitize_tag(na_policy)}")
    if match_view:
        extras.append("view")
    if one_to_one:
        extras.append("1to1")
    if outdir_tag:
        extras.append(_sanitize_tag(outdir_tag)[:24])
    extra = ("_" + "_".join(extras)) if extras else ""
    name = f"cf_match_{enc}_{ctag}_{grp}_{a}_vs_{b}_cosine_{sim}{extra}"
    return os.path.join(base_dir, name)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gen_jsonl", required=True)
    ap.add_argument(
        "--meta_json",
        default="/a2il/data/mbhosale/MrFair/physionet.org/mimc-cxr-jpeg/gpt4-reports/chat_test_MIMIC_CXR_all_dem_clean_2K.json",
        help="Path to chat_test_*.json (meta).",
    )
    ap.add_argument(
        "--image_root",
        default="/data_local1/mbhosale/MrFair/mimc-cxr-jpeg/files/",
        help="Path to mimic-cxr-jpeg/files (contains p10,p11,...).",
    )
    ap.add_argument("--image_prefix_to_strip", default="mimic/")

    # Join control
    ap.add_argument("--join_key", default="auto", choices=["auto", "id", "subject_study"])

    # Text fields for RadGraph
    ap.add_argument("--ref_field_gen", default="reference")
    ap.add_argument("--hyp_field_gen", default="prediction")
    ap.add_argument("--use_gen_text_for_radgraph", action="store_true", default=True)

    ap.add_argument("--ref_field_meta", default="impression")
    ap.add_argument("--hyp_field_meta", default="conversations:gpt")

    # Encoder config (defaults aligned with microsoft/llava-rad files) :contentReference[oaicite:4]{index=4}
    ap.add_argument(
        "--vision_tower_config",
        default="/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json",
        help="Path to biomedclipcxr_518.json (or your local biomedclip_cxr_518.json).",
    )
    ap.add_argument(
        "--vision_tower_checkpoint",
        default="/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava-rad_hf/biomedclipcxr_518_checkpoint.pt",
        help="Path to biomedclipcxr_518_checkpoint.pt",
    )

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--fp16", action="store_true")

    # Matching controls
    ap.add_argument("--group_col", default="gender")
    ap.add_argument("--a_val", default="F")
    ap.add_argument("--b_val", default="M")
    ap.add_argument("--constraint_labels",
                    default="Edema,Pleural Effusion,Cardiomegaly,Enlarged Cardiomediastinum")
    ap.add_argument("--match_view", action="store_true")
    ap.add_argument("--view_col", default="view")
    ap.add_argument(
        "--one_to_one",
        dest="one_to_one",
        action="store_true",
        default=True,
        help="Use 1-to-1 matching (default).",
    )
    ap.add_argument(
        "--many_to_one",
        dest="one_to_one",
        action="store_false",
        help="Disable 1-to-1 matching (allow reusing B samples).",
    )
    ap.add_argument("--na_policy", default="match", choices=["match", "drop", "wildcard"])
    ap.add_argument("--min_shared_known_constraints", type=int, default=0,
                    help="In wildcard NA mode, require at least this many constraint labels to be non-NA on BOTH sides."
                         " Use 1 to avoid unconstrained 'none' pairs.")
    ap.add_argument("--constraint_state_mode", default=None, choices=["full", "omit_na"],
                    help="How to report constraint_state in outputs. 'omit_na' omits labels where either side is NA.")
    ap.add_argument("--allowed_states", default="pos,neg,unc,na")

    # Pair quality / metric scaling
    ap.add_argument("--min_cosine_sim", type=float, default=0.0,
                    help="Minimum cosine similarity required to accept a pair. Pairs below this are dropped.")
    ap.add_argument("--sweep", action="store_true",
                    help="If set, sweep multiple cosine thresholds and write one output folder per threshold.")
    ap.add_argument("--sweep_values", default=None,
                    help="Comma-separated cosine thresholds to sweep. Default: 0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--greenscore_scale", type=float, default=100.0,
                    help="Scale factor applied to GreenScore in matched-pairs outputs (e.g., 100 for 0-100).")
    ap.add_argument("--radgraph_scale", type=float, default=100.0,
                    help="Scale factor applied to RadGraph F1 in matched-pairs outputs (e.g., 100 for 0-100).")

    # RadGraph
    ap.add_argument("--reward_level", default="partial", choices=["simple", "partial", "complete"])
    ap.add_argument("--radgraph_cache_csv", default=None)
    ap.add_argument("--skip_radgraph", action="store_true")

    # Output
    ap.add_argument("--outdir", default=None,
                    help="Output directory. If omitted, auto-named next to gen_jsonl.")
    ap.add_argument("--outdir_base", default=None,
                    help="Base directory to place auto-named outputs. Defaults to dirname(gen_jsonl).")
    ap.add_argument("--auto_outdir", action="store_true",
                    help="If set, treat --outdir as a base prefix and append an auto-name suffix.")
    ap.add_argument("--outdir_tag", default=None,
                    help="Optional short tag appended to auto-named output folder (kept short).")

    args = ap.parse_args()

    # Parse these early so auto_outdir can encode them
    constraint_labels = [x.strip() for x in args.constraint_labels.split(",") if x.strip()]
    allowed_states = [x.strip() for x in args.allowed_states.split(",") if x.strip()]

    if args.constraint_state_mode is None:
        # In wildcard mode NA is treated as unknown; omitting NA produces clearer reports.
        args.constraint_state_mode = "omit_na" if args.na_policy == "wildcard" else "full"

    sweep_values = _parse_sweep_values(args.sweep_values)
    is_sweep = bool(args.sweep or (sweep_values is not None))
    if is_sweep:
        if not sweep_values:
            sweep_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # In sweep mode, treat --outdir as a base directory (not a full run directory).
        sweep_base = args.outdir_base or args.outdir or os.path.dirname(os.path.abspath(args.gen_jsonl))
        args.outdir_base = sweep_base
        ensure_dir(sweep_base)
        # Use a single radgraph cache for the sweep unless user overrides.
        if (not args.skip_radgraph) and (args.radgraph_cache_csv is None):
            args.radgraph_cache_csv = os.path.join(sweep_base, "radgraph_cache.csv")
    else:
        # Auto-name outputs + cache paths (to avoid passing many args)
        if args.outdir is None:
            args.outdir = _auto_outdir(
                gen_jsonl=args.gen_jsonl,
                vision_tower_config=args.vision_tower_config,
                group_col=args.group_col,
                a_val=args.a_val,
                b_val=args.b_val,
                min_cosine_sim=args.min_cosine_sim,
                match_view=args.match_view,
                one_to_one=args.one_to_one,
                constraint_labels=constraint_labels,
                outdir_base=args.outdir_base,
                outdir_tag=args.outdir_tag,
                na_policy=args.na_policy,
            )
        elif args.auto_outdir:
            base = args.outdir
            args.outdir = _auto_outdir(
                gen_jsonl=args.gen_jsonl,
                vision_tower_config=args.vision_tower_config,
                group_col=args.group_col,
                a_val=args.a_val,
                b_val=args.b_val,
                min_cosine_sim=args.min_cosine_sim,
                match_view=args.match_view,
                one_to_one=args.one_to_one,
                constraint_labels=constraint_labels,
                outdir_base=base,
                outdir_tag=args.outdir_tag,
                na_policy=args.na_policy,
            )

        ensure_dir(args.outdir)

        # Default cache goes inside outdir (so each run has its own calibrated cache)
        if (not args.skip_radgraph) and (args.radgraph_cache_csv is None):
            args.radgraph_cache_csv = os.path.join(args.outdir, "radgraph_cache.csv")

    # ---- locate config file robustly
    if args.vision_tower_config is None:
        # common local repo locations
        candidates = [
            "llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclipcxr_518.json",
            "llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json",
            "biomedclipcxr_518.json",
            "biomedclip_cxr_518.json",
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if found is None:
            raise FileNotFoundError(
                "Could not find BiomedCLIP-CXR config json. "
                "Pass --vision_tower_config explicitly (biomedclipcxr_518.json)."
            )
        args.vision_tower_config = found
    else:
        if not os.path.exists(args.vision_tower_config):
            raise FileNotFoundError(f"--vision_tower_config not found: {args.vision_tower_config}")

    # -------------------------
    # Load files
    # -------------------------
    gen_rows = read_jsonl(args.gen_jsonl)
    gen_df = pd.DataFrame(gen_rows)

    if "id" not in gen_df.columns:
        raise ValueError("gen_jsonl must contain 'id'")

    # Ensure subject_id/study_id exist for fallback join
    if "subject_id" not in gen_df.columns or "study_id" not in gen_df.columns:
        ss = gen_df["id"].apply(parse_subject_study_from_id)
        gen_df["subject_id"] = ss.apply(lambda t: t[0])
        gen_df["study_id"] = ss.apply(lambda t: t[1])

    gen_df["reference_gen"] = gen_df.apply(lambda r: extract_text_from_row(r, args.ref_field_gen), axis=1)
    gen_df["prediction_gen"] = gen_df.apply(lambda r: extract_text_from_row(r, args.hyp_field_gen), axis=1)

    meta_rows = read_json(args.meta_json)
    meta_df = pd.DataFrame(meta_rows)

    if "id" not in meta_df.columns:
        raise ValueError("meta_json must contain 'id'")

    if "subject_id" not in meta_df.columns or "study_id" not in meta_df.columns:
        ss = meta_df["id"].apply(parse_subject_study_from_id)
        meta_df["subject_id"] = ss.apply(lambda t: t[0])
        meta_df["study_id"] = ss.apply(lambda t: t[1])

    # Flatten chexpert_labels dict into columns
    flat = meta_df["chexpert_labels"].apply(flatten_chexpert_dict)
    flat_df = pd.DataFrame(list(flat))
    meta_df = pd.concat([meta_df.drop(columns=["chexpert_labels"], errors="ignore"), flat_df], axis=1)

    meta_df["reference_meta"] = meta_df.apply(lambda r: extract_text_from_row(r, args.ref_field_meta), axis=1)
    meta_df["prediction_meta"] = meta_df.apply(lambda r: extract_text_from_row(r, args.hyp_field_meta), axis=1)

    meta_df["image_abs"] = meta_df["image"].apply(
        lambda p: resolve_image_path(p, args.image_root, args.image_prefix_to_strip)
    )

    # Diagnostics
    gen_ids = set(gen_df["id"].astype(str))
    meta_ids = set(meta_df["id"].astype(str))
    print(f"[diag] gen rows={len(gen_df)} meta rows={len(meta_df)} id-overlap={len(gen_ids & meta_ids)}")

    gen_ss = set(zip(gen_df["subject_id"], gen_df["study_id"]))
    meta_ss = set(zip(meta_df["subject_id"], meta_df["study_id"]))
    print(f"[diag] subj-study overlap={len(gen_ss & meta_ss)}")

    # -------------------------
    # Join
    # -------------------------
    meta_keep_cols = ["id", "subject_id", "study_id", "image_abs", args.view_col] + CHEXPERT_LABELS + [
        "reference_meta", "prediction_meta"
    ]
    if args.join_key in ["auto", "id"]:
        df = gen_df.merge(meta_df[meta_keep_cols], on="id", how="inner")
        if args.join_key == "auto" and len(df) == 0:
            print("[diag] join on id gave 0 -> fallback join on (subject_id,study_id)")
            df = gen_df.merge(meta_df[meta_keep_cols], on=["subject_id", "study_id"], how="inner")
    else:
        df = gen_df.merge(meta_df[meta_keep_cols], on=["subject_id", "study_id"], how="inner")

    # Filter to existing images
    df = df[df["image_abs"].apply(lambda p: isinstance(p, str) and os.path.exists(p))].reset_index(drop=True)
    print(f"[info] rows after join+image check: {len(df)}")

    if len(df) == 0:
        # show a few sample path attempts
        sample = meta_df[["image", "image_abs"]].head(5)
        print("[fatal] 0 rows survived. Likely image_root/prefix mismatch.")
        print(sample.to_string(index=False))
        return

    # In sweep mode, we write outputs per-threshold; in single-run mode we write to args.outdir here.
    if not is_sweep:
        df.to_csv(os.path.join(args.outdir, "merged_joined.csv"), index=False)

    # -------------------------
    # Load encoder + embed images (always recompute per run)
    # -------------------------
    model, preprocess = load_biomedclip_cxr_from_cfg(
        args.vision_tower_config, args.vision_tower_checkpoint,
        device=args.device, fp16=args.fp16
    )
    image_paths = df["image_abs"].tolist()
    emb = encode_images(model, preprocess, image_paths, device=args.device,
                        batch_size=args.batch_size, fp16=args.fp16)

    if not is_sweep:
        np.save(os.path.join(args.outdir, "image_embeddings.npy"), emb)

    # -------------------------
    # RadGraph per sample
    # -------------------------
    if args.skip_radgraph:
        df["radgraph_f1"] = np.nan
    else:
        if args.use_gen_text_for_radgraph:
            refs = df["reference_gen"].fillna("").tolist()
            hyps = df["prediction_gen"].fillna("").tolist()
        else:
            refs = df["reference_meta"].fillna("").tolist()
            hyps = df["prediction_meta"].fillna("").tolist()

        if args.radgraph_cache_csv and os.path.exists(args.radgraph_cache_csv):
            cache = pd.read_csv(args.radgraph_cache_csv).drop_duplicates("id")
            m = dict(zip(cache["id"].astype(str), cache["radgraph_f1"].astype(float)))
            df["radgraph_f1"] = [float(m.get(str(i), np.nan)) for i in df["id"].tolist()]
        else:
            scorer = build_radgraph_scorer(reward_level=args.reward_level)
            _, reward_list = scorer(refs=refs, hyps=hyps)
            df["radgraph_f1"] = [float(r[0]) for r in reward_list]
            if args.radgraph_cache_csv:
                pd.DataFrame({"id": df["id"].astype(str), "radgraph_f1": df["radgraph_f1"]}).to_csv(
                    args.radgraph_cache_csv, index=False
                )

    if not is_sweep:
        df.to_csv(os.path.join(args.outdir, "per_sample_metrics.csv"), index=False)

    # -------------------------
    # Match pairs (A vs B)
    # -------------------------
    a_val_norm = _normalize_group_value(args.a_val)
    b_val_norm = _normalize_group_value(args.b_val)
    df = df.copy()
    df["__group_norm__"] = df[args.group_col].map(_normalize_group_value)

    def _run_one_threshold(threshold: float, outdir: str):
        ensure_dir(outdir)
        # Save common artifacts in each output dir for consistency
        df.to_csv(os.path.join(outdir, "merged_joined.csv"), index=False)
        np.save(os.path.join(outdir, "image_embeddings.npy"), emb)
        df.to_csv(os.path.join(outdir, "per_sample_metrics.csv"), index=False)

        mask = df["__group_norm__"].isin([a_val_norm, b_val_norm])
        df2_local = df.loc[mask].copy()
        emb2_local = emb[df2_local.index.to_numpy()]
        df2_local = df2_local.reset_index(drop=True)

        pairs_local = make_pairs(
            df2_local, emb2_local,
            group_col="__group_norm__", a_val=a_val_norm, b_val=b_val_norm,
            constraint_labels=constraint_labels,
            match_view=args.match_view, view_col=args.view_col,
            one_to_one=args.one_to_one,
            na_policy=args.na_policy,
            allowed_states=allowed_states,
            min_cosine_sim=float(threshold),
            min_shared_known_constraints=int(args.min_shared_known_constraints),
        )

        # Pair metrics
        rows_local = []
        for (i, j, sim, key) in pairs_local:
            ra = df2_local.iloc[i]
            rb = df2_local.iloc[j]

            gs_scale = float(args.greenscore_scale)
            ga_raw = float(ra.get("greenscore", np.nan))
            gb_raw = float(rb.get("greenscore", np.nan))
            ga = (ga_raw * gs_scale) if (not math.isnan(ga_raw)) else np.nan
            gb = (gb_raw * gs_scale) if (not math.isnan(gb_raw)) else np.nan
            rg_scale = float(args.radgraph_scale)
            rga_raw = float(ra.get("radgraph_f1", np.nan))
            rgb_raw = float(rb.get("radgraph_f1", np.nan))
            rga = (rga_raw * rg_scale) if (not math.isnan(rga_raw)) else np.nan
            rgb = (rgb_raw * rg_scale) if (not math.isnan(rgb_raw)) else np.nan

            rows_local.append({
                "id_A": ra["id"],
                "id_B": rb["id"],
                f"{args.group_col}_A": ra[args.group_col],
                f"{args.group_col}_B": rb[args.group_col],
                "cosine_sim": float(sim),
                "constraint_labels": ",".join(constraint_labels),
                "constraint_state": format_constraint_state_for_pair(
                    ra, rb, constraint_labels, mode=args.constraint_state_mode
                ),
                "view_A": ra.get(args.view_col, ""),
                "view_B": rb.get(args.view_col, ""),
                "greenscore_A": ga,
                "greenscore_B": gb,
                "abs_delta_greenscore": float(abs(ga - gb)) if (not math.isnan(ga) and not math.isnan(gb)) else np.nan,
                "signed_delta_greenscore_A_minus_B": float(ga - gb) if (not math.isnan(ga) and not math.isnan(gb)) else np.nan,
                "radgraph_f1_A": rga,
                "radgraph_f1_B": rgb,
                "abs_delta_radgraph_f1": float(abs(rga - rgb)) if (not math.isnan(rga) and not math.isnan(rgb)) else np.nan,
                "signed_delta_radgraph_A_minus_B": float(rga - rgb) if (not math.isnan(rga) and not math.isnan(rgb)) else np.nan,
            })

        pair_df_local = pd.DataFrame(rows_local)
        pair_df_local.to_csv(os.path.join(outdir, "matched_pairs.csv"), index=False)

        summary_local = {
            "n_samples_used": int(len(df2_local)),
            "n_pairs": int(len(pair_df_local)),
            "constraint_labels": constraint_labels,
            "match_view": bool(args.match_view),
            "one_to_one": bool(args.one_to_one),
            "na_policy": args.na_policy,
            "constraint_state_mode": args.constraint_state_mode,
            "min_shared_known_constraints": int(args.min_shared_known_constraints),
            "min_cosine_sim": float(threshold),
            "greenscore_scale": float(args.greenscore_scale),
            "radgraph_scale": float(args.radgraph_scale),
            "cosine_sim": summarize_series(pair_df_local.get("cosine_sim", pd.Series(dtype=float))),
            "greenscore_abs_delta": summarize_series(pair_df_local.get("abs_delta_greenscore", pd.Series(dtype=float))),
            "radgraph_abs_delta": summarize_series(pair_df_local.get("abs_delta_radgraph_f1", pd.Series(dtype=float))),
            "greenscore_signed_delta_mean": float(pair_df_local["signed_delta_greenscore_A_minus_B"].dropna().mean()) if len(pair_df_local) else None,
            "radgraph_signed_delta_mean": float(pair_df_local["signed_delta_radgraph_A_minus_B"].dropna().mean()) if len(pair_df_local) else None,
            "by_constraint_state": {},
        }

        if len(pair_df_local) > 0:
            for state, g in pair_df_local.groupby("constraint_state"):
                summary_local["by_constraint_state"][state] = {
                    "n_pairs": int(len(g)),
                    "cosine_sim": summarize_series(g["cosine_sim"]) if "cosine_sim" in g.columns else {"n": 0},
                    "greenscore_abs_delta": summarize_series(g["abs_delta_greenscore"]),
                    "radgraph_abs_delta": summarize_series(g["abs_delta_radgraph_f1"]),
                }

        with open(os.path.join(outdir, "summary.json"), "w") as f:
            json.dump(summary_local, f, indent=2)

        print("[done] wrote outputs to:", outdir)

    if is_sweep:
        for thr in sweep_values:
            outdir_thr = _auto_outdir(
                gen_jsonl=args.gen_jsonl,
                vision_tower_config=args.vision_tower_config,
                group_col=args.group_col,
                a_val=args.a_val,
                b_val=args.b_val,
                min_cosine_sim=float(thr),
                match_view=args.match_view,
                one_to_one=args.one_to_one,
                constraint_labels=constraint_labels,
                outdir_base=args.outdir_base,
                outdir_tag=args.outdir_tag,
                na_policy=args.na_policy,
            )
            _run_one_threshold(float(thr), outdir_thr)
        return

    # Single-threshold run
    _run_one_threshold(float(args.min_cosine_sim), args.outdir)
    return


if __name__ == "__main__":
    main()