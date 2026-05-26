#!/usr/bin/env python3
"""Run CE-loss-weight ablations (ablations2) safely and reproducibly.

This mirrors `scripts/run_mi_lambda_ablations.py`, but instead of sweeping
`*_lmda` it sweeps `ce_loss_weight` while keeping all other fairness settings
identical to a template config.

It generates one fairness config JSON per run, then launches
`scripts/finetune_lora_mi.sh` with FAIRNESS_CONFIG pointing to that JSON.

Creates a run directory containing:
- the exact fairness config used (fairness_config.json)
- a copy of the template config as provided (template_config.original.json)
- stdout/stderr log from the bash script (train.log)
- a metadata JSON (meta.json) and overall SUMMARY.json

Default output location (artifacts/logs):
  results/ablations2/ce_loss_weight/<timestamp>/

Default checkpoint root passed to finetune script:
    ./checkpoints/ablations2

Example:
    python3 scripts/run_ce_weight_ablations2.py --epochs 3 --bsz 6 --seed 42

    # Or override checkpoint root:
    python3 scripts/run_ce_weight_ablations2.py --output-dir ./checkpoints/ablations2 --epochs 3 --bsz 6

"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    tee_path: Optional[Path] = None,
) -> int:
    """Run a command, optionally teeing stdout+stderr to a file."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert proc.stdout is not None

    out_f = None
    if tee_path is not None:
        tee_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = tee_path.open("w", encoding="utf-8")

    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if out_f is not None:
                out_f.write(line)
        return proc.wait()
    finally:
        try:
            if out_f is not None:
                out_f.flush()
                out_f.close()
        except Exception:
            pass


def _try_git_commit(repo_root: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _parse_weights(values: Optional[List[str]]) -> List[float]:
    if not values:
        return [0.1, 2.0, 3.5, 5.0]
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except ValueError as e:
            raise ValueError(f"Invalid --ce-weights value: {v}") from e
    return out


_OUTPUT_DIR_RE = re.compile(r"--output_dir\s+(\S+)")


def _infer_checkpoint_dir_from_log(log_path: Path) -> Optional[Path]:
    """Best-effort: parse DeepSpeed command line from train.log and extract --output_dir."""
    try:
        if not log_path.exists():
            return None
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        matches = _OUTPUT_DIR_RE.findall(text)
        if not matches:
            return None
        # take the last match (most likely the actual launch command)
        return Path(matches[-1])
    except Exception:
        return None


def _looks_like_trained_checkpoint_dir(p: Path) -> bool:
    """Heuristic: consider a run complete if the output dir contains HF trainer artifacts."""
    try:
        if not p.exists() or not p.is_dir():
            return False
        # Typical HF/DeepSpeed outputs
        if (p / "trainer_state.json").exists():
            return True
        # Checkpoint directories
        for _ in p.glob("checkpoint-*"):
            return True
        # Some runs may only leave adapter weights at top-level
        if (p / "adapter_model.bin").exists() or (p / "adapter_model.safetensors").exists():
            return True
        return False
    except Exception:
        return False


def _diff_except(
    base: Dict[str, Any],
    other: Dict[str, Any],
    *,
    allow_keys: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Return key-> {base, other} for any differences excluding allow_keys.

    This is a shallow diff (top-level keys), which is sufficient for asserting
    we only changed one hyperparameter in the fairness config.
    """
    diffs: Dict[str, Dict[str, Any]] = {}
    keys = set(base.keys()) | set(other.keys())
    allowed = set(allow_keys)
    for k in sorted(keys):
        if k in allowed:
            continue
        if base.get(k) != other.get(k):
            diffs[k] = {"base": base.get(k), "other": other.get(k)}
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CE-loss-weight ablations (writes artifacts under results/ablations2)."
    )
    parser.add_argument(
        "--template-config",
        default=None,
        help=(
            "Template fairness config JSON. Defaults to "
            "configs/fairness_finetune_mimic_cxr_mi.json"
        ),
    )
    parser.add_argument(
        "--finetune-script",
        default=None,
        help="Finetune script path. Defaults to scripts/finetune_lora_mi.sh",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "First arg passed to finetune script (checkpoint root). "
            "Defaults to ./checkpoints/ablations2"
        ),
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help=(
            "Where to write ablation logs/config copies. "
            "Defaults to results/ablations2/ce_loss_weight/<timestamp>"
        ),
    )
    parser.add_argument(
        "--ce-weights",
        nargs="+",
        default=None,
        help="List of CE loss weights to sweep (space-separated). Defaults: 0.1 2 3.5 5",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default=None,
        help="Optional: pass as 2nd positional arg to finetune script.",
    )
    parser.add_argument(
        "--bsz",
        type=str,
        default=None,
        help="Optional: pass as 3rd positional arg to finetune script.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining experiments if one fails.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "If a run directory already has meta.json with exit_code==0, skip rerunning it. "
            "Useful with --runs-root to resume partially completed sweeps."
        ),
    )
    parser.add_argument(
        "--skip-if-checkpoint-exists",
        action="store_true",
        help=(
            "Skip a run if we can detect that its checkpoint output dir already exists and looks complete, "
            "even if meta.json exit_code was non-zero (e.g., post-training script error)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (exported as SEED to finetune script).",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    template_config = (
        Path(args.template_config)
        if args.template_config
        else repo_root / "configs" / "fairness_finetune_mimic_cxr_mi.json"
    ).resolve()
    finetune_script = (
        Path(args.finetune_script)
        if args.finetune_script
        else repo_root / "scripts" / "finetune_lora_mi.sh"
    ).resolve()
    output_dir = (
        Path(args.output_dir) if args.output_dir else repo_root / "checkpoints" / "ablations2"
    ).resolve()

    # Make sure checkpoint root exists early so the finetune script doesn't fail on mkdir issues.
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = (
        Path(args.runs_root)
        if args.runs_root
        else (repo_root / "results" / "ablations2" / "ce_loss_weight" / ts)
    ).resolve()

    if not template_config.exists():
        raise FileNotFoundError(f"Template config not found: {template_config}")
    if not finetune_script.exists():
        raise FileNotFoundError(f"Finetune script not found: {finetune_script}")

    template = _load_json(template_config)
    weights = _parse_weights(args.ce_weights)

    git_commit = _try_git_commit(repo_root)

    runs_root.mkdir(parents=True, exist_ok=True)

    summary_path = runs_root / "SUMMARY.json"
    if summary_path.exists():
        try:
            summary = _load_json(summary_path)
        except Exception:
            summary = {}
    else:
        summary = {}

    # Initialize/refresh summary header fields but preserve any existing runs.
    summary_runs = summary.get("runs")
    if not isinstance(summary_runs, list):
        summary_runs = []
    summary = {
        "created_utc": summary.get("created_utc") or _utc_now_iso(),
        "repo_root": str(repo_root),
        "template_config": str(template_config),
        "finetune_script": str(finetune_script),
        "output_dir": str(output_dir),
        "git_commit": git_commit,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": args.seed,
        "ce_weights": weights,
        "runs": summary_runs,
    }
    _dump_json(summary_path, summary)

    existing_by_id: Dict[str, Dict[str, Any]] = {}
    for r in summary_runs:
        if isinstance(r, dict) and isinstance(r.get("run_id"), str):
            existing_by_id[r["run_id"]] = r

    total = len(weights)

    for idx, w in enumerate(weights, start=1):
        run_id = f"{idx:02d}_ce_loss_weight_{w}"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        fairness_cfg = dict(template)
        fairness_cfg["ce_loss_weight"] = w

        # Safety: ensure we only changed ce_loss_weight.
        diffs = _diff_except(template, fairness_cfg, allow_keys=["ce_loss_weight"])
        if diffs:
            raise RuntimeError(
                "Unexpected fairness config changes beyond ce_loss_weight: "
                + ", ".join(diffs.keys())
            )

        fairness_cfg_path = run_dir / "fairness_config.json"
        _dump_json(fairness_cfg_path, fairness_cfg)

        shutil.copy2(template_config, run_dir / "template_config.original.json")

        # Build command: bash scripts/finetune_lora_mi.sh <output_dir> [epochs] [bsz]
        cmd = ["bash", str(finetune_script), str(output_dir)]
        if args.epochs is not None:
            cmd.append(str(args.epochs))
        if args.bsz is not None:
            cmd.append(str(args.bsz))

        env = os.environ.copy()
        env["FAIRNESS_CONFIG"] = str(fairness_cfg_path)
        env["RUN_TAG"] = f"ce_loss_weight_{w}"
        env["SEED"] = str(args.seed)
        env["RUN_NAME_FILE"] = str(run_dir / "run_name.txt")

        log_path = run_dir / "train.log"
        meta_path = run_dir / "meta.json"

        if (args.skip_completed or args.skip_if_checkpoint_exists) and meta_path.exists():
            try:
                existing_meta = _load_json(meta_path)
                exit_code_existing = int(existing_meta.get("exit_code", 1))

                inferred_ckpt = _infer_checkpoint_dir_from_log(log_path)
                if inferred_ckpt is not None:
                    existing_meta.setdefault("checkpoint_dir_inferred", str(inferred_ckpt))
                    existing_meta.setdefault(
                        "checkpoint_inferred_looks_complete",
                        _looks_like_trained_checkpoint_dir(inferred_ckpt),
                    )

                should_skip = False
                if args.skip_completed and exit_code_existing == 0:
                    should_skip = True
                if (
                    not should_skip
                    and args.skip_if_checkpoint_exists
                    and inferred_ckpt is not None
                    and _looks_like_trained_checkpoint_dir(inferred_ckpt)
                ):
                    should_skip = True

                if should_skip:
                    print("\n" + "=" * 80)
                    print(f"[{idx}/{total}] SKIP (completed): {run_id}")
                    existing_by_id[run_id] = existing_meta
                    # Persist summary in case it was missing this run.
                    summary = _load_json(summary_path)
                    runs_list = summary.get("runs")
                    if not isinstance(runs_list, list):
                        runs_list = []
                    # Replace or append
                    replaced = False
                    for i, rr in enumerate(runs_list):
                        if isinstance(rr, dict) and rr.get("run_id") == run_id:
                            runs_list[i] = existing_meta
                            replaced = True
                            break
                    if not replaced:
                        runs_list.append(existing_meta)
                    summary["runs"] = runs_list
                    _dump_json(summary_path, summary)
                    continue
            except Exception:
                pass

        meta: Dict[str, Any] = {
            "run_id": run_id,
            "ce_loss_weight": w,
            "seed": args.seed,
            "start_utc": _utc_now_iso(),
            "cwd": str(repo_root),
            "command": cmd,
            "env_overrides": {
                "FAIRNESS_CONFIG": str(fairness_cfg_path),
                "RUN_TAG": f"ce_loss_weight_{w}",
                "SEED": str(args.seed),
            },
            "git_commit": git_commit,
        }
        _dump_json(meta_path, meta)

        print("\n" + "=" * 80)
        print(f"[{idx}/{total}] Starting {run_id}")
        print(f"  ce_loss_weight={w}")
        print(f"  FAIRNESS_CONFIG={fairness_cfg_path}")
        print(f"  Logging to {log_path}")
        sys.stdout.flush()

        t0 = time.time()
        exit_code = _run_cmd(cmd, cwd=repo_root, env=env, tee_path=log_path)
        dt = time.time() - t0

        # Capture the run_name that finetune script writes (prefer per-run file)
        run_name: Optional[str] = None
        try:
            per_run_name_path = run_dir / "run_name.txt"
            if per_run_name_path.exists():
                run_name = per_run_name_path.read_text(encoding="utf-8").strip() or None
            else:
                legacy = repo_root / "run_name"
                if legacy.exists():
                    run_name = legacy.read_text(encoding="utf-8").strip() or None
                    shutil.copy2(legacy, per_run_name_path)
        except Exception:
            pass

        meta.update(
            {
                "end_utc": _utc_now_iso(),
                "duration_sec": dt,
                "exit_code": exit_code,
                "llava_run_name": run_name,
                "checkpoint_dir_guess": str(output_dir / run_name) if run_name else None,
            }
        )

        inferred_ckpt = _infer_checkpoint_dir_from_log(log_path)
        if inferred_ckpt is not None:
            meta["checkpoint_dir_inferred"] = str(inferred_ckpt)
            meta["checkpoint_inferred_looks_complete"] = _looks_like_trained_checkpoint_dir(inferred_ckpt)
        _dump_json(meta_path, meta)

        # Update global summary incrementally
        try:
            summary = _load_json(summary_path)
            runs_list = summary.get("runs")
            if not isinstance(runs_list, list):
                runs_list = []
            replaced = False
            for i, rr in enumerate(runs_list):
                if isinstance(rr, dict) and rr.get("run_id") == run_id:
                    runs_list[i] = meta
                    replaced = True
                    break
            if not replaced:
                runs_list.append(meta)
            summary["runs"] = runs_list
            _dump_json(summary_path, summary)
        except Exception:
            pass

        if exit_code != 0:
            print(f"[{idx}/{total}] FAILED (exit={exit_code}): {run_id}")
            if not args.continue_on_error:
                print("Stopping (use --continue-on-error to keep going).")
                return exit_code
        else:
            print(f"[{idx}/{total}] OK: {run_id} ({dt/60.0:.1f} min)")

    print("\nAll CE-weight ablation runs complete.")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
