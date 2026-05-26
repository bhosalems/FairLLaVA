#!/usr/bin/env python3
"""Run MI lambda ablations safely and reproducibly.

Generates one fairness config JSON per run, then launches
`scripts/finetune_lora_mi.sh` with FAIRNESS_CONFIG pointing to that JSON.

Creates a run directory containing:
- the exact fairness config used
- stdout/stderr log from the bash script
- a metadata JSON (command, timings, exit code, git commit)

This is intentionally minimal: it does not modify training code, and it
does not try to parse WandB or checkpoint structure.
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
from typing import Any, Dict, List, Optional


ATTR_TO_KEY = {
    "age": "age_lmda",
    "gender": "gender_lmda",
    "race": "race_lmda",
}


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MI lambda ablations (12 runs).")
    parser.add_argument(
        "--template-config",
        default=None,
        help=(
            "Template fairness config JSON. Defaults to "
            "configs/fairness_finetune_mimic_cxr_mi_lambda_ablate.json"
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
            "Defaults to ./checkpoints"
        ),
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help=(
            "Where to write ablation logs/config copies. "
            "Defaults to <output-dir>/mi_lambda_ablations/<timestamp>"
        ),
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (exported as SEED to finetune script).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining experiments if one fails.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    template_config = (
        Path(args.template_config)
        if args.template_config
        else repo_root / "configs" / "fairness_finetune_mimic_cxr_mi_lambda_ablate.json"
    ).resolve()
    finetune_script = (
        Path(args.finetune_script)
        if args.finetune_script
        else repo_root / "scripts" / "finetune_lora_mi.sh"
    ).resolve()
    output_dir = (Path(args.output_dir) if args.output_dir else repo_root / "checkpoints").resolve()

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = (
        Path(args.runs_root)
        if args.runs_root
        else (output_dir / "mi_lambda_ablations" / ts)
    ).resolve()

    if not template_config.exists():
        raise FileNotFoundError(f"Template config not found: {template_config}")
    if not finetune_script.exists():
        raise FileNotFoundError(f"Finetune script not found: {finetune_script}")

    template = _load_json(template_config)

    sweep_values = {
        "age": [0.1, 0.2, 2, 4],
        "gender": [0.2, 0.6, 2, 4],
        "race": [0.1, 0.6, 2, 4],
    }

    git_commit = _try_git_commit(repo_root)

    runs: List[Dict[str, Any]] = []
    for attr, values in sweep_values.items():
        for v in values:
            runs.append({"attr": attr, "lambda": v})

    runs_root.mkdir(parents=True, exist_ok=True)

    summary_path = runs_root / "SUMMARY.json"
    summary: Dict[str, Any] = {
        "created_utc": _utc_now_iso(),
        "repo_root": str(repo_root),
        "template_config": str(template_config),
        "finetune_script": str(finetune_script),
        "output_dir": str(output_dir),
        "git_commit": git_commit,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "seed": args.seed,
        "runs": [],
    }
    _dump_json(summary_path, summary)

    total = len(runs)
    for idx, spec in enumerate(runs, start=1):
        attr = spec["attr"]
        lam = spec["lambda"]
        key = ATTR_TO_KEY[attr]

        run_id = f"{idx:02d}_{attr}_lmda_{lam}"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        fairness_cfg = dict(template)
        fairness_cfg["target_dem"] = [attr]
        fairness_cfg[key] = lam

        fairness_cfg_path = run_dir / "fairness_config.json"
        _dump_json(fairness_cfg_path, fairness_cfg)

        # Copy the template for reference (exactly as provided)
        shutil.copy2(template_config, run_dir / "template_config.original.json")

        # Build command: bash scripts/finetune_lora_mi.sh <output_dir> [epochs] [bsz]
        cmd = ["bash", str(finetune_script), str(output_dir)]
        if args.epochs is not None:
            cmd.append(str(args.epochs))
        if args.bsz is not None:
            cmd.append(str(args.bsz))

        env = os.environ.copy()
        env["FAIRNESS_CONFIG"] = str(fairness_cfg_path)
        env["RUN_TAG"] = f"{attr}_lmda_{lam}"
        env["SEED"] = str(args.seed)
        env["RUN_NAME_FILE"] = str(run_dir / "run_name.txt")

        log_path = run_dir / "train.log"
        meta_path = run_dir / "meta.json"

        meta: Dict[str, Any] = {
            "run_id": run_id,
            "attr": attr,
            "lambda": lam,
            "fairness_key": key,
            "seed": args.seed,
            "start_utc": _utc_now_iso(),
            "cwd": str(repo_root),
            "command": cmd,
            "env_overrides": {
                "FAIRNESS_CONFIG": str(fairness_cfg_path),
                "RUN_TAG": f"{attr}_lmda_{lam}",
                "SEED": str(args.seed),
            },
            "git_commit": git_commit,
        }
        _dump_json(meta_path, meta)

        print("\n" + "=" * 80)
        print(f"[{idx}/{total}] Starting {run_id}")
        print(f"  FAIRNESS_CONFIG={fairness_cfg_path}")
        print(f"  Logging to {log_path}")
        sys.stdout.flush()

        t0 = time.time()
        exit_code = _run_cmd(cmd, cwd=repo_root, env=env, tee_path=log_path)
        dt = time.time() - t0

        # Capture the run_name that finetune script writes (best-effort)
        run_name_path = repo_root / "run_name"
        run_name: Optional[str] = None
        # Capture the run_name that finetune script writes (prefer per-run file)
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
        _dump_json(meta_path, meta)

        # Update global summary incrementally
        try:
            summary = _load_json(summary_path)
            summary["runs"].append(meta)
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

    print("\nAll ablation runs complete.")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
