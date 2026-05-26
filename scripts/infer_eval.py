#!/usr/bin/env python3
import os
import sys
import subprocess
import multiprocessing
from pathlib import Path
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def get_gpu_count():
    """Get number of available GPUs"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        return len(result.stdout.strip().split('\n'))
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi not found or failed. Defaulting to 1 GPU.")
        return 1

def run_eval_worker(args_dict):
    """Run a single evaluation worker on a specific GPU"""
    cmd = [
        sys.executable, "-m", "llava.eval.model_mimic_cxr",
        "--query_file", args_dict["query_file"],
        "--loader", args_dict["loader"],
        "--image_folder", args_dict["image_folder"],
        "--conv_mode", args_dict["conv_mode"],
        "--prediction_file", args_dict["prediction_file"],
        "--temperature", str(args_dict["temperature"]),
        "--model_path", args_dict["model_path"],
        "--chunk_idx", str(args_dict["chunk_idx"]),
        "--num_chunks", str(args_dict["num_chunks"]),
        "--batch_size", str(args_dict["batch_size"]),
        # "--fairness_config", str(args_dict.get("fairness_config", "")),
        "--group_by_length"
    ]

    if args_dict.get("model_base"):
        cmd += ["--model_base", args_dict["model_base"]]
    
    env = os.environ.copy()
    gpu_id = args_dict.get("gpu_id", 0)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"Starting worker {args_dict['chunk_idx']}/{args_dict['num_chunks']} on GPU {gpu_id}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"Worker {args_dict['chunk_idx']} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Worker {args_dict['chunk_idx']} failed with return code {e.returncode}")
        raise

def merge_predictions(prediction_file, num_chunks, prediction_dir):
    """Merge prediction files from all workers"""
    parts = []
    for i in range(num_chunks):
        part_file = f"{prediction_file}_{i}.jsonl"
        if os.path.exists(part_file):
            parts.append(part_file)
    
    print(f"Found {len(parts)}/{num_chunks} shard files.")
    
    if len(parts) == 0:
        print(f"No prediction parts found matching {prediction_file}_*.jsonl", file=sys.stderr)
        return False
    
    merged_file = os.path.join(prediction_dir, "merged_preds.jsonl")
    total_lines = 0
    with open(merged_file, 'w') as outf:
        for part_file in parts:
            print(f"Reading {part_file}...")
            with open(part_file, 'r') as inf:
                lines = 0
                for line in inf:
                    outf.write(line)
                    lines += 1
                print(f"  Added {lines} lines from {part_file}")
                total_lines += lines
    
    print(f"Merged {len(parts)} files into {merged_file} ({total_lines} total lines)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run LLaVA-Rad evaluation across multiple GPUs")
    parser.add_argument("--model_base", default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--prediction_dir", default=None)
    parser.add_argument("--query_file", 
                       default="/a2il/data/mbhosale/MrFair/physionet.org/mimc-cxr-jpeg/gpt4-reports/chat_test_MIMIC_CXR_all_dem.json")
    parser.add_argument("--image_folder", 
                       default="/data_local1/mbhosale/MrFair/mimc-cxr-jpeg/files/")
    parser.add_argument("--loader", default="mimic_test_findings")
    parser.add_argument("--conv_mode", default="v1")
    parser.add_argument("--fairness_finetune", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fairness_only", action="store_true", help="Only merge existing prediction files")
    parser.add_argument("--skip_merge", action="store_true", help="Skip merging step")
    parser.add_argument("--skip_post_eval", action="store_true", help="Skip all post-inference evaluation/stratification")
    parser.add_argument("--skip_gap_es", action="store_true", help="Skip fairness gap/ES summary (summary_fairness_gap.csv, summary_es_metrics.csv)")
    parser.add_argument(
        "--gap_es_metrics",
        default="BLEU-1,BLEU-4,F1-RadGraph,GreenScore,ROUGE-L",
        help="Comma-separated metrics for gap/ES computation (CXR only; metrics themselves are not rescaled except internally for gap/ES).",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    # parser.add_argument("--fairness_config", type=str, required=True, help="Path to fairness config JSON (for loading DAC modules)")
    
    args = parser.parse_args()
    if args.model_base is None or args.model_base.strip().lower() in {"", "none", "null"}:
        args.model_base = None

    # Allow model_path to be a HuggingFace repo id (namespace/repo_name) or a local path
    model_path = Path(args.model_path).expanduser()
    is_hf_repo = ("/" in args.model_path and not model_path.exists())
    if not is_hf_repo:
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                "--model_path does not exist on disk. "
                "If you intended a local checkpoint path, provide an existing absolute path (or a path relative to the repo). "
                "If you intended a HuggingFace Hub model, pass a repo id like 'namespace/repo_name' (no leading '/'). "
                f"Got: {args.model_path} (resolved: {model_path})"
            )
        args.model_path = str(model_path)
    # else: leave args.model_path as the repo id string
    
    # Set defaults
    if args.run_name is None:
        args.run_name = f"{args.prefix}llavarad"
    
    if args.prediction_dir is None:
        args.prediction_dir = f"/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/results/{args.run_name}"
    
    prediction_file = os.path.join(args.prediction_dir, "test")
    
    # Create output directory
    Path(args.prediction_dir).mkdir(parents=True, exist_ok=True)
    
    num_chunks = 1 # get_gpu_count() # TODO mahesh Please remove this, some issues with multi GPU setup, setting 1 for now.
    print(f"Using {num_chunks} GPUs for evaluation")

    if not args.fairness_only:
        # Prepare worker arguments
        worker_args = []
        for idx in range(num_chunks):
            worker_args.append({
                "query_file": args.query_file,
                "loader": args.loader,
                "image_folder": args.image_folder,
                "conv_mode": args.conv_mode,
                "prediction_file": f"{prediction_file}_{idx}.jsonl",
                "temperature": args.temperature,
                "model_path": args.model_path,
                "model_base": args.model_base,
                "chunk_idx": idx,
                "num_chunks": num_chunks,
                "batch_size": args.batch_size,
                "gpu_id": idx,
                "fairness_finetune": args.fairness_finetune,
                # "fairness_config": args.fairness_config 
            })
        
        # Run workers in parallel
        with multiprocessing.Pool(processes=num_chunks) as pool:
            try:
                pool.map(run_eval_worker, worker_args)
                print("All workers completed successfully")
            except Exception as e:
                print(f"Error during evaluation: {e}")
                return 1
    
    # Merge predictions
    if not args.skip_merge:
        success = merge_predictions(prediction_file, num_chunks, args.prediction_dir)
        if not success:
            return 1
        
    if args.skip_post_eval:
        print("Skipping post-eval (requested via --skip_post_eval)")
        print("Evaluation completed successfully")
        return 0

    ds = (args.dataset or "").strip().lower()

    # Radiology fairness evaluation pipeline (existing)
    fairness_script = "/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/scripts/evaluate_fairness.sh"
    if ds in {"mimic-cxr", "padchest"}:
        if os.path.exists(fairness_script):
            print("Running fairness evaluation...")
            subprocess.run(["bash", fairness_script, args.prediction_dir, args.prefix, args.dataset], check=True)
        else:
            print(f"Fairness script not found at {fairness_script}; skipping.")

        if not args.skip_gap_es:
            print("Computing fairness gap + ES summary...")
            subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_gap_es.py",
                    "--pred_dir",
                    args.prediction_dir,
                    "--dataset",
                    args.dataset,
                    "--report_metrics",
                    args.gap_es_metrics,
                ],
                check=True,
            )

    # HAM10000: stratify + run classification fairness evaluation
    elif ds in {"ham10000", "ham"}:
        print("Running HAM10000 stratification...")
        subprocess.run([
            sys.executable,
            "scripts/stratify_results.py",
            "--pred_dir",
            args.prediction_dir,
            "--dataset",
            "ham10000",
        ], check=True)

        print("Running HAM10000 fairness evaluation...")
        subprocess.run([
            sys.executable,
            "scripts/eval_ham_fairness.py",
            "--pred_dir",
            args.prediction_dir,
            "--query_file",
            args.query_file,
        ], check=True)

    else:
        print(f"No post-eval configured for dataset={args.dataset}; skipping.")
    
    print("Evaluation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
    


# Typical command to runt he inference and evaluation
# CUDA_VISIBLE_DEVICES=0 python scripts/infer_eval.py  --model_base lmsys/vicuna-7b-v1.5 --prefix "mimic_all_pretrained_mi_ablation" 
# --query_file /a2il/data/mbhosale/MrFair/physionet.org/mimc-cxr-jpeg/gpt4-reports/chat_test_MIMIC_CXR_all_dem_clean_2K.json 
# --model_path /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/checkpoints/biomedclip_cxr_518-lora-1e-1e-4-20251119182850 --fairness_finetune False 
# --batch_size 8 --prediction_dir /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/results/mimic_all_pretrained_mi_ablation --loader mimic_test_findings 
# --image_folder /data_local1/mbhosale/MrFair/mimc-cxr-jpeg/files/ --dataset mimic-cxr