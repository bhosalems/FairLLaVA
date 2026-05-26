#!/bin/bash

set -euo pipefail

model_base=lmsys/vicuna-7b-v1.5
model_path= # microsoft/llava-rad # <-- Change this
prefix="mi_only_img_d0_" # <-- Change this for the correct logging also in Wandb

run_name="${prefix}llavarad"
run_name="${4:-$run_name}"
model_base="${1:-$model_base}"
model_path="${2:-$model_path}"
prediction_dir="${3:-/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/results/$run_name}"
prediction_file=$prediction_dir/test


query_file=/a2il/data/mbhosale/MrFair/physionet.org/mimc-cxr-jpeg/gpt4-reports/chat_test_MIMIC_CXR_all_dem.json # TODO remove small

image_folder="/data_local1/mbhosale/MrFair/mimc-cxr-jpeg/files/" # "/a2il/data/mbhosale/MrFair/physionet.org/mimc-cxr-jpeg/files/"
loader="mimic_test_findings"
conv_mode="v1"

CHUNKS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
pids=()
for (( idx=0; idx<$CHUNKS; idx++ ))
do
    CUDA_VISIBLE_DEVICES=$idx python -m llava.eval.model_mimic_cxr \
        --query_file ${query_file} \
        --loader ${loader} \
        --image_folder ${image_folder} \
        --conv_mode ${conv_mode} \
        --prediction_file ${prediction_file}_${idx}.jsonl \
        --temperature 0 \
        --model_path ${model_path} \
        --model_base ${model_base} \
        --chunk_idx ${idx} \
        --num_chunks ${CHUNKS} \
        --batch_size 8 \
        --group_by_length &
        # >/dev/null 2>&1 &  # silence noisy workers; remove this redirection if you want logs
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

# shopt -s nullglob
# parts=( "${prediction_file}"_*.jsonl )
# echo "Found ${#parts[@]}/${CHUNKS} shard files."
# if ((${#parts[@]}==0)); then
#   echo "No prediction parts found matching ${prediction_file}_*.jsonl" >&2
#   exit 1
# fi

# cat -- "${parts[@]}" > "${prediction_dir}/merged_preds.jsonl"
# echo "Merged ${#parts[@]} files into ${prediction_dir}/merged_preds.jsonl"

# bash /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/scripts/evaluate_fairness_MIMIC_CXR.sh ${prediction_dir} ${prefix}