#!/bin/bash

set -e
set -o pipefail

model_base=lmsys/vicuna-7b-v1.5
model_path=microsoft/llava-rad
prefix="baseline_reeval" # <-- Change this for the correct logging also in Wandb

run_name="${prefix}llavarad"
run_name="${4:-$run_name}"
model_base="${1:-$model_base}"
model_path="${2:-$model_path}"
prediction_dir="${3:-/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/results/$run_name}"
prediction_file=$prediction_dir/test


query_file=/a2il/data/mbhosale/MrFair/physionet.org/mimc-cxr-jpeg/gpt4-reports/chat_test_MIMIC_CXR_all_dem.json # TODO remove small

# image_folder=/PATH_TO/physionet.org/files/mimic-cxr-jpg/2.0.0/files
loader="mimic_test_findings"
conv_mode="v1"

CHUNKS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

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
done

wait

cat ${prediction_file}_*.jsonl > mimic_cxr_preds.jsonl

pushd llava/eval/rrg_eval
WANDB_PROJECT="llava" WANDB_RUN_ID="llava-eval-$(date +%Y%m%d%H%M%S)" WANDB_RUN_GROUP=evaluate CUDA_VISIBLE_DEVICES=0 \
    python run.py ../../../mimic_cxr_preds.jsonl --run_name ${run_name} --output_dir ../../../${prediction_dir}/eval
popd

rm mimic_cxr_preds.jsonl