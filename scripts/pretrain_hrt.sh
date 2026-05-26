#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

model_base=lmsys/vicuna-7b-v1.5
output_dir="${1:-./checkpoints}"

data_path="/a2il/data/mbhosale/MrFair/dataset/amazon_english_llava_train.json"
loader="iam_train" # just same as iam so reuse
image_folder="/data_local1/mbhosale/MrFair/Amazon_English_only_Images/"
################## Run name ##################
vision_tower="openai/clip-vit-large-patch14-336"
# vision_tower_config=""
# vision_tower_checkpoint="" 

epoch="${2:-6}"
bsz="${3:-12}"
grad_acc="${4:-4}"
lr="1e-3"
schedule="pt-${epoch}e"
run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"
echo $run_name > run_name
################## Run name ##################

# Global batch size should be 256*grad_acc
# Trained on 6 GPUS A6000, max batch size 8. 

WANDB_RUN_ID="llava-pt-$(date +%Y%m%d%H%M%S)" WANDB_PROJECT="llava" WANDB_RUN_GROUP=pre-train \
    deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${model_base} \
    --version plain \
    --data_path ${data_path} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${output_dir}/${run_name} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${grad_acc} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${run_name}
