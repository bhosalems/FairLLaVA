#!/bin/bash

# This is for normal LLaVA finetuning without debiasing
# Set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1

model_base="lmsys/vicuna-7b-v1.5" 
output_dir="${1:-./checkpoints}"

PROJECTOR="/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/checkpoints/openai/clip-vit-large-patch14-336-pt-3e-1e-3-20251117161822/mm_projector.bin" #"/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/checkpoints/llavarad-merged/mm_projector.bin" # generated using pretrain.sh
vision_tower="openai/clip-vit-large-patch14-336"
# vision_tower_config="llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json"
# vision_tower_checkpoint="/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava-rad_hf/biomedclipcxr_518_checkpoint.pt"
################## VICUNA ##################


################## Data ##################
data_path="/a2il/data/mbhosale/MrFair/dataset/amazon_english_llava_original.json"
loader="iam_train" # just same as iam so reuse
image_folder="/a2il/data/mbhosale/MrFair/dataset/Amazon_English_only_Images"
################## Data ##################

################## Run name ##################
epoch="${3:-3}"
bsz="${3:-6}"
lr="1e-4"
schedule="lora-${epoch}e"
export run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"
echo $run_name > run_name
PORT=$(shuf -i 20000-29999 -n 1)
################## Run name ##################


# Batch size is set for 8-GPU machines, for 6 GPU set the grad acc steps to 3 for the same BS of 6.
WANDB_PROJECT="llava" WANDB_RUN_ID="llava-ft-$(date +%Y%m%d%H%M%S)" WANDB_RUN_GROUP=fine-tune \
    deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_alpha 128 \
    --model_name_or_path ${model_base} \
    --version $PROMPT_VERSION \
    --data_path ${data_path} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --pretrain_mm_mlp_adapter ${PROJECTOR} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${output_dir}/${run_name} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 20 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --run_name ${run_name} \
    --output_hidden_states True \
    --fairness_finetune True \
    --fairness_labels True \
    --fairness_config "/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/configs/fairness_finetune_hrt_mi.json" \