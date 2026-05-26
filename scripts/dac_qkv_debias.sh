#!/bin/bash

# Set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1

PROJECTOR="/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/checkpoints/llavarad-merged/mm_projector.bin"
model_name_or_path="/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/checkpoints/biomedclip_cxr_518-lora-3e-1e-4-20251014222727" #lmsys/vicuna-7b-v1.5 ===> For fairness finetuning we do not have to start at Vicuna weights. We can rather use llava-rad directly.
output_dir="${1:-./checkpoints}"

vision_tower="biomedclip_cxr_518"
vision_tower_config="llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json"
vision_tower_checkpoint="/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava-rad_hf/biomedclipcxr_518_checkpoint.pt" #"/data_local1/mbhosale/MrFair/mimc-cxr-jpeg/biomedclipcxr_518_checkpoint.pt"
################## VICUNA ##################


################## Data ##################
data_path="/a2il/data/mbhosale/MrFair/physionet.org/mimc-cxr-jpeg/gpt4-reports/chat_train_MIMIC_CXR_all_dem_clean.json"
loader="mimic_train_findings"
image_folder=/data_local1/mbhosale/MrFair/mimc-cxr-jpeg/files 
################## Data ##################

################## Run name ##################
epoch="${2:-1}"
bsz="${3:-6}"
lr="1e-4"
schedule="lora-${epoch}e"
export run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"
echo $run_name > run_name
################## Run name ##################

# Add at the top of dac_pretrain.sh (after shebang):
set -x  # Print each command before execution
echo "=== RUNNING DAC_PRETRAIN.SH ==="
echo "PWD: $(pwd)"
echo "Model: ${model_name_or_path}"
echo "Data: ${data_path}"
echo "Epochs: ${epoch}"
echo "Batch Size: ${bsz}"
echo "LR: ${lr}"

# Calculate steps per epoch
# Formula: steps_per_epoch = ceil(num_samples / (batch_size * num_gpus * grad_accum_steps))


# Batch size is set for 8-GPU machines, for 6 GPU set the grad acc steps to 3 for the same BS of 6.
WANDB_PROJECT="llava" WANDB_RUN_ID="llava-ft-$(date +%Y%m%d%H%M%S)" WANDB_RUN_GROUP=fine-tune \
    deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_alpha 128 \
    --model_name_or_path ${model_name_or_path} \
    --version $PROMPT_VERSION \
    --freeze_backbone False \
    --data_path ${data_path} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --vision_tower_config ${vision_tower_config} \
    --vision_tower_checkpoint ${vision_tower_checkpoint} \
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
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
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
    --fairness_config /home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/configs/fairness_finetune_mimic_cxr.json \
    --seed 42