#!/bin/bash
NAME=$1

# export WANDB_DISABLED="true"
export WANDB_PROJECT="eagle"
export WANDB_RUN_ID=${NAME}
export WANDB_RESUME="allow"

echo "MASTER_ADDR=$MASTER_ADDR"
n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "node rank:" $SLURM_PROCID

python -m torch.distributed.run \
    --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port 25031 \
    train_mem.py \
    --deepspeed ./scripts/zero1.json \
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --version qwen_1_5 \
    --data_path llava-one-vision-sampled/llava-ov-image-merge.json \
    --image_folder dataset/llava-one-vision-sampled/ \
    --vision_tower "google/siglip-so400m-patch14-384" \
    --pretrain_mm_mlp_adapter ./checkpoints-pretrain/eagle-qwen2-7b-pretrain/mm_projector.bin \
    --mm_projector_type patch_mergerv2 \
    --tune_mm_mlp_adapter False \
    --mm_use_4_vision_tokens False \
    --video_frames 64 \
    --fps 1 \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints-finetune/eagle-qwen2-7b-finetune-uni-ov-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none \
    --run_name ${NAME}


