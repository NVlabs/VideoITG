export HF_HOME=$(realpath ~/.cache/huggingface)

python3 -m accelerate.commands.launch \
        --num_processes=8 \
        -m lmms_eval \
        --model eagle_grounding \
        --model_args target_fps=1,grounding_folder=./grounding_videomme,pretrained=./checkpoints-finetune/eagle-qwen2-7b-finetune-grounding,num_frames=512 \
        --tasks videomme \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix videomme \
        --output_path ./logs/  
