export HF_HOME=$(realpath ~/.cache/huggingface)

python3 -m accelerate.commands.launch \
        --num_processes=8 \
        -m lmms_eval \
        --tasks videomme \
        --model videoitg \
        --model_args target_fps=1,output_dir=./videomme_result_512,pretrained=nvidia/VideoITG-8B,num_frames=512 \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix videomme \
        --output_path ./logs/  