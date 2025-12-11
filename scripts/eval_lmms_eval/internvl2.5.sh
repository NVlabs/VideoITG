export HF_HOME=$(realpath ~/.cache/huggingface)

python3 -m accelerate.commands.launch \
        --num_processes=8 \
        -m lmms_eval \
        --model internvl2 \
        --model_args frame_indices_jsonl=results\videomme_result_512.jsonl,pretrained=OpenGVLab/InternVL2_5-8B,modality=video,num_frame=32 \
        --tasks videomme \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix videomme \
        --output_path ./logs/  
