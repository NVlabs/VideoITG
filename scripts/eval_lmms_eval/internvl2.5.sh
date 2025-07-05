export HF_HOME=$(realpath ~/.cache/huggingface)

python3 -m accelerate.commands.launch \
        --num_processes=8 \
        -m lmms_eval \
        --model internvl2 \
        --model_args grounding_files='grounding_videomme.json',pretrained=OpenGVLab/InternVL2_5-8B,modality=video \
        --tasks videomme \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix videomme \
        --output_path ./logs/  