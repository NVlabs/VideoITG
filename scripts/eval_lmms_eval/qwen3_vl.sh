export HF_HOME=$(realpath ~/.cache/huggingface)

# Example: evaluate Qwen3-VL with VideoITG-selected frames (VideoMME).
# 1) Run VideoITG to generate per-video frame scores (see videomme_grounding.sh).
# 2) Select Top-K frames and write them into `frame_indices_jsonl` (one line per sample).
# 3) Run this script.

python3 -m accelerate.commands.launch \
        --num_processes=8 \
        -m lmms_eval \
        --model qwen3_vl \
        --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,frame_indices_jsonl=results/videomme_result_512.jsonl,max_num_frames=32 \
        --tasks videomme \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix videomme \
        --output_path ./logs/

