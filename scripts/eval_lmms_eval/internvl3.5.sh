export HF_HOME=$(realpath ~/.cache/huggingface)

python3 -m accelerate.commands.launch \
        --num_processes=8 \
        -m lmms_eval \
        --model internvl3_5 \
        --model_args pretrained=OpenGVLab/InternVL3_5-8B,modality=video,frame_indices_jsonl=results/videomme_result_512.jsonl \
        --tasks videomme \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix videomme \
        --output_path ./logs/  


# 2025-11-14 00:58:10.757 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: short:  78.4%
# 2025-11-14 00:58:10.758 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: medium:  65.9%
# 2025-11-14 00:58:10.758 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: long:  59.0%

# | Tasks  |Version|Filter|n-shot|         Metric          |   | Value |   |Stderr|
# |--------|-------|------|-----:|-------------------------|---|------:|---|------|
# |videomme|Yaml   |none  |     0|videomme_perception_score|↑  |67.7778|±  |   N/A|

# |Tasks|Version|Filter|n-shot|       Metric        |   |Value |   |Stderr|
# |-----|-------|------|-----:|---------------------|---|-----:|---|------|
# |mlvu |Yaml   |none  |     0|mlvu_perception_score|↑  |74.149|±  |   N/A|

# |       Tasks        |Version|Filter|n-shot|Metric |   |Value |   |Stderr|
# |--------------------|-------|------|-----:|-------|---|-----:|---|------|
# |longvideobench_val_v|Yaml   |none  |     0|lvb_acc|↑  |0.6567|±  |   N/A|

# |      Tasks      |Version|Filter|n-shot|         Metric         |   |Value|   |Stderr|
# |-----------------|-------|------|-----:|------------------------|---|----:|---|------|
# |cgbench_subtitles|Yaml   |none  |     0|cgbench_perception_score|↑  | 47.6|±  |   N/A|