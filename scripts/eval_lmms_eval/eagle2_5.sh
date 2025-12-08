export HF_HOME=$(realpath ~/.cache/huggingface)

python3 -m accelerate.commands.launch \
        --num_processes=8 \
        -m lmms_eval \
        --model eagle2_5 \
        --model_args pretrained=nvidia/Eagle2.5-8B,modality=video,num_frame=32,frame_indices_jsonl=results/cg_bench_512_results.jsonl \
        --tasks cgbench_subtitles \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix cgbench_subtitles \
        --output_path ./logs/  

# 64 frame
# |       Tasks        |Version|Filter|n-shot|Metric |   |Value |   |Stderr|
# |--------------------|-------|------|-----:|-------|---|-----:|---|------|
# |longvideobench_val_v|Yaml   |none  |     0|lvb_acc|↑  |0.6515|±  |   N/A|

# videomme eagle2.5-8b
# | Tasks  |Version|Filter|n-shot|         Metric          |   | Value |   |Stderr|
# |--------|-------|------|-----:|-------------------------|---|------:|---|------|
# |videomme|Yaml   |none  |     0|videomme_perception_score|↑  |66.2593|±  |   N/A|
# 2025-09-20 01:37:59.598 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: short:  78.8%
# 2025-09-20 01:37:59.598 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: medium:  64.1%                                                   
# 2025-09-20 01:37:59.599 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: long:  55.9%   

# mlvu_dev eagle2.5-8b
# | Tasks  |Version|Filter|n-shot|       Metric       |   | Value |   |Stderr|
# |--------|-------|------|-----:|--------------------|---|------:|---|------|
# |mlvu_dev|Yaml   |none  |     0|mlvu_percetion_score|↑  |66.6465|±  |   N/A|


# 2025-09-23 10:38:18.327 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: short:  80.0%
# 2025-09-23 10:38:18.327 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: medium:  67.8%
# 2025-09-23 10:38:18.328 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: long:  60.3%   
# | Tasks  |Version|Filter|n-shot|         Metric          |   | Value |   |Stderr|
# |--------|-------|------|-----:|-------------------------|---|------:|---|------|
# |videomme|Yaml   |none  |     0|videomme_perception_score|↑  |69.3704|±  |   N/A|

# 2025-11-12 17:37:51.773 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: short:  80.3%
# 2025-11-12 17:37:51.774 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: medium:  65.9%
# 2025-11-12 17:37:51.774 | INFO     | utils:videomme_aggregate_results:315 - Evaluation on video Type: long:  58.4%
# | Tasks  |Version|Filter|n-shot|         Metric          |   | Value |   |Stderr|
# |--------|-------|------|-----:|-------------------------|---|------:|---|------|
# |videomme|Yaml   |none  |     0|videomme_perception_score|↑  |68.2222|±  |   N/A|