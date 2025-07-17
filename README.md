# VideoITG: Improving Multimodal Video Understanding with Instructed Temporal Grounding

---

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Model License](https://img.shields.io/badge/MODEL%20License-CC%20By%20NC%204.0-red.svg)](MODEL_LICENSE)
<a href='https://github.com/exiawsh/VideoITG-PAGE'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

[arXiv]() / [Demo]()



## Introduction

<div align="center">
<img src="assets/teaser.png" width="90%">
</div>

VideoITG is an innovative approach to video understanding, designed to enhance the performance of Video Large Language Models (Video-LLMs) through informed frame selection. It tackles the complexities of real-world video scenarios by aligning frame sampling with user instructions. VideoITG employs a comprehensive pipeline that includes detailed clip-level description generation, question-guided clip retrieval, and task-specific frame selection. This results in a robust dataset of 40K videos and 480K annotations. The plug-and-play model leverages visual language alignment and reasoning, achieving superior results across multimodal benchmarks, particularly in tasks requiring precise temporal grounding.



## Updates
- [2025/04] Release the training and evaluation code of VideoITG. The pretrained model is available at [Model Card](#model-card)


## Contents
- [Models & Performance](#models--performance)
- [Visual Examples](#visual-examples)
- [Install](#install)
- [Training Data](#training-data)
- [Checkpoint Preparation](#checkpoint-preparation)
- [Training](#training)
- [Evaluation](#evaluation)


## Models & Performance
Here is the model trained on our organized 1.8M supervised fine-tuning data.
| Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | VideoLLM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Frames | LongVideoBench | MLVU | VideoMME | CG-Bench |
|----------|------------|-------------|:------:|:-------:|:----------:|:----------:|
| VideoITG-7B | InternVL2.5-8B  | 32 | 61.9 (+2.9%) | 75.0 (+7.8%) |  67.3 (+4.0%)  |    46.7 (+7.0%)    |
| VideoITG-7B | InternVL2.5-26B | 32 | 63.0 (+1.0%) | 78.9 (+6.1%) |  69.9 (+2.5)  |   48.7 (+6.0%)    |
| VideoITG-7B | LLaVA-Video-7B  | 32 | 61.6 (3.6%) | 74.6 (+8.6%) |  66.1 (+3.0%)  |      42.8 (+9.0%)     |
| VideoITG-7B | LLaVA-Video-7B  | 64 | 60.9 (+7.4%) | 76.3 (+7.6%) |  66.4 (+1.9%)  |    42.9 (8.1%)   |


## Visual Examples

<div align="center">
<img src="assets/VQA1.png" width="80%">
</div><br>

<div align="center">
<img src="assets/VQA2.png" width="80%">
</div><br>



## Install
Please following the guide here to prepare the environment on **Linux OS**.
<!-- currently does not support windows and MacOS -->

1. Clone this repository
```bash
git clone https://github.com/NVlabs/VideoITG.git
cd VideoITG
```

2. Create environment and install package
```Shell
conda create -n videoitg python=3.12 -y
conda activate videoitg
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
```

3. Install additional packages for training cases
```bash
pip install flash-attn==2.4.2.dev3 --no-build-isolation
```

## Training Data

### VideoLLM Data
For VideoLLM training, wew use the same data and stragety as LLaVA-Video, including the [Pretraining Data](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K), [OV SFT Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) and [LLaVA-Video-178K Data](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K).


### VideoITG Data


## Checkpoint Preparation
We recommend using the VideoLLM checkpoints we provided [here]() to reproduce our results.

## Training
You can train the model following:

```bash
bash scripts/videoitg/finetune-uni-64frame-qwen2-7b-grounding.sh finetune 16
```

In default we use 128 NVIDIA A100 80G GPU to conduct the training. Please modify the `per_device_train_batch_size` and `gradient_accumulation_steps` if you are using different amount of GPUs. The training for VideoITG requires 4 hours.


### Notes
If you have limited GPU resources or memory, please considering the following:

- use gradient accumulation and reduce the per-device batch size

## Evaluation

### Evaluation with LMMs-Eval
For evaluation, we use Videomme as an example.
First, using this command to run our VideoITG model and get the instructed grounding results.

```bash
bash scripts/eval_lmms_eval/videomme_grounding.sh $REPO_ID_OR_LOCAL_PATH $MODEL_NAME $CONV_MODE
```
### Notes
In our paper, we report the results of CG-Bench mini, which includes 3,000 QA pairs.

## Citation
If you find this project useful, please cite our work:
```
@article{
}
```


## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon.
- [Eagle](https://github.com/NVlabs/EAGLE): the codebase we built upon. Thanks for the great pioneer open-source project!
- [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): many thanks to the LMMs-Lab for their wonderful and easy-to-use evaluation tools!
- [LLaVA-Video-178K](https://llava-vl.github.io/blog/2024-09-30-llava-video/): we train our model with the data from LLaVA-Video-178k.

