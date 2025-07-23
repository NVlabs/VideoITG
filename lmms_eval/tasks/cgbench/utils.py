# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adopted from lmms-eval from https://github.com/EvolvingLMMs-Lab/lmms-eval. Below is the original copyright:
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import decord
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "cgbench_subtitles.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    total_frame = len(vr)
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame

def cgbench_doc_to_text_subtitle(doc, lmms_eval_specific_kwargs=None):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_uid"] + ".mp4"
    subtitle_path = os.path.join(cache_dir, "cg_subtitles", doc["video_uid"] + ".srt")
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below: \n"
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        try:
            frame_num = lmms_eval_specific_kwargs["frame_num"]
            subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
            if frame_num == -1:
                frame_num = total_frame
            uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

            subtitle_by_frame_idx = []
            for frame_idx in uniform_sampled_frames:
                for idx, title in enumerate(subtitle_by_frame):
                    if frame_idx < title[1] and frame_idx >= title[0]:
                        subtitle_by_frame_idx.append(idx)
            subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

            textlist = []
            for idx in subtitle_by_frame_idx:
                raw_text = subtitle_by_frame[idx][2]
                try:
                    textlist.append(raw_text)
                except:
                    continue
            subtitle_text = "\n".join(textlist)
            subtitle = subtitle_text
        except:
            subtitle = "No subtitles available"

    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter of the correct option."
    question = doc["question"]
    option = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(doc["choices"])])
    question = question + "\n" + option
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + "The best answer is:"
    return full_prompt


def cgbench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_uid"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def cgbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter of the correct option."
    question = doc["question"]
    option = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(doc["choices"])])
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
    # full_prompt = question
    return full_prompt


# Frames + Subs
# This video's subtitles are listed below:
# 【subtitles】

# Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:
# Frames / Frames + Audio
# Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
# 【question】
# The best answer is:



def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEFGHIJKLMN]", s):
        return ""

    matches = re.search(r"[ABCDEFGHIJKLMN]", s)
    if matches is None:
        return ""
    return matches[0]



def cgbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)

    category = doc["domain"]
    sub_category = doc["sub_category"]
    data_dict = {"question_id": doc["qid"], "duration": doc["duration"], "category": category, "sub_category": sub_category, "pred_answer": pred_ans, "answer": doc["right_answer"]}

    return {f"cgbench_perception_score": data_dict}


def cgbench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    answered = 0
    correct = 0
    for result in results:
        answered += 1
        correct += result["pred_answer"] == result["answer"]
    eval_logger.info(f"Overall Performance: {100 * correct / answered if answered > 0 else 0 : .1f}%")
    return 100 * correct / answered if answered > 0 else 0
