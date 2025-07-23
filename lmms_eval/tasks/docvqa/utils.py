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
import json
import os

from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def docvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def docvqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def docvqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    return {"anls": {"questionId": int(questionId), "answer": pred}, "submission": {"questionId": int(questionId), "answer": pred}}


def docvqa_test_aggregate_results(results, args):
    # save results as json
    path = generate_submission_file("docvqa_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved to {path}")
