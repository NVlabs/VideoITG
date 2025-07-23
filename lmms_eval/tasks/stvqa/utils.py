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

from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def stvqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def stvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def stvqa_process_results(doc, results):
    answer = results[0]
    return {"submission": {"question_id": int(doc["question_id"]), "answer": answer}}


def stvqa_aggregate_submissions(results, args):
    file = generate_submission_file("stvqa_test_for_submission.json", args)
    with open(file, "w") as f:
        json.dump(results, f)
    logger.info(f"Results saved to {file}")
