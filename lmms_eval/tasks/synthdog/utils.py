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
from lmms_eval.tasks.synthdog.donut_evaluator import JSONParseEvaluator

evaluator = JSONParseEvaluator()


def synthdog_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [doc["image"].convert("RGB")]


def synthdog_doc_to_target(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [json.loads(doc["ground_truth"])["gt_parse"]["text_sequence"]]


def synthdog_process_results(doc, results):
    pred = {"output": results[0].lower().strip()}
    gt_ans = json.loads(doc["ground_truth"])["gt_parse"]

    predictions = []
    ground_truths = []
    accs = []

    score = evaluator.cal_acc(pred, gt_ans)

    accs.append(score)

    predictions.append(pred)
    ground_truths.append(gt_ans)

    return {
        "tree_edit_distance": {"score": score, "prediction": pred, "ground_truth": gt_ans},
    }


def synthdog_aggregate_ted(results, args):
    final_score = 0
    for result in results:
        final_score += result["score"]
    return final_score
