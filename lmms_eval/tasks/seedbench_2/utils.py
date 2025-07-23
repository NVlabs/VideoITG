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


def seed_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["image"]]


def parse_choice_img(choice: str, img_token: str):
    if "jpg" in choice or "png" in choice:
        return img_token
    return choice


def seed_doc_to_text(doc, model_specific_kwargs=None):
    question = doc["question"]
    question.replace("<img>", model_specific_kwargs["img_token"])
    question += "\n" + f"A. {parse_choice_img(doc['choice_a'], model_specific_kwargs['img_token'])}\n"
    question += f"B. {parse_choice_img(doc['choice_b'], model_specific_kwargs['img_token'])}\n"
    question += f"C. {parse_choice_img(doc['choice_c'], model_specific_kwargs['img_token'])}\n"
    question += f"D. {parse_choice_img(doc['choice_d'], model_specific_kwargs['img_token'])}"
    if doc["data_type"] == "Image Generation":
        num_img_in_question = len(doc["data_id"]) - 4
        prepend_tokens = [model_specific_kwargs["img_token"]] * num_img_in_question
        question = " ".join(prepend_tokens) + "\n" + question
    return f"{question}\n{model_specific_kwargs['post_prompt']}"


def seed_process_result(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = doc["answer"]
    data_type = doc["data_type"].split(" ")
    data_type = "_".join(data_type)

    return {f"seed_{data_type}": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}, f"seed_all": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}}


def seed_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"] == result["answer"]:
            total_correct += 1
        total_count += 1
    return total_correct / total_count if total_count != 0 else 0


def seed_aggregation_result_all(results):
    score = seed_aggregation_result(results)
    stored_results = []
    for result in results:
        stored_results.append({"question_id": result["question_id"], "prediction": result["pred"]})
    with open("./seed_submission.json", "w") as f:
        json.dump(stored_results, f, indent=4)
    print("Storing files for seed_submission ...")

    return score
