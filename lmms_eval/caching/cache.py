# Adopted from lmms-eval from https://github.com/EvolvingLMMs-Lab/lmms-eval. Below is the original copyright:
# MIT License

# Copyright (c) 2024 LMMs-Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import hashlib
import os

import dill

from lmms_eval.loggers.utils import _handle_non_serializable
from lmms_eval.utils import eval_logger

MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

OVERRIDE_PATH = os.getenv("LM_HARNESS_CACHE_PATH")


PATH = OVERRIDE_PATH if OVERRIDE_PATH else f"{MODULE_DIR}/.cache"

# This should be sufficient for uniqueness
HASH_INPUT = "EleutherAI-lm-evaluation-harness"

HASH_PREFIX = hashlib.sha256(HASH_INPUT.encode("utf-8")).hexdigest()

FILE_SUFFIX = f".{HASH_PREFIX}.pickle"


def load_from_cache(file_name):
    try:
        path = f"{PATH}/{file_name}{FILE_SUFFIX}"

        with open(path, "rb") as file:
            cached_task_dict = dill.loads(file.read())
            return cached_task_dict

    except Exception:
        eval_logger.debug(f"{file_name} is not cached, generating...")
        pass


def save_to_cache(file_name, obj):
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    file_path = f"{PATH}/{file_name}{FILE_SUFFIX}"

    serializable_obj = []

    for item in obj:
        sub_serializable_obj = []
        for subitem in item:
            if hasattr(subitem, "arguments"):  # we need to handle the arguments specially since doc_to_visual is callable method and not serializable
                serializable_arguments = tuple(arg if not callable(arg) else None for arg in subitem.arguments)
                subitem.arguments = serializable_arguments
            sub_serializable_obj.append(_handle_non_serializable(subitem))
        serializable_obj.append(sub_serializable_obj)

    eval_logger.debug(f"Saving {file_path} to cache...")
    with open(file_path, "wb") as file:
        file.write(dill.dumps(serializable_obj))


# NOTE the "key" param is to allow for flexibility
def delete_cache(key: str = ""):
    files = os.listdir(PATH)

    for file in files:
        if file.startswith(key) and file.endswith(FILE_SUFFIX):
            file_path = f"{PATH}/{file}"
            os.unlink(file_path)
