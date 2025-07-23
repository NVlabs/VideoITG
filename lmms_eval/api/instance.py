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
from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class Instance:
    request_type: Literal["loglikelihood", "generate_until", "generate_until_multi_round"]
    arguments: tuple
    idx: int
    metadata: Tuple[str, int, int] = field(default_factory=lambda: (None, None, None))  # TODO: better typehints here
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    # initialized after init
    task_name: str = None
    doc_id: str = None
    repeats: str = None
    doc: dict = None

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata["task"], self.metadata["doc_id"], self.metadata["repeats"]

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
