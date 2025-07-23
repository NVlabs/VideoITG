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
from collections import Counter

from lmms_eval.api.filter import Filter


class TakeFirstFilter(Filter):
    def __init__(self) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    def apply(self, resps, docs):
        """
        Assuming each entry of `resps` is a list of model responses, we discard all but the first response.
        """
        return map(lambda r: r[0], resps)


class TakeKFilter(Filter):
    def __init__(self, *args, **kwargs) -> None:
        self.k = kwargs.pop("k")

        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # check we have at least k responses per doc, else we can't take the first k
        assert len(resps[0]) >= self.k, f"Need at least {self.k} responses per doc to take first {self.k}, but got {len(resps[0])} only! Please increase TaskConfig.repeats ."
        return map(lambda r: r[: self.k], resps)


class MajorityVoteFilter(Filter):
    def __init__(self) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    def apply(self, resps, docs):
        """
        Each entry of `resps` is a list of model responses.
        We select the response that occurs most frequently in each entry of `resps`.
        """

        def select_majority(resp):
            counts = Counter(resp)
            vote = counts.most_common(1)[0][0]
            return vote

        return map(lambda r: [select_majority(r)], resps)
