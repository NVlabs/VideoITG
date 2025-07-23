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
from lmms_eval.api.filter import Filter


class LowercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.lower() for resp in inst]

        return [filter_set(resp) for resp in resps]


class UppercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.upper() for resp in inst]

        return [filter_set(resp) for resp in resps]


class MapFilter(Filter):
    def __init__(self, mapping_dict: dict = {}, default_value=None) -> None:
        """
        Initializes the MapFilter with a given mapping dictionary and default value.

        Args:
        - mapping_dict (dict): A dictionary containing the key-value mappings.
                               Default is an empty dictionary.
        - default_value (Any): The value to be returned when a key is not found in the mapping_dict.
                               Default is None.

        Example:
        mapper = MapFilter({'A': 1, 'B': 2}, default_value=0)
        """
        assert isinstance(mapping_dict, dict), "Provided mapping_dict is not a dictionary"
        self.mapping_dict = mapping_dict
        self.default_value = default_value

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp, self.default_value) for resp in inst]

        return [filter_set(resp) for resp in resps]
