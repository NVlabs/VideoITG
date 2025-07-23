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
import os

import yaml

splits = ["val2014"]
tasks = ["vqa"]

if __name__ == "__main__":
    dump_tasks = []
    for task in tasks:
        for split in splits:
            yaml_dict = {"group": f"ok_vqa", "task": f"ok_vqa_{split}", "include": f"_default_template_{task}_yaml", "test_split": split}
            if split == "train":
                yaml_dict.pop("group")
            else:
                dump_tasks.append(f"ok_vqa_{split}")

            save_path = f"./ok_vqa_{split}.yaml"
            print(f"Saving to {save_path}")
            with open(save_path, "w") as f:
                yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    group_dict = {"group": "ok_vqa", "task": dump_tasks}

    with open("./_ok_vqa.yaml", "w") as f:
        yaml.dump(group_dict, f, default_flow_style=False, indent=4)
