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
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MLPProjector(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        vision_token_num: int = 8192,
        vision_min_num: int = 1,
    ):
        super().__init__()

        self.vision_min_num = vision_min_num
        self.vision_token_num = vision_token_num
        self.out_projection = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim
        self.dim = dim

    def forward(self, x) -> torch.Tensor:
        x_out = []

        for i, batch in enumerate(x):
            if batch.ndim == 2:
                batch = batch.unsqueeze(0)
            T, P, C = batch.shape
            ori_HW = int(P**0.5)
            hidden_states = batch
            HW = math.floor((self.vision_token_num / T)**0.5)
            # print("min_num", self.vision_min_num)
            # print("token_num", self.vision_token_num)
            if self.training:
                HW = min(torch.randint(self.vision_min_num, HW+1, (1,)).item(), ori_HW)
            else:
                HW = min(HW, ori_HW)

            if HW < ori_HW:
                if T > 512:
                    fast_states = []
                    for i in range(0, T, 512):
                        end_idx = min(i + 512, T)
                        curr_states = F.interpolate(hidden_states[i:end_idx].view(end_idx-i, ori_HW, ori_HW, C).permute(0, 3, 1, 2), 
                                                  size=(HW, HW), mode='bilinear', align_corners=False)
                        fast_states.append(curr_states)
                    fast_states = torch.cat(fast_states, dim=0).contiguous()
                else:
                    fast_states = F.interpolate(hidden_states.view(T, ori_HW, ori_HW, C).permute(0, 3, 1, 2), 
                                              size=(HW, HW), mode='bilinear', align_corners=False).contiguous()
                fast_states = fast_states.permute(0, 2, 3, 1).reshape(T, -1,  C)
            else:
                fast_states = hidden_states
            
            out = self.out_projection(fast_states)
            x_out.append(out.flatten(0, 1))

        return x_out