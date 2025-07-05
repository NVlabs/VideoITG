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
#    Copyright 2024 Hao Zhang
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


from typing import List, Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import inspect
from transformers import AutoConfig, AutoModel

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..eagle_archv1 import EagleMetaModel, EagleMetaForCausalLM
from transformers import Qwen2Config, Qwen2PreTrainedModel, Qwen2Model
from transformers.utils import logging
logger = logging.get_logger(__name__)

from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2, Qwen2DecoderLayer
import time
class CustomQwen2FlashAttention2(Qwen2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super(CustomQwen2FlashAttention2, self).__init__(*args, **kwargs)
        self.is_causal = False
        
class CustomQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super(CustomQwen2DecoderLayer, self).__init__(config, layer_idx)
        self.self_attn = CustomQwen2FlashAttention2(config, layer_idx)

class EagleQwenGConfig(Qwen2Config):
    model_type = "eagle_QwenG"


class EagleQwenGModel(EagleMetaModel, Qwen2Model):
    config_class = EagleQwenGConfig

    def __init__(self, config: Qwen2Config):
        super(EagleQwenGModel, self).__init__(config)
        self.layers = nn.ModuleList(
            [CustomQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class EagleQwenG(Qwen2PreTrainedModel, EagleMetaForCausalLM):
    config_class = EagleQwenGConfig

    def __init__(self, config):
        super(EagleQwenG, self).__init__(config)
        config.model_type = "Eagle_qwenG"
        config.rope_scaling = None

        self.model = EagleQwenGModel(config)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        grounding_labels=None,
        token_types: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, token_types) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.out_proj(hidden_states)
            return logits, labels

        else:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
            batch_size = token_types.shape[0]
            seq_length = token_types.shape[1]
            image_positions = (token_types == 3)

            image_logits = []

            hidden_states = outputs[0]
            for b in range(batch_size):
                hidden_states_batch = hidden_states[b][image_positions[b]].reshape(images[b].shape[0], -1, hidden_states.shape[-1]).mean(dim=1)
                logits_batch = self.out_proj(hidden_states_batch)
                image_logits.append(logits_batch)

            loss = None
            if grounding_labels is not None:
                image_logits = torch.cat(image_logits, dim=0).float().view(-1)
                grounding_labels = torch.cat(grounding_labels).to(image_logits.device).float().view(-1)

                positive_samples = torch.sum(grounding_labels)
                negative_samples = grounding_labels.numel() - positive_samples
                pos_weight = torch.sqrt(negative_samples / max(1, positive_samples)).to(hidden_states.device)

                loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.min(torch.tensor(5.0, device=hidden_states.device), pos_weight))

                loss = loss_fct(image_logits, grounding_labels)

                if not return_dict:
                    output = (image_logits,) + outputs[1:]
                    return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=image_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )



AutoConfig.register("eagle_QwenG", EagleQwenGConfig)
AutoModel.register(EagleQwenGConfig, EagleQwenG)