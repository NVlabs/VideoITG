import os
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
import numpy as np
import math
from datetime import timedelta
from transformers import AutoConfig
from huggingface_hub import snapshot_download
import requests
import json
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

import subprocess

from loguru import logger as eval_logger

from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from eagle.conversation import conv_templates, SeparatorStyle
from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import av
from av.codec.context import CodecContext
from PIL import Image
import time
def record_video_length_stream(container, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return frames


# This one works for all types of video
def record_video_length_packet(container):
    frames = []
    # https://github.com/PyAV-Org/PyAV/issues/1269
    # https://www.cnblogs.com/beyond-tester/p/17641872.html
    # context = CodecContext.create("libvpx-vp9", "r")
    for packet in container.demux(video=0):
        for frame in packet.decode():
            frames.append(frame)
    return frames

def get_seq_frames(total_frames, original_fps, target_fps, num_frm, multiple=1):
    sample_fps = round(original_fps / target_fps)
    frame_idx = [i for i in range(0, total_frames, sample_fps)]
    if len(frame_idx) < num_frm:
        while len(frame_idx) % multiple != 0:
            frame_idx.append(0)
        # If we have fewer frames than num_frm, just return all the frames
        return frame_idx 
    scale = len(frame_idx) / num_frm
    uniform_idx = [int(i * scale) for i in range(num_frm)]
    frame_idx = [frame_idx[i] for i in uniform_idx]
    return frame_idx

def load_video(video_path, num_frames=64, target_fps=1):
    try:
        video = VideoReader(video_path, ctx=cpu(0))
        fps = video.get_avg_fps()
        total_frames = len(video)
       
        sampled_frames = get_seq_frames(int(total_frames), fps, target_fps, num_frames)

        frames = video.get_batch(sampled_frames).asnumpy()
        images = [Image.fromarray(frame) for frame in frames]
    except:
        video = av.open(video_path)
        video_stream = video.streams.video[0]
        fps = float(video_stream.average_rate)
        
        if "webm" not in video_path and "mkv" not in video_path:
            try:
                total_frames = video_stream.frames
            except:
                frames = record_video_length_packet(video)
                total_frames = len(frames)
        else:
            frames = record_video_length_packet(video) 
            total_frames = len(frames)
            
        sampled_frames = get_seq_frames(int(total_frames), fps, target_fps, num_frames)
        
        frames = []
        video.seek(0)
        for i, frame in enumerate(video.decode(video=0)):
            if i in sampled_frames:
                frames.append(frame.to_ndarray(format='rgb24'))
        images = [Image.fromarray(frame) for frame in frames]
        video.close()
        
    return images, sampled_frames

@register_model("eagle_grounding")
class Eagle_grounding(lmms):
    def __init__(
        self,
        pretrained: str = "/lustre/fs12/portfolios/llmservice/users/shihaow/dev/temporal_grounding/checkpoints-finetune",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_flash_attn=True,
        device_map="",
        conv_template="qwen_1_5",
        use_cache=True,
        truncate_context=False,
        num_frames: int = 512,
        target_fps: int = 1,
        multiple: int = 1,
        output_dir = "./longvideobench_val_v_grounding_patchv1",
        **kwargs,
    ) -> None:
        super().__init__()

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        self.pretrained = pretrained
        # self.model_path = snapshot_download(self.pretrained)
        self.model_path = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.num_frames = num_frames
        self.output_dir = output_dir
        
        accelerator.wait_for_everyone()
        self._tokenizer, self._model, self.image_processor, self._max_length = load_pretrained_model(
            self.model_path,
            None,
            self.model_name,
            device_map=self.device_map,
            use_flash_attn=use_flash_attn,
        )

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        self.target_fps = target_fps
        self.multiple = multiple
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests):
        os.makedirs(self.output_dir, exist_ok=True)
        res = []
        total_requests = len(requests)

        pbar = tqdm(total=total_requests, disable=(self.rank != 0), desc="Model Responding")

        def process_request(contexts, gen_kwargs, doc_to_visual, doc_id, task, split):
            outputs = {}
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]

            visuals = self.flatten(visuals)
            videos = []
            for visual in visuals:
                if isinstance(visual, Image.Image):
                    image_tensor = self.image_processor.preprocess(visual, return_tensors="pt")["pixel_values"].half().cuda()
                    videos.append(image_tensor)
                else:
                    video, sampled_frames = load_video(visual, self.num_frames, self.target_fps)
                    video_tensor = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                    videos.append(video_tensor)     

            prompt = DEFAULT_IMAGE_TOKEN + contexts + "\n"
            
            # prompt = DEFAULT_IMAGE_TOKEN + "Analyze the video, focusing on its cinematography and narrative structure." + "\n"

            print(prompt)
            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            with torch.inference_mode():
                response = self.model(
                input_ids,
                attention_mask=attention_masks,
                images=videos)
                logits = response.logits[0].sigmoid().view(-1)
                values, indices = torch.sort(logits, descending=True)
                indices = indices.tolist()
                values = values.tolist()
                frame_idx = [sampled_frames[i] for i in indices]
                
                values = [round(v, 2) for v in values]

            outputs['index'] = frame_idx
            outputs['logits'] = values
            outputs['num_frames'] = len(videos)
            outputs['contexts'] = contexts
            outputs['video_path'] = visual
        
            with open(os.path.join(self.output_dir, f"""{doc_id}.json"""), 'w') as f:
                json.dump(outputs, f, indent=4)
 

            return outputs

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            outputs = process_request(contexts, gen_kwargs, doc_to_visual, doc_id, task, split)
            pbar.update(1)
            res.append(outputs)

        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        return super().loglikelihood(requests)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size
    
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids
    
    def generate_until_multi_round(self, requests) -> List[str]:
        return self.generate_until(requests)