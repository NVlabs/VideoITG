import logging
from typing import List, Tuple, Dict, Optional
import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from PIL import Image
import PIL
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")

# AutoModel._tp_plan = []

@register_model("eagle2_5")
class Eagle2_5(lmms):
    def __init__(
        self,
        pretrained: str = "nvidia/Eagle2.5-8B",
        sys_prompt: str = None,
        sys_prompt_version: str = 'eagle2_5',
        device: str = "cuda",
        device_map: str = "cuda",
        batch_size: int = 1,
        num_frame: int = 64,
        num_layers=None,
        frame_indices_jsonl: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained
        self.num_frame = num_frame
        self.docid_to_indices: Dict[int, List[int]] = {}
        
        if sys_prompt is None:
            if sys_prompt_version == 'eagle2_5':
                self.sys_prompt = "You are a helpful assistant."
            elif sys_prompt_version == 'eagle2_6':
                self.sys_prompt = "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers. Please answer in detail."
        else:
            self.sys_prompt = sys_prompt
        

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for Eagle 2.5 / 2.6, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        # timeout = timedelta(weeks=52)

        # accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timeout)])
        # print("before accelerator")
        accelerator = Accelerator()
        
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
                
        else:
            if device_map == "auto":
                eval_logger.info("Using device_map:auto with num_processes=1")
            else:
                eval_logger.info(f"Using single device: {self._device}")

            self._device = torch.device(device)
            self.device_map = device_map
            self._rank = 0
            self._world_size = 1

        # print("Device map:", self.device_map)
        self._model = AutoModel.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True
        )

        self._model.eval()  # Set model to evaluation mode
        self._config = self._model.config


        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        self._tokenizer = self.processor.tokenizer
        self._tokenizer.padding_side = "left"

        # Optionally load precomputed frame indices from a JSONL file
        if frame_indices_jsonl is not None and os.path.isfile(frame_indices_jsonl):
            try:
                with open(frame_indices_jsonl, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_no, line in enumerate(f, start=1):
                        stripped = line.strip()
                        if not stripped:
                            # Skip empty/whitespace-only lines
                            continue
                        try:
                            record = json.loads(stripped)
                        except json.JSONDecodeError as e:
                            eval_logger.warning(f"Failed to parse JSON at {frame_indices_jsonl}:{line_no}: {e}")
                            continue
                        if 'doc_id' in record and 'index' in record and isinstance(record['index'], list):
                            try:
                                did = int(record['doc_id'])
                                idx_list = [int(x) for x in record['index']]
                                # If multiple entries per doc_id exist, keep the first seen
                                if did not in self.docid_to_indices:
                                    self.docid_to_indices[did] = idx_list
                            except Exception as e:
                                eval_logger.warning(f"Failed to load frame indices record at {frame_indices_jsonl}:{line_no}: {e}")
                                continue
                if self.docid_to_indices:
                    eval_logger.info(f"Loaded frame indices for {len(self.docid_to_indices)} doc_ids from {frame_indices_jsonl}")
                else:
                    eval_logger.warning(f"No valid frame indices loaded from {frame_indices_jsonl}")
            except Exception as e:
                eval_logger.warning(f"Failed to load frame indices jsonl {frame_indices_jsonl}: {e}")

        if accelerator.num_processes > 1:
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")

            assert accelerator.distributed_type in [
                DistributedType.FSDP, 
                DistributedType.MULTI_GPU, 
                DistributedType.DEEPSPEED
            ], "Unsupported distributed type. Only DDP, FSDP and DEEPSPEED are supported."
            
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                ds_config = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **ds_config)
                eval_logger.info("Using DEEPSPEED. Ensure you ran `accelerate config` with zero stage 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            
        self.accelerator = accelerator

        print(f"Accelerator distributed type: {self.accelerator.distributed_type}")
        # if accelerator.num_processes > 1:
        #     accelerator.wait_for_everyone()

    # @property
    # def config(self):
    #     # return the associated transformers.AutoConfig for the given pretrained model.
    #     return self.worker.llm.config

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
        return self.tokenizer.eos_token_id

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

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def read_video_custom(self, video_path, max_frames: int):
        if isinstance(video_path, (list, tuple)):
            vp = video_path[0]
        else:
            vp = video_path
        vr = VideoReader(vp, ctx=cpu(0))
        total_frames = len(vr)
        max_frames_num = min(max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, max_frames_num, dtype=int).tolist()
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # TCHW
        avg_fps = vr.get_avg_fps()
        timestamps = [i / avg_fps for i in indices]
        sampled_fps = avg_fps * max_frames_num / max(total_frames, 1e-6)
        return video_tensor, sampled_fps, timestamps

    def read_video_with_indices(self, video_path, indices: List[int], max_frames: int):
        if isinstance(video_path, (list, tuple)):
            vp = video_path[0]
        else:
            vp = video_path
        vr = VideoReader(vp, ctx=cpu(0))
        total_frames = len(vr)
        # 先采样前max_frames帧（顺序保留），再排序
        sanitized: List[int] = []
        count = 0
        for idx in indices:
            if not isinstance(idx, (int, np.integer)):
                continue
            ii = int(idx)
            if 0 <= ii < total_frames:
                sanitized.append(ii)
                count += 1
            if count >= max_frames:
                break
        if len(sanitized) == 0:
            # 如果没有有效帧，回退到均匀采样
            return self.read_video_custom(video_path, max_frames=max_frames)
        sanitized = sorted(sanitized)
        frames = vr.get_batch(sanitized).asnumpy()
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
        avg_fps = vr.get_avg_fps()
        timestamps = [i / avg_fps for i in sanitized]
        # 采样帧率定义为采样帧数除以时间跨度
        if len(sanitized) > 1:
            duration_s = (max(sanitized) - min(sanitized)) / max(avg_fps, 1e-6)
            sampled_fps = len(sanitized) / max(duration_s, 1e-6)
        else:
            sampled_fps = avg_fps
        return video_tensor, sampled_fps, timestamps

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            if "<image>" in contexts:
                contexts = contexts.replace("<image>", "")
            if "<video>" in contexts:
                contexts = contexts.replace("<video>", "")

            messages = [
                {"role": "system", "content": self.sys_prompt},
                {'role': 'user', 'content': []}
            ]

            image_inputs = []
            video_inputs = []
            video_kwargs = {"fps": [], "timestamps": []}
            max_frames = int(gen_kwargs.get("num_frame", self.num_frame))
            for visual in visuals:
                if isinstance(visual, str):
                    messages[-1]["content"].append({"type": "video", "video": visual})
                    # Prefer JSONL-provided indices per doc_id when available
                    if self.docid_to_indices:
                        use_indices = self.docid_to_indices.get(int(doc_id))
                        if use_indices is not None and isinstance(use_indices, list) and len(use_indices) > 0:
                            video_tensor, sampled_fps, sampled_timestamps = self.read_video_with_indices(
                                visual, use_indices, max_frames=max_frames
                            )
                        else:
                            video_tensor, sampled_fps, sampled_timestamps = self.read_video_custom(visual, max_frames=max_frames)
                    else:
                        video_tensor, sampled_fps, sampled_timestamps = self.read_video_custom(visual, max_frames=max_frames)
                    video_inputs.append(video_tensor)
                    video_kwargs["fps"].append(sampled_fps)
                    video_kwargs["timestamps"].append(sampled_timestamps)
                elif isinstance(visual, PIL.Image.Image):
                    messages[-1]["content"].append({"type": "image", "image": visual})
                    image_inputs.append(visual)

            messages[-1]["content"].append({"type": "text", "text": contexts})

            # print("Messages:", messages)

            text_list = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
            inputs = self.processor(
                text=text_list,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                videos_kwargs=video_kwargs,
            )
            
            inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            gen_kwargs['do_sample'] = True if gen_kwargs.get("temperature", 0) > 0 else False

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    pad_token_id=self.eot_token_id,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    do_sample=gen_kwargs["do_sample"]
                )
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                res.append(output_text)
            
            pbar.update(1)
        
        pbar.close()
        return res


    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood calculation not implemented for Eagle2.5-HF")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented for Eagle2.5-HF")
