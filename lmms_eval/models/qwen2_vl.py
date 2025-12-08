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
import base64
import json
import os
from io import BytesIO
from typing import List, Optional, Tuple, Union, Dict

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("qwen2_vl")
class Qwen2_VL(lmms):
    """
    Qwen2_VL Model
    "https://github.com/QwenLM/Qwen2-VL"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        max_pixels: int = 12845056,
        min_pixels: int = 3136,
        max_num_frames: int = 8,
        frame_indices_jsonl: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if use_flash_attention_2:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype="auto",
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype="auto",
                device_map=self.device_map
            ).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        # 限制每帧最大像素，保持长宽比（用户需求：不超过 768*28*28 像素）
        self.max_pixels_per_frame = 768 * 28 * 28

        # doc_id -> list(frame indices)
        self.docid_to_indices: Dict[int, List[int]] = {}
        if frame_indices_jsonl is not None and os.path.isfile(frame_indices_jsonl):
            try:
                with open(frame_indices_jsonl, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_no, line in enumerate(f, start=1):
                        stripped = line.strip()
                        if not stripped:
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
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return getattr(self, "_max_length", None)

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    # ---------- Video reading helpers (decord) ----------

    def _resize_frames_keep_aspect(self, frames: np.ndarray, max_pixels_per_frame: int) -> np.ndarray:
        """
        将视频帧按长宽比统一缩放，确保单帧像素数量 H*W 不超过 max_pixels_per_frame。
        输入/输出: frames 形状为 (T, H, W, C), dtype=uint8。
        """
        if frames is None or frames.size == 0:
            return frames

        if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
            # 非预期形状，直接返回
            return frames

        T, H, W, C = frames.shape
        if H * W <= max_pixels_per_frame:
            return frames

        scale = (max_pixels_per_frame / float(H * W)) ** 0.5
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))

        # 由于四舍五入可能超过上限，做一次收缩
        while new_h * new_w > max_pixels_per_frame and (new_h > 1 or new_w > 1):
            if new_h >= new_w and new_h > 1:
                new_h -= 1
            elif new_w > 1:
                new_w -= 1
            else:
                break

        resized = np.empty((T, new_h, new_w, C), dtype=np.uint8)
        for t in range(T):
            pil_img = Image.fromarray(frames[t])
            pil_img = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
            resized[t] = np.asarray(pil_img, dtype=np.uint8)
        return resized

    def _read_video_uniform(self, video_path: str, max_frames: int) -> np.ndarray:
        """
        Uniformly sample up to max_frames frames from video and return (T, H, W, C) uint8 numpy array.
        保证帧数为偶数，如果是奇数，在右边repeat一个新的frame
        """
        if isinstance(video_path, (list, tuple)):
            vp = video_path[0]
        else:
            vp = video_path
        vr = decord.VideoReader(vp, ctx=decord.cpu(0))
        total = len(vr)
        if total <= 0:
            return np.empty((0,), dtype=np.uint8)
        max_frames_num = min(max_frames, total)
        indices = np.linspace(0, total - 1, max_frames_num, dtype=int).tolist()
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C), uint8
        # 缩放以满足每帧像素上限且保持长宽比
        frames = self._resize_frames_keep_aspect(frames, self.max_pixels_per_frame)
        # 保证帧数为偶数，如果是奇数，在右边repeat一个新的frame
        if frames.shape[0] % 2 == 1:
            last_frame = np.expand_dims(frames[-1], axis=0)
            frames = np.concatenate([frames, last_frame], axis=0)
        return frames

    def _read_video_with_indices(self, video_path: str, indices: List[int], max_frames: int) -> np.ndarray:
        """
        Sample frames according to provided indices:
        - take indices in their given order until we reach max_frames (filter invalid)
        - then sort ascending and read frames
        Return (T, H, W, C) uint8 numpy array.
        If no valid frames -> fallback to uniform sampling.
        保证帧数为偶数，如果是奇数，在右边repeat一个新的frame
        """
        if isinstance(video_path, (list, tuple)):
            vp = video_path[0]
        else:
            vp = video_path
        vr = decord.VideoReader(vp, ctx=decord.cpu(0))
        total = len(vr)
        if total <= 0:
            return np.empty((0,), dtype=np.uint8)

        sanitized: List[int] = []
        for idx in indices:
            if isinstance(idx, (int, np.integer)):
                ii = int(idx)
                if 0 <= ii < total:
                    sanitized.append(ii)
                    if len(sanitized) >= max_frames:
                        break

        if len(sanitized) == 0:
            return self._read_video_uniform(video_path, max_frames=max_frames)

        sanitized = sorted(sanitized)
        frames = vr.get_batch(sanitized).asnumpy()  # (T, H, W, C), uint8
        # 缩放以满足每帧像素上限且保持长宽比
        frames = self._resize_frames_keep_aspect(frames, self.max_pixels_per_frame)
        # 保证帧数为偶数，如果是奇数，在右边repeat一个新的frame
        if frames.shape[0] % 2 == 1:
            last_frame = np.expand_dims(frames[-1], axis=0)
            frames = np.concatenate([frames, last_frame], axis=0)
        return frames

    # ---------- Local process_vision_info (no external dependency) ----------
    def _process_vision_info(self, messages, visuals, doc_ids) -> Tuple[Optional[List], Optional[List]]:
        """
        构建与messages对齐的image_inputs和video_inputs（batch-first）。
        - images: 每个样本为PIL.Image.Image的列表
        - videos: 每个样本为torch.Tensor (T, C, H, W) 或 None
        如果整个batch都没有图片/视频，则返回None
        """
        image_inputs: List[Optional[List[Image.Image]]] = []
        video_inputs: List[Optional[torch.Tensor]] = []

        n = len(messages)
        for i in range(n):
            vis = visuals[i] if i < len(visuals) else None
            images_for_i: Optional[List[Image.Image]] = None
            video_for_i: Optional[np.ndarray] = None

            # Handle visuals types
            if isinstance(vis, str) and vis.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                # Choose indices by doc_id if provided
                idxs = None
                if doc_ids is not None and i < len(doc_ids):
                    try:
                        did = int(doc_ids[i])
                        idxs = self.docid_to_indices.get(did, None)
                    except Exception:
                        idxs = None

                if idxs is not None and isinstance(idxs, list) and len(idxs) > 0:
                    video_for_i = self._read_video_with_indices(vis, idxs, max_frames=self.max_num_frames)
                else:
                    video_for_i = self._read_video_uniform(vis, max_frames=self.max_num_frames)

            elif isinstance(vis, Image.Image):
                images_for_i = [vis]

            elif isinstance(vis, (list, tuple)) and all(isinstance(v, Image.Image) for v in vis):
                images_for_i = list(vis)

            # Append results
            image_inputs.append(images_for_i)
            video_inputs.append(video_for_i)

        # If entire batch has no images/videos, convert lists to None to match HF processor expectations
        if all(imgs is None for imgs in image_inputs):
            image_inputs_out = None
        else:
            # Replace None with empty list to keep batch shape
            image_inputs_out = [imgs if imgs is not None else [] for imgs in image_inputs]

        if all(v is None for v in video_inputs):
            video_inputs_out = None
        else:
            # 保持batch对齐，None用空tensor代替
            video_inputs_out = [v if v is not None else torch.empty((0,)) for v in video_inputs]

        return image_inputs_out, video_inputs_out

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")
                if "<video>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<video>", "")

            messages = []
            for i, context in enumerate(contexts):
                msg = [{"role": "system", "content": "You are a helpful assistant."}]
                # Build content that the chat template expects
                user_content = []
                visual = visuals[i] if i < len(visuals) else None
                if isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    user_content.append({"type": "video", "video": visual, "max_pixels": self.max_pixels})
                elif isinstance(visual, Image.Image):
                    # Put a base64 image placeholder into content; real pixels go through 'images' to processor
                    base64_image = visual.convert("RGB")
                    buffer = BytesIO()
                    base64_image.save(buffer, format="JPEG")
                    base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    user_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                    image_content = []
                    for v in visual:
                        v_rgb = v.convert("RGB")
                        buf = BytesIO()
                        v_rgb.save(buf, format="JPEG")
                        base64_string = base64.b64encode(buf.getvalue()).decode("utf-8")
                        image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                    user_content.extend(image_content)

                user_content.append({"type": "text", "text": context})
                msg.append({"role": "user", "content": user_content})
                messages.append(msg)

            # Build text prompts
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

            # Custom process_vision_info: read images/videos, with decord + frame indices
            # doc_id is a tuple aligned with contexts
            doc_ids_list = list(doc_id)
            image_inputs, video_inputs = self._process_vision_info(messages, visuals, doc_ids_list)

            # Prepare inputs for processor
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")