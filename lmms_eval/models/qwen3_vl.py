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
import json
import os
import time
from typing import List, Optional, Tuple, Union, Dict

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("qwen3_vl")
class Qwen3VL(lmms):
    """
    Qwen3_VL Model with frame indices support
    "https://github.com/QwenLM/Qwen3-VL"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        total_pixels: int = 32000 * 32 * 32,
        max_num_frames: int = 32,
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

        # Load model
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device_map,
        }
        if use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = AutoModelForImageTextToText.from_pretrained(
            pretrained,
            **model_kwargs
        ).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.total_pixels = total_pixels
        self.max_num_frames = max_num_frames
        self._tokenizer = self.processor.tokenizer

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _read_video_with_indices(self, video_path: str, indices: List[int], max_frames: int) -> Tuple[List[Image.Image], Dict]:
        """
        Read video frames according to provided indices.
        First take the first max_frames indices, then validate and sort them.
        Returns a list of PIL Images and video metadata.
        """
        if isinstance(video_path, (list, tuple)):
            vp = video_path[0]
        else:
            vp = video_path

        vr = decord.VideoReader(vp, ctx=decord.cpu(0))
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps())

        if total_frames <= 0:
            return [], {}

        # First, take the first max_frames indices (similar to internvl3_5.py)
        # Then validate and filter them
        frame_indices = None
        if indices is not None and len(indices) > 0:
            try:
                # Take first max_frames indices, then validate and sort
                frame_indices = sorted(
                    int(i)
                    for i in indices[:max_frames]
                    if isinstance(i, (int, np.integer)) and 0 <= int(i) < total_frames
                )
            except Exception as e:
                eval_logger.warning(
                    f"Failed to use frame indices from jsonl for video {video_path}: {e}"
                )
                frame_indices = None

        if frame_indices is None or len(frame_indices) == 0:
            # Fallback to uniform sampling
            max_frames_num = min(max_frames, total_frames)
            frame_indices = np.linspace(0, total_frames - 1, max_frames_num, dtype=int).tolist()

        # Read frames
        frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C), uint8

        # Convert to PIL Images
        image_list = [Image.fromarray(frame).convert("RGB") for frame in frames]

        # Create video metadata
        video_metadata = {
            "fps": video_fps,
            "frames_indices": frame_indices,
            "total_num_frames": total_frames,
            "video_backend": "decord",
        }

        return image_list, video_metadata

    def _read_video_uniform(self, video_path: str, max_frames: int) -> Tuple[List[Image.Image], Dict]:
        """
        Uniformly sample frames from video.
        Returns a list of PIL Images and video metadata.
        """
        if isinstance(video_path, (list, tuple)):
            vp = video_path[0]
        else:
            vp = video_path

        vr = decord.VideoReader(vp, ctx=decord.cpu(0))
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps())

        if total_frames <= 0:
            return [], {}

        max_frames_num = min(max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, max_frames_num, dtype=int).tolist()
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C), uint8

        # Convert to PIL Images
        image_list = [Image.fromarray(frame).convert("RGB") for frame in frames]

        # Create video metadata
        video_metadata = {
            "fps": video_fps,
            "frames_indices": indices,
            "total_num_frames": total_frames,
            "video_backend": "decord",
        }

        return image_list, video_metadata

    def _process_vision_info_with_indices(self, messages, visuals, doc_ids) -> Tuple[Optional[List], Optional[List], Optional[List]]:
        """
        Process vision info with support for frame indices from jsonl.
        Returns (image_inputs, video_inputs, video_metadatas)
        """
        image_inputs: List[Optional[List[Image.Image]]] = []
        video_inputs: List[Optional[List[Image.Image]]] = []
        video_metadatas: List[Optional[Dict]] = []

        n = len(messages)
        for i in range(n):
            vis = visuals[i] if i < len(visuals) else None
            images_for_i: Optional[List[Image.Image]] = None
            video_for_i: Optional[List[Image.Image]] = None
            video_metadata_for_i: Optional[Dict] = None

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
                    video_for_i, video_metadata_for_i = self._read_video_with_indices(
                        vis, idxs, max_frames=self.max_num_frames
                    )
                else:
                    video_for_i, video_metadata_for_i = self._read_video_uniform(
                        vis, max_frames=self.max_num_frames
                    )

            elif isinstance(vis, Image.Image):
                images_for_i = [vis]

            elif isinstance(vis, (list, tuple)) and all(isinstance(v, Image.Image) for v in vis):
                images_for_i = list(vis)

            # Append results
            image_inputs.append(images_for_i)
            video_inputs.append(video_for_i)
            video_metadatas.append(video_metadata_for_i)

        # Convert to format expected by processor
        if all(imgs is None for imgs in image_inputs):
            image_inputs_out = None
        else:
            image_inputs_out = [imgs if imgs is not None else [] for imgs in image_inputs]

        if all(v is None for v in video_inputs):
            video_inputs_out = None
            video_metadatas_out = None
        else:
            video_inputs_out = [v if v is not None else [] for v in video_inputs]
            video_metadatas_out = [m if m is not None else {} for m in video_metadatas]

        return image_inputs_out, video_inputs_out, video_metadatas_out

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

            # Pre-process videos to extract frames according to indices
            doc_ids_list = list(doc_id)
            processed_visuals = []
            video_metadatas_list = []
            
            for i, visual in enumerate(visuals):
                if isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    # Choose indices by doc_id if provided
                    idxs = None
                    if i < len(doc_ids_list):
                        try:
                            did = int(doc_ids_list[i])
                            idxs = self.docid_to_indices.get(did, None)
                        except Exception:
                            idxs = None
                    
                    if idxs is not None and isinstance(idxs, list) and len(idxs) > 0:
                        frames, metadata = self._read_video_with_indices(
                            visual, idxs, max_frames=self.max_num_frames
                        )
                    else:
                        frames, metadata = self._read_video_uniform(
                            visual, max_frames=self.max_num_frames
                        )
                    
                    # Replace video path with frame list
                    processed_visuals.append(frames if frames else visual)
                    video_metadatas_list.append(metadata)
                else:
                    processed_visuals.append(visual)
                    video_metadatas_list.append(None)

            # Build messages for qwen3vl
            messages = []
            for i, context in enumerate(contexts):
                user_content = []
                visual = processed_visuals[i] if i < len(processed_visuals) else None

                if isinstance(visual, list) and all(isinstance(v, Image.Image) for v in visual):
                    # This is a list of frames from video
                    user_content.append({
                        "type": "video",
                        "video": visual,  # qwen_vl_utils supports list of PIL Images
                        "total_pixels": self.total_pixels,
                    })
                elif isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    # Fallback: video path (shouldn't happen after processing, but just in case)
                    user_content.append({
                        "type": "video",
                        "video": visual,
                        "total_pixels": self.total_pixels,
                    })
                elif isinstance(visual, Image.Image):
                    user_content.append({"type": "image", "image": visual})
                elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                    for v in visual:
                        user_content.append({"type": "image", "image": v})

                user_content.append({"type": "text", "text": context})
                messages.append([{"role": "user", "content": user_content}])

            # Use qwen_vl_utils.process_vision_info to process the messages
            # It will handle frame lists automatically
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            # Split videos and metadatas if needed
            if videos is not None:
                # If videos is a list of tuples (video, metadata), split them
                if len(videos) > 0 and isinstance(videos[0], tuple):
                    videos, video_metadatas_from_utils = zip(*videos)
                    videos = list(videos)
                    video_metadatas_from_utils = list(video_metadatas_from_utils)
                else:
                    video_metadatas_from_utils = [None] * len(videos)
                
                # Prefer our custom metadatas (with correct frames_indices) over process_vision_info's
                for i in range(len(videos)):
                    if i < len(video_metadatas_list) and video_metadatas_list[i]:
                        video_metadatas_from_utils[i] = video_metadatas_list[i]
            else:
                video_metadatas_from_utils = None

            # Apply chat template
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]

            # Prepare inputs
            inputs = self.processor(
                text=texts,
                images=images,
                videos=videos,
                video_metadata=video_metadatas_from_utils,
                return_tensors="pt",
                do_resize=False,  # qwen-vl-utils already resized
                **video_kwargs
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Generation kwargs
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # Record frame counts for each sample
            frame_counts = []
            for i, visual in enumerate(processed_visuals):
                if isinstance(visual, list) and all(isinstance(v, Image.Image) for v in visual):
                    # Video frames
                    frame_counts.append(len(visual))
                elif isinstance(visual, Image.Image):
                    # Single image
                    frame_counts.append(1)
                elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                    # Multiple images
                    frame_counts.append(len(visual))
                else:
                    # No visual or video path (will be processed by processor)
                    frame_counts.append(0)

            # Record inference time
            inference_start_time = time.time()
            
            # Generate
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_kwargs["max_new_tokens"],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"] if gen_kwargs["temperature"] > 0 else None,
                top_p=gen_kwargs["top_p"] if gen_kwargs["temperature"] > 0 else None,
                num_beams=gen_kwargs["num_beams"],
                use_cache=self.use_cache,
            )
            
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            # Log frame counts and inference time for each sample
            for i, (ans, context, frame_count) in enumerate(zip(answers, contexts, frame_counts)):
                # Calculate per-sample inference time (distributed evenly across batch)
                per_sample_time = inference_time / len(answers) if len(answers) > 0 else inference_time
                eval_logger.info(
                    f"Sample {i} (doc_id={doc_ids_list[i] if i < len(doc_ids_list) else 'N/A'}): "
                    f"frames={frame_count}, inference_time={per_sample_time:.4f}s"
                )
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            
            # Log batch-level statistics
            if len(frame_counts) > 0:
                eval_logger.info(
                    f"Batch stats: total_samples={len(frame_counts)}, "
                    f"avg_frames={sum(frame_counts)/len(frame_counts):.2f}, "
                    f"total_inference_time={inference_time:.4f}s, "
                    f"avg_per_sample_time={inference_time/len(frame_counts):.4f}s"
                )

        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

