import logging
import math
import os
import json
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def build_transform(input_size: int):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(tile) for tile in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices


def load_video(
    video_path,
    bound=None,
    input_size=448,
    max_num=1,
    num_segments=32,
    doc_id=None,
    docid_to_indices: Optional[Dict[int, List[int]]] = None,
):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)

    frame_indices = None
    if docid_to_indices is not None and doc_id is not None:
        try:
            did = int(doc_id)
            if (
                did in docid_to_indices
                and isinstance(docid_to_indices[did], list)
                and len(docid_to_indices[did]) > 0
            ):
                frame_indices = sorted(
                    int(i)
                    for i in docid_to_indices[did][:num_segments]
                    if isinstance(i, (int, np.integer))
                )
        except Exception as e:
            eval_logger.warning(
                f"Failed to use frame indices from jsonl for doc_id={doc_id}: {e}"
            )

    if frame_indices is None or len(frame_indices) == 0:
        frame_indices = get_index(
            bound, fps, max_frame, first_idx=0, num_segments=num_segments
        )

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img_tiles = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img_tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def _extract_layer_count_from_config(model_name: str) -> Optional[int]:
    try:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
    except Exception as exc:
        eval_logger.warning(
            f"Failed to load config for {model_name} to infer layer count: {exc}"
        )
        return None

    candidates = [
        "num_hidden_layers",
        "n_layers",
        "num_layers",
        "text_config.num_hidden_layers",
        "language_config.num_hidden_layers",
    ]

    for candidate in candidates:
        value = config
        try:
            for attr in candidate.split("."):
                value = getattr(value, attr)
            if isinstance(value, int):
                return int(value)
        except AttributeError:
            continue
    return None


def split_model(model_name, num_layers=None):
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        return "auto"

    if num_layers is None:
        num_layers = _extract_layer_count_from_config(model_name)

    if num_layers is None:
        fallback_layers = {
            "InternVL3_5-8B": 32,
            "InternVL3_5-4B": 32,
            "InternVL3_5-2B": 24,
            "InternVL3_5-1B": 24,
            "InternVL3_5-12B": 40,
            "InternVL3_5-26B": 48,
        }
        num_layers = fallback_layers.get(model_name, 80)

    device_map: Dict[str, int] = {}
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, per_gpu in enumerate(num_layers_per_gpu):
        for _ in range(per_gpu):
            if layer_cnt >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1

    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[
        f"language_model.model.layers.{max(num_layers - 1, 0)}"
    ] = 0

    return device_map


@register_model("internvl3_5")
class InternVL3_5(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda",
        batch_size: str = "1",
        num_frame: int = 32,
        max_num: int = 1,
        grounding_files: Optional[str] = None,
        num_layers=None,
        frame_indices_jsonl: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.max_num_tiles = max_num
        self.path = pretrained
        self.num_frame = num_frame
        self.modality = modality
        self.docid_to_indices: Dict[int, List[int]] = {}

        if frame_indices_jsonl is not None and os.path.isfile(frame_indices_jsonl):
            try:
                with open(frame_indices_jsonl, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, start=1):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            record = json.loads(stripped)
                        except json.JSONDecodeError as e:
                            eval_logger.warning(
                                f"Failed to parse JSON at {frame_indices_jsonl}:{line_no}: {e}"
                            )
                            continue
                        if "doc_id" in record and "index" in record and isinstance(record["index"], list):
                            try:
                                did = int(record["doc_id"])
                                idx_list = [int(x) for x in record["index"]]
                                if did not in self.docid_to_indices:
                                    self.docid_to_indices[did] = idx_list
                            except Exception as e:
                                eval_logger.warning(
                                    f"Failed to load frame indices record at {frame_indices_jsonl}:{line_no}: {e}"
                                )
                                continue
                if self.docid_to_indices:
                    eval_logger.info(
                        f"Loaded frame indices for {len(self.docid_to_indices)} doc_ids from {frame_indices_jsonl}"
                    )
                else:
                    eval_logger.warning(
                        f"No valid frame indices loaded from {frame_indices_jsonl}"
                    )
            except Exception as e:
                eval_logger.warning(
                    f"Failed to load frame indices jsonl {frame_indices_jsonl}: {e}"
                )
        elif frame_indices_jsonl is not None:
            eval_logger.warning(
                f"frame_indices_jsonl not found or not a file: {frame_indices_jsonl}"
            )

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for InternVL3.5, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            derived_device_map = split_model(
                pretrained.split("/")[-1], num_layers=num_layers
            )
            self.device_map = derived_device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self._config = None
        try:
            self._config = AutoConfig.from_pretrained(
                self.path, trust_remote_code=True
            )
        except Exception as exc:
            eval_logger.warning(
                f"Failed to load config for {self.path}: {exc}"
            )

        self._model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self.device_map,
            load_in_8bit=False,
            use_flash_attn=kwargs.get("use_flash_attn", True),
        ).eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.path, trust_remote_code=True, use_fast=False
        )

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                ds_kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **ds_kwargs
                )
                eval_logger.info(
                    "Detected DistributedType.DEEPSPEED. Ensure you run `accelerate config` and set zero stage to 0."
                )

            if accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)

            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            if accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(
                f"Using {accelerator.num_processes} devices with tensor parallelism"
            )
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
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
        return self._model

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
        return [j for i in input for j in i]

    def _prepare_image_tensor(self, visual):
        image_tensor = load_image(
            visual, max_num=self.max_num_tiles
        ).to(torch.bfloat16).cuda()
        return image_tensor

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = [k for k in gen_kwargs if k not in DEFAULT_GEN_KWARGS]
            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            if self.modality == "image":
                if visuals:
                    visuals_tensors = [
                        self._prepare_image_tensor(visual) for visual in visuals
                    ]
                    pixel_values = torch.cat(visuals_tensors, dim=0)
                    num_patches_list = [visual.size(0) for visual in visuals_tensors]
                    image_tokens = " ".join(["<image>"] * len(visuals_tensors))
                    contexts = image_tokens + "\n" + contexts
                else:
                    pixel_values = None
                    num_patches_list = None
                response, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    contexts,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
            elif self.modality == "video":
                pixel_values_list = []
                num_patches_list = []
                video_prefix = ""
                frame_count = 0

                for visual in visuals:
                    if isinstance(visual, Image.Image):
                        image_tensor = self._prepare_image_tensor(visual)
                        pixel_values_list.append(image_tensor)
                        num_patches_list.append(image_tensor.size(0))
                        video_prefix += f"Frame{frame_count + 1}: <image>\n"
                        frame_count += 1
                    else:
                        pixel_values, patches = load_video(
                            visual,
                            num_segments=self.num_frame,
                            doc_id=doc_id,
                            docid_to_indices=self.docid_to_indices
                            if hasattr(self, "docid_to_indices")
                            else None,
                            max_num=self.max_num_tiles,
                        )
                        pixel_values_list.append(pixel_values)
                        num_patches_list.extend(patches)
                        video_prefix += "".join(
                            [
                                f"Frame{i + frame_count + 1}: <image>\n"
                                for i in range(len(patches))
                            ]
                        )
                        frame_count += len(patches)

                pixel_values = torch.cat(pixel_values_list, dim=0).to(
                    torch.bfloat16
                ).cuda()
                question = video_prefix + contexts
                response, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
            else:
                raise ValueError(f"Unsupported modality: {self.modality}")

            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for InternVL3.5")

    def generate_until_multi_round(self, requests) -> List[str]:
        return self.generate_until(requests)

