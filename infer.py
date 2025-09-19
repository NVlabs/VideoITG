import torch
from decord import VideoReader, cpu
from eagle.mm_utils import tokenizer_image_token, get_model_name_from_path
from eagle.model.builder import load_pretrained_model
from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
import os
from PIL import Image
import numpy as np

# 指定所有设备为cuda:0
DEVICE = torch.device("cuda:0")

# 加载模型
topk_model_path = "nvidia/VideoITG-8B"
topk_tokenizer, topk_model, image_processor, _ = load_pretrained_model(
    topk_model_path,
    None,
    get_model_name_from_path(topk_model_path),
    device_map="cuda:0",
)
topk_model.half().eval().to(DEVICE)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def pad_sequence(tokenizer, input_ids, batch_first, padding_value):
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids

def get_frame_indices(total_frames, original_fps, target_fps, num_frm):
    sample_fps = max(1, round(original_fps / target_fps))
    frame_idx = [i for i in range(0, total_frames, sample_fps)]
    if len(frame_idx) < num_frm:
        return frame_idx 
    scale = 1.0 * len(frame_idx) / num_frm
    uniform_idx = [round((i + 1) * scale - 1) for i in range(num_frm)]
    frame_idx = [frame_idx[i] for i in uniform_idx]
    return frame_idx

def read_video_decord(video_path, num_frm=16, target_fps=2):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)  # 使用4线程读取
    total_frames = len(vr)
    original_fps = vr.get_avg_fps()
    indices = get_frame_indices(total_frames, original_fps, target_fps, num_frm)
    frames = vr.get_batch(indices)
    vr.seek(0)
    return frames.asnumpy(), indices, round(total_frames/original_fps)

def topk_selection(prompt, video_path, num_topk):
    video, frame_idx, total_seconds = read_video_decord(video_path, num_frm=512, target_fps=2)
    video_tensor = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().to(DEVICE)
    processed_video = [video_tensor]
    total_frames = video_tensor.shape[0]
    
    # 生成输入
    prompt = DEFAULT_IMAGE_TOKEN + prompt + "\n"
    input_ids = tokenizer_image_token(prompt, topk_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    pad_token_id = topk_tokenizer.pad_token_id if topk_tokenizer.pad_token_id is not None else topk_tokenizer.eos_token_id
    input_ids = pad_sequence(topk_tokenizer, [input_ids], batch_first=True, padding_value=pad_token_id).to(DEVICE)
    attention_mask = input_ids.ne(pad_token_id).to(DEVICE)
    
    with torch.inference_mode():
        response = topk_model(
                input_ids,
                attention_mask=attention_mask,
                images=processed_video,
            )
        logits = response.logits[0].sigmoid().view(-1)
        
        values, indices = torch.sort(logits, descending=True)
        indices = indices.tolist()
        values = values.tolist()
        selected_indices = [frame_idx[i] for i in indices[:num_topk]]
    selected_indices.sort()
    return selected_indices

def save_selected_frames(video_path, selected_indices, save_dir="./vis"):
    """
    将选中的帧保存为JPG到指定目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    for idx, frame_idx in enumerate(selected_indices):
        img = vr[frame_idx].asnumpy()
        img_pil = Image.fromarray(img)
        save_path = os.path.join(save_dir, f"frame_{idx:03d}_idx{frame_idx}.jpg")
        img_pil.save(save_path, "JPEG")
    print(f"已保存{len(selected_indices)}帧到{save_dir}")

def main():
    prompt = "Which IMAX movie isn't in the video? A. The Hunger Games: B. Catching Fire; C. The Dark Knight; D. Oppenheimer; E. Dune \n Please respond with only the letter of the correct answer."
    video_path = "/lustre/fs12/portfolios/llmservice/users/shihaow/dev/demo/assets/imax.mp4"
    num_topk = 32
    selected_indices = topk_selection(prompt, video_path, num_topk)
    print(selected_indices)
    save_selected_frames(video_path, selected_indices, save_dir="./vis")

if __name__ == "__main__":
    main()