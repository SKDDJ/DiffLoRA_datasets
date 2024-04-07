# from google.colab import drive
# drive.mount('/content/drive')

# !pip install diffusers omegaconf peft -qqq
# !git clone https://github.com/TencentARC/PhotoMaker.git

import torch
import numpy as np
import random
import os
from PIL import Image
import argparse
import time 

from labml import logger, monit, lab, tracker

from diffusers.utils import load_image
from diffusers import DDIMScheduler
from huggingface_hub import hf_hub_download


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.chdir("/root/shiym_proj/DiffLook/")
from photomaker import PhotoMakerStableDiffusionXLPipeline

device = "cuda"

# gloal variable and function
def image_grid(imgs, rows, cols, size_after_resize):
    assert len(imgs) == rows*cols

    w, h = size_after_resize, size_after_resize

    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        img = img.resize((w,h))
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def load_all_image_paths(images_directory, start_id=''):
    """
    从指定的目录加载所有图片的路径，并从指定的ID开始。
    """
    files = os.listdir(images_directory)
    files = sorted(files)
    if start_id:
        files = files[files.index(f"{start_id}.png"):]  # 从start_id开始
    return [os.path.join(images_directory, f) for f in files if f.endswith('.png')]

## Note that the trigger word `img` must follow the class word for personalization
def read_and_process_file(file_path, add_suffix=False, suffix_text=" a woman img"):
    """
    读取指定路径的文本文件，逐行处理并返回处理后的列表。
    
    参数:
    - file_path: 文件的路径。
    - add_suffix: 是否在每行末尾添加指定字段。默认为False。
    - suffix_text: 需要添加到每行末尾的文本。仅当add_suffix为True时生效。
    
    返回:
    - 一个包含处理后每一行内容的列表。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    
    if add_suffix:
        lines = [line + suffix_text for line in lines]
    
    return lines


# Download models
lora_path = "/root/shiym_proj/DiffLook/checkpoints/Photomaker/xl_more_art-full.safetensors"
# !wget -O /checkpoints/Photomaker/xl_more_art-full.safetensors https://civitai.com/api/download/models/152309?type=Model&format=SafeTensor

photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

from diffusers import LCMScheduler

base_model_path = '/root/.cache/huggingface/hub/models--SG161222--RealVisXL_V3.0/snapshots/11ee564ebf4bd96d90ed5d473cb8e7f2e6450bcf'
# base_model_path = 'SG161222/RealVisXL_V3.0'
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16, 
).to(device)

# Load PhotoMaker checkpoint
pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt),
    subfolder="",
    weight_name=os.path.basename(photomaker_ckpt),
    trigger_word="img"
)

# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

print("Loading lora...")
pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path), adapter_name="xl_more_art-full")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")

pipe.set_adapters(["photomaker", "xl_more_art-full", "lcm"], adapter_weights=[1.0,0.5,1.0])
# pipe.set_adapters(["photomaker",  "lcm"], adapter_weights=[1.0, 1.0])
pipe.fuse_lora()

generator = torch.Generator(device=device).manual_seed(500)


parser = argparse.ArgumentParser()
parser.add_argument('--prompt_path', help='hhh', default='/root/shiym_proj/DiffLook/data/prompt/prompt.txt')
parser.add_argument('--negative_prompt_path', help='', default='/root/shiym_proj/DiffLook/data/prompt/negative.txt')
parser.add_argument('--images_directory', help='', default='/root/shiym_proj/DiffLook/data/part1')
parser.add_argument('--start_id', type=str, help='Start generation from the image with this ID', default='')
parser.add_argument('--save_path', help='', default='/root/shiym_proj/DiffLook/outputs/part1')
parser.add_argument('--suffix_text', help='', default='a woman img')
# parser.add_argument('--', help='', default='')

args = parser.parse_args()

prompt_path = args.prompt_path
negative_prompt_path = args.negative_prompt_path
images_directory = args.images_directory
save_path = args.save_path
start_id = args.start_id  # 获取start_id参数
suffix_text = args.suffix_text

# 使用定义的函数读取并处理两个文件
prompts = read_and_process_file(prompt_path, add_suffix=True, suffix_text=suffix_text)
negative_prompt = read_and_process_file(negative_prompt_path)
# 读取所有图片的路径
# image_paths = load_all_image_paths(images_directory)
image_paths = load_all_image_paths(images_directory, start_id=start_id)

with monit.section("Generating images"):
    for image_path in image_paths:
        # 以下是对每一张图片执行生成过程的代码片段
        person_name = os.path.splitext(os.path.basename(image_path))[0]  # 假设每张图片以人物名称命名
        person_save_path = os.path.join(save_path, person_name)
        
        if os.path.exists(person_save_path):
            logger.log(f"Directory for {person_name} already exists. Skipping generation for this ID.")
        else:
            # 如果文件夹不存在，则创建文件夹，并进行图片生成
            os.makedirs(person_save_path, exist_ok=True)  # 为每个人创建一个目录

            input_id_image = [load_image(image_path)]  # 注意，现在input_id_images变为了单张图片的list
            
            #Parameter setting
            num_steps = 10
            style_strength_ratio = 20
            start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
            if start_merge_step > 30:
                start_merge_step = 30
                
            # for idx, prompt in enumerate(prompts):
            for idx, prompt in monit.enum(f"{person_name}",prompts):
                images = pipe(
                    prompt=prompt,
                    input_id_images=input_id_image,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=1,
                    num_inference_steps=num_steps,
                    start_merge_step=start_merge_step,
                    generator=generator,
                    guidance_scale=1
                ).images

                for image_idx, image in enumerate(images):
                    # 文件名包括人物名称、提示索引和图片索引
                    file_name = f"{person_name}_prompt{idx:03d}_image{image_idx:02d}.png"
                    image.save(os.path.join(person_save_path, file_name))



