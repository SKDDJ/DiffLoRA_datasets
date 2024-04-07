#!/bin/bash

# 启动第一个命令并重定向输出
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=2 python main.py \
    --prompt_path="/root/shiym_proj/DiffLook/data/prompt/prompt.txt" \
    --negative_prompt_path="/root/shiym_proj/DiffLook/data/prompt/negative.txt" \
    --images_directory="/root/shiym_proj/DiffLook/data/part1" \
    --start_id="20" \
    --suffix_text="a woman img" \
    --save_path="/root/shiym_proj/DiffLook/outputs/part1" > part1.log 2>&1 &

# 启动第二个命令并重定向输出
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 python main.py \
    --prompt_path="/root/shiym_proj/DiffLook/data/prompt/prompt.txt" \
    --negative_prompt_path="/root/shiym_proj/DiffLook/data/prompt/negative.txt" \
    --images_directory="/root/shiym_proj/DiffLook/data/part2" \
    --save_path="/root/shiym_proj/DiffLook/outputs/part2" > part2.log 2>&1 &

# # 启动第三个命令并重定向输出
# HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=7 python main.py \
#     --prompt_path="/root/shiym_proj/DiffLook/data/prompt/prompt.txt" \
#     --negative_prompt_path="/root/shiym_proj/DiffLook/data/prompt/negative.txt" \
#     --images_directory="/root/shiym_proj/DiffLook/data/part3" \
#     --save_path="/root/shiym_proj/DiffLook/outputs/part3" > part3.log 2>&1 &

wait # 等待所有后台命令结束
