import os
import shutil

# 图片的源目录
source_directory = "/root/shiym_proj/DiffLook/data/img"

# 目标目录列表
target_directories = [
    "/root/shiym_proj/DiffLook/data/part1",
    "/root/shiym_proj/DiffLook/data/part2",
    "/root/shiym_proj/DiffLook/data/part3"
]

# 创建目标目录
for directory in target_directories:
    os.makedirs(directory, exist_ok=True)

# 获取所有图片，并按文件名排序，确保顺序
all_images = sorted([f for f in os.listdir(source_directory) if f.endswith('.png')],
                    key=lambda x: int(x.split('.')[0]))  # 按照数字顺序排序

# 计算每一部分应该含有的图片数量
images_per_part = len(all_images) // len(target_directories)

# 分配图片到各个目标目录
for i, image in enumerate(all_images):
    # 计算当前图片应该在哪个部分
    part_index = i // images_per_part
    # 如果部分索引超出目标目录的数量（即在最后一部分），将其归入最后一个目标目录
    if part_index >= len(target_directories):
        part_index = len(target_directories) - 1
    # 目标文件路径
    target_path = os.path.join(target_directories[part_index], image)
    # 源文件路径
    source_path = os.path.join(source_directory, image)
    # 复制文件
    shutil.move(source_path, target_path)

print("Images have been successfully distributed.")
