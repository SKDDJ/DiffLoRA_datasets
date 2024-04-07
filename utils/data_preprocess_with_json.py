import os
import shutil
import json
from tqdm import tqdm

def copy_female_images_and_jsons(src_json_path, src_img_path, dest_img_path, dest_json_path):
    json_files = [file for file in os.listdir(src_json_path) if file.endswith('.json')]
    
    for json_file in tqdm(json_files, desc="Processing JSON Files"):
        full_json_path = os.path.join(src_json_path, json_file)
        with open(full_json_path, 'r') as f:
            data = json.load(f)

        # 适配到你的JSON结构
        # 假设我们的数据在列表的第一个元素中
        if data and isinstance(data, list) and data[0]['faceAttributes']['gender'] == 'female':
            img_file_name = json_file.replace('.json', '.png')
            full_img_path = os.path.join(src_img_path, img_file_name)
            full_dest_img_path = os.path.join(dest_img_path, img_file_name)
            full_dest_json_path = os.path.join(dest_json_path, json_file)
            
            shutil.copy2(full_img_path, full_dest_img_path)
            shutil.copy2(full_json_path, full_dest_json_path)
            print(f"Copied {img_file_name} to {dest_img_path}")
            print(f"Copied {json_file} to {dest_json_path}")

src_json_path = '/root/shiym_proj/ffhq-dataset/ffhq-features-dataset/json/'
src_img_path = '/root/shiym_proj/ffhq-dataset/thumbnails/thumbnails128x128/'
dest_img_path = '/root/shiym_proj/DiffLook/data/img/'
dest_json_path = '/root/shiym_proj/DiffLook/data/json/'

for path in [dest_img_path, dest_json_path]:
    if not os.path.exists(path):
        os.makedirs(path)

copy_female_images_and_jsons(src_json_path, src_img_path, dest_img_path, dest_json_path)
