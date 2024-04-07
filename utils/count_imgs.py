# this file is used to count the number of imgs of female or male 
import os 

# dest_img_path = '/root/shiym_proj/DiffLook/data/img/'
src_json_path = '/root/shiym_proj/DiffLook/male_data/json/'

json_files = [file for file in os.listdir(src_json_path) if file.endswith('.json')]

print("lens of json_files: ", len(json_files))
#lens of json_files:  38388 (female)
#lens of json_files:  31083 (male)
