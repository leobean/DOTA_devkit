import os
import sys
import shutil
# split train set. For accelerate train, use half images to train, %s.txt/2==0 or %s.txt/2==1

dir_path = "/data/konglingbin/DOTA/DOTA10/train_split_1024_gap200"
copy_to_dir_path = "/data/konglingbin/DOTA/DOTA10/train_split_1024_gap200_half"
filenames = os.listdir(os.path.join(dir_path, 'images'))

if not os.path.exists(copy_to_dir_path):
    os.makedirs(copy_to_dir_path)
    os.makedirs(os.path.join(copy_to_dir_path, 'images'))
    os.makedirs(os.path.join(copy_to_dir_path, 'labelTxt'))

for filename in filenames:
    num = filename.split("_")[0]
    num_int = int(num[-1])
    if num_int % 2 == 1:
        image_copy_name = os.path.join('images', filename)
        txt_copy_name = os.path.join('labelTxt', filename.split('.')[0] + '.txt')
        #print(image_copy_name)
        shutil.copyfile(os.path.join(dir_path, image_copy_name), os.path.join(copy_to_dir_path, image_copy_name))        
        shutil.copyfile(os.path.join(dir_path, txt_copy_name), os.path.join(copy_to_dir_path, txt_copy_name))        
