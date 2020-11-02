import sys
import os

source_dir = '/home/konglingbin/project/dota/CenterNet/exp/ctdet/acblock_resnet18_dota10_1024/result_dota_merge'
new_dir = '/home/konglingbin/project/dota/CenterNet/exp/ctdet/acblock_resnet18_dota10_1024/result_dota_merge_after_smallobj_filter'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

files = os.listdir(source_dir)
for item in files:
    filename = os.path.join(source_dir, item)
    with open(filename, 'r') as file:
        with open(os.path.join(new_dir, item), 'w') as file2:
            for line in file.readlines():
                words = line.strip().split()
                if len(words) < 5:
                    continue
                if float(words[4])-float(words[2]) > 3 and float(words[5])-float(words[3]) > 3:
                    file2.write(line)
                    
