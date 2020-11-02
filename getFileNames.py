import os

path = "/data/konglingbin/DOTA/DOTA/val/images"
files = os.listdir(path)
for file in files:
    print(file.strip().split('.')[0])



