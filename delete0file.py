import os
import shutil
def del_empty_file(path):
    files = os.listdir(path)
    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            if os.path.getsize(os.path.join(path, file)) > 0:
                continue
            shutil.move("/data/konglingbin/DOTA/DOTA/DOTA512/train_split_gap100/labelTxt/%s" % file, "/data/konglingbin/DOTA/DOTA/DOTA512/train_split_gap100_empty/labelTxt/")
            name = file.split(".")[0]
            name = "%s.png" % name
            shutil.move("/data/konglingbin/DOTA/DOTA/DOTA512/train_split_gap100/images/%s" % name, "/data/konglingbin/DOTA/DOTA/DOTA512/train_split_gap100_empty/images/")

if __name__ == '__main__':
    del_empty_file("/data/konglingbin/DOTA/DOTA/DOTA512/train_split_gap100/labelTxt")


