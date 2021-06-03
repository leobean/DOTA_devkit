# DOTA_devkit
dota_dataset tools, including ImageSplit, DOTA2COCO, ResultMerge...
DOTA发布的工具包原文链接如下，含安装说明和介绍：https://captain-whu.github.io/DOTA/code.html   https://github.com/CAPTAIN-WHU/DOTA_devkit

## 各主要文件的功能如下：
- DOTA2COCO.py 将DOTA数据集的格式转成COCO格式，DOTA的格式如下：https://captain-whu.github.io/DOTA/dataset.html
- ImgSplit.py 将图像进行切割，可配置切割的小图尺寸、重叠像素值
- ImgSplit_multi_process_filter_no_airport.py 过滤部分图像，如不在机场范围内的就不再切割；若某张图像没检测到机场，就全图切割。
- ResultMerge.py 将小图上的检测结果，还原到原始图像坐标中
- dota_evaluation_task.py 专门针对DOTA数据集的检测结果进行评测，计算各类别AP和mAP。
