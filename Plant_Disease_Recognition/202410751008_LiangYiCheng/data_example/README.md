# 目录说明
该文件夹存储了训练、推理用到的各种数据
- train文件夹下是训练用到的各种图片、标注数据（共244张照片，各子文件夹仅保留两张图片）
  数据集来源：
  1. https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/data
  2. 自己拍摄的校内各种树叶照片
  使用Roboflow网页手动标注
  任务：目标检测，检测输入照片中的叶子，并判断为diseased、healthy两个类别
  train：valid：test的图片分配比例为7：2：1
- input、output文件夹存储了推理时的输入输出照片（被.gitignore忽略）