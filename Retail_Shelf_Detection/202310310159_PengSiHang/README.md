# 零售货架商品检测项目

基于 YOLOv8 模型的零售货架商品密集目标检测系统，使用 SKU110K 数据集训练。

## 项目背景

零售货架商品检测是零售行业中的一个重要应用场景，可用于自动化商品盘点、陈列分析和库存管理。本项目旨在构建一个高效的商品检测系统，能够在复杂的零售环境中准确识别密集排列的商品。

## 数据集介绍

SKU110K 数据集包含 11,762 张零售货架图片，标注了商品的边界框位置。该数据集主要特点：

- 密集小目标：商品在货架上密集排列，且相对图像尺寸较小
- 高度相似：许多商品外观相似，增加了检测难度
- 真实场景：图像来自真实零售环境，包含各种光照和视角变化

数据集分为三个子集：

- 训练集：8,219 张图像
- 验证集：588 张图像
- 测试集：2,936 张图像

## 模型选择

本项目使用 YOLOv8n（Nano 版）作为目标检测模型，该模型具有以下优势：

- 轻量级设计，适合低算力设备部署
- 高检测精度和实时性能的平衡
- 对小目标检测有较好的支持

## 项目结构

```
├── code/                   # 代码目录
│   ├── main.py             # 主入口脚本
│   ├── train.py            # 模型训练脚本
│   ├── evaluate.py         # 模型评估脚本
│   ├── predict.py          # 预测脚本
│   └── yolotest.py         # 测试脚本
├── data/                   # 数据目录
│   └── SKU110K/            # SKU110K数据集
│       ├── images/         # 图像文件
│       ├── annotations/    # 原始标注
│       └── yolo_format/    # YOLO格式标注
├── yolov8n.pt              # YOLOv8n预训练模型
└── README.md               # 项目说明文档
```

## 安装依赖

```bash
pip install ultralytics opencv-python matplotlib numpy
```

## 使用方法

### 训练模型

```bash
cd code
python main.py train
```

### 评估模型

评估模型性能：

```bash
python main.py eval
```

评估并可视化结果：

```bash
python main.py eval --viz --samples 10
```

### 预测

使用摄像头进行实时预测：

```bash
python main.py predict
```

使用图像进行预测：

```bash
python main.py predict --source /path/to/image.jpg --save
```

使用视频进行预测：

```bash
python main.py predict --source /path/to/video.mp4 --save
```

### 参数说明

训练相关参数在`train.py`中设置：

- `epochs`: 训练轮数
- `imgsz`: 输入图像大小
- `batch`: 批次大小
- 等其他超参数

预测相关参数：

- `--source`: 输入源（图像、视频或摄像头）
- `--model`: 模型路径
- `--conf`: 置信度阈值
- `--iou`: NMS IOU 阈值
- `--save`: 是否保存结果
- `--device`: 设备选择

## 模型优化

针对 SKU110K 数据集中的密集小目标检测问题，本项目采用了以下优化策略：

- 多尺度训练 (0.5x-1.5x 缩放)
- 输入分辨率设置为 640x640
- 使用 Focal Loss 关注难分类样本
- 优化 NMS 处理密集目标

## 参考资料

1. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
2. [SKU110K 数据集](https://github.com/eg4000/SKU110K_CVPR19)
3. [YOLO 论文](https://arxiv.org/abs/2207.02696)
