# 零售货架目标检测项目

本项目旨在实现零售货架上商品的自动检测与识别，使用 YOLOv8 模型在 SKU110K 数据集上进行训练和评估。该项目解决了密集小目标检测的挑战，可应用于零售商品自动盘点和陈列分析等场景。

## 项目结构

```
retail_shelf/
├── config/                    # 配置文件目录
│   └── sku110k.yaml           # SKU110K数据集和模型配置
├── data/                      # 数据集目录
│   └── SKU110K/               # SKU110K数据集
├── models/                    # 模型保存目录
├── results/                   # 结果保存目录
├── scripts/                   # 脚本目录
│   ├── download_data.py       # 数据下载脚本
│   ├── prepare_data.py        # 数据预处理脚本
│   ├── train.py               # 训练脚本
│   ├── val.py                 # 验证脚本
│   └── test.py                # 测试脚本
├── utils/                     # 工具函数目录
│   ├── general.py             # 通用工具函数
│   ├── logger.py              # 日志模块
│   └── plots.py               # 绘图工具
└── main.py                    # 主程序入口
```

## 环境设置

本项目需要以下依赖：

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Matplotlib
- Pandas

可以使用 conda 创建环境并安装依赖：

```bash
# 创建conda环境
conda create -n retail_shelf python=3.8 -y

# 激活环境
conda activate retail_shelf

# 安装依赖
pip install numpy opencv-python matplotlib torch torchvision ultralytics
```

## 数据集

本项目使用 SKU110K 数据集，该数据集包含 11,762 张零售货架图片，标注了商品的边界框。对于本项目，我们选取含饮料/零食的子集进行处理。

### 数据集获取

可以使用提供的下载脚本获取数据集：

```bash
python scripts/download_data.py
```

### 数据集预处理

下载后，需要将数据集转换为 YOLO 格式：

```bash
python scripts/prepare_data.py
```

## 模型训练

使用以下命令开始训练：

```bash
python main.py --mode train --weights yolov8n.pt --epochs 100 --batch-size 16 --img-size 640
```

训练参数说明：

- `--weights`: 预训练模型路径，默认为 yolov8n.pt（nano 版本适合低算力环境）
- `--epochs`: 训练轮数
- `--batch-size`: 批量大小
- `--img-size`: 输入图像尺寸
- `--device`: CUDA 设备选择，如'0'或'0,1'等

## 模型评估

```bash
python main.py --mode val --weights ./results/train/weights/best.pt
```

## 模型测试与推理

在测试图像上进行推理：

```bash
python main.py --mode test --weights ./results/train/weights/best.pt --source ./data/test_images
```

## 小目标检测优化

为了优化密集小目标检测效果，本项目采用了以下策略：

1. 输入分辨率设置为 640x640
2. 使用多尺度训练（0.5x-1.5x 缩放）
3. 优化 Anchor Box 尺寸以适应小目标
4. 使用 Focal Loss 解决类别不平衡问题
5. 改进 NMS（非极大值抑制）算法以处理密集目标

## 结果展示

训练过程和检测效果将保存在`results`目录下，包括：

- 训练日志和指标曲线
- 验证集上的评估结果
- 测试图像的检测结果

## 参考资源

1. SKU110K 数据集: [github.com/eg4000/SKU110K_CVPR19](https://github.com/eg4000/SKU110K_CVPR19)
2. YOLOv8: [ultralytics.com/yolov8](https://docs.ultralytics.com/)
3. 论文: "SKU-110K: Precise Detection of Objects in Dense Scenes" - Goldman et al. CVPR 2019
