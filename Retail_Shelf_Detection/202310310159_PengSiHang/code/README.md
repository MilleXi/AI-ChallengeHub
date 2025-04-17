# 零售货架商品检测代码说明

本目录包含了零售货架商品检测系统的全部源代码，该系统基于 YOLOv8 实现了对 SKU110K 数据集中密集排列商品的检测。

## 文件结构与功能说明

1. `main.py`: 项目主入口脚本，提供统一的命令行接口

   - 支持三种模式：训练模式、评估模式和预测模式
   - 统一管理命令行参数

2. `train.py`: 模型训练脚本

   - 基于 YOLOv8n 预训练模型进行迁移学习
   - 针对小目标密集检测场景优化参数
   - 自动适应硬件环境，调整训练参数

3. `evaluate.py`: 模型评估脚本

   - 计算模型在测试集上的性能指标
   - 支持可视化检测结果
   - 生成评估报告和预测可视化图像

4. `predict.py`: 预测脚本

   - 支持图像、视频和摄像头实时检测
   - 可调整置信度阈值和 NMS 参数
   - 支持检测结果的保存与展示

5. `convert_annotations.py`: 数据集转换脚本

   - 将 SKU110K 数据集的原始 CSV 格式标注转换为 YOLO 格式
   - 生成训练、验证和测试集的标注文件
   - 创建数据集配置文件(data.yaml)

6. `test_detection.py`: 测试脚本

   - 加载训练好的模型进行批量测试
   - 支持自定义测试图像目录
   - 生成并保存测试结果

7. `yolotest.py`: 简单测试脚本
   - YOLOv8 基本功能测试示例
   - 展示 YOLOv8 API 的基本用法

## 使用方法

### 数据准备

在使用系统前，需要先准备数据集并转换格式：

```bash
python convert_annotations.py
```

### 模型训练

通过 main.py 的 train 模式启动训练：

```bash
python main.py train
```

### 模型评估

评估模型性能：

```bash
python main.py eval
```

带可视化的评估：

```bash
python main.py eval --viz --samples 10
```

### 预测

使用图像进行预测：

```bash
python main.py predict --source image.jpg --save
```

使用视频进行预测：

```bash
python main.py predict --source video.mp4 --save
```

使用摄像头进行实时预测：

```bash
python main.py predict
```

## 技术特点

1. **小目标密集检测优化**：

   - 优化 NMS 处理算法
   - 调整置信度阈值以提高召回率
   - 使用矩形训练提高效率

2. **内存和 GPU 优化**：

   - 支持混合精度训练(AMP)
   - 动态调整批次大小和图像尺寸
   - 优化数据加载，减少内存占用

3. **易用性设计**：
   - 统一的命令行接口
   - 自动选择最佳模型进行预测
   - 详细的日志输出和进度显示

## 系统要求

- Python 3.8+
- PyTorch 1.7+
- Ultralytics 8.0.0+
- CUDA 支持（推荐用于训练）
- 内存：至少 8GB
- 磁盘空间：至少 20GB（包含数据集）
