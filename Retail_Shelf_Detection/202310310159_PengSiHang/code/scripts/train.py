#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO

# 添加项目根目录到系统路径
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.logger import LOGGER
from utils.general import set_logging

def train(args):
    """
    训练YOLOv8模型
    """
    # 加载配置文件
    cfg_path = args.cfg
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 设置训练参数
    epochs = args.epochs or cfg.get('epochs', 100)
    batch_size = args.batch_size or cfg.get('batch', 16)
    img_size = args.img_size or cfg.get('imgsz', 640)
    
    # 创建YOLO模型
    model = YOLO(args.weights)
    
    # 输出训练参数
    LOGGER.info(f"开始训练, 配置:")
    LOGGER.info(f"- 模型: {args.weights}")
    LOGGER.info(f"- 数据集: {args.data}")
    LOGGER.info(f"- 配置文件: {args.cfg}")
    LOGGER.info(f"- 训练轮数: {epochs}")
    LOGGER.info(f"- 批量大小: {batch_size}")
    LOGGER.info(f"- 输入尺寸: {img_size}")
    LOGGER.info(f"- 设备: {args.device}")
    
    # 开始训练
    results = model.train(
        data=args.cfg,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=args.device,
        workers=args.workers,
        project=os.path.join(ROOT, 'results'),
        name='train',
        # 针对小目标检测的优化
        rect=True,  # 使用矩形训练 (批次中填充到相同尺寸)
        mosaic=1.0,  # Mosaic数据增强概率
        mixup=0.1,   # Mixup数据增强概率
        copy_paste=0.0,  # 复制粘贴增强概率
        degrees=0.0,  # 旋转增强范围
        translate=0.2,  # 平移增强范围
        scale=0.5,  # 缩放增强范围
        fliplr=0.5,  # 水平翻转概率
        perspective=0.0,  # 透视变换增强范围
        hsv_h=0.015,  # HSV-H增强系数
        hsv_s=0.7,    # HSV-S增强系数
        hsv_v=0.4,    # HSV-V增强系数
    )
    
    LOGGER.info(f"训练完成! 结果保存在 {os.path.join(ROOT, 'results', 'train')}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='训练YOLOv8模型')
    parser.add_argument('--data', type=str, default='./data/SKU110K', help='数据集路径')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='预训练模型路径')
    parser.add_argument('--cfg', type=str, default='./config/sku110k.yaml', help='模型配置文件')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='批量大小')
    parser.add_argument('--img-size', type=int, default=None, help='输入图像尺寸')
    parser.add_argument('--device', default='', help='cuda设备，如0或0,1,2,3或cpu')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    set_logging()
    args = parse_args()
    train(args) 