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

def validate(args):
    """
    验证YOLOv8模型性能
    """
    # 加载配置文件
    cfg_path = args.cfg
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 设置验证参数
    img_size = args.img_size or cfg.get('imgsz', 640)
    batch_size = args.batch_size or cfg.get('batch', 16)
    
    # 加载模型
    if os.path.exists(args.weights):
        model = YOLO(args.weights)
    else:
        LOGGER.error(f"找不到模型权重文件: {args.weights}")
        sys.exit(1)
    
    # 输出验证参数
    LOGGER.info(f"开始验证, 配置:")
    LOGGER.info(f"- 模型: {args.weights}")
    LOGGER.info(f"- 数据集: {args.data}")
    LOGGER.info(f"- 配置文件: {args.cfg}")
    LOGGER.info(f"- 批量大小: {batch_size}")
    LOGGER.info(f"- 输入尺寸: {img_size}")
    LOGGER.info(f"- 设备: {args.device}")
    
    # 验证模型
    results = model.val(
        data=args.cfg,
        batch=batch_size,
        imgsz=img_size,
        device=args.device,
        workers=args.workers,
        project=os.path.join(ROOT, 'results'),
        name='val',
        save_json=True,  # 保存结果为JSON格式
        save_conf=True,  # 保存置信度
        save_txt=True,   # 保存预测结果为文本
        iou=0.65,        # IoU阈值，对于密集小目标可能需要调低
        conf=0.25,       # 置信度阈值
        max_det=300,     # 每张图像最大检测数量，对于密集场景需要增大
    )
    
    # 输出验证结果
    metrics = results.box
    LOGGER.info(f"验证完成!")
    LOGGER.info(f"mAP@0.5: {metrics.maps[0]:.4f}")
    LOGGER.info(f"mAP@0.5:0.95: {metrics.map:.4f}")
    LOGGER.info(f"Precision: {metrics.p:.4f}")
    LOGGER.info(f"Recall: {metrics.r:.4f}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='验证YOLOv8模型性能')
    parser.add_argument('--data', type=str, default='./data/SKU110K', help='数据集路径')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件')
    parser.add_argument('--cfg', type=str, default='./config/sku110k.yaml', help='模型配置文件')
    parser.add_argument('--batch-size', type=int, default=None, help='批量大小')
    parser.add_argument('--img-size', type=int, default=None, help='输入图像尺寸')
    parser.add_argument('--device', default='', help='cuda设备，如0或0,1,2,3或cpu')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    set_logging()
    args = parse_args()
    validate(args) 