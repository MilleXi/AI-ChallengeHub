#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import argparse
import cv2
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# 添加项目根目录到系统路径
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.logger import LOGGER
from utils.general import set_logging
from utils.plots import draw_detections

def test(args):
    """
    使用训练好的模型在测试图像上进行推理
    """
    # 加载配置文件
    cfg_path = args.cfg
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 设置参数
    img_size = args.img_size or cfg.get('imgsz', 640)
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres
    
    # 加载模型
    if os.path.exists(args.weights):
        model = YOLO(args.weights)
    else:
        LOGGER.error(f"找不到模型权重文件: {args.weights}")
        sys.exit(1)
    
    # 输出测试参数
    LOGGER.info(f"开始测试, 配置:")
    LOGGER.info(f"- 模型: {args.weights}")
    LOGGER.info(f"- 数据路径: {args.source}")
    LOGGER.info(f"- 输入尺寸: {img_size}")
    LOGGER.info(f"- 置信度阈值: {conf_thres}")
    LOGGER.info(f"- IoU阈值: {iou_thres}")
    LOGGER.info(f"- 设备: {args.device}")
    
    # 创建结果目录
    save_dir = os.path.join(ROOT, 'results', 'test')
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取测试图像列表
    if os.path.isdir(args.source):
        img_list = glob.glob(os.path.join(args.source, '*.jpg')) + \
                  glob.glob(os.path.join(args.source, '*.jpeg')) + \
                  glob.glob(os.path.join(args.source, '*.png'))
    else:
        img_list = [args.source]
    
    LOGGER.info(f"找到 {len(img_list)} 张测试图像")
    
    # 对每张图像进行推理
    for img_path in tqdm(img_list, desc="测试进行中"):
        img_name = os.path.basename(img_path)
        
        # 推理
        results = model(
            img_path, 
            imgsz=img_size,
            conf=conf_thres,
            iou=iou_thres,
            device=args.device,
            verbose=False,
            max_det=300,  # 对于密集场景增加最大检测数量
        )
        
        # 保存结果
        result_img = results[0].plot()
        
        # 保存带有检测结果的图像
        cv2.imwrite(os.path.join(save_dir, img_name), result_img)
    
    LOGGER.info(f"测试完成! 结果保存在 {save_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行推理测试')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件')
    parser.add_argument('--source', type=str, required=True, help='测试图像路径或目录')
    parser.add_argument('--cfg', type=str, default='./config/sku110k.yaml', help='模型配置文件')
    parser.add_argument('--img-size', type=int, default=None, help='输入图像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS的IoU阈值')
    parser.add_argument('--device', default='', help='cuda设备，如0或0,1,2,3或cpu')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    set_logging()
    args = parse_args()
    test(args) 