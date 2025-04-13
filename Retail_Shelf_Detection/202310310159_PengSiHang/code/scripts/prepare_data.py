#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import random
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

# 添加项目根目录到系统路径
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.logger import LOGGER
from utils.general import set_logging

def convert_bbox_to_yolo(img_width, img_height, bbox):
    """
    将SKU110K的边界框格式转换为YOLO格式
    SKU110K格式: [x_min, y_min, x_max, y_max]
    YOLO格式: [x_center, y_center, width, height]，所有值都相对于图像尺寸归一化到0-1
    """
    x_min, y_min, x_max, y_max = bbox
    
    # 计算中心坐标和宽高
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # 归一化
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]

def process_annotations(ann_file, img_dir, output_dir, class_id=0):
    """
    处理SKU110K注释文件，转换为YOLO格式
    """
    # 读取注释文件
    df = pd.read_csv(ann_file)
    
    # 获取所有唯一的图像ID
    img_ids = df['image_name'].unique()
    
    for img_id in tqdm(img_ids, desc="处理图像"):
        # 获取图像的所有边界框
        img_df = df[df['image_name'] == img_id]
        
        # 读取图像以获取尺寸
        img_path = os.path.join(img_dir, img_id)
        if not os.path.exists(img_path):
            LOGGER.warning(f"图像 {img_path} 不存在，跳过")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            LOGGER.warning(f"无法读取图像 {img_path}，跳过")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 准备YOLO格式的标签文件
        label_path = os.path.join(output_dir, os.path.splitext(img_id)[0] + '.txt')
        
        with open(label_path, 'w') as f:
            for _, row in img_df.iterrows():
                # 获取边界框坐标
                x_min = row['x1']
                y_min = row['y1']
                x_max = row['x2']
                y_max = row['y2']
                
                # 转换为YOLO格式
                yolo_bbox = convert_bbox_to_yolo(img_width, img_height, [x_min, y_min, x_max, y_max])
                
                # 写入标签文件 (class_id, x_center, y_center, width, height)
                f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

def create_data_splits(image_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    创建训练、验证和测试集划分
    """
    # 获取所有图像文件
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 随机打乱
    random.shuffle(all_images)
    
    # 计算每个集合的大小
    total = len(all_images)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # 划分数据集
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]
    
    # 创建数据集列表文件
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    # 打印数据集统计信息
    LOGGER.info(f"数据集划分 - 训练集: {len(train_images)}, 验证集: {len(val_images)}, 测试集: {len(test_images)}")
    
    # 创建数据集列表文件
    for split_name, images in splits.items():
        with open(os.path.join(output_dir, f"{split_name}.txt"), 'w') as f:
            for img in images:
                f.write(f"./images/{img}\n")

def prepare_sku110k(data_dir='./data/SKU110K', output_dir='./data/SKU110K/yolo_format', 
                    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    准备SKU110K数据集，转换为YOLO格式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # 数据集路径
    img_dir = os.path.join(data_dir, 'images')
    ann_file = os.path.join(data_dir, 'annotations', 'annotations.csv')
    
    # 处理注释文件
    LOGGER.info("转换注释为YOLO格式...")
    process_annotations(ann_file, img_dir, os.path.join(output_dir, 'labels'))
    
    # 复制图像到输出目录
    LOGGER.info("复制图像到输出目录...")
    for img_file in tqdm(os.listdir(img_dir), desc="复制图像"):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy(
                os.path.join(img_dir, img_file),
                os.path.join(output_dir, 'images', img_file)
            )
    
    # 创建数据集划分
    LOGGER.info("创建数据集划分...")
    create_data_splits(
        os.path.join(output_dir, 'images'),
        output_dir,
        train_ratio,
        val_ratio,
        test_ratio
    )
    
    LOGGER.info(f"数据集准备完成! 转换后的数据保存在 {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='准备SKU110K数据集，转换为YOLO格式')
    parser.add_argument('--data-dir', type=str, default='./data/SKU110K', help='原始数据集路径')
    parser.add_argument('--output-dir', type=str, default='./data/SKU110K/yolo_format', help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    set_logging()
    args = parse_args()
    prepare_sku110k(
        args.data_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    ) 