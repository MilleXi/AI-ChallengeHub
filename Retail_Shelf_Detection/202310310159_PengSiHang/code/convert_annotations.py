#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import sys

def convert_csv_to_yolo(csv_file, output_dir, images_dir, class_id=0):
    """
    将SKU110K数据集的CSV标注转换为YOLO格式
    
    参数:
    - csv_file: CSV标注文件路径
    - output_dir: 输出YOLO格式标注的目录
    - images_dir: 图像文件目录
    - class_id: 物体类别ID (默认为0，表示所有物体为同一类)
    
    YOLO格式:
    - 每个图像对应一个txt文件
    - 每行格式为: class_id x_center y_center width height
    - 所有值都是归一化的 (0-1)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_file}")
    try:
        # 尝试不使用列名读取CSV
        df = pd.read_csv(csv_file, header=None)
        print(f"成功读取CSV文件，共 {len(df)} 行数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 如果CSV没有标题行，我们需要设置列名
    # 根据readme.txt的描述，列名应该是: image_name,x1,y1,x2,y2,class,image_width,image_height
    # 但根据实际输出看，列名可能是:
    # [图像名称, x1, y1, x2, y2, 'object', 宽度, 高度]
    df.columns = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class_name', 'image_width', 'image_height']
    
    # 显示前几行数据，检查格式
    print("数据预览:")
    print(df.head())
    
    # 统计需要处理的图像数量
    image_count = len(df['image_name'].unique())
    print(f"需要处理的图像数量: {image_count}")
    
    # 根据图像名称分组，为每个图像创建一个YOLO标注文件
    image_groups = df.groupby('image_name')
    
    # 使用tqdm显示进度条
    processed_count = 0
    for image_name, group in tqdm(image_groups, desc="转换标注"):
        # 检查图像是否存在
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"警告: 图像 {image_name} 不存在于 {images_dir}")
            continue
        
        # 创建YOLO格式标注文件
        yolo_file = os.path.join(output_dir, Path(image_name).stem + '.txt')
        
        # 获取图像尺寸
        img_width = group['image_width'].iloc[0]
        img_height = group['image_height'].iloc[0]
        
        # 确保图像尺寸是数值
        try:
            img_width = float(img_width)
            img_height = float(img_height)
        except ValueError:
            print(f"警告: 图像 {image_name} 的尺寸无效: 宽={img_width}, 高={img_height}")
            continue
        
        with open(yolo_file, 'w') as f:
            # 处理该图像的每个边界框
            for _, row in group.iterrows():
                # 获取边界框坐标
                try:
                    x1, y1, x2, y2 = float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])
                except ValueError:
                    print(f"警告: 图像 {image_name} 中存在无效坐标: x1={row['x1']}, y1={row['y1']}, x2={row['x2']}, y2={row['y2']}")
                    continue
                
                # 检查坐标有效性
                if x1 >= x2 or y1 >= y2:
                    print(f"警告: 图像 {image_name} 中存在无效边界框: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                # 转换为YOLO格式 (中心点坐标和宽高，并归一化)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # 确保值在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # 写入YOLO格式的一行
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"已处理 {processed_count}/{image_count} 张图像")
    
    print(f"转换完成! 共处理 {processed_count} 张图像，YOLO格式标注已保存到 {output_dir}")
    return True

def create_yolo_dataset_structure(root_dir, dest_dir):
    """
    创建YOLO格式的数据集目录结构
    
    参数:
    - root_dir: SKU110K数据集根目录
    - dest_dir: YOLO格式数据集输出目录
    """
    # 检查目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误: 根目录 {root_dir} 不存在")
        return False
    
    print(f"正在创建YOLO格式数据集结构...")
    
    # 创建输出目录
    yolo_images_dir = os.path.join(dest_dir, 'images')
    yolo_labels_dir = os.path.join(dest_dir, 'labels')
    
    # 创建训练、验证和测试子目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(yolo_images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(yolo_labels_dir, split), exist_ok=True)
    
    # 转换标注文件
    annotations_dir = os.path.join(root_dir, 'annotations')
    images_dir = os.path.join(root_dir, 'images')
    
    success = True
    for split in ['train', 'val', 'test']:
        csv_file = os.path.join(annotations_dir, f'annotations_{split}.csv')
        if not os.path.exists(csv_file):
            print(f"错误: 标注文件 {csv_file} 不存在")
            success = False
            continue
        
        split_images_dir = os.path.join(images_dir, split)
        if not os.path.exists(split_images_dir):
            print(f"错误: 图像目录 {split_images_dir} 不存在")
            success = False
            continue
        
        # 转换CSV标注为YOLO格式
        labels_dir = os.path.join(yolo_labels_dir, split)
        result = convert_csv_to_yolo(csv_file, labels_dir, split_images_dir)
        if not result:
            success = False
    
    if success:
        print(f"已创建YOLO格式数据集结构，路径: {dest_dir}")
        
        # 生成训练、验证和测试集的图像路径文件
        for split in ['train', 'val', 'test']:
            images_list_file = os.path.join(dest_dir, f'{split}.txt')
            split_images_dir = os.path.join(images_dir, split)
            
            if os.path.exists(split_images_dir):
                with open(images_list_file, 'w') as f:
                    for img_file in os.listdir(split_images_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            # 使用绝对路径
                            img_path = os.path.abspath(os.path.join(split_images_dir, img_file))
                            f.write(f"{img_path}\n")
                print(f"已生成{split}集图像列表: {images_list_file}")
        
        # 创建数据集配置文件
        train_txt = os.path.abspath(os.path.join(dest_dir, 'train.txt'))
        val_txt = os.path.abspath(os.path.join(dest_dir, 'val.txt'))
        test_txt = os.path.abspath(os.path.join(dest_dir, 'test.txt'))
        
        yaml_content = f"""
# SKU110K数据集配置
train: {train_txt}
val: {val_txt}
test: {test_txt}

# 类别数量和名称
nc: 1  # 类别数量
names: ['item']  # 类别名称
"""
        
        yaml_file = os.path.join(dest_dir, 'data.yaml')
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"已创建数据集配置文件: {yaml_file}")
        
        return True
    else:
        print("转换过程中出现错误，请检查上述日志")
        return False

def main():
    # 设置数据集路径
    root_dir = os.path.abspath('.')
    sku110k_dir = os.path.join(root_dir, 'data', 'SKU110K')
    output_dir = os.path.join(root_dir, 'data', 'SKU110K', 'yolo_format')
    
    if not os.path.exists(sku110k_dir):
        print(f"错误: SKU110K数据集目录 {sku110k_dir} 不存在")
        return
    
    # 创建YOLO格式数据集
    success = create_yolo_dataset_structure(sku110k_dir, output_dir)
    
    if success:
        print("="*50)
        print("转换完成! 现在您可以使用以下命令训练YOLOv8模型:")
        print(f"python code/main.py train")
        print("="*50)
    else:
        print("转换失败，请检查错误日志并解决问题")

if __name__ == "__main__":
    main() 