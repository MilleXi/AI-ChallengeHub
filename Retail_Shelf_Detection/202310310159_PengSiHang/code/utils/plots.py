#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

def draw_detections(image, detections, class_names, conf_threshold=0.25):
    """
    在图像上绘制检测结果
    
    Args:
        image: 原始图像
        detections: 检测结果，格式为 [x1, y1, x2, y2, conf, class_id]
        class_names: 类别名称列表
        conf_threshold: 置信度阈值，低于此值的检测结果不会绘制
    
    Returns:
        绘制了检测框的图像
    """
    # 复制图像以避免修改原图
    img_draw = image.copy()
    
    # 检查是否有检测结果
    if detections is None or len(detections) == 0:
        return img_draw
    
    # 遍历每个检测结果
    for det in detections:
        # 获取边界框坐标、置信度和类别
        x1, y1, x2, y2, conf, class_id = det
        
        # 如果置信度低于阈值，跳过
        if conf < conf_threshold:
            continue
        
        # 确保class_id是整数
        class_id = int(class_id)
        
        # 获取类别名称
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        # 随机生成颜色 (基于类别ID)
        color = generate_color(class_id)
        
        # 绘制边界框
        cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # 准备标签文本
        label = f"{class_name} {conf:.2f}"
        
        # 获取标签的尺寸
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 绘制标签背景
        cv2.rectangle(
            img_draw, 
            (int(x1), int(y1) - label_height - 5), 
            (int(x1) + label_width, int(y1)), 
            color, 
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            img_draw,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img_draw

def generate_color(class_id):
    """
    基于类别ID生成一个稳定的颜色
    
    Args:
        class_id: 类别ID
    
    Returns:
        RGB颜色元组，格式为 (B, G, R)
    """
    np.random.seed(int(class_id) + 42)
    color = np.random.randint(0, 256, 3).tolist()
    return color

def plot_one_box(box, img, color=(128, 128, 128), label=None, line_thickness=3):
    """
    在图像上绘制单个边界框
    
    Args:
        box: 边界框坐标，格式为 [x1, y1, x2, y2]
        img: 要绘制的图像
        color: 边界框颜色，BGR格式
        label: 可选标签文本
        line_thickness: 线条粗细
    
    Returns:
        None (直接修改输入图像)
    """
    # 绘制边界框
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # 线条粗细
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    # 绘制标签
    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # 标签背景
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_results(file='results.csv'):
    """
    绘制训练结果
    
    Args:
        file: 结果CSV文件路径
    
    Returns:
        None (显示/保存图像)
    """
    try:
        import pandas as pd
        
        # 读取CSV文件
        data = pd.read_csv(file)
        
        # 创建图像
        fig, ax = plt.subplots(2, 5, figsize=(18, 8))
        ax = ax.flatten()
        
        # 设置图像标题
        s = ['训练/验证损失', '精确率', '召回率', 'mAP@0.5', 'mAP@0.5:0.95',
             '边界框损失', '分类损失', '目标损失', '学习率', '数据集大小']
        
        for i, metric in enumerate(['loss', 'precision', 'recall', 'mAP_0.5', 'mAP_0.5:0.95',
                                   'box_loss', 'cls_loss', 'obj_loss', 'lr', 'dataset_size']):
            if metric in data.columns:
                ax[i].plot(data['epoch'], data[metric], '.-', linewidth=2, markersize=8)
                ax[i].set_title(s[i])
            else:
                ax[i].set_visible(False)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig('results.png', dpi=200)
        plt.close()
    
    except Exception as e:
        print(f'警告: 绘制结果出错: {e}')

def plot_confusion_matrix(confusion_matrix, names, file_name='confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    Args:
        confusion_matrix: 混淆矩阵
        names: 类别名称列表
        file_name: 保存的文件名
    
    Returns:
        None (保存图像)
    """
    try:
        # 设置图像大小和颜色映射
        array = confusion_matrix / (confusion_matrix.sum(1, keepdims=True) + 1e-6)
        array[array < 0.005] = np.nan  # 忽略小值
        
        fig, ax = plt.figure(figsize=(12, 9)), plt.subplot(111)
        im = ax.imshow(array, cmap='Blues')
        
        # 添加标签
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticklabels(names)
        
        # 添加颜色条和标题
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('预测')
        ax.set_ylabel('真实')
        ax.set_title('混淆矩阵')
        
        # 在单元格中添加文本
        for i in range(len(names)):
            for j in range(len(names)):
                text = ax.text(j, i, f'{confusion_matrix[i, j]}',
                           ha="center", va="center", color="white" if array[i, j] > 0.5 else "black")
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(file_name, dpi=250)
        plt.close()
    
    except Exception as e:
        print(f'警告: 绘制混淆矩阵出错: {e}') 