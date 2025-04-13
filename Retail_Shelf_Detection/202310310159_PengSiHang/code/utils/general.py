#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import random
import numpy as np
import torch
import yaml

from utils.logger import LOGGER

def set_logging(rank=-1, verbose=True):
    """
    设置日志级别
    """
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN
    )
    return LOGGER

def seed_everything(seed=42):
    """
    设置随机种子，确保结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    LOGGER.info(f'已设置随机种子为 {seed} 以确保可复现性')

def load_yaml(file_path):
    """
    加载YAML配置文件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def save_yaml(file_path, data):
    """
    保存数据为YAML文件
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False)

def xyxy2xywh(x):
    """
    将边界框从(x1, y1, x2, y2)格式转换为(x, y, w, h)格式
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    """
    将边界框从(x, y, w, h)格式转换为(x1, y1, x2, y2)格式
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def clip_coords(boxes, img_shape):
    """
    裁剪边界框坐标以确保在图像范围内
    """
    # 裁剪边界框 xyxy 为图像尺寸 (高, 宽)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将坐标从img1_shape缩放到img0_shape
    """
    if ratio_pad is None:  # 计算从img0_shape到img1_shape的比例
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def select_device(device='', batch_size=None):
    """
    选择设备 (CPU 或 GPU)
    """
    device = str(device).lower()
    cpu = device == 'cpu'
    
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制使用CPU
    elif device:  # 例如: '0' 或 '0,1,2,3'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # 设置可见的CUDA设备
        assert torch.cuda.is_available(), f'CUDA不可用，接收到无效的GPU: {device}'
    
    if not cpu and torch.cuda.is_available():
        devices = device.split(',') if device else '0'  # 例如:['0'] 或 ['0', '1', ...]
        n = len(devices)  # 设备数量
        if n > 1 and batch_size:  # 检查批量大小是否可被设备数整除
            assert batch_size % n == 0, f'批量大小 {batch_size} 不能被GPU数量 {n} 整除'
        space = ' ' * len(device)
        LOGGER.info(f'使用{space + ("CPU" if device == "cpu" else f"GPU {device}")}')
    else:
        LOGGER.info('使用CPU')
    
    return torch.device('cuda:0' if not cpu and torch.cuda.is_available() else 'cpu') 