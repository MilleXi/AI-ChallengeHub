#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import requests
import zipfile
from pathlib import Path
import shutil

# 添加项目根目录到系统路径
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.logger import LOGGER
from utils.general import set_logging

def download_file(url, save_path):
    """
    下载文件并显示进度
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    downloaded = 0
    
    LOGGER.info(f"开始下载: {url}")
    LOGGER.info(f"文件大小: {total_size / (1024 * 1024):.2f} MB")
    
    with open(save_path, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
            downloaded += len(data)
            # 显示下载进度
            done = int(50 * downloaded / total_size)
            sys.stdout.write("\r[%s%s] %d%%" % ('█' * done, ' ' * (50 - done), int(100 * downloaded / total_size)))
            sys.stdout.flush()
    print()
    LOGGER.info(f"下载完成: {save_path}")

def extract_zip(zip_path, extract_path):
    """
    解压ZIP文件
    """
    LOGGER.info(f"解压文件: {zip_path} 到 {extract_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    LOGGER.info("解压完成")

def download_sku110k(data_dir='./data/SKU110K', download_subsets=True):
    """
    下载SKU110K数据集
    
    Args:
        data_dir: 数据集保存路径
        download_subsets: 是否只下载包含饮料/零食的子集
    """
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # GitHub仓库URL
    github_url = "https://github.com/eg4000/SKU110K_CVPR19"
    
    # 数据集下载URL（假设数据集有一个直接下载链接）
    # 实际需要根据仓库情况修改此URL
    dataset_url = "https://github.com/eg4000/SKU110K_CVPR19/releases/download/v1.0/SKU110K_fixed.tar.gz"
    
    # 下载数据集压缩文件
    zip_path = os.path.join(data_dir, "SKU110K_fixed.tar.gz")
    
    if not os.path.exists(zip_path):
        LOGGER.info("开始下载SKU110K数据集...")
        download_file(dataset_url, zip_path)
    else:
        LOGGER.info(f"数据集压缩文件已存在: {zip_path}")
    
    # 解压数据集
    if not os.path.exists(os.path.join(data_dir, "images")) or not os.path.exists(os.path.join(data_dir, "annotations")):
        LOGGER.info("解压数据集...")
        extract_zip(zip_path, data_dir)
    else:
        LOGGER.info("数据集已解压")
    
    if download_subsets:
        LOGGER.info("准备筛选包含饮料/零食的2000张子集...")
        # 这里需要添加筛选逻辑，这可能需要额外的脚本或人工挑选
        # 这只是一个占位符，实际实现需要根据具体情况修改
    
    LOGGER.info("数据集准备完成！")

def parse_args():
    parser = argparse.ArgumentParser(description='下载SKU110K数据集')
    parser.add_argument('--data-dir', type=str, default='./data/SKU110K', help='数据集保存路径')
    parser.add_argument('--download-subsets', action='store_true', help='是否只下载包含饮料/零食的子集')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    set_logging()
    args = parse_args()
    download_sku110k(args.data_dir, args.download_subsets) 