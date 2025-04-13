#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path

from utils.general import set_logging
from utils.logger import LOGGER

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/SKU110K', help='数据集路径')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='预训练模型路径')
    parser.add_argument('--cfg', type=str, default='./config/sku110k.yaml', help='模型配置文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批量大小')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', default='', help='cuda设备，如0或0,1,2,3或cpu')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test', 'export'], help='运行模式')
    return parser.parse_args()

def main(args):
    """主函数"""
    # 设置日志
    set_logging()
    LOGGER.info(f'开始 {args.mode} 模式...')

    if args.mode == 'train':
        from scripts.train import train
        train(args)
    elif args.mode == 'val':
        from scripts.val import validate
        validate(args)
    elif args.mode == 'test':
        from scripts.test import test
        test(args)
    elif args.mode == 'export':
        from scripts.export import export_model
        export_model(args)
    else:
        LOGGER.error(f'未知模式: {args.mode}')

if __name__ == '__main__':
    args = parse_args()
    main(args) 