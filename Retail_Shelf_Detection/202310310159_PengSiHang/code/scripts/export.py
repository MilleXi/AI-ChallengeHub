#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
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

def export_model(args):
    """
    导出模型为不同格式（ONNX, TensorRT等）
    """
    # 检查模型文件是否存在
    if not os.path.exists(args.weights):
        LOGGER.error(f"找不到模型权重文件: {args.weights}")
        sys.exit(1)
    
    # 创建结果目录
    save_dir = os.path.join(ROOT, 'results', 'export')
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型
    model = YOLO(args.weights)
    
    # 输出导出参数
    LOGGER.info(f"开始导出模型, 配置:")
    LOGGER.info(f"- 源模型: {args.weights}")
    LOGGER.info(f"- 导出格式: {args.format}")
    LOGGER.info(f"- 输入尺寸: {args.img_size}")
    LOGGER.info(f"- 半精度: {args.half}")
    LOGGER.info(f"- 动态: {args.dynamic}")
    LOGGER.info(f"- 简化: {args.simplify}")
    LOGGER.info(f"- 保存目录: {save_dir}")
    
    # 导出模型
    exported_model = model.export(
        format=args.format,
        imgsz=args.img_size,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        workspace=args.workspace,
        device=args.device,
        nms=True,
    )
    
    LOGGER.info(f"模型导出完成! 保存在 {exported_model}")
    
    return exported_model

def parse_args():
    parser = argparse.ArgumentParser(description='导出YOLOv8模型为部署格式')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件')
    parser.add_argument('--format', type=str, default='onnx', 
                        choices=['torchscript', 'onnx', 'openvino', 'engine', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle'],
                        help='导出格式')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度')
    parser.add_argument('--dynamic', action='store_true', help='动态ONNX轴')
    parser.add_argument('--simplify', action='store_true', help='简化ONNX模型')
    parser.add_argument('--opset', type=int, default=12, help='ONNX操作集版本')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT工作空间大小(GB)')
    parser.add_argument('--device', default='0', help='cuda设备，如0或0,1,2,3或cpu')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    set_logging()
    args = parse_args()
    export_model(args) 