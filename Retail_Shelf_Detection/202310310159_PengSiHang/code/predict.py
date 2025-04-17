from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import torch
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description="零售货架商品检测")
    parser.add_argument("--source", type=str, default=None, help="输入图像或视频路径，默认使用摄像头")
    parser.add_argument("--model", type=str, default=None, help="模型路径，默认使用最新训练的模型")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU阈值")
    parser.add_argument("--save", action="store_true", help="是否保存结果")
    parser.add_argument("--device", type=str, default="", help="设备选择 (例如 'cpu', '0', '0,1,2,3')")
    return parser.parse_args()

def predict(source=None, model_path=None, conf=0.25, iou=0.45, save=True, device=""):
    """使用YOLOv8模型进行预测"""
    try:
        ROOT_DIR = os.path.abspath('../')
        
        # 检查CUDA是否可用
        if not device:
            if torch.cuda.is_available():
                device = '0'  # 使用第一个GPU
                print(f"预测将使用GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("警告：未检测到可用的GPU，将使用CPU预测（速度会较慢）")
        
        if model_path is None:
            # 默认使用最新训练的模型
            runs_dir = os.path.join(ROOT_DIR, 'runs/SKU110K_detection')
            if os.path.exists(runs_dir):
                model_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                latest_model = sorted(model_dirs)[-1] if model_dirs else None
                
                if latest_model:
                    model_path = os.path.join(runs_dir, latest_model, 'weights/best.pt')
                    print(f"使用最新训练的模型: {model_path}")
                else:
                    # 如果没有训练过模型，使用原始的预训练模型
                    model_path = os.path.join(ROOT_DIR, 'yolov8n.pt')
                    print(f"未找到训练模型，使用预训练模型: {model_path}")
            else:
                model_path = os.path.join(ROOT_DIR, 'yolov8n.pt')
                print(f"未找到训练目录，使用预训练模型: {model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"错误：模型文件 {model_path} 不存在！")
            return
        
        # 加载模型
        model = YOLO(model_path)
        
        # 设置源（默认为摄像头）
        if source is None:
            source = 0  # 使用摄像头
            print("使用摄像头进行实时预测...")
        else:
            print(f"使用输入源: {source}")
        
        # 运行预测
        results = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            device=device,
            show=True,
            stream=True,
            verbose=False
        )
        
        # 处理结果
        if isinstance(source, (int, str)) and (source == 0 or source.isdigit()):
            # 如果是摄像头，进行实时预测显示
            for result in results:
                frame = result.orig_img
                # 在图像上绘制检测结果
                annotated_frame = result.plot()
                
                # 显示带有检测结果的帧
                cv2.imshow("零售货架商品检测", annotated_frame)
                
                # 按下'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            # 释放摄像头并关闭所有窗口
            cv2.destroyAllWindows()
        else:
            # 如果是图像或视频，结果已经保存到默认位置
            print(f"预测结果已保存到 {os.path.join(os.getcwd(), 'runs/detect')}")
    
    except KeyboardInterrupt:
        print("\n预测被用户中断")
    except Exception as e:
        print(f"\n预测过程中发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    args = parse_args()
    predict(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        device=args.device
    ) 