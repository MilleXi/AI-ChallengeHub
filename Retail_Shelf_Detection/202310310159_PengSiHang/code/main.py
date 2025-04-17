import argparse
import os
import sys
from train import train_model
from evaluate import evaluate_model, visualize_predictions
from predict import predict

def parse_args():
    parser = argparse.ArgumentParser(description="零售货架商品检测 - YOLOv8")
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")
    
    # 训练模式
    train_parser = subparsers.add_parser("train", help="训练模型")
    
    # 评估模式
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("--model", type=str, default=None, help="模型路径，默认使用最新训练的模型")
    eval_parser.add_argument("--viz", action="store_true", help="是否可视化预测结果")
    eval_parser.add_argument("--samples", type=int, default=5, help="可视化样本数量")
    
    # 预测模式
    pred_parser = subparsers.add_parser("predict", help="使用模型进行预测")
    pred_parser.add_argument("--source", type=str, default=None, help="输入图像或视频路径，默认使用摄像头")
    pred_parser.add_argument("--model", type=str, default=None, help="模型路径，默认使用最新训练的模型")
    pred_parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    pred_parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU阈值")
    pred_parser.add_argument("--save", action="store_true", help="是否保存结果")
    pred_parser.add_argument("--device", type=str, default="", help="设备选择 (例如 'cpu', '0', '0,1,2,3')")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode == "train":
        print("开始训练模型...")
        train_model()
        
    elif args.mode == "eval":
        print("评估模型性能...")
        evaluate_model(args.model)
        
        if args.viz:
            print(f"可视化 {args.samples} 个预测结果...")
            visualize_predictions(args.model, args.samples)
            
    elif args.mode == "predict":
        print("使用模型进行预测...")
        predict(
            source=args.source,
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            device=args.device
        )
    
    else:
        print("请指定运行模式: train, eval, 或 predict")
        print("例如: python main.py train")
        print("      python main.py eval --viz")
        print("      python main.py predict --source /path/to/image.jpg")
        sys.exit(1)

if __name__ == "__main__":
    main() 