from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# 设置项目路径
ROOT_DIR = os.path.abspath('.')  # 当前目录
print(f"项目根目录: {ROOT_DIR}")

# 使用转换好的YOLO格式数据集
YOLO_FORMAT_DIR = os.path.join(ROOT_DIR, 'data', 'SKU110K', 'yolo_format')
CONFIG_YAML = os.path.join(YOLO_FORMAT_DIR, 'data.yaml')
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, 'data', 'SKU110K', 'images', 'test')

def evaluate_model(model_path=None):
    """评估训练好的模型在测试集上的性能"""
    # 检查配置文件是否存在
    if not os.path.exists(CONFIG_YAML):
        print(f"错误: 数据配置文件 {CONFIG_YAML} 不存在！")
        print("请先运行 python code/convert_annotations.py 转换数据集")
        return None
        
    if model_path is None:
        # 默认使用最新训练的模型
        runs_dir = os.path.join(ROOT_DIR, 'runs/SKU110K_detection')
        if os.path.exists(runs_dir):
            # 检查weights目录
            weights_dir = os.path.join(runs_dir, 'weights')
            if os.path.exists(weights_dir):
                best_model = os.path.join(weights_dir, 'best.pt')
                if os.path.exists(best_model):
                    model_path = best_model
                    print(f"使用模型: {model_path}")
            
            # 如果没找到weights子目录，检查训练子目录
            if model_path is None:
                model_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                latest_model = sorted(model_dirs)[-1] if model_dirs else None
                
                if latest_model:
                    model_path = os.path.join(runs_dir, latest_model, 'weights/best.pt')
                    if os.path.exists(model_path):
                        print(f"使用模型: {model_path}")
        
        if model_path is None:
            # 如果没有训练过模型，使用原始的预训练模型
            model_path = os.path.join(ROOT_DIR, 'yolov8n.pt')
            print(f"未找到训练模型，使用预训练模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return None
    
    # 加载模型
    try:
        model = YOLO(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 评估模型
    try:
        print(f"开始在测试集上评估模型...")
        print(f"使用配置文件: {CONFIG_YAML}")
        results = model.val(data=CONFIG_YAML, split='test')
        
        print(f"\n测试结果汇总:")
        try:
            print(f"mAP50-95: {results.box.map:.4f}")
            print(f"mAP50: {results.box.map50:.4f}")
            print(f"mAP75: {results.box.map75:.4f}")
            print(f"精确率: {results.box.p.mean():.4f}")
            print(f"召回率: {results.box.r.mean():.4f}")
        except:
            print("无法获取详细评估指标，可能是因为测试集未提供真实标签")
        
        return results
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_predictions(model_path=None, num_samples=5):
    """可视化模型在测试集上的预测结果"""
    if model_path is None:
        # 默认使用最新训练的模型
        runs_dir = os.path.join(ROOT_DIR, 'runs/SKU110K_detection')
        if os.path.exists(runs_dir):
            # 检查weights目录
            weights_dir = os.path.join(runs_dir, 'weights')
            if os.path.exists(weights_dir):
                best_model = os.path.join(weights_dir, 'best.pt')
                if os.path.exists(best_model):
                    model_path = best_model
                    print(f"使用模型: {model_path}")
            
            # 如果没找到weights子目录，检查训练子目录
            if model_path is None:
                model_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                latest_model = sorted(model_dirs)[-1] if model_dirs else None
                
                if latest_model:
                    model_path = os.path.join(runs_dir, latest_model, 'weights/best.pt')
                    if os.path.exists(model_path):
                        print(f"使用模型: {model_path}")
        
        if model_path is None:
            # 如果没有训练过模型，使用原始的预训练模型
            model_path = os.path.join(ROOT_DIR, 'yolov8n.pt')
            print(f"未找到训练模型，使用预训练模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return None
    
    # 加载模型
    try:
        model = YOLO(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 获取测试图像
    test_images = []
    
    if os.path.exists(TEST_IMAGES_DIR):
        print(f"从测试目录加载图像: {TEST_IMAGES_DIR}")
        for file in os.listdir(TEST_IMAGES_DIR):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(TEST_IMAGES_DIR, file))
    else:
        # 如果测试目录不存在，尝试从测试集文件列表读取
        test_file = os.path.join(YOLO_FORMAT_DIR, 'test.txt')
        if os.path.exists(test_file):
            print(f"从测试集文件列表加载图像: {test_file}")
            with open(test_file, 'r') as f:
                for line in f:
                    img_path = line.strip()
                    if os.path.exists(img_path):
                        test_images.append(img_path)
    
    if not test_images:
        print(f"错误: 未找到任何测试图像!")
        return None
    
    print(f"找到 {len(test_images)} 张测试图像")
    
    # 选择几个样本进行可视化
    if len(test_images) > num_samples:
        test_samples = np.random.choice(test_images, num_samples, replace=False)
    else:
        test_samples = test_images
    
    plt.figure(figsize=(15, 12))
    for i, img_path in enumerate(test_samples):
        if i >= num_samples:
            break
            
        print(f"预测图像 {i+1}/{len(test_samples)}: {os.path.basename(img_path)}")
        
        try:
            # 进行预测
            results = model(img_path, conf=0.3, iou=0.5, agnostic_nms=True)
            
            # 获取预测结果图像
            result_img = results[0].plot()
            
            # 显示图像
            plt.subplot(min(3, (num_samples+1)//2), min(2, num_samples), i+1)
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title(f"样本 {i+1}")
            plt.axis('off')
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            continue
    
    # 保存图像
    save_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictions.png"))
    plt.close()
    
    print(f"可视化结果已保存至 {os.path.join(save_dir, 'predictions.png')}")
    return os.path.join(save_dir, "predictions.png")

if __name__ == "__main__":
    # 评估模型
    results = evaluate_model()
    
    # 可视化预测结果
    if results is not None:
        visualize_predictions() 