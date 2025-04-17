from ultralytics import YOLO
import os
import cv2
import sys
import glob

# 设置数据目录和保存目录
current_dir = os.path.abspath('.')
TEST_DIR = os.path.join(current_dir, 'data', 'test')
SAVE_DIR = os.path.join(TEST_DIR, 'output')
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"测试图片目录: {TEST_DIR}")
print(f"创建保存目录: {SAVE_DIR}")

# 设置模型路径
root_dir = os.path.abspath('.')

# 寻找最佳模型
model_path = None
runs_dir = os.path.join(root_dir, 'runs/SKU110K_detection')
if os.path.exists(runs_dir):
    weights_path = os.path.join(runs_dir, 'weights', 'best.pt')
    if os.path.exists(weights_path):
        model_path = weights_path
        print(f"找到模型: {model_path}")

# 如果没找到训练好的模型，使用预训练模型
if not model_path or not os.path.exists(model_path):
    model_path = os.path.join(root_dir, 'yolov8n.pt')
    print(f"使用预训练模型: {model_path}")

# 加载模型
try:
    model = YOLO(model_path)
    print(f"成功加载模型: {model_path}")
except Exception as e:
    print(f"加载模型失败: {e}")
    sys.exit(1)

# 获取测试目录中的所有图像，使用正确的方式避免重复
image_files = []
valid_extensions = ['.jpg', '.jpeg', '.png']

# 使用os.listdir遍历目录并过滤图像文件，避免Windows不区分大小写导致的重复
for file in os.listdir(TEST_DIR):
    file_lower = file.lower()
    if any(file_lower.endswith(ext) for ext in valid_extensions):
        # 跳过output目录中的文件
        if os.path.isdir(os.path.join(TEST_DIR, file)):
            continue
        image_files.append(os.path.join(TEST_DIR, file))

if not image_files:
    print(f"在目录 {TEST_DIR} 中未找到任何图像文件！")
    sys.exit(1)

print(f"在目录 {TEST_DIR} 中找到 {len(image_files)} 个图像文件")

# 对每个图像进行预测
for i, image_path in enumerate(image_files):
    print(f"\n处理图像 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
    
    try:
        # 进行预测，使用640x640分辨率以及其他提高精度的参数
        results = model.predict(
            source=image_path, 
            save=True, 
            project=SAVE_DIR, 
            name="",
            imgsz=640,           # 设置输入分辨率为640x640
            conf=0.3,            # 降低置信度阈值以检测更多物体
            iou=0.5,             # 调整IOU阈值以改进NMS
            augment=True,        # 启用测试时增强
            agnostic_nms=True,   # 使用类别无关的NMS
            retina_masks=True    # 使用更精细的掩码（对分割模型有效）
        )
        
        result_img = results[0].plot()
        
        # YOLOv8会自动保存结果到SAVE_DIR/{image_name}
        save_path = os.path.join(SAVE_DIR, os.path.basename(image_path))
        print(f"预测结果已保存至: {save_path}")
        
        # 显示图像（如果在有GUI的环境中运行）
        try:
            cv2.imshow("Detection Result", result_img)
            # 等待键盘输入，按ESC键退出循环
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC键
                break
        except Exception as e:
            print(f"无法显示图像窗口，但已保存结果: {e}")
    
    except Exception as e:
        print(f"处理图像 {image_path} 时发生错误: {e}")
        continue

# 关闭所有窗口
cv2.destroyAllWindows()
print("\n所有图像处理完成！") 