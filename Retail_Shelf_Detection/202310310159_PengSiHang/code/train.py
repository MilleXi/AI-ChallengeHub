from ultralytics import YOLO
import os
import torch
import sys

# 设置项目路径
ROOT_DIR = os.path.abspath('.')  # 当前目录
print(f"项目根目录: {ROOT_DIR}")

# 使用转换好的YOLO格式数据集
YOLO_FORMAT_DIR = os.path.join(ROOT_DIR, 'data', 'SKU110K', 'yolo_format')
CONFIG_YAML = os.path.join(YOLO_FORMAT_DIR, 'data.yaml')

def train_model():
    try:
        # 检查配置文件是否存在
        if not os.path.exists(CONFIG_YAML):
            print(f"错误: 数据配置文件 {CONFIG_YAML} 不存在！")
            print("请先运行 python code/convert_annotations.py 转换数据集")
            return None
        
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            device = '0'  # 使用第一个GPU
            print(f"训练将使用GPU: {torch.cuda.get_device_name(0)}")
            # 显示可用显存
            free_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU总显存: {free_mem:.2f} GB")
        else:
            device = 'cpu'
            print("警告：未检测到可用的GPU，将使用CPU训练（速度会很慢）")
        
        # 加载预训练模型
        model = YOLO(os.path.join(ROOT_DIR, 'yolov8n.pt'))
        print(f"成功加载预训练模型: {os.path.join(ROOT_DIR, 'yolov8n.pt')}")
        
        # 开始训练
        print(f"使用配置文件: {CONFIG_YAML}")
        print("开始训练...")
        
        results = model.train(
            data=CONFIG_YAML,
            epochs=50,           # 训练轮数
            imgsz=512,           # 输入图像大小（减小以节省显存）
            batch=4,             # 批次大小（减小以节省显存）
            device=device,       # GPU设备，动态选择
            workers=2,           # 数据加载线程数（减少以降低内存使用）
            patience=10,         # 早停耐心值
            save=True,           # 保存训练结果
            project=os.path.join(ROOT_DIR, 'runs'),  # 结果保存目录
            name='SKU110K_detection',  # 实验名称
            exist_ok=True,       # 是否覆盖已有结果
            pretrained=True,     # 使用预训练权重
            optimizer='SGD',     # 优化器
            verbose=True,        # 是否打印详细信息
            seed=42,             # 随机种子
            deterministic=True,  # 确定性训练
            multi_scale=False,   # 关闭多尺度训练以节省显存
            rect=True,           # 使用矩形训练以提高效率
            cos_lr=True,         # 余弦学习率调度
            lr0=0.01,            # 初始学习率
            lrf=0.01,            # 最终学习率因子
            momentum=0.937,      # SGD动量
            weight_decay=0.0005, # 权重衰减
            warmup_epochs=3.0,   # 预热轮数
            warmup_momentum=0.8, # 预热动量
            warmup_bias_lr=0.1,  # 预热偏置学习率
            box=7.5,             # 边界框损失权重
            cls=0.5,             # 类别损失权重
            dfl=1.5,             # 分布焦点损失权重
            nbs=64,              # 标称批次大小
            overlap_mask=False,  # 关闭重叠掩码以节省显存
            mask_ratio=4,        # 掩码比例
            dropout=0.0,         # dropout率
            val=True,            # 是否在训练期间进行验证
            cache=False,         # 是否缓存图像以加快训练速度
            fraction=0.5,        # 使用50%的训练数据（根据显存情况调整）
            amp=True             # 启用混合精度训练，减少显存使用
        )
        return results
    except KeyboardInterrupt:
        print("\n训练被用户中断。已保存最新模型。")
        return None
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    train_model()