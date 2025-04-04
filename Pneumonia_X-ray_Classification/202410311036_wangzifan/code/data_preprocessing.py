import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 设置数据路径
train_dir = r'C:\Users\wzf20\Desktop\chest_xray_pneumonia\data\train'
val_dir = r'C:\Users\wzf20\Desktop\chest_xray_pneumonia\data\val'
test_dir = r'C:\Users\wzf20\Desktop\chest_xray_pneumonia\data\test'

# 图像目标尺寸
img_height = 150
img_width = 150

# 定义数据增强和预处理变换
train_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomAffine(
        degrees=40,
        translate=(0.2, 0.2),
        shear=20,           # shear角度
        scale=(0.8, 1.2)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    # 如果需要，可以添加归一化 transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor()
])

test_transform = val_transform

def get_dataloaders(batch_size=32):
    # 加载训练和验证数据集（要求文件夹结构为每个类别一个子文件夹）
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"训练集大小: {len(train_dataset)} 样本")
    print(f"验证集大小: {len(val_dataset)} 样本")
    
    return train_loader, val_loader

def get_test_loader(batch_size=32):
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader

if __name__ == '__main__':
    # 测试数据加载是否正常
    get_dataloaders(batch_size=32)




#pip3 install torch torchvision torchaudio