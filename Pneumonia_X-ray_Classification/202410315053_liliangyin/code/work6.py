# =============== 导入库 ===============
import cv2  # OpenCV用于图像处理
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from sklearn.model_selection import train_test_split

# =============== 配置参数 ===============
DATA_DIR = "E:\WorkData"  # 数据集路径
IMG_SIZE = 224                # 图像统一缩放尺寸
BATCH_SIZE = 32               # 批大小根据显存调整
NUM_EPOCHS = 10               # 训练轮数
LEARNING_RATE = 0.001         # 学习率
SEED = 42                     # 随机种子

# 设置设备（优先使用GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)

# =============== 数据加载器 ===============
class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.transform = transform
        self.image_paths = []
        
        # 遍历目录收集图像路径和标签
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, mode, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpeg', '.jpg', '.png')):
                    self.image_paths.append((
                        os.path.join(class_dir, img_file),
                        class_idx
                    ))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        
        # 使用OpenCV读取并预处理图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 灰度读取
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))     # 统一尺寸
        image = cv2.equalizeHist(image)                     # 直方图均衡化
        
        # 转换为PyTorch张量并应用增强
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 数据增强配置
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),      # 随机旋转±10度
    transforms.RandomHorizontalFlip(),  # 水平翻转增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # 单通道归一化
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# 创建数据集和数据加载器
train_dataset = ChestXRayDataset(DATA_DIR, train_transform, 'train')
test_dataset = ChestXRayDataset(DATA_DIR, test_transform, 'test')

# 划分训练集和验证集（8:2）
train_dataset, val_dataset = train_test_split(
    train_dataset, 
    test_size=0.2,
    random_state=SEED
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# =============== 模型定义 ===============
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 输出尺寸减半
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 56 * 56, 256)  # 224/2/2=56
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self, x):
        # [batch_size, 1, 224, 224]
        x = self.pool(F.relu(self.conv1(x)))  # -> [b,32,112,112]
        x = self.pool(F.relu(self.conv2(x)))  # -> [b,64,56,56]
        x = torch.flatten(x, 1)               # -> [b,64*56*56]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))              # -> [b,256]
        x = self.fc2(x)                      # -> [b,2]
        return x

# =============== 训练配置 ===============
model = PneumoniaCNN().to(device)
criterion = nn.CrossEntropyLoss()          # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============== 训练函数 ===============
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计指标
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# =============== 主训练循环 ===============
print(f"开始训练，使用设备：{device}")
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # 训练阶段
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    
    # 验证阶段
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    
    # 打印进度
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
    print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("=> 保存新最佳模型")

# =============== 测试评估 ===============
print("\n测试集最终评估：")
model.load_state_dict(torch.load("best_model.pth"))
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"测试准确率：{test_acc:.2%}")