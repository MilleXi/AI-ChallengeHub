# Data
https://drive.grand-challenge.org/
# Requirements
```numpy 1.26.4```

```opencv-python 4.11.0.86```

```torch 2.6.0 (cuda 12.6)```
# Progress
* 3.29 dataset utils.py
* 3.29 尝试通过边缘检测进行预处理, 发现部分细小血管无法检测到
* 3.30 使用[U-Net](https://arxiv.org/pdf/1505.04597), 部分代码参考[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
* 4.2 可以较准确预测，但是边缘和中心亮点会被误识别