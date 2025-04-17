# SKU110K 示例数据

本目录包含 SKU110K 数据集的示例图像，用于展示零售货架商品检测系统的输入数据。

## 数据集简介

SKU110K 数据集是一个专门用于商品密集检测的大规模数据集，包含了约 11,000 张零售货架图像，标注了超过 170 万个商品边界框。该数据集的主要特点是商品排列密集、外观相似，是零售场景目标检测的重要数据集。

## 目录结构

```
SKU110K_example/
├── images/              # 图像文件目录
│   ├── train/           # 训练集示例图像
│   ├── val/             # 验证集示例图像
│   └── test/            # 测试集示例图像
└── annotations/         # 标注文件
```

## 示例说明

本目录中的示例图像是从完整 SKU110K 数据集中选取的代表性样本，保持了原始数据集的以下特征：

1. **密集排列**：货架上的商品密集排列，边界框重叠严重
2. **小目标**：单个商品在整个图像中所占比例较小
3. **外观相似**：许多商品包装颜色和外形相似，增加识别难度
4. **环境变化**：图像来自多个零售店，光照和视角各不相同

## 数据来源

完整的 SKU110K 数据集由以色列特拉维夫大学发布，原始数据集可从以下地址获取：
[SKU110K Dataset](https://github.com/eg4000/SKU110K_CVPR19)

```
@inproceedings{goldman2019dense,
  title={Precise Detection in Densely Packed Scenes},
  author={Goldman, Eran and Herzig, Roei and Eisenschtat, Aviv and Goldberger, Jacob and Hassner, Tal},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5227--5236},
  year={2019}
}
```

## 使用说明

这些示例图像仅用于演示目的。要使用完整数据集训练模型，请下载原始 SKU110K 数据集，并使用`convert_annotations.py`脚本转换为 YOLO 格式。
