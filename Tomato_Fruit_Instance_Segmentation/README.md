# **Tomato Fruit Instance Segmentation**

## **Background**
This project involves detecting and segmenting tomato fruits, which can help in assessing fruit maturity and predicting yield in agricultural automation.

## **Dataset**
- **Laboro Tomato Dataset**: 2,842 images of tomato fruits with instance-level segmentation annotations.
- **Source**: [Laboro Tomato Dataset](https://github.com/laboroai/Laboro-Tomato)

## **Reference Models**
- **Mask R-CNN with ResNet50-FPN**: A powerful model for instance segmentation tasks.
- **YOLOv8-seg**: A lightweight version of YOLOv8 tailored for instance segmentation tasks.

## **Difficulty**: ★★★  
(The challenge lies in detecting overlapping tomatoes and performing instance segmentation accurately.)

## **Tips**:
- Use **NMS** (Non-Maximum Suppression) with a carefully selected IoU threshold to avoid false positives in overlapping instances.
- Data augmentation (such as **scaling** and **translation**) can help the model generalize better when detecting overlapping fruits.
- Ensure your instance segmentation mask is accurate for each tomato and handle instances of overlap effectively during training.