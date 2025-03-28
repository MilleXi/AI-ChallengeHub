# **Retail Shelf Detection**

## **Background**
In retail, detecting products on shelves automatically can help with inventory management and shelf optimization.

## **Dataset**
- **SKU-110k**: 11,762 images with annotated bounding boxes for retail products (mainly drinks and snacks).
- **Subset for Practice**: Choose a subset of 2,000 images with specific categories like drinks or snacks.
- **Source**: [SKU-110k Dataset](https://github.com/eg4000/SKU110K_CVPR19)

## **Reference Models**
- **SSD with MobileNet Backbone**: A good lightweight model for detecting objects in images.
- **YOLOv8n (Nano)**: A tiny version of YOLOv8, perfect for resource-constrained environments.

## **Difficulty**: ★★☆  
(The challenge is mainly in detecting small and densely packed products.)

## **Tips**:
- Since the dataset has dense and small targets, use **multi-scale training** and adjust **Anchor Box sizes** to improve detection.
- Enable **Focal Loss** to prioritize small or difficult-to-detect objects.
- Fine-tuning NMS (Non-Maximum Suppression) is critical to reduce overlap and improve precision for densely packed items.

