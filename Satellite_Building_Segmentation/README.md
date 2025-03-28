# **Satellite Building Segmentation**

## **Background**
This project aims to detect and segment buildings from satellite images, a key task for urban planning and environmental monitoring.

## **Dataset**
- **Inria Aerial Image Labeling Dataset**: 360 satellite images (512x512 resolution) labeled for building segmentation.
- **Source**: [Inria Aerial Image Dataset](https://project.inria.fr/aerialimagelabeling/)

## **Reference Models**
- **Feature Pyramid Network (FPN)**: This model is excellent for segmenting objects of various sizes.
- **Fast-SCNN**: A lightweight and real-time segmentation network suitable for fast applications.

## **Difficulty**: ★★★  
(Handling high-resolution images and avoiding overfitting with a small dataset is the main challenge.)

## **Tips**:
- With only 360 images, the dataset is small, so **data augmentation** (rotation, flipping, etc.) will be essential.
- You might want to divide the high-resolution images into smaller patches (256x256) to manage memory usage and improve training speed.
- Consider using **pre-trained models** to leverage knowledge from ImageNet for better performance with limited data.
