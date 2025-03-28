# **Retinal Vessel Segmentation**

## **Background**
This project focuses on the segmentation of blood vessels from retinal images, which is crucial for diagnosing eye diseases like diabetic retinopathy.

## **Dataset**
- **DRIVE Dataset**: 40 retinal images with annotated vessel masks.
- **Source**: [DRIVE Dataset](https://drive.grand-challenge.org/)

## **Reference Models**
- **U-Net with Attention Gate**: Designed to focus on small structures, improving the accuracy of vessel detection.
- **LinkNet**: A lightweight model that is fast and effective for medical image segmentation.

## **Difficulty**: ★★★☆  
(Challenges include working with low-contrast images and requiring high precision in structure segmentation.)

## **Tips**:
- Retinal images often have low contrast, so you can use **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to improve contrast and highlight the blood vessels.
- Consider using **Attention Gates** to focus on fine details in the image.
- Use **Dice Loss** to optimize for the overlap between predicted and ground truth vessels.
