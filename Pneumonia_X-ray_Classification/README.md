# **Pneumonia X-ray Classification**

## **Background**
The goal of this project is to assist doctors in quickly screening chest X-ray images for signs of pneumonia, which can be a challenging task due to the subtle nature of early-stage pneumonia symptoms.

## **Dataset**
- **ChestX-ray8**: 5,863 X-ray images with binary classification (Normal/Pneumonia)
- **Source**: [Kaggle - Chest X-ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## **Reference Models**
- **MobileNetV3**: A lightweight model suitable for medical image classification. You can fine-tune it by freezing the pre-trained layers and adjusting the final layers.
- **EfficientNet-B0**: A balance between speed and accuracy, making it a good choice for classification tasks with a smaller dataset.

## **Difficulty**: ★★☆  
(Challenges include handling the imbalance of medical images, especially for pneumonia detection.)

## **Tips**:
Medical images often have an imbalance in class distribution, particularly in the pneumonia X-ray dataset. To tackle this, consider:
- Using **oversampling** techniques to augment the number of positive (pneumonia) samples.
- Employing **elastic deformations** or **random rotations** to enhance the training set.
- Using **weighted cross-entropy loss** (with a higher weight for positive samples) to help the model focus on the underrepresented class.
