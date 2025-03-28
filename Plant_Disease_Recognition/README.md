# **Plant Disease Recognition**

## **Background**
This project aims to detect diseases in plant leaves, which can help automate the monitoring of crop health in agricultural settings.

## **Dataset**
- **PlantVillage Dataset**: 54,000 leaf images with 38 classes (diseased/healthy) from various crops.
- **Subset for Practice**: Choose 3-5 plant species (e.g., tomato, potato) for a smaller subset (~2,000 images).
- **Source**: [PlantVillage Dataset](https://plantvillage.psu.edu/)

## **Reference Models**
- **Vision Transformer (Tiny)**: A lightweight model for smaller datasets, it works well for image classification tasks.
- **EfficientNet-B3**: A good balance between performance and computational cost, suitable for medium-sized datasets.

## **Difficulty**: ★★☆  
(Challenges include dealing with imbalanced classes for some diseases.)

## **Tips**:
- Plant diseases are often imbalanced. You can use **data augmentation** techniques like rotation, cropping, and color jittering.
- Apply **class-weighted loss** to ensure that less-represented diseases are not overlooked during training.
- Consider techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) for further improving class balance.
