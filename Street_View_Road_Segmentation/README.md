# **Street View Road Segmentation**

## **Background**
This project involves identifying drivable areas in street view images, which is essential for autonomous driving systems to make real-time decisions.

## **Dataset**
- **Cityscapes Subset**: A subset of 500 images with detailed pixel-wise annotations, categorized into 19 classes (e.g., road, sidewalk, building, etc.).
- **Source**: [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

## **Reference Models**
- **U-Net with ResNet18 Backbone**: Ideal for small sample segmentation tasks, providing good results with fewer images.
- **DeepLabv3+ Mobile**: A lightweight model tailored for real-time segmentation with mobile devices.

## **Difficulty**: ★★★  
(Requires handling multi-class pixel-level segmentation, a complex problem in computer vision.)

## **Tips**:
- Ensure each class has a sufficient number of samples, especially for less frequent categories. Using a **weighted loss function** can help.
- Consider augmentations like **random scaling**, **flipping**, and **cropping** to artificially increase dataset size.
- Check your dataset balance for rare classes and adjust training strategies to avoid overfitting on more frequent classes.
