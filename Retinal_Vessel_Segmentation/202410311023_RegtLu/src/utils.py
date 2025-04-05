import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import os
import cv2
from PIL import Image


class TestDataset(Dataset):
    def __init__(self, data_path):
        self.image_paths = os.path.join(data_path, "images")
        self.mask_paths = os.path.join(data_path, "mask")
        self.file_list = sorted(
            [f for f in os.listdir(self.image_paths) if f.endswith(".tif")]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_name = self.file_list[idx].replace(".tif", "")
        image_file = os.path.join(self.image_paths, f"{base_name}.tif")
        mask_file = os.path.join(self.mask_paths, f"{base_name}_mask.gif")
        image_origin = TF.to_tensor(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE))
        image_color = cv2.imread(image_file)
        mask = TF.to_tensor(Image.open(mask_file).convert("L"))
        padding = (11, 12, 2, 2)
        image = nn.ZeroPad2d(padding)(image_origin)
        mask_padding = nn.ZeroPad2d(padding)(mask)
        new_image = cv2.equalizeHist(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) * (mask.squeeze().numpy() > 0))
        new_image = torch.tensor(new_image).unsqueeze(0) / 255.0
        new_image = nn.ZeroPad2d(padding)(new_image)
        image = (image + new_image) / 2
        return image * (mask_padding > 0), mask, image_origin, image_color


class TrainingDataset(Dataset):
    def __init__(self, data_path):
        self.image_paths = os.path.join(data_path, "images")
        self.mask_paths = os.path.join(data_path, "mask")
        self.manual = os.path.join(data_path, "1st_manual")
        self.file_list = sorted(
            [f for f in os.listdir(self.image_paths) if f.endswith(".tif")]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_name = self.file_list[idx].replace(".tif", "")
        image_file = os.path.join(self.image_paths, f"{base_name}.tif")
        mask_file = os.path.join(self.mask_paths, f"{base_name}_mask.gif")
        manual_file = os.path.join(
            self.manual, f"{base_name.strip('_training')}_manual1.gif"
        )
        image = TF.to_tensor(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE))
        manual = TF.to_tensor(Image.open(manual_file).convert("L"))
        mask = TF.to_tensor(Image.open(mask_file).convert("L"))
        padding = (11, 12, 2, 2)
        image = nn.ZeroPad2d(padding)(image)
        mask_padding = nn.ZeroPad2d(padding)(mask)
        new_image = cv2.equalizeHist(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) * (mask.squeeze().numpy() > 0))
        new_image = torch.tensor(new_image).unsqueeze(0) / 255.0
        new_image = nn.ZeroPad2d(padding)(new_image)
        image = (image + new_image) / 2
        return image * (mask_padding > 0), manual * (mask > 0), mask


if __name__ == "__main__":
    train_dataset = TrainingDataset(
        r"Retinal_Vessel_Segmentation\202410311023_RegtLu\data_example\training"
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataset = TestDataset(
        r"Retinal_Vessel_Segmentation\202410311023_RegtLu\data_example\test"
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for batch_idx, train_data in enumerate(train_loader):
        image1, image2, image3= train_data
        image1 = image1.squeeze().numpy().astype(np.float32)
        image2 = image2.squeeze().numpy().astype(np.float32)
        image3 = image3.squeeze().numpy().astype(np.float32)
        cv2.imshow("Image", image1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Image", image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Image", image3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    exit()
    for batch_idx, test_data in enumerate(test_loader):
        image1, image2, image3, image4 = test_data
        image1 = image1.squeeze().numpy().astype(np.float32)
        image2 = image2.squeeze().numpy().astype(np.float32)
        image3 = image3.squeeze().numpy().astype(np.float32)
        image4 = image4.squeeze().numpy()
        cv2.imshow("Image", image1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Image", image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Image", image3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Image", image4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
