import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
import cv2
from PIL import Image

class TestDataset(Dataset):
    def __init__(self, data_path):
        self.image_paths = os.path.join(data_path, 'images')
        self.mask_paths = os.path.join(data_path, 'mask')
        self.file_list = sorted([f for f in os.listdir(self.image_paths) if f.endswith('.tif')])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_file = os.path.join(self.image_paths, self.file_list[idx])
        mask_file = os.path.join(self.mask_paths, self.file_list[idx].replace('.tif', '_mask.gif'))
        image = cv2.imread(image_file)
        mask = np.array(Image.open(mask_file).convert("L"))
        return image * np.expand_dims(mask > 0, axis=-1)


class TrainingDataset(Dataset):
    def __init__(self, data_path):
        self.image_paths = os.path.join(data_path, 'images')
        self.mask_paths = os.path.join(data_path, 'mask')
        self.manual = os.path.join(data_path, '1st_manual')
        self.file_list = sorted([f for f in os.listdir(self.image_paths) if f.endswith('.tif')])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_name = self.file_list[idx].replace('.tif', '')
        image_file = os.path.join(self.image_paths, f'{base_name}.tif')
        mask_file = os.path.join(self.mask_paths, f'{base_name}_mask.gif')
        manual_file = os.path.join(self.manual, f'{base_name.strip('_training')}_manual1.gif')
        image = cv2.imread(image_file)
        manual = np.array(Image.open(manual_file).convert("RGB"))
        mask = np.array(Image.open(mask_file).convert("L"))
        return image * np.expand_dims(mask > 0, axis=-1), manual * np.expand_dims(mask > 0, axis=-1)


if __name__ == "__main__":
    train_dataset = TrainingDataset(r'Retinal_Vessel_Segmentation\202410311023_RegtLu\data_example\training')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataset = TestDataset(r'Retinal_Vessel_Segmentation\202410311023_RegtLu\data_example\test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for batch_idx, train_data in enumerate(train_loader):
        image1,image2 = train_data
        cv2.imshow('Image', image1.squeeze().numpy().astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('Image', image2.squeeze().numpy().astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
    for batch_idx, test_data in enumerate(test_loader):
        image=test_data.squeeze()
        cv2.imshow('Image', image.squeeze().numpy().astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break