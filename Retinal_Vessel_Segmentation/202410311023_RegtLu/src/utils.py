import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from PIL import Image

def resize_image(image, size=(572, 572)):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

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
        image = resize_image(image)
        mask = resize_image(mask)
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
        manual = np.array(Image.open(manual_file).convert("L"))
        mask = np.array(Image.open(mask_file).convert("L"))
        image = resize_image(image)
        manual = resize_image(manual)
        mask = resize_image(mask)
        return image * np.expand_dims(mask > 0, axis=-1), manual * np.expand_dims(mask > 0, axis=-1)

if __name__ == "__main__":
    train_dataset = TrainingDataset(r'Retinal_Vessel_Segmentation\202410311023_RegtLu\data_example\training')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataset = TestDataset(r'Retinal_Vessel_Segmentation\202410311023_RegtLu\data_example\test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for batch_idx, train_data in enumerate(train_loader):
        image1,image2 = train_data
        image1=image1.squeeze().numpy().astype(np.uint8)
        image2=image2.squeeze().numpy().astype(np.uint8)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        mean = np.mean(image1)
        image1 = image1 - mean
        image1 =np.clip(image1, 0, 255)
        image1 = image1.astype(np.uint8)
        image1=cv2.Canny(image1, 25, 50,apertureSize=3,L2gradient=True)
        cv2.imshow('Image', image1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('Image', image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
    for batch_idx, test_data in enumerate(test_loader):
        image=test_data.squeeze().numpy().astype(np.uint8)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break