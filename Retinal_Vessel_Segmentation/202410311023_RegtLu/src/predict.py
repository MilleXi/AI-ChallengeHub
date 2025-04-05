import os
import numpy as np
import torch
import cv2
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_data = utils.TestDataset(
    r"Retinal_Vessel_Segmentation\202410311023_RegtLu\data_example\test"
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
model = torch.load(os.path.join(os.path.dirname(__file__), "model.pth"), weights_only=False).to(device)

model.eval()
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        image = data[0].to(device)
        mask = data[1].squeeze().numpy().astype(np.float32)
        image_origin = data[3].squeeze().numpy()
        predict = model(image)[:, :, 2:-2, 11:-12]
        temp = predict.detach().cpu().squeeze().numpy().astype(np.float32)
        temp = temp * mask
        temp = np.concatenate((np.stack([temp]*3, axis=-1)* 255,image_origin), axis=1)
        filename = f'test{batch_idx+1}.png'
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result', filename)
        cv2.imwrite(filepath, temp)
