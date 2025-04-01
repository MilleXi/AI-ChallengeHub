import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unet
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = unet.UNet(1,1)
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


train_data = utils.TrainingDataset(r'Retinal_Vessel_Segmentation\202410311023_RegtLu\data_example\training')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)

for i in range(5):
    for batch_idx, data in enumerate(train_loader):
        image = data[0].to('cpu')
        target = data[1].to('cpu')
        mask = data[2].squeeze().numpy().astype(np.float32)
        image_1 = image[:, :, :292, :292]  # 左上
        image_2 = image[:, :, :292, 292:]  # 右上
        image_3 = image[:, :, 292:, :292]  # 左下
        image_4 = image[:, :, 292:, 292:]  # 右下
        predict_1 = model(image_1)
        predict_2 = model(image_2)
        predict_3 = model(image_3)
        predict_4 = model(image_4)
        batch_size, channels, _, _ = predict_1.shape
        predict = torch.zeros((batch_size, channels, 584, 565), device=device)
        predict[:, :, 0:292, 0:292] = predict_1
        predict[:, :, 0:292, 273:565] = predict_2
        predict[:, :, 292:584, 0:292] = predict_3
        predict[:, :, 292:584, 273:565] = predict_4
        predict[:, :, 0:292, 273:292] = (predict_1[:, :, 0:292, 273:292] + predict_2[:, :, 0:292, 0:19]) / 2
        predict[:, :, 292:584, 273:292] = (predict_3[:, :, 0:292, 273:292] + predict_4[:, :, 0:292, 0:19]) / 2
        predict = predict.to('cpu') 
        predict * torch.tensor(mask > 0, dtype=torch.float32, device='cpu')
        loss = loss_func(predict, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Batch {batch_idx}: Loss = {loss.item()}")