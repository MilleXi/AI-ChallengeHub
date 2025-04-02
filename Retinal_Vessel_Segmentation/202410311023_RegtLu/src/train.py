import torch
import torch.nn as nn
import numpy as np
import unet
import utils
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = unet.UNet(1, 1).to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
train_data = utils.TrainingDataset(
    r"Retinal_Vessel_Segmentation\202410311023_RegtLu\data_example\training"
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)

for i in range(10):
    for batch_idx, data in enumerate(train_loader):
        image = data[0].to(device)
        target = data[1].to(device)
        mask = data[2].squeeze().numpy().astype(np.float32)
        predict = model(image)[:, :, 2:-2, 11:-12]
        predict * torch.tensor(mask > 0, dtype=torch.float32, device=device)
        loss = loss_func(predict, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Batch {batch_idx}: Loss = {loss.item()}")

torch.save(model, os.path.join(os.path.dirname(__file__), "model.pth"))
