import torch
import torch.nn as nn
import numpy as np
import unet
import utils
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = unet.UNet(1, 1).to(device)
loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_data = utils.TrainingDataset(
    r"Retinal_Vessel_Segmentation\202410311023_RegtLu\data_example\training"
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)

def train():
    best_loss = float("inf")
    for batch_idx, data in enumerate(train_loader):
        image = data[0].to(device)
        target = data[1].to(device)
        mask = data[2].squeeze().numpy().astype(np.float32)
        predict = model(image)[:, :, 2:-2, 11:-12]
        predict * torch.tensor(mask > 0, dtype=torch.float32, device=device)
        loss = loss_func(predict, target)
        if loss < best_loss:
            best_loss = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Best Loss = {best_loss}")

for i in range(20):
    train()

torch.save(model, os.path.join(os.path.dirname(__file__), "model.pth"))
