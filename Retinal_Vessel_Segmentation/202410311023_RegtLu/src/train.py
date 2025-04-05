import torch
import models
import utils
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.UNet(1, 1).to(device)
loss_func = models.DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_data = utils.TrainingDataset(
    r"Retinal_Vessel_Segmentation\202410311023_RegtLu\data_example\training"
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=False)

def train() -> float:
    best_loss = float("inf")
    for batch_idx, data in enumerate(train_loader):
        image = data[0].to(device)
        target = data[1].to(device)
        mask = data[2].to(device)
        predict = model(image)[:, :, 2:-2, 11:-12]
        loss = loss_func(predict, target, mask)
        if loss < best_loss:
            best_loss = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return best_loss.item()

for i in range(50):
    best_loss = train()
    print(f"Epoch {i+1}: Loss = {best_loss}")

torch.save(model, os.path.join(os.path.dirname(__file__), "model.pth"))
