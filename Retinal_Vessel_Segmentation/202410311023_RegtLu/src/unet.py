import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_double(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_double = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv_double(x)
    

class downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsampling = nn.Sequential(
        nn.MaxPool2d(2),
        conv_double(in_channels, out_channels))

    def forward(self, x):
        return self.downsampling(x)


class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsampling=nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_double=conv_double(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsampling(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv_double(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1=conv_double(in_channels, 64)
        self.l2=downsampling(64,128)
        self.l3=downsampling(128,256)
        self.l4=downsampling(256,512)
        self.l5=downsampling(512,1024)
        self.l5_ = upsampling(1024, 512)
        self.l4_ = upsampling(512, 256)
        self.l3_ = upsampling(256, 128)
        self.l2_ = upsampling(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        x5 = self.l5(x4)
        x = self.l5_(x5, x4)
        x = self.l4_(x, x3)
        x = self.l3_(x, x2)
        x = self.l2_(x, x1)
        logits = self.out(x)
        return logits