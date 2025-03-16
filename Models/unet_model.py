import torch

from Models.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.size())
        x2 = self.down1(x1)
        # print(x2.size())
        x3 = self.down2(x2)
        # print(x3.size())
        x4 = self.down3(x3)
        # print(x4.size())
        x5 = self.down4(x4)
        # print(x5.size())
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1))




    import cv2
    from skimage.io import imread
    img_name = '/Users/kongxiangyu/Downloads/Kvasir-SEG/masks/cju0qkwl35piu0993l0dewei2.jpg'
    img = cv2.imread(img_name)
    im = imread(img_name)




    labels = torch.cat([torch.arange(16) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    import numpy as np
    a = np.array([[1.0, 1.0],
                  [0.9, 0.9]])
    b = np.array([[1.1, 1.0],
                  [0.95, 0.91]])
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    re = cos(torch.flatten(a), torch.flatten(b))

    input = torch.randn(size=(16, 3, 352, 352))
    model = UNet(n_channels=3, n_classes=3, bilinear=True)
    output = model(input)
    print()

    net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    print()