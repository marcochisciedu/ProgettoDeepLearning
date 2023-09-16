import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_classes, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.leakyReLU(self.conv1(x))
        x = self.leakyReLU(self.conv2(x))
        x = self.leakyReLU(self.conv3(x))
        x = self.leakyReLU(self.conv4(x))
        x = self.conv5(x)
        return x
