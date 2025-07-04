import torch
from torch import nn

class NINBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class NIN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            NINBlock(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            NINBlock(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            NINBlock(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            NINBlock(384, num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


# model = NIN(num_classes=10)
# X = torch.rand(1, 1, 224, 224)
# for layer in model.net:
#     X = layer(X)
#     print(f"{layer.__class__.__name__:20} output shape: {X.shape}")
