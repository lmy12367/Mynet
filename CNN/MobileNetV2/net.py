import torch
from torch import nn as nn

def _makedivisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)

    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConBNReLu(nn.Sequential):
    def __init__(self, in_channel, out_channel,
                 kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConBNReLu, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size,
                     stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6()
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            layers.append(ConBNReLu(in_channel, hidden_channel, kernel_size=1))
        
        layers.extend([
            ConBNReLu(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_class=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _makedivisible(32 * alpha, round_nearest)
        last_channel = _makedivisible(1280 * alpha, round_nearest)

        invert_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        features.append(ConBNReLu(3, input_channel, stride=2))
        
        for t, c, n, s in invert_residual_setting:
            output_channel = _makedivisible(c * alpha, round_nearest)

            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(
                    input_channel,
                    output_channel,
                    stride,
                    expand_ratio=t
                ))
                input_channel = output_channel
        
        features.append(ConBNReLu(input_channel, last_channel, 1))
        
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_class)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
