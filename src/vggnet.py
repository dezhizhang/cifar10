import torch
import torch.nn as nn


class VGGbase(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # block1
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # block2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # block3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # block4
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 👉 核心：自适应，不用再算尺寸
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)

        # 👉 不管输入多大，这里都会变成 [B, 512, 1, 1]
        out = self.pool(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out


def VGGNet():
    return VGGbase()