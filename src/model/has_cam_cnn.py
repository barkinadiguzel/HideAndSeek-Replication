import torch
import torch.nn as nn
from src.layers.conv_block import ConvBlock
from src.layers.pooling import GlobalAveragePooling

class HAS_CAM_CNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )
        self.gap = GlobalAveragePooling()
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        gap_out = self.gap(features)
        out = self.classifier(gap_out)
        return out, features
