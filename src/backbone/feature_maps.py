import torch.nn as nn
from src.backbone.cnn_blocks import CNNBlock

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layer1 = CNNBlock(in_channels, 64)
        self.layer2 = CNNBlock(64, 128)
        self.layer3 = CNNBlock(128, 256)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 
