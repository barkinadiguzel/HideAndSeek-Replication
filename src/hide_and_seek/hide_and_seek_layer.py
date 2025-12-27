import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.config import PATCH_SIZE, P_HIDE, MEAN_PIXEL

class HideAndSeek(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, p_hide=P_HIDE, mean_pixel=MEAN_PIXEL):
        super().__init__()
        self.patch_size = patch_size
        self.p_hide = p_hide
        self.mean_pixel = torch.tensor(mean_pixel).view(3,1,1)

    def forward(self, x):
        if not self.training:
            return x
        B, C, H, W = x.shape
        S = self.patch_size
        for i in range(0, H, S):
            for j in range(0, W, S):
                if np.random.rand() < self.p_hide:
                    x[:, :, i:i+S, j:j+S] = self.mean_pixel.to(x.device)
        return x
