import torch.nn as nn

def get_normalization(name, num_features):
    name = name.lower()
    if name == "batch":
        return nn.BatchNorm2d(num_features)
    elif name == "layer":
        return nn.LayerNorm([num_features, 1, 1])
    else:
        raise ValueError(f"Unsupported normalization: {name}")
