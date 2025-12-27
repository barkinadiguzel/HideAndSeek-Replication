import torch.nn as nn

def get_activation(name="relu"):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")
