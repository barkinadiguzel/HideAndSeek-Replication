import torch
import numpy as np

def generate_cam(feature_maps, classifier_weights, class_idx):
    weights = classifier_weights[class_idx]  
    cam = (weights.view(1,-1,1,1) * feature_maps).sum(dim=1)  
    cam = F.relu(cam)
    cam = cam / cam.max()  
    return cam.detach().cpu().numpy()
