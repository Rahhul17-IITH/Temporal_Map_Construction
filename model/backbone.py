import torch
from torch import nn
from ultralytics import YOLO

class YOLOv8Backbone(nn.Module):
    def __init__(self, model_size='s'):
        super().__init__()
        # Load pretrained YOLOv8 model
        self.model = YOLO(f'yolov8{model_size}.pt')
        # Extract backbone layers
        self.backbone = self.model.model.model[:10]  # First 10 layers form the backbone
        
    def forward(self, x):
        features = []
        for layer in self.backbone:
            x = layer(x)
            features.append(x)
        return features