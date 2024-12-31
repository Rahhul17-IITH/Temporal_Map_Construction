import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from ultralytics import YOLO

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # Handle dimensions
        if x1.dim() < 4:
            x1 = x1.unsqueeze(0)
        if x2.dim() < 4:
            x2 = x2.unsqueeze(0)
        
        # Upsample x1
        x1 = F.interpolate(x1, scale_factor=self.scale_factor, 
                          mode='bilinear', align_corners=True)
        
        # Match dimensions
        if x2.shape[-2:] != x1.shape[-2:]:
            x2 = F.interpolate(x2, size=x1.shape[-2:], 
                             mode='bilinear', align_corners=True)
        
        # Concatenate and process
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# class CamEncode(nn.Module):
#     def __init__(self, C):
#         super(CamEncode, self).__init__()
#         self.C = C

#         self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
#         self.up1 = Up(320+112, self.C)

#     def get_eff_depth(self, x):
#         # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
#         endpoints = dict()

#         # Stem
#         x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
#         prev_x = x

#         # Blocks
#         for idx, block in enumerate(self.trunk._blocks):
#             drop_connect_rate = self.trunk._global_params.drop_connect_rate
#             if drop_connect_rate:
#                 drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
#             x = block(x, drop_connect_rate=drop_connect_rate)
#             if prev_x.size(2) > x.size(2):
#                 endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
#             prev_x = x

#         # Head
#         endpoints['reduction_{}'.format(len(endpoints)+1)] = x
#         x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
#         return x

#     def forward(self, x):
#         return self.get_eff_depth(x)

class CamEncode(nn.Module):
    def __init__(self, C):
        super(CamEncode, self).__init__()
        self.C = C
        
        # Initialize YOLOv8 backbone
        yolo_model = YOLO('yolov8s.pt')
        self.backbone = yolo_model.model.model[:10]
        
        # Channel adaptation layers
        self.adapt_512 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.adapt_256 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(384, self.C, 3, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)
        
        # Extract features through backbone
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            features.append(x)
        
        # Get P3 and P4 features
        p4 = features[-2]  # 512 channels
        p3 = features[-3]  # 256 channels
        
        # Adapt channel dimensions
        p4_adapted = self.adapt_512(p4)  # 512 -> 256
        p3_adapted = self.adapt_256(p3)  # 256 -> 128
        
        # Upsample p4 to match p3 spatial dimensions
        p4_upsampled = F.interpolate(
            p4_adapted, 
            size=p3_adapted.shape[-2:],
            mode='bilinear',
            align_corners=True
        )
        
        # Concatenate features
        combined = torch.cat([p4_upsampled, p3_adapted], dim=1)  # 384 channels
        
        # Final processing
        out = self.fusion(combined)  # 384 -> C channels
        
        # Reshape back to batch format
        _, C, H, W = out.shape
        out = out.view(B, N, C, H, W)
        
        return out

class BevEncode(nn.Module):
    def __init__(self, inC, outC, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=37):
        super(BevEncode, self).__init__()

        # Remove the original Swin Transformer and replace with a more flexible backbone
        self.trunk = nn.Sequential(
            nn.Conv2d(inC, 96, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(inC, 96, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)

        # Keep the rest of the architecture the same
        self.up1 = Up(96 + 384, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

        # Rest of the initialization remains the same
        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(96 + 384, 256, scale_factor=4)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )

        self.direction_pred = direction_pred
        if direction_pred:
            self.up1_direction = Up(96 + 384, 256, scale_factor=4)
            self.up2_direction = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, direction_dim, kernel_size=1, padding=0),
