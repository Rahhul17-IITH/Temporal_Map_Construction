import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet50


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class ONNXSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CamEncode(nn.Module):
    def __init__(self, C):
        super(CamEncode, self).__init__()
        self.C = C
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320+112, self.C)
        
        # Replace Swish implementations for ONNX compatibility
        self.trunk._swish = ONNXSwish()
        for block in self.trunk._blocks:
            if hasattr(block, '_swish'):
                block._swish = ONNXSwish()

    def replace_swish_for_onnx(self):
        """Replace all Swish implementations with ONNX-compatible version"""
        self.trunk._swish = ONNXSwish()
        for block in self.trunk._blocks:
            if hasattr(block, '_swish'):
                block._swish = ONNXSwish()

    def get_eff_depth(self, x):
        endpoints = dict()
        
        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        return self.get_eff_depth(x)

def prepare_model_for_onnx_export(model):
    """Prepare model for ONNX export by replacing all Swish implementations"""
    def replace_swish(module):
        for name, child in module.named_children():
            if isinstance(child, CamEncode):
                child.replace_swish_for_onnx()
            else:
                replace_swish(child)
    
    replace_swish(model)
    return model



class BevEncode(nn.Module):
    def __init__(self, inC, outC, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=37):
        super(BevEncode, self).__init__()
        trunk = resnet50(pretrained=False)
        
        # Update initial layers to adapt input channels
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        # Extract layers from ResNet-50
        self.layer1 = trunk.layer1  # Output: 256 channels
        self.layer2 = trunk.layer2  # Output: 512 channels
        self.layer3 = trunk.layer3  # Output: 1024 channels

        # Adjust Up layers for ResNet-50's feature dimensions
        self.up1 = Up(256 + 1024, 512, scale_factor=4)  # Combine layer3 and layer1 features
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1)
        )

        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(256 + 1024, 512, scale_factor=4)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1)
            )

        self.direction_pred = direction_pred
        if direction_pred:
            self.up1_direction = Up(256 + 1024, 512, scale_factor=4)
            self.up2_direction = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, direction_dim, kernel_size=1)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # Features from layer1 (256 channels)
        x = self.layer2(x1)  # Features from layer2 (512 channels)
        x2 = self.layer3(x)  # Features from layer3 (1024 channels)

        x = self.up1(x2, x1)  # Combine layer3 and layer1 features
        x_output = self.up2(x)

        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1)
            x_embedded = self.up2_embedded(x_embedded)
        else:
            x_embedded = None

        if self.direction_pred:
            x_direction = self.up1_direction(x2, x1)
            x_direction = self.up2_direction(x_direction)
        else:
            x_direction = None

        return x_output, x_embedded, x_direction