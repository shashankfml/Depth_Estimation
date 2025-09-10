"""
Model architectures for SPAD depth estimation.
Includes UNet, Multi-scale network, and custom architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class UNet(nn.Module):
    """Enhanced U-Net with BatchNorm for depth estimation."""
    
    def __init__(self):
        super().__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(config.INPUT_CHANNELS, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Decoder with bilinear upsampling
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1)
        )
        self.dec3 = conv_block(512, 256)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1)
        )
        self.dec2 = conv_block(256, 128)
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1)
        )
        self.dec1 = conv_block(128, 64)
        
        self.out = nn.Conv2d(64, config.OUTPUT_CHANNELS, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.out(d1))


class CoarseNet(nn.Module):
    """Coarse network for multi-scale depth estimation (Eigen et al.)."""
    
    def __init__(self):
        super(CoarseNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(config.INPUT_CHANNELS, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 15 * 15, 4096),
            nn.ReLU()
        )
        self.fc7 = nn.Linear(4096, 64 * 64)
        self.output_size = (64, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = x.view(-1, 1, *self.output_size)
        return x


class FineNet(nn.Module):
    """Fine network for multi-scale depth estimation (Eigen et al.)."""
    
    def __init__(self):
        super(FineNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(config.INPUT_CHANNELS, 63, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(63 + 1, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.conv4 = nn.Conv2d(64, 1, kernel_size=5, padding=2)

    def forward(self, image, coarse_depth):
        x = self.conv1(image)
        x = torch.cat((x, coarse_depth), dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class MultiScaleDepthNet(nn.Module):
    """Multi-scale depth estimation network (Eigen et al.)."""
    
    def __init__(self):
        super(MultiScaleDepthNet, self).__init__()
        self.coarse = CoarseNet()
        self.fine = FineNet()

    def forward(self, x):
        coarse = self.coarse(x)
        refined = self.fine(x, coarse)
        return coarse, refined


class DepthNetGray(nn.Module):
    """Custom depth estimation network with layer normalization."""
    
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(config.INPUT_CHANNELS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Bottleneck
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Decoder
        self.upconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1)

        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv7 = nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1)

        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv8 = nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1)

        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv9 = nn.Conv2d(32 + 32, 16, kernel_size=3, padding=1)

        # Final prediction
        self.final = nn.Conv2d(16, config.OUTPUT_CHANNELS, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x1 = F.layer_norm(x1, x1.shape[1:])
        x1 = F.relu(x1)

        x2 = F.max_pool2d(x1, 2)
        x2 = self.conv2(x2)
        x2 = F.layer_norm(x2, x2.shape[1:])
        x2 = F.relu(x2)

        x3 = F.max_pool2d(x2, 2)
        x3 = self.conv3(x3)
        x3 = F.layer_norm(x3, x3.shape[1:])
        x3 = F.relu(x3)

        x4 = F.max_pool2d(x3, 2)
        x4 = self.conv4(x4)
        x4 = F.layer_norm(x4, x4.shape[1:])
        x4 = F.relu(x4)

        # Bottleneck
        x5 = F.max_pool2d(x4, 2)
        x5 = self.conv5(x5)
        x5 = F.layer_norm(x5, x5.shape[1:])
        x5 = F.relu(x5)

        # Decoder
        d4 = self.upconv4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.conv6(d4)
        d4 = F.layer_norm(d4, d4.shape[1:])
        d4 = F.relu(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.conv7(d3)
        d3 = F.layer_norm(d3, d3.shape[1:])
        d3 = F.relu(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv8(d2)
        d2 = F.layer_norm(d2, d2.shape[1:])
        d2 = F.relu(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.conv9(d1)
        d1 = F.layer_norm(d1, d1.shape[1:])
        d1 = F.relu(d1)

        out = self.final(d1)
        return out


def get_model(model_name='unet'):
    """
    Get model by name.
    
    Args:
        model_name (str): Name of the model ('unet', 'multiscale', 'depthnet')
        
    Returns:
        nn.Module: The requested model
    """
    models = {
        'unet': UNet,
        'multiscale': MultiScaleDepthNet,
        'depthnet': DepthNetGray
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()
