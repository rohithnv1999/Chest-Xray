"""
DenseNet-121 Architecture with Attention Mechanisms
Implements Squeeze-and-Excitation and Spatial Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation attention module"""
    
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module with 7x7 convolution"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class AttentionDenseNet121(nn.Module):
    """DenseNet-121 with integrated attention mechanisms"""
    
    def __init__(self, num_classes=14, pretrained=True, dropout_rate=0.4):
        super(AttentionDenseNet121, self).__init__()
        
        self.densenet = models.densenet121(pretrained=pretrained)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        
        self.se1 = SqueezeExcitation(256)
        self.se2 = SqueezeExcitation(512)
        self.se3 = SqueezeExcitation(1024)
        self.se4 = SqueezeExcitation(1024)
        
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        features = self.densenet.features.conv0(x)
        features = self.densenet.features.norm0(features)
        features = self.densenet.features.relu0(features)
        features = self.densenet.features.pool0(features)
        
        features = self.densenet.features.denseblock1(features)
        features = self.se1(features)
        features = self.densenet.features.transition1(features)
        
        features = self.densenet.features.denseblock2(features)
        features = self.se2(features)
        features = self.densenet.features.transition2(features)
        
        features = self.densenet.features.denseblock3(features)
        features = self.se3(features)
        features = self.densenet.features.transition3(features)
        
        features = self.densenet.features.denseblock4(features)
        features = self.se4(features)
        features = self.densenet.features.norm5(features)
        
        attention_features = self.spatial_attention(features)
        pooled = self.global_pool(attention_features)
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.classifier(pooled)
        
        if return_features:
            return logits, attention_features
        else:
            return logits
    
    def get_attention_maps(self, x):
        with torch.no_grad():
            _, features = self.forward(x, return_features=True)
            attention = torch.mean(features, dim=1, keepdim=True)
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            return attention


def create_model(config, pretrained=True):
    """Factory function to create the model"""
    model = AttentionDenseNet121(
        num_classes=config.NUM_CLASSES,
        pretrained=pretrained,
        dropout_rate=config.DROPOUT_RATE
    )
    
    model = model.to(config.DEVICE)
    
    if config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    return model


def count_parameters(model):
    """Count trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {total_params - trainable_params:,}")
    
    return trainable_params


if __name__ == "__main__":
    from config import Config
    config = Config()
    model = create_model(config, pretrained=True)
    count_parameters(model)
    dummy_input = torch.randn(2, 3, 512, 512).to(config.DEVICE)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    output, features = model(dummy_input, return_features=True)
    print(f"Features shape: {features.shape}")
    attention = model.get_attention_maps(dummy_input)
    print(f"Attention map shape: {attention.shape}")
    print("\n✓ Model test passed!")