"""
DeepLabV3+ with MobileNetV3-Large Segmentation Model
Author: Amr Gamal
Organization: EME Internship Portfolio

Builds a DeepLabV3+ semantic segmentation model with a MobileNetV3-Large backbone
and Atrous Spatial Pyramid Pooling (ASPP) for high-accuracy foliage segmentation.
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights


def create_deeplabv3_model(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    """
    Instantiates DeepLabV3+ with MobileNetV3-Large backbone.
    Replaces the default 21-class classifier with a custom segmentation head
    for binary foliage extraction (background vs leaf).
    """
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    # Replace final classifier layer
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    return model


class CombinedDiceBCEWithLogitsLoss(nn.Module):
    """
    Hybrid loss function combining Binary Cross-Entropy (BCE) with Dice Loss
    to optimize both pixel-level cross-entropy and area-level boundary overlap.
    """
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        # Sigmoid activation for Dice computation
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
