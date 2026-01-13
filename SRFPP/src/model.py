from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Simple CNN approach via transfer learning:
    - ResNet18 pretrained on ImageNet
    - Replace final FC layer for num_classes
    """
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.inference_mode()
def predict_proba(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """
    batch: (N, C, H, W)
    returns: (N, num_classes) probabilities
    """
    logits = model(batch)
    return torch.softmax(logits, dim=1)


