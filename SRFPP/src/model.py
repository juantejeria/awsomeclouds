from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int, pretrained: bool = True, dropout: float = 0.4) -> nn.Module:
    """
    CNN via transfer learning:
    - ResNet18 pretrained on ImageNet
    - Replace final FC layer with a head that includes Dropout + hidden layer
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout probability for the classifier head (0 to disable)
    """
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    in_features = model.fc.in_features  # 512 for ResNet18

    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.75),  # Slightly less dropout on second layer
            nn.Linear(256, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all layers except the classifier head (model.fc).
    Used in Phase 1 of gradual fine-tuning to preserve ImageNet features.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_last_block(model: nn.Module) -> None:
    """
    Unfreeze layer4 (last residual block) + classifier head.
    Used in Phase 2 of gradual fine-tuning to adapt high-level features.
    """
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


@torch.inference_mode()
def predict_proba(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """
    batch: (N, C, H, W)
    returns: (N, num_classes) probabilities
    """
    logits = model(batch)
    return torch.softmax(logits, dim=1)


@torch.inference_mode()
def extract_embeddings(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """
    Extract 512-dim feature embeddings from ResNet18's avgpool layer
    (before the classification head).

    These embeddings serve as a "fingerprint" for open-set recognition:
    known individuals cluster near their class centroid, while unknown
    individuals fall far from all centroids.

    Args:
        model: ResNet18-based model (must have conv1..layer4, avgpool)
        batch: (N, C, H, W) input tensor

    Returns:
        (N, 512) L2-normalized embedding vectors
    """
    x = model.conv1(batch)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)          # (N, 512, 1, 1)
    x = x.view(x.size(0), -1)    # (N, 512)
    # L2-normalize so cosine similarity = dot product
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    return x


