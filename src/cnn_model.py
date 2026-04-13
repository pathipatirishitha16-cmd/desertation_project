"""
cnn_model.py
============
ResNet-50 architecture for primary 14-class chest X-ray disease detection.
Pre-trained on ImageNet. Final layer replaced with custom classification head.

Usage:
    from cnn_model import build_resnet50
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

NUM_CLASSES    = 15    # 14 NIH diseases + Hernia
EMBEDDING_DIM  = 1024


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 for multi-label chest X-ray classification.
    Architecture:
        ResNet-50 backbone (pretrained ImageNet)
        → AdaptiveAvgPool2d
        → Linear(2048 → 1024)   ← embedding layer for GAT
        → ReLU + Dropout(0.5)
        → Linear(1024 → 14)
        → Sigmoid (for multi-label)
    """

    def __init__(self,
                 num_classes:   int   = NUM_CLASSES,
                 dropout:       float = 0.5,
                 pretrained:    bool  = True):
        super().__init__()

        # Load pretrained backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove original FC layer – keep up to the global average pool
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # Output: (batch, 2048, 1, 1)

        # Custom classification head
        self.embedding = nn.Sequential(
            nn.Flatten(),                           # (batch, 2048)
            nn.Linear(2048, EMBEDDING_DIM),         # (batch, 1024)
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, num_classes),  # (batch, 14)
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            logits:    (batch, 14)  – sigmoid probabilities
            embedding: (batch, 1024) – for GAT input
        """
        feat      = self.features(x)        # (B, 2048, 1, 1)
        embedding = self.embedding(feat)    # (B, 1024)
        logits    = self.classifier(embedding)  # (B, 14)
        return logits, embedding

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the 1024-dim embedding (used during GAT training)."""
        with torch.no_grad():
            feat      = self.features(x)
            embedding = self.embedding(feat)
        return embedding


def build_resnet50(num_classes: int   = NUM_CLASSES,
                   dropout:     float = 0.5,
                   pretrained:  bool  = True) -> ResNet50Classifier:
    """Factory function to build the ResNet-50 classifier."""
    model = ResNet50Classifier(num_classes=num_classes,
                                dropout=dropout,
                                pretrained=pretrained)
    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"[INFO] ResNet-50 | Total params   : {total_params:,}")
    print(f"[INFO] ResNet-50 | Trainable params: {trainable_params:,}")
    return model