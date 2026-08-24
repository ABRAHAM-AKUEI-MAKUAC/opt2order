"""Small VGG-style CNN for CIFAR-10.

Architecture per Section 3.5 of the thesis:
  three blocks of [Conv 3x3, Conv 3x3, MaxPool 2x2]
  then a two-layer fully connected classifier head.
Channel widths chosen to land near ~1M parameters total.
"""
from __future__ import annotations

import torch.nn as nn


class CifarVGG(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_cifar_vgg(**kwargs) -> nn.Module:
    return CifarVGG(**kwargs)
