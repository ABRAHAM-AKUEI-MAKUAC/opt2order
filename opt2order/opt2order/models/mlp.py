"""MLP for MNIST.

Architecture from Table 3.1 of the thesis: 784 -> 512 -> 256 -> 128 -> 10
with ReLU activations, totalling roughly 530,000 parameters.
"""
from __future__ import annotations

import torch.nn as nn


class MnistMLP(nn.Module):
    def __init__(self, in_dim: int = 784, hidden=(512, 256, 128),
                 out_dim: int = 10):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def build_mnist_mlp(**kwargs) -> nn.Module:
    return MnistMLP(**kwargs)
