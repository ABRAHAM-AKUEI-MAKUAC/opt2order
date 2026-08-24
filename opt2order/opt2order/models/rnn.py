"""Two-layer LSTM for the Hochreiter-Schmidhuber addition task.

Section 3.5 / Table 3.1 of the thesis: 128 hidden units per layer,
~270k parameters, MSE loss on a scalar predicted sum.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdditionLSTM(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 128,
                 num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, 2)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)   # (B,)


def build_addition_lstm(**kwargs) -> nn.Module:
    return AdditionLSTM(**kwargs)
