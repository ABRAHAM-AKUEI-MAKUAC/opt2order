"""Base class for all optimizers in opt2order.

Provides parameter management and the flat-vector helpers used by the
second-order methods.  Subclasses override ``step``.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


class CurvatureOptimizer(Optimizer):
    """Base class with flat-parameter helpers.

    All curvature-aware optimizers inherit from this.  First-order
    baselines (SGDM, Adam) also inherit so they share the same step()
    closure protocol.
    """

    def __init__(self, params, defaults):
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # Flat-vector helpers
    # ------------------------------------------------------------------
    def _params_list(self) -> List[Tensor]:
        return [p for g in self.param_groups for p in g["params"]
                if p.requires_grad]

    def _flat_params(self) -> Tensor:
        return torch.cat([p.data.view(-1) for p in self._params_list()])

    def _flat_grads(self) -> Tensor:
        pieces = []
        for p in self._params_list():
            if p.grad is None:
                pieces.append(torch.zeros_like(p).view(-1))
            else:
                pieces.append(p.grad.data.view(-1))
        return torch.cat(pieces)

    def _set_flat_params(self, flat: Tensor) -> None:
        offset = 0
        for p in self._params_list():
            n = p.numel()
            p.data.copy_(flat[offset:offset + n].view_as(p))
            offset += n

    def _add_flat_to_params(self, flat: Tensor, alpha: float = 1.0) -> None:
        offset = 0
        for p in self._params_list():
            n = p.numel()
            p.data.add_(flat[offset:offset + n].view_as(p), alpha=alpha)
            offset += n

    def step(self, closure: Optional[Callable] = None):
        raise NotImplementedError
