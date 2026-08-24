"""SGD with momentum: Algorithm 3.1 of the thesis.

This is implemented in opt2order rather than reusing torch.optim.SGD so
that it shares the closure protocol of the second-order optimizers and
appears in the optimizer registry.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from .base import CurvatureOptimizer


class SGDMomentum(CurvatureOptimizer):
    def __init__(self, params, lr: float = 0.05, momentum: float = 0.9,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0.0:
                    g = g.add(p.data, alpha=wd)
                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data)
                v = state["v"]
                v.mul_(mom).add_(g)
                p.data.add_(v, alpha=-lr)
        return float(loss.item()) if loss is not None else None
