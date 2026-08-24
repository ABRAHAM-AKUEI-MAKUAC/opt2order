"""Adam optimizer: Algorithm 3.2 of the thesis."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from .base import CurvatureOptimizer


class Adam(CurvatureOptimizer):
    def __init__(self, params, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0.0:
                    g = g.add(p.data, alpha=wd)
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return float(loss.item()) if loss is not None else None
