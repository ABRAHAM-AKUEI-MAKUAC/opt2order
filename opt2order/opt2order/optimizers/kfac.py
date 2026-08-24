"""K-FAC optimizer: Algorithm 3.6 of the thesis.

Implements K-FAC for ``nn.Linear`` and ``nn.Conv2d`` layers using:
  - forward / full-backward hooks to capture activations a^l and
    pre-activation gradients g^l (Section 4.1.5).
  - exponential moving averages with decay 0.95 of the Kronecker
    factors A = E[a a^T] (with appended ones for bias) and S = E[g g^T].
  - periodic inversion every ``inv_freq`` steps (T_I in the thesis).
  - per-layer update  d^l = (S + sqrt(lambda) I)^{-1} grad_W (A + sqrt(lambda) I)^{-1}.
  - parameters not associated with a tracked layer (e.g. BatchNorm
    weights) fall back to a plain SGD update with the same learning rate.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import CurvatureOptimizer


class _KFACModule:
    """Tracks Kronecker factors for one Linear or Conv2d layer."""

    def __init__(self, layer: nn.Module, decay: float = 0.95):
        self.layer = layer
        self.decay = decay
        self.A: Optional[Tensor] = None       # input covariance
        self.S: Optional[Tensor] = None       # gradient covariance
        self._handle_f = layer.register_forward_pre_hook(self._save_input)
        self._handle_b = layer.register_full_backward_hook(self._save_grad)

    def remove_hooks(self):
        self._handle_f.remove()
        self._handle_b.remove()

    # ----------------------------------------------------------- input
    def _save_input(self, module: nn.Module, inputs):
        a = inputs[0].detach()
        if isinstance(module, nn.Conv2d):
            # Unfold to (B*L, C_in*kH*kW) where L = #spatial output positions
            a = F.unfold(a, kernel_size=module.kernel_size,
                         dilation=module.dilation,
                         padding=module.padding,
                         stride=module.stride)
            # a: (B, C_in*kH*kW, L)  ->  (B*L, C_in*kH*kW)
            a = a.transpose(1, 2).reshape(-1, a.size(1))
        elif isinstance(module, nn.Linear):
            a = a.reshape(-1, a.size(-1))   # collapse leading dims
        else:
            return
        if module.bias is not None:
            ones = torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=1)
        new_A = (a.t() @ a) / a.size(0)
        if self.A is None:
            self.A = new_A.detach()
        else:
            self.A = self.decay * self.A + (1 - self.decay) * new_A.detach()

    # ----------------------------------------------------------- grad
    def _save_grad(self, module: nn.Module, grad_input, grad_output):
        g = grad_output[0].detach()
        if isinstance(module, nn.Conv2d):
            # g shape: (B, C_out, H, W) -> (B*H*W, C_out)
            g = g.permute(0, 2, 3, 1).reshape(-1, g.size(1))
            # When a batched loss is averaged across spatial locations,
            # the per-spatial-position gradients are 1/(H*W) too small
            # for a Fisher estimate; multiply by the loss-batch scale
            # (B*H*W) to recover per-sample contributions.
            g = g * g.size(0)
        elif isinstance(module, nn.Linear):
            g = g.reshape(-1, g.size(-1))
            g = g * g.size(0)
        else:
            return
        new_S = (g.t() @ g) / g.size(0)
        if self.S is None:
            self.S = new_S.detach()
        else:
            self.S = self.decay * self.S + (1 - self.decay) * new_S.detach()


class KFAC(CurvatureOptimizer):
    """Kronecker-Factored Approximate Curvature."""

    def __init__(self, params, model: Optional[nn.Module] = None,
                 lr: float = 1e-2,
                 damping: float = 1e-2,
                 inv_freq: int = 10,
                 decay: float = 0.95,
                 weight_decay: float = 0.0):
        if model is None:
            raise ValueError("KFAC requires the model so it can install hooks. "
                             "Pass model=... in build_optimizer kwargs.")
        defaults = dict(lr=lr, damping=damping, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = lr
        self.damping = damping
        self.inv_freq = inv_freq
        self.decay = decay
        self.weight_decay = weight_decay
        self.model = model
        self.modules: List[_KFACModule] = []
        self._cached_inv: Dict[int, tuple[Tensor, Tensor]] = {}
        self._step_count = 0
        self._tracked_params = set()
        for layer in model.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                self.modules.append(_KFACModule(layer, decay=decay))
                self._tracked_params.add(id(layer.weight))
                if layer.bias is not None:
                    self._tracked_params.add(id(layer.bias))

    # ------------------------------------------------------------------
    def _maybe_update_inverses(self) -> None:
        if self._step_count % self.inv_freq != 0:
            return
        d = self.damping ** 0.5
        for m in self.modules:
            if m.A is None or m.S is None:
                continue
            A_reg = m.A + d * torch.eye(m.A.size(0), device=m.A.device,
                                        dtype=m.A.dtype)
            S_reg = m.S + d * torch.eye(m.S.size(0), device=m.S.device,
                                        dtype=m.S.dtype)
            try:
                A_inv = torch.linalg.inv(A_reg)
                S_inv = torch.linalg.inv(S_reg)
            except RuntimeError:
                # Fall back to pseudo-inverse on rare singular factors.
                A_inv = torch.linalg.pinv(A_reg)
                S_inv = torch.linalg.pinv(S_reg)
            self._cached_inv[id(m.layer)] = (A_inv, S_inv)

    # ------------------------------------------------------------------
    def step(self, closure: Optional[Callable] = None):
        loss_val = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                loss_val = float(loss.item())

        self._maybe_update_inverses()

        # ---- per-layer K-FAC update for tracked parameters ---------
        for m in self.modules:
            if id(m.layer) not in self._cached_inv:
                continue
            A_inv, S_inv = self._cached_inv[id(m.layer)]
            layer = m.layer
            grad_W = layer.weight.grad
            if grad_W is None:
                continue
            if isinstance(layer, nn.Conv2d):
                grad_W_2d = grad_W.view(grad_W.size(0), -1)
            else:
                grad_W_2d = grad_W
            if layer.bias is not None and layer.bias.grad is not None:
                grad_b = layer.bias.grad.view(-1, 1)
                grad_full = torch.cat([grad_W_2d, grad_b], dim=1)
            else:
                grad_full = grad_W_2d

            if self.weight_decay != 0.0:
                W_2d = (layer.weight.view(layer.weight.size(0), -1)
                        if isinstance(layer, nn.Conv2d) else layer.weight)
                if layer.bias is not None:
                    W_full = torch.cat([W_2d, layer.bias.view(-1, 1)], dim=1)
                else:
                    W_full = W_2d
                grad_full = grad_full + self.weight_decay * W_full

            # Natural gradient direction in matrix form.
            d_full = S_inv @ grad_full @ A_inv

            # Slice back into weight and bias.
            if layer.bias is not None:
                d_W_2d = d_full[:, :-1]
                d_b = d_full[:, -1]
                with torch.no_grad():
                    if isinstance(layer, nn.Conv2d):
                        layer.weight.data.add_(d_W_2d.view_as(layer.weight),
                                               alpha=-self.lr)
                    else:
                        layer.weight.data.add_(d_W_2d, alpha=-self.lr)
                    layer.bias.data.add_(d_b, alpha=-self.lr)
            else:
                with torch.no_grad():
                    if isinstance(layer, nn.Conv2d):
                        layer.weight.data.add_(d_full.view_as(layer.weight),
                                               alpha=-self.lr)
                    else:
                        layer.weight.data.add_(d_full, alpha=-self.lr)

        # ---- SGD fallback for non-K-FAC parameters (e.g. BatchNorm) ----
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if id(p) in self._tracked_params:
                        continue
                    if p.grad is None:
                        continue
                    g = p.grad.data
                    if self.weight_decay != 0.0:
                        g = g.add(p.data, alpha=self.weight_decay)
                    p.data.add_(g, alpha=-self.lr)

        self._step_count += 1
        return loss_val

    def __del__(self):
        # Best-effort hook cleanup.
        try:
            for m in self.modules:
                m.remove_hooks()
        except Exception:
            pass
