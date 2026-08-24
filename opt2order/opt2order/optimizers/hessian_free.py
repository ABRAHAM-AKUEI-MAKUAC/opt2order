"""Hessian-Free truncated-Newton optimizer.

Implements Algorithm 3.5 / Listing A.1 of the thesis:
  - Newton system solved by CG using Hessian-vector products from
    double backpropagation (see opt2order.curvature.oracle.hvp).
  - Tikhonov damping that is updated at each outer iteration via the
    Levenberg-Marquardt rule (ratio of actual to predicted decrease).
  - Backtracking line search.
  - CG warm-started from 0.95 * previous direction.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from .base import CurvatureOptimizer
from ..curvature.cg import conjugate_gradient


class HessianFree(CurvatureOptimizer):
    def __init__(self, params,
                 damping: float = 1e-2,
                 cg_max_iter: int = 20,
                 cg_tol: float = 1e-3,
                 ls_max_iter: int = 10,
                 ls_armijo: float = 1e-4,
                 lm_decrease: float = 1.5,
                 lm_increase: float = 1.5,
                 use_gn: bool = False):
        defaults = dict(damping=damping, cg_max_iter=cg_max_iter,
                        cg_tol=cg_tol, ls_max_iter=ls_max_iter,
                        ls_armijo=ls_armijo)
        super().__init__(params, defaults)
        self.damping = damping
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.ls_max_iter = ls_max_iter
        self.ls_armijo = ls_armijo
        self.lm_decrease = lm_decrease
        self.lm_increase = lm_increase
        self.use_gn = use_gn   # if True, use Gauss-Newton matrix instead of H
        self._prev_direction: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, closure: Callable[..., Tensor]):
        """``closure`` must accept create_graph=False/True kwargs.

        It zeros gradients, runs forward + backward, and returns the loss
        with a fresh graph.  When create_graph=True is requested, the
        closure must build the graph in such a way that the loss can be
        differentiated again (i.e. don't call .backward(), just return
        the loss tensor with grad-tracking).
        """
        params = self._params_list()

        # 1. Loss + gradient with graph retained for HVP.
        with torch.enable_grad():
            loss = closure(create_graph=True)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_g = torch.cat([g.contiguous().view(-1) for g in grads])
        loss_val = float(loss.detach().item())

        # 2. Operator A v = H v + lambda v.
        damping = self.damping

        def A_op(v: Tensor) -> Tensor:
            inner = torch.dot(flat_g, v)
            hv = torch.autograd.grad(inner, params, retain_graph=True,
                                     allow_unused=True)
            pieces = []
            for h, p in zip(hv, params):
                pieces.append(torch.zeros_like(p).view(-1)
                              if h is None else h.contiguous().view(-1))
            Hv = torch.cat(pieces)
            return Hv + damping * v

        # 3. CG for d in A d = -g.
        x0 = (0.95 * self._prev_direction
              if self._prev_direction is not None else None)
        d, n_cg, _ = conjugate_gradient(
            A_op, -flat_g.detach(), x0=x0,
            tol=self.cg_tol, max_iter=self.cg_max_iter,
        )
        self._prev_direction = d.clone()

        # 4. Predicted decrease from the quadratic model:
        #      m(d) - m(0) = g^T d + 0.5 d^T (H d).
        # Recompute H d = A d - lambda d.
        Hd = A_op(d) - damping * d
        gd = float(torch.dot(flat_g.detach(), d))
        dHd = float(torch.dot(d, Hd))
        predicted_decrease = -(gd + 0.5 * dHd)

        # 5. Backtracking line search.
        old_params = self._flat_params().clone()
        alpha = 1.0
        new_loss_val = loss_val
        for _ in range(self.ls_max_iter):
            self._set_flat_params(old_params + alpha * d)
            with torch.no_grad():
                with torch.enable_grad():
                    new_loss = closure(create_graph=False)
            new_loss_val = float(new_loss.detach().item())
            if new_loss_val <= loss_val + self.ls_armijo * alpha * gd:
                break
            alpha *= 0.5
        else:
            # No decrease found; revert.
            self._set_flat_params(old_params)
            new_loss_val = loss_val
            alpha = 0.0

        # 6. Levenberg-Marquardt damping update.
        actual = loss_val - new_loss_val
        if predicted_decrease > 1e-12:
            ratio = actual / predicted_decrease
            if ratio > 0.75:
                self.damping = max(self.damping / self.lm_decrease, 1e-8)
            elif ratio < 0.25:
                self.damping = min(self.damping * self.lm_increase, 1e3)

        return new_loss_val
