"""Damped Newton optimizer: Algorithm 3.3 of the thesis.

Materializes the full Hessian and solves (H + lambda I) d = -g at each
iteration.  Suitable only for tiny problems (Rosenbrock, small logistic
regression).  Used as a sanity check on the curvature oracles.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from .base import CurvatureOptimizer
from ..curvature.oracle import hvp


class DampedNewton(CurvatureOptimizer):
    def __init__(self, params, damping: float = 1e-4,
                 ls_max_iter: int = 25, c1: float = 1e-4,
                 backtrack: float = 0.5):
        defaults = dict(damping=damping, ls_max_iter=ls_max_iter, c1=c1)
        super().__init__(params, defaults)
        self.damping = damping
        self.ls_max_iter = ls_max_iter
        self.c1 = c1
        self.backtrack = backtrack

    def _full_hessian(self, closure: Callable[..., Tensor]) -> Tensor:
        """Build H by N HVPs against the standard basis.  N must be small."""
        params = self._params_list()
        n = sum(p.numel() for p in params)
        device = params[0].device
        dtype = params[0].dtype
        H = torch.zeros(n, n, device=device, dtype=dtype)
        eye = torch.eye(n, device=device, dtype=dtype)

        def loss_fn():
            return closure(create_graph=True)

        for i in range(n):
            col = hvp(loss_fn, params, eye[i])
            H[:, i] = col.detach()
        # Symmetrize to remove tiny numerical asymmetry.
        return 0.5 * (H + H.t())

    def step(self, closure: Callable[..., Tensor]):
        params = self._params_list()
        n = sum(p.numel() for p in params)
        if n > 2000:
            raise RuntimeError(
                f"DampedNewton materializes a {n}x{n} Hessian; refusing "
                "to run on a problem this large.  Use HessianFree, L-BFGS, "
                "or K-FAC instead.")

        # 1. Loss + gradient at current params.
        with torch.enable_grad():
            loss = closure(create_graph=False)
            loss_val = float(loss.detach().item())
        grad = self._flat_grads()

        # 2. Build the Hessian.
        H = self._full_hessian(closure)
        H_reg = H + self.damping * torch.eye(
            n, device=H.device, dtype=H.dtype)

        # 3. Solve.
        try:
            d = torch.linalg.solve(H_reg, -grad)
        except RuntimeError:
            d = -grad   # fall back to steepest descent

        # 4. Armijo line search.
        x0 = self._flat_params().clone()
        gd = float(torch.dot(grad, d))
        alpha = 1.0
        new_loss_val = loss_val
        for _ in range(self.ls_max_iter):
            self._set_flat_params(x0 + alpha * d)
            with torch.enable_grad():
                new_loss = closure(create_graph=False)
            new_loss_val = float(new_loss.detach().item())
            if new_loss_val <= loss_val + self.c1 * alpha * gd:
                break
            alpha *= self.backtrack
        else:
            self._set_flat_params(x0)
            new_loss_val = loss_val

        return new_loss_val
