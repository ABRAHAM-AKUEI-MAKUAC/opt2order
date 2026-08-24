"""L-BFGS optimizer: Algorithm 3.4 of the thesis.

Implements the two-loop recursion shown in Section 4.1.4 with:
  - bounded history of m pairs (s_k, y_k)
  - curvature-pair filtering (s · y > eps_curv)
  - weak-Wolfe line search (Armijo + curvature)

Note that this is a deterministic L-BFGS suitable for full-batch or
large-batch training; in the stochastic regime it should be combined
with deterministic curvature pairs (§4.7.3 of the thesis).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch import Tensor

from .base import CurvatureOptimizer


class LBFGS(CurvatureOptimizer):
    def __init__(self, params, lr: float = 1.0, history_size: int = 20,
                 max_ls: int = 25, c1: float = 1e-4, c2: float = 0.9,
                 eps_curv: float = 1e-10):
        defaults = dict(lr=lr, history_size=history_size, max_ls=max_ls,
                        c1=c1, c2=c2, eps_curv=eps_curv)
        super().__init__(params, defaults)
        self._s: List[Tensor] = []
        self._y: List[Tensor] = []
        self._rho: List[Tensor] = []
        self._prev_flat_params: Optional[Tensor] = None
        self._prev_flat_grad: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Two-loop recursion
    # ------------------------------------------------------------------
    def _two_loop(self, grad: Tensor) -> Tensor:
        q = grad.clone()
        alphas: List[Tensor] = []
        for s, y, rho in zip(reversed(self._s), reversed(self._y),
                             reversed(self._rho)):
            a = rho * torch.dot(s, q)
            alphas.append(a)
            q = q - a * y
        if self._y:
            s_last, y_last = self._s[-1], self._y[-1]
            yy = torch.dot(y_last, y_last) + 1e-20
            gamma = torch.dot(s_last, y_last) / yy
        else:
            gamma = torch.tensor(1.0, device=grad.device, dtype=grad.dtype)
        r = gamma * q
        for s, y, rho, a in zip(self._s, self._y, self._rho,
                                reversed(alphas)):
            b = rho * torch.dot(y, r)
            r = r + (a - b) * s
        return -r   # search direction d = -B^{-1} g

    # ------------------------------------------------------------------
    # Wolfe line search
    # ------------------------------------------------------------------
    def _line_search(self, closure: Callable, x0: Tensor,
                     g0: Tensor, d: Tensor,
                     loss0: float, max_ls: int,
                     c1: float, c2: float) -> tuple[float, Tensor, float]:
        """Return (alpha, new_grad, new_loss).

        Uses bracketing backtracking with the (weak) Wolfe conditions.
        Falls back to the smallest alpha that yields any decrease.
        """
        gd0 = float(torch.dot(g0, d))
        if gd0 >= 0:
            # Direction is not a descent direction; reset history and
            # fall back to steepest descent.
            d = -g0
            gd0 = float(torch.dot(g0, d))

        alpha = 1.0
        best_alpha, best_loss = 0.0, loss0
        best_grad = g0.clone()

        for _ in range(max_ls):
            self._set_flat_params(x0 + alpha * d)
            with torch.enable_grad():
                loss = closure()
                loss_val = float(loss.item())
            armijo_ok = loss_val <= loss0 + c1 * alpha * gd0

            # Get gradient at trial point.
            new_g = self._flat_grads()
            curvature_ok = float(torch.dot(new_g, d)) >= c2 * gd0

            if loss_val < best_loss:
                best_loss, best_alpha = loss_val, alpha
                best_grad = new_g.clone()

            if armijo_ok and curvature_ok:
                return alpha, new_g, loss_val
            alpha *= 0.5

        # Could not satisfy Wolfe; commit to the best decrease seen.
        self._set_flat_params(x0 + best_alpha * d)
        return best_alpha, best_grad, best_loss

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, closure: Callable):
        group = self.param_groups[0]
        m = group["history_size"]
        max_ls = group["max_ls"]
        c1, c2 = group["c1"], group["c2"]
        eps_curv = group["eps_curv"]

        # 1. Initial loss and gradient at current params.
        with torch.enable_grad():
            loss = closure()
            loss_val = float(loss.item())
        grad = self._flat_grads()
        x0 = self._flat_params()

        # 2. Compute search direction.
        d = self._two_loop(grad)

        # 3. Line search.
        alpha, new_grad, new_loss = self._line_search(
            closure, x0, grad, d, loss_val, max_ls, c1, c2)

        # 4. Update curvature pairs.
        s = alpha * d
        y = new_grad - grad
        sy = float(torch.dot(s, y))
        if sy > eps_curv:
            self._s.append(s.detach())
            self._y.append(y.detach())
            self._rho.append(torch.tensor(1.0 / sy, device=s.device,
                                          dtype=s.dtype))
            if len(self._s) > m:
                self._s.pop(0); self._y.pop(0); self._rho.pop(0)
        # Otherwise reject the pair to keep B positive definite.

        return new_loss
