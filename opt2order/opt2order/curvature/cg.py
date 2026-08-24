"""Conjugate gradient inner solver for Hessian-free optimization.

Implements the CG variant from Section 4.1.3 of the thesis, with:
  - Tikhonov damping (caller-supplied, since CG operator already includes it)
  - early termination on relative residual
  - best-iterate fallback if iteration is truncated

The operator ``A_op`` should already include any damping the caller wants
(this is how Hessian-free uses it: A_op(v) = G v + lambda v).
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor


def conjugate_gradient(
    A_op: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Tensor | None = None,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> Tuple[Tensor, int, float]:
    """Solve A x = b with conjugate gradients.

    Parameters
    ----------
    A_op
        Function v -> A v.  Must be linear and (in exact arithmetic)
        symmetric positive definite for CG to converge.
    b
        Right-hand side, flat 1-D tensor.
    x0
        Optional warm start (e.g., previous CG solution scaled by 0.95).
    tol
        Stop when ||residual|| / ||b|| < tol.
    max_iter
        Maximum CG iterations.

    Returns
    -------
    (x, iters, residual_norm)
        Best iterate found, number of iterations executed, and the
        residual norm at that iterate.
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    Ax = A_op(x) if x0 is not None else torch.zeros_like(b)
    r = b - Ax
    p = r.clone()
    rs_old = torch.dot(r, r)

    b_norm = b.norm().clamp(min=1e-30)
    best_x = x.clone()
    best_rnorm = rs_old.sqrt().item()

    for i in range(max_iter):
        Ap = A_op(p)
        denom = torch.dot(p, Ap)
        if not torch.isfinite(denom) or denom.abs() < 1e-20:
            # Indefinite or zero curvature direction -- bail with best so far.
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        rnorm = rs_new.sqrt().item()
        if rnorm < best_rnorm:
            best_rnorm = rnorm
            best_x = x.clone()
        if rnorm / b_norm.item() < tol:
            return x, i + 1, rnorm
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return best_x, max_iter, best_rnorm
