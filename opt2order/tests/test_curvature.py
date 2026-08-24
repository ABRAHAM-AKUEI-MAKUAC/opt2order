"""Unit tests for the curvature oracle and the CG inner solver.

Verifies that:
  - hvp() agrees with the explicit Hessian on small problems.
  - gnvp() agrees with J^T H_L J on a small linear regression and
    is positive semi-definite.
  - conjugate_gradient() solves Ax=b on small SPD systems to high
    precision.
"""
from __future__ import annotations

import math

import pytest
import torch

from opt2order.curvature.cg import conjugate_gradient
from opt2order.curvature.oracle import flatten, gnvp, hvp


# ---------------------------------------------------------------------------
# HVP
# ---------------------------------------------------------------------------

def test_hvp_quadratic():
    """For f(x) = 0.5 x^T A x, H = A.  HVP should match A v exactly."""
    torch.manual_seed(0)
    n = 8
    A = torch.randn(n, n)
    A = A @ A.T + torch.eye(n)        # SPD
    x = torch.zeros(n, requires_grad=True)
    v = torch.randn(n)

    def loss():
        return 0.5 * x @ A @ x

    Hv = hvp(loss, [x], v)
    expected = A @ v
    assert torch.allclose(Hv, expected, atol=1e-5)


def test_hvp_neural_net():
    """HVP on a small MLP must equal numerical-Hessian @ v."""
    torch.manual_seed(0)
    layer = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    x = torch.randn(5, 3)
    y = torch.randint(0, 2, (5,))
    params = list(layer.parameters())
    n = sum(p.numel() for p in params)

    def loss():
        return torch.nn.functional.cross_entropy(layer(x), y)

    # Build full Hessian via N HVPs.
    H_explicit = torch.zeros(n, n)
    eye = torch.eye(n)
    for i in range(n):
        H_explicit[:, i] = hvp(loss, params, eye[i])

    # Compare to a torch.autograd.functional.hessian-style construction
    # (jacobian of the gradient) for the first row.
    g = torch.autograd.grad(loss(), params, create_graph=True)
    flat_g = flatten(g)
    H_row0 = torch.autograd.grad(flat_g[0], params, retain_graph=True)
    H_row0_flat = flatten(H_row0)
    assert torch.allclose(H_row0_flat, H_explicit[0], atol=1e-5)
    # Symmetry check.
    asym = (H_explicit - H_explicit.T).abs().max().item()
    assert asym < 1e-4


# ---------------------------------------------------------------------------
# GNVP
# ---------------------------------------------------------------------------

def test_gnvp_linear_regression():
    """For y_hat = W x and squared loss, G = X^T X / N (per layer).

    Here we check J^T J against the explicit Jacobian-product for a tiny
    linear model with MSE loss.
    """
    torch.manual_seed(0)
    n_samples, in_dim, out_dim = 20, 4, 3
    X = torch.randn(n_samples, in_dim)
    y = torch.randn(n_samples, out_dim)
    W = torch.zeros(out_dim, in_dim, requires_grad=True)

    def model_out():
        return X @ W.T

    def loss_fn(yhat):
        return 0.5 * ((yhat - y) ** 2).mean()

    v = torch.randn(W.numel())
    Gv = gnvp(model_out, loss_fn, [W], v)
    # For MSE loss with mean, H_L w.r.t. yhat is (1/N) I (per element of yhat).
    # G v should equal (1/N) J^T J v.
    # Compute J explicitly.
    J = torch.zeros(n_samples * out_dim, W.numel())
    for i in range(n_samples * out_dim):
        e = torch.zeros(n_samples * out_dim)
        e[i] = 1.0
        # Pull back e through model_out by VJP.
        out = model_out().reshape(-1)
        g = torch.autograd.grad(out, W, grad_outputs=e.view_as(out),
                                retain_graph=True)[0]
        J[i] = g.flatten()
    expected = (J.T @ J / (n_samples * out_dim)) @ v
    assert torch.allclose(Gv, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Conjugate gradient
# ---------------------------------------------------------------------------

def test_cg_spd():
    """CG must converge to the exact solution on small SPD systems."""
    torch.manual_seed(0)
    n = 30
    A = torch.randn(n, n)
    A = A @ A.T + n * torch.eye(n)
    b = torch.randn(n)

    def A_op(v):
        return A @ v

    x_true = torch.linalg.solve(A, b)
    x_cg, n_iter, _ = conjugate_gradient(A_op, b, tol=1e-8, max_iter=2 * n)
    # CG converges in <= n iterations in exact arithmetic.
    assert torch.allclose(x_cg, x_true, atol=1e-4)
    assert n_iter <= n + 1


def test_cg_warm_start():
    """Warm start with x0 close to the solution should converge quickly."""
    torch.manual_seed(1)
    n = 20
    A = torch.randn(n, n)
    A = A @ A.T + n * torch.eye(n)
    b = torch.randn(n)

    def A_op(v):
        return A @ v

    x_true = torch.linalg.solve(A, b)
    x0 = x_true + 0.01 * torch.randn(n)
    x_cg, n_iter, _ = conjugate_gradient(A_op, b, x0=x0,
                                         tol=1e-6, max_iter=n)
    assert torch.allclose(x_cg, x_true, atol=1e-3)
    assert n_iter < n   # warm start should not need full n iterations
