"""End-to-end optimizer smoke tests on tiny problems.

Each test runs an optimizer for a small number of iterations on a
problem where every method should make rapid progress, and checks
that the loss decreases substantially.  These are not
performance-comparison tests; they exist to catch interface
regressions and gradient-flow bugs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from opt2order.data.synthetic import logistic_regression_data
from opt2order.optimizers import (Adam, DampedNewton, HessianFree, KFAC,
                                  LBFGS, SGDMomentum)


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    X, y = logistic_regression_data(n_samples=400, n_features=20,
                                    n_classes=3, seed=seed)
    model = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 3))
    return model, X, y


def _initial_loss(model, X, y):
    with torch.no_grad():
        return float(F.cross_entropy(model(X), y).item())


def test_sgdm_decreases_loss():
    model, X, y = _setup()
    opt = SGDMomentum(model.parameters(), lr=0.1, momentum=0.9)
    L0 = _initial_loss(model, X, y)
    for _ in range(50):
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            return loss
        opt.step(closure)
    L1 = _initial_loss(model, X, y)
    assert L1 < 0.5 * L0, f"SGDM did not reduce loss enough: {L0:.3f} -> {L1:.3f}"


def test_adam_decreases_loss():
    model, X, y = _setup()
    opt = Adam(model.parameters(), lr=1e-2)
    L0 = _initial_loss(model, X, y)
    for _ in range(50):
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            return loss
        opt.step(closure)
    L1 = _initial_loss(model, X, y)
    assert L1 < 0.5 * L0


def test_lbfgs_decreases_loss():
    model, X, y = _setup()
    opt = LBFGS(model.parameters(), history_size=10, max_ls=20)
    L0 = _initial_loss(model, X, y)
    for _ in range(20):
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            return loss
        opt.step(closure)
    L1 = _initial_loss(model, X, y)
    assert L1 < 0.3 * L0   # L-BFGS should converge fast on this problem


def test_hessian_free_decreases_loss():
    model, X, y = _setup()
    opt = HessianFree(model.parameters(), damping=1e-2, cg_max_iter=15)
    L0 = _initial_loss(model, X, y)
    for _ in range(15):
        def closure(create_graph=False):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X), y)
            if not create_graph:
                loss.backward()
            return loss
        opt.step(closure)
    L1 = _initial_loss(model, X, y)
    assert L1 < 0.3 * L0


def test_kfac_decreases_loss():
    model, X, y = _setup()
    opt = KFAC(model.parameters(), model=model, lr=0.1,
               damping=1e-2, inv_freq=1)
    L0 = _initial_loss(model, X, y)
    for _ in range(40):
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            return loss
        opt.step(closure)
    L1 = _initial_loss(model, X, y)
    assert L1 < 0.5 * L0


def test_damped_newton_decreases_loss():
    """Newton on a tiny model only."""
    torch.manual_seed(0)
    X, y = logistic_regression_data(n_samples=200, n_features=8,
                                    n_classes=3, seed=0)
    model = nn.Linear(8, 3)
    opt = DampedNewton(model.parameters(), damping=1e-4)
    L0 = _initial_loss(model, X, y)
    for _ in range(8):
        def closure(create_graph=False):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X), y)
            if not create_graph:
                loss.backward()
            return loss
        opt.step(closure)
    L1 = _initial_loss(model, X, y)
    # Newton should crush this convex problem.
    assert L1 < 0.2 * L0
