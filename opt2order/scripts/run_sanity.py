"""Sanity checks: Rosenbrock function and logistic regression.

Reproduces Section 4.2 of the thesis.  Runs multiple optimizers on
problems for which the answer is known, then prints a tidy summary.

Usage:
    python -m scripts.run_sanity --problem rosenbrock --dim 2
    python -m scripts.run_sanity --problem logistic
    python -m scripts.run_sanity --problem all
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opt2order.data.synthetic import logistic_regression_data
from opt2order.optimizers import (Adam, DampedNewton, HessianFree, LBFGS,
                                  SGDMomentum, KFAC)


# ---------------------------------------------------------------------------
# Rosenbrock
# ---------------------------------------------------------------------------

def rosenbrock(x: torch.Tensor, a: float = 1.0, b: float = 100.0) -> torch.Tensor:
    """Differentiable Rosenbrock; works for any dim >= 2."""
    return ((a - x[:-1]) ** 2 + b * (x[1:] - x[:-1] ** 2) ** 2).sum()


def run_rosenbrock(dim: int = 2, max_iters: int = 5000,
                   tol: float = 1e-8) -> List[Dict]:
    starts = {2: torch.tensor([-1.2, 1.0]),
              10: torch.cat([torch.tensor([-1.2, 1.0]),
                             torch.zeros(8)])}
    x0 = starts.get(dim, torch.full((dim,), -0.5))
    results = []

    for opt_name in ["newton", "lbfgs", "hf", "sgdm", "adam"]:
        x = x0.clone().requires_grad_(True)
        if opt_name == "newton":
            opt = DampedNewton([x], damping=1e-6)
            cap = 100
        elif opt_name == "lbfgs":
            opt = LBFGS([x], history_size=10)
            cap = 200
        elif opt_name == "hf":
            opt = HessianFree([x], damping=1e-3, cg_max_iter=50)
            cap = 200
        elif opt_name == "sgdm":
            opt = SGDMomentum([x], lr=1e-3, momentum=0.9)
            cap = max_iters
        elif opt_name == "adam":
            opt = Adam([x], lr=3e-3)
            cap = max_iters
        else:
            continue

        def closure(create_graph=False):
            opt.zero_grad(set_to_none=True)
            loss = rosenbrock(x)
            if not create_graph:
                loss.backward()
            return loss

        t0 = time.time()
        last_loss = float("inf")
        n_iter = 0
        for it in range(cap):
            try:
                loss_val = opt.step(closure)
            except RuntimeError as e:
                results.append(dict(optimizer=opt_name, dim=dim,
                                    iters=n_iter, time=time.time() - t0,
                                    final_loss=last_loss,
                                    converged=False, error=str(e)))
                break
            if loss_val is None:
                with torch.no_grad():
                    loss_val = float(rosenbrock(x).item())
            n_iter = it + 1
            last_loss = float(loss_val)
            # Convergence check by loss value.
            if last_loss < tol:
                break

        results.append(dict(optimizer=opt_name, dim=dim,
                            iters=n_iter, time=time.time() - t0,
                            final_loss=last_loss,
                            converged=last_loss < tol))
    return results


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

class LogReg(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.lin = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.lin(x)


def run_logistic_regression(max_iters: int = 1000,
                            tol_loss: float = 1e-2,
                            seed: int = 0) -> List[Dict]:
    torch.manual_seed(seed)
    X, y = logistic_regression_data(n_samples=5000, n_features=50,
                                    n_classes=3, seed=seed)
    results = []
    for opt_name in ["newton", "lbfgs", "hf", "kfac", "adam"]:
        torch.manual_seed(seed)
        model = LogReg(50, 3)
        if opt_name == "newton":
            opt = DampedNewton(model.parameters(), damping=1e-6)
            cap = 50
        elif opt_name == "lbfgs":
            opt = LBFGS(model.parameters(), history_size=20)
            cap = 100
        elif opt_name == "hf":
            opt = HessianFree(model.parameters(), damping=1e-3,
                              cg_max_iter=20)
            cap = 100
        elif opt_name == "kfac":
            opt = KFAC(model.parameters(), model=model,
                       lr=0.5, damping=1e-3, inv_freq=1)
            cap = 200
        elif opt_name == "adam":
            opt = Adam(model.parameters(), lr=1e-1)
            cap = max_iters
        else:
            continue

        def closure(create_graph=False):
            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            if not create_graph:
                loss.backward()
            return loss

        t0 = time.time()
        last_loss = float("inf")
        n_iter = 0
        for it in range(cap):
            loss_val = opt.step(closure)
            if loss_val is None:
                with torch.no_grad():
                    loss_val = float(F.cross_entropy(model(X), y).item())
            n_iter = it + 1
            last_loss = float(loss_val)
            if last_loss < tol_loss:
                break
        results.append(dict(optimizer=opt_name, iters=n_iter,
                            time=time.time() - t0, final_loss=last_loss,
                            converged=last_loss < tol_loss))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["rosenbrock", "logistic", "all"],
                        default="all")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--out", default="runs/sanity.json")
    args = parser.parse_args()

    out: Dict[str, list] = {}
    if args.problem in ("rosenbrock", "all"):
        print(f"=== Rosenbrock (dim={args.dim}) ===")
        res = run_rosenbrock(dim=args.dim)
        out["rosenbrock"] = res
        for r in res:
            converged = "OK" if r["converged"] else "no"
            print(f"  {r['optimizer']:>8s}: iters={r['iters']:>4d}  "
                  f"final_loss={r['final_loss']:.3e}  "
                  f"time={r['time']:.2f}s  converged={converged}")

    if args.problem in ("logistic", "all"):
        print("=== Logistic regression ===")
        res = run_logistic_regression()
        out["logistic"] = res
        for r in res:
            converged = "OK" if r["converged"] else "no"
            print(f"  {r['optimizer']:>8s}: iters={r['iters']:>4d}  "
                  f"final_loss={r['final_loss']:.4f}  "
                  f"time={r['time']:.2f}s  converged={converged}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
