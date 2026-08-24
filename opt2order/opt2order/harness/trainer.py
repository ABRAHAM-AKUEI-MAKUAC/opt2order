"""Training harness.

Coordinates dataset, model, optimizer, and metrics for a single run.
Three closure protocols coexist:

  - "simple"   : zero-grad, forward, backward, return loss tensor.
                 Used by SGDM, Adam, K-FAC.
  - "lbfgs"    : zero-grad, forward, backward, return loss tensor.
                 Re-evaluable any number of times in a single step()
                 (the line search calls it repeatedly).
  - "hf"       : accepts create_graph kwarg; returns the loss with
                 a graph that can be differentiated again.

The trainer dispatches on the optimizer name to pick the right one.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..data import build_loaders
from ..models import build_model
from ..optimizers import build_optimizer
from .config import ExperimentConfig
from .metrics import MetricsLogger


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _get_loss_fn(name: str) -> Callable[[Tensor, Tensor], Tensor]:
    if name == "cross_entropy":
        return F.cross_entropy
    if name == "mse":
        return F.mse_loss
    raise ValueError(f"Unknown loss '{name}'")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn,
             device: str, classification: bool) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total += bs
        if classification:
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1) if classification else float("nan")
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Closure factories for each optimizer style
# ---------------------------------------------------------------------------

def _make_simple_closure(model, optimizer, x, y, loss_fn,
                         grad_clip: Optional[float]):
    def closure():
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        return loss
    return closure


def _make_lbfgs_closure(model, optimizer, x, y, loss_fn,
                        grad_clip: Optional[float]):
    # Re-evaluable closure for the Wolfe line search.
    def closure():
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        return loss
    return closure


def _make_hf_closure(model, optimizer, x, y, loss_fn,
                     grad_clip: Optional[float]):
    def closure(create_graph: bool = False):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        if not create_graph:
            # When we don't need a second derivative graph, populate
            # .grad here so that the optimizer can read it if needed.
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        return loss
    return closure


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_one_run(cfg: ExperimentConfig, seed: Optional[int] = None,
                  download_data: bool = True) -> Dict[str, Any]:
    """Run one training experiment end-to-end and return its summary."""
    if seed is None:
        seed = cfg.seed
    set_seed(seed)

    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[harness] CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    # ---- Data -------------------------------------------------------------
    data_kwargs = dict(cfg.data_kwargs)
    data_kwargs.setdefault("batch_size", cfg.batch_size)
    data_kwargs.setdefault("seed", seed)
    if cfg.dataset in ("mnist", "cifar10"):
        data_kwargs.setdefault("download", download_data)
    train_loader, val_loader, test_loader = build_loaders(cfg.dataset,
                                                          **data_kwargs)

    # ---- Model ------------------------------------------------------------
    model = build_model(cfg.model, **cfg.model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[harness] {cfg.name}: model={cfg.model} ({n_params:,} params), "
          f"optimizer={cfg.optimizer}, device={device}, seed={seed}")

    # ---- Loss -------------------------------------------------------------
    loss_fn = _get_loss_fn(cfg.loss)
    classification = cfg.loss == "cross_entropy"

    # ---- Optimizer --------------------------------------------------------
    opt_kwargs = dict(cfg.optimizer_kwargs)
    if cfg.optimizer.lower() == "kfac":
        opt_kwargs["model"] = model
    optimizer = build_optimizer(cfg.optimizer,
                                model.parameters(), **opt_kwargs)

    # ---- Closure protocol -------------------------------------------------
    optname = cfg.optimizer.lower()
    if optname in ("hf", "hessian_free"):
        make_closure = _make_hf_closure
    elif optname == "lbfgs":
        make_closure = _make_lbfgs_closure
    else:
        make_closure = _make_simple_closure

    # ---- Logger -----------------------------------------------------------
    out_dir = Path(cfg.output_dir) / cfg.name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    logger = MetricsLogger(out_path=log_path,
                           target_accuracy=cfg.target_accuracy,
                           target_loss=cfg.target_loss)

    # ---- Training loop ----------------------------------------------------
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, running_n = 0.0, 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            closure = make_closure(model, optimizer, x, y, loss_fn,
                                   cfg.grad_clip)
            loss_val = optimizer.step(closure)
            if loss_val is None:
                # Some optimizers (SGDM/Adam) don't return loss when no
                # closure is requested; recompute for logging.
                with torch.no_grad():
                    model.eval()
                    out = model(x)
                    loss_val = float(loss_fn(out, y).item())
                    model.train()
            running_loss += float(loss_val) * y.size(0)
            running_n += y.size(0)
            if batch_idx % cfg.log_every == 0:
                logger.log_step(loss=float(loss_val))

        train_loss = running_loss / max(running_n, 1)
        if cfg.eval_every_epoch:
            val_loss, val_acc = evaluate(model, val_loader, loss_fn,
                                         device, classification)
            print(f"  epoch {epoch+1:>3d}/{cfg.epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
            logger.log_epoch(epoch + 1, train_loss, val_loss, val_acc)
        else:
            logger.log_epoch(epoch + 1, train_loss, None, None)

    # ---- Final test evaluation -------------------------------------------
    test_loss, test_acc = evaluate(model, test_loader, loss_fn,
                                   device, classification)
    print(f"[harness] FINAL test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    summary = logger.summary({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "n_params": n_params,
        "optimizer": cfg.optimizer,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "seed": seed,
    })
    return summary
