"""YAML-based experiment configuration.

A minimal schema that's expressive enough for the experiments described
in Chapter 4 without pulling in heavy config frameworks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ExperimentConfig:
    name: str
    dataset: str                       # mnist, cifar10, addition
    model: str                         # mnist_mlp, cifar_vgg, addition_lstm
    optimizer: str                     # sgdm, adam, lbfgs, hf, kfac
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    data_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    epochs: int = 30
    batch_size: int = 128
    eval_every_epoch: bool = True
    target_accuracy: Optional[float] = None
    target_loss: Optional[float] = None
    grad_clip: Optional[float] = None
    device: str = "cuda"
    seed: int = 1
    output_dir: str = "runs"
    log_every: int = 50

    # ---- task-shaped extras ----
    loss: str = "cross_entropy"        # cross_entropy or mse
    closure_supports_create_graph: bool = False  # set True for HF


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if "data_kwargs" not in raw:
        raw["data_kwargs"] = {}
    if "optimizer_kwargs" not in raw:
        raw["optimizer_kwargs"] = {}
    if "model_kwargs" not in raw:
        raw["model_kwargs"] = {}
    return ExperimentConfig(**raw)
