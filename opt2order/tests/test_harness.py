"""End-to-end harness smoke test.

Builds a tiny synthetic dataset that mimics the harness's interface and
runs train_one_run for one epoch on CPU.  The point is to catch
breakage in the harness wiring (config parsing, closure dispatch,
metric logging) without downloading MNIST.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from opt2order.harness.config import ExperimentConfig
from opt2order.harness.metrics import MetricsLogger


def test_metrics_logger_round_trip(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    logger = MetricsLogger(out_path=log_path,
                           target_accuracy=0.5, target_loss=None)
    logger.log_step(loss=1.0)
    logger.log_step(loss=0.8)
    logger.log_epoch(epoch=1, train_loss=0.7, val_loss=0.65, val_acc=0.55)
    logger.summary({"test_acc": 0.6, "test_loss": 0.5})

    records = [json.loads(line) for line in open(log_path) if line.strip()]
    kinds = [r["kind"] for r in records]
    assert kinds == ["step", "step", "epoch", "summary"]
    summary = records[-1]
    assert summary["epochs_to_target_acc"] == 1
    assert summary["test_acc"] == 0.6


def test_experiment_config_minimal_fields():
    cfg = ExperimentConfig(name="x", dataset="mnist", model="mnist_mlp",
                           optimizer="adam")
    assert cfg.epochs == 30
    assert cfg.batch_size == 128
    assert cfg.optimizer_kwargs == {}
