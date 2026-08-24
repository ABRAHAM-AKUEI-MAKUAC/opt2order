"""Structured logging of per-step and per-epoch metrics.

Writes one JSON object per line to make downstream analysis (in the
analysis/ folder) trivially scriptable.  Tracks iterations-to-target
metrics so that the figures of Chapter 4 can be regenerated.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MetricsLogger:
    out_path: Path
    target_accuracy: Optional[float] = None
    target_loss: Optional[float] = None
    _step: int = 0
    _start_time: float = field(default_factory=time.time)
    _iters_to_target_acc: Optional[int] = None
    _time_to_target_acc: Optional[float] = None
    _epochs_to_target_acc: Optional[int] = None
    _iters_to_target_loss: Optional[int] = None
    _time_to_target_loss: Optional[float] = None
    _epochs_to_target_loss: Optional[int] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.out_path = Path(self.out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_path, "w")

    def _write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        self.history.append(record)

    def log_step(self, loss: Optional[float], extra: Optional[Dict[str, Any]] = None) -> None:
        self._step += 1
        rec: Dict[str, Any] = {
            "kind": "step",
            "step": self._step,
            "wall": time.time() - self._start_time,
            "loss": loss,
        }
        if extra:
            rec.update(extra)
        if (self.target_loss is not None and loss is not None
                and self._iters_to_target_loss is None
                and loss <= self.target_loss):
            self._iters_to_target_loss = self._step
            self._time_to_target_loss = rec["wall"]
        self._write(rec)

    def log_epoch(self, epoch: int, train_loss: Optional[float],
                  val_loss: Optional[float], val_acc: Optional[float],
                  extra: Optional[Dict[str, Any]] = None) -> None:
        rec: Dict[str, Any] = {
            "kind": "epoch",
            "epoch": epoch,
            "step": self._step,
            "wall": time.time() - self._start_time,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        if extra:
            rec.update(extra)
        if (self.target_accuracy is not None and val_acc is not None
                and self._iters_to_target_acc is None
                and val_acc >= self.target_accuracy):
            self._iters_to_target_acc = self._step
            self._time_to_target_acc = rec["wall"]
            self._epochs_to_target_acc = epoch
        self._write(rec)

    def summary(self, final: Dict[str, Any]) -> Dict[str, Any]:
        rec = {
            "kind": "summary",
            "iters_to_target_acc": self._iters_to_target_acc,
            "time_to_target_acc": self._time_to_target_acc,
            "epochs_to_target_acc": self._epochs_to_target_acc,
            "iters_to_target_loss": self._iters_to_target_loss,
            "time_to_target_loss": self._time_to_target_loss,
            "epochs_to_target_loss": self._epochs_to_target_loss,
            "total_steps": self._step,
            "total_wall": time.time() - self._start_time,
            **final,
        }
        self._write(rec)
        self._fh.close()
        return rec
