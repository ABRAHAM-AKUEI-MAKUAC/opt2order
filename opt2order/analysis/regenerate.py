"""Regenerate the figures and tables of Chapter 4 from raw log files.

Reads runs/*/seed*/log.jsonl produced by the harness, aggregates over
seeds, and writes:
  - analysis/figures/<task>_acc_vs_epoch.png
  - analysis/figures/<task>_acc_vs_wallclock.png
  - analysis/figures/rosenbrock_loss.png      (if scripts/run_sanity.py was run)
  - analysis/tables/<task>_results.csv
  - analysis/summary.json    (machine-readable rollup)

Usage:
    python -m analysis.regenerate
    python -m analysis.regenerate --runs-dir runs --out-dir analysis
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(log_path: Path) -> Dict[str, Any]:
    """Load one run's log.jsonl into structured arrays."""
    epochs, wall_at_epoch, val_acc, val_loss, train_loss = [], [], [], [], []
    summary: Optional[Dict[str, Any]] = None
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "epoch":
                epochs.append(rec["epoch"])
                wall_at_epoch.append(rec["wall"])
                val_acc.append(rec.get("val_acc"))
                val_loss.append(rec.get("val_loss"))
                train_loss.append(rec.get("train_loss"))
            elif rec.get("kind") == "summary":
                summary = rec
    return {
        "epochs": epochs,
        "wall_at_epoch": wall_at_epoch,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "summary": summary,
    }


def discover_runs(runs_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Group runs by experiment name."""
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for log_path in runs_dir.glob("*/seed*/log.jsonl"):
        exp_name = log_path.parent.parent.name
        seed_str = log_path.parent.name
        try:
            seed = int(seed_str.replace("seed", ""))
        except ValueError:
            seed = -1
        run = load_run(log_path)
        run["name"] = exp_name
        run["seed"] = seed
        out[exp_name].append(run)
    return dict(out)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def median_iqr(xs: List[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    m = statistics.median(xs)
    if len(xs) >= 4:
        q1 = statistics.quantiles(xs, n=4)[0]
        q3 = statistics.quantiles(xs, n=4)[2]
        iqr = q3 - q1
    else:
        iqr = max(xs) - min(xs)
    return m, iqr


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute median/IQR of summary metrics across seeds."""
    test_acc = [r["summary"]["test_acc"] for r in runs
                if r["summary"] and r["summary"].get("test_acc") is not None]
    test_loss = [r["summary"]["test_loss"] for r in runs
                 if r["summary"] and r["summary"].get("test_loss") is not None]
    epochs_to = [r["summary"].get("epochs_to_target_acc") for r in runs
                 if r["summary"]]
    time_to = [r["summary"].get("time_to_target_acc") for r in runs
               if r["summary"]]
    iters_loss = [r["summary"].get("iters_to_target_loss") for r in runs
                  if r["summary"]]
    time_loss = [r["summary"].get("time_to_target_loss") for r in runs
                 if r["summary"]]

    acc_m, acc_i = median_iqr(test_acc)
    loss_m, loss_i = median_iqr(test_loss)
    e2t_m, e2t_i = median_iqr([e for e in epochs_to if e is not None])
    t2t_m, t2t_i = median_iqr([t for t in time_to if t is not None])
    il_m, il_i = median_iqr([e for e in iters_loss if e is not None])
    tl_m, tl_i = median_iqr([t for t in time_loss if t is not None])

    return {
        "n_seeds": len(runs),
        "test_acc_median": acc_m, "test_acc_iqr": acc_i,
        "test_loss_median": loss_m, "test_loss_iqr": loss_i,
        "epochs_to_target_acc_median": e2t_m,
        "epochs_to_target_acc_iqr": e2t_i,
        "time_to_target_acc_median": t2t_m,
        "time_to_target_acc_iqr": t2t_i,
        "iters_to_target_loss_median": il_m,
        "iters_to_target_loss_iqr": il_i,
        "time_to_target_loss_median": tl_m,
        "time_to_target_loss_iqr": tl_i,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

OPTIMIZER_COLORS = {
    "sgdm": "#1f77b4",
    "adam": "#ff7f0e",
    "lbfgs": "#2ca02c",
    "hf": "#d62728",
    "kfac": "#9467bd",
    "newton": "#7f7f7f",
}


def _color_for(name: str) -> str:
    for key, color in OPTIMIZER_COLORS.items():
        if key in name.lower():
            return color
    return "#444444"


def plot_accuracy_curves(task: str, runs_by_exp: Dict[str, List[Dict[str, Any]]],
                         out_dir: Path, x_axis: str = "epoch") -> Optional[Path]:
    """One figure per task, one curve per optimizer (median across seeds)."""
    relevant = {n: rs for n, rs in runs_by_exp.items()
                if n.startswith(task)}
    if not relevant:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for exp_name, runs in sorted(relevant.items()):
        all_y, all_x = [], []
        for r in runs:
            if not r["val_acc"] or all(v is None for v in r["val_acc"]):
                continue
            xs = (r["epochs"] if x_axis == "epoch" else r["wall_at_epoch"])
            ys = [v if v is not None else float("nan") for v in r["val_acc"]]
            all_x.append(xs)
            all_y.append(ys)
        if not all_y:
            continue
        # Align on shortest run.
        L = min(len(y) for y in all_y)
        ys_arr = [y[:L] for y in all_y]
        xs_arr = [x[:L] for x in all_x]
        x_mean = [sum(xs) / len(xs) for xs in zip(*xs_arr)]
        y_med = [statistics.median(ys) for ys in zip(*ys_arr)]
        ax.plot(x_mean, y_med, label=exp_name.replace(task + "_", ""),
                color=_color_for(exp_name), lw=2)

    ax.set_xlabel("Epoch" if x_axis == "epoch" else "Wall-clock time (s)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(f"{task.upper()} — accuracy vs {x_axis}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    suffix = "epoch" if x_axis == "epoch" else "wallclock"
    out_path = out_dir / f"{task}_acc_vs_{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_rnn_loss_curves(runs_by_exp: Dict[str, List[Dict[str, Any]]],
                         out_dir: Path) -> Optional[Path]:
    """RNN task uses loss not accuracy; plot val_loss vs epoch."""
    relevant = {n: rs for n, rs in runs_by_exp.items()
                if n.startswith("addition_")}
    if not relevant:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for exp_name, runs in sorted(relevant.items()):
        all_y, all_x = [], []
        for r in runs:
            if not r["val_loss"] or all(v is None for v in r["val_loss"]):
                continue
            all_x.append(r["epochs"])
            all_y.append([v if v is not None else float("nan")
                          for v in r["val_loss"]])
        if not all_y:
            continue
        L = min(len(y) for y in all_y)
        ys_arr = [y[:L] for y in all_y]
        xs_arr = [x[:L] for x in all_x]
        x_mean = [sum(xs) / len(xs) for xs in zip(*xs_arr)]
        y_med = [statistics.median(ys) for ys in zip(*ys_arr)]
        ax.plot(x_mean, y_med, label=exp_name.replace("addition_", ""),
                color=_color_for(exp_name), lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.set_yscale("log")
    ax.set_title("Synthetic addition task — validation loss")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "addition_loss_vs_epoch.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_table(task: str, runs_by_exp: Dict[str, List[Dict[str, Any]]],
                out_dir: Path) -> Optional[Path]:
    relevant = {n: rs for n, rs in runs_by_exp.items() if n.startswith(task)}
    if not relevant:
        return None
    out_path = out_dir / f"{task}_results.csv"
    with open(out_path, "w") as f:
        f.write("optimizer,n_seeds,test_acc_median,test_acc_iqr,"
                "epochs_to_target_median,epochs_to_target_iqr,"
                "time_to_target_median,time_to_target_iqr,"
                "iters_to_target_loss_median,time_to_target_loss_median\n")
        for name, runs in sorted(relevant.items()):
            agg = aggregate(runs)
            short = name.replace(task + "_", "")
            f.write(f"{short},{agg['n_seeds']},"
                    f"{agg['test_acc_median']:.4f},{agg['test_acc_iqr']:.4f},"
                    f"{agg['epochs_to_target_acc_median']:.2f},"
                    f"{agg['epochs_to_target_acc_iqr']:.2f},"
                    f"{agg['time_to_target_acc_median']:.2f},"
                    f"{agg['time_to_target_acc_iqr']:.2f},"
                    f"{agg['iters_to_target_loss_median']:.2f},"
                    f"{agg['time_to_target_loss_median']:.2f}\n")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="analysis")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    runs_by_exp = discover_runs(runs_dir)
    if not runs_by_exp:
        print(f"No runs found under {runs_dir}.  "
              f"Run scripts/run_experiment.py or scripts/run_sweep.py first.")
        return 1
    print(f"Discovered {sum(len(v) for v in runs_by_exp.values())} runs "
          f"across {len(runs_by_exp)} experiments.")

    # Figures
    written = []
    for task in ("mnist", "cifar"):
        for axis in ("epoch", "wallclock"):
            p = plot_accuracy_curves(task, runs_by_exp, fig_dir, x_axis=axis)
            if p is not None:
                written.append(p)
    p = plot_rnn_loss_curves(runs_by_exp, fig_dir)
    if p is not None:
        written.append(p)

    # Tables
    for task in ("mnist", "cifar", "addition"):
        p = write_table(task, runs_by_exp, tab_dir)
        if p is not None:
            written.append(p)

    # Master summary JSON.  Replace NaN with None so the output is
    # standards-compliant JSON (json.dump emits NaN by default but
    # most consumers reject it).
    def _clean(x):
        import math
        if isinstance(x, float) and math.isnan(x):
            return None
        if isinstance(x, dict):
            return {k: _clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_clean(v) for v in x]
        return x

    summary: Dict[str, Any] = {}
    for name, runs in runs_by_exp.items():
        summary[name] = aggregate(runs)
        summary[name]["n_seeds_observed"] = sum(
            1 for r in runs if r["summary"] is not None)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_clean(summary), f, indent=2)
    written.append(out_dir / "summary.json")

    print("Wrote:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
