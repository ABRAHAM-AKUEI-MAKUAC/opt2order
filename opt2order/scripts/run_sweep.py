"""Run the full experimental sweep of Chapter 4.

By default, runs every config in configs/{mnist,cifar10,rnn}/*.yaml across
five seeds (1..5), exactly as described in Section 3.7 of the thesis.
A subset can be selected with --tasks.

Usage:
    python -m scripts.run_sweep                       # everything
    python -m scripts.run_sweep --tasks mnist         # only MNIST
    python -m scripts.run_sweep --tasks mnist cifar10 # MNIST + CIFAR
    python -m scripts.run_sweep --seeds 1 2           # fewer seeds
    python -m scripts.run_sweep --quick               # 2 epochs / 2 seeds, smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List

from opt2order.harness import load_config, train_one_run


def _gather_configs(tasks: List[str], configs_root: Path) -> List[Path]:
    out = []
    for task in tasks:
        d = configs_root / task
        if not d.is_dir():
            print(f"  (no configs found for task '{task}' at {d})")
            continue
        out.extend(sorted(d.glob("*.yaml")))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*",
                        default=["mnist", "cifar10", "rnn"],
                        help="Subdirectories of configs/ to include.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    parser.add_argument("--configs-root", default="configs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="2 epochs and 2 seeds; smoke test only.")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true",
                        default=True,
                        help="Skip individual failed runs (default).")
    args = parser.parse_args()

    if args.quick:
        args.seeds = args.seeds[:2]

    configs_root = Path(args.configs_root)
    config_files = _gather_configs(args.tasks, configs_root)
    if not config_files:
        print("No configs to run.")
        return 1

    print(f"Sweep: {len(config_files)} configs × {len(args.seeds)} seeds = "
          f"{len(config_files) * len(args.seeds)} runs.")
    for c in config_files:
        print(f"  {c}")

    all_summaries = []
    for cfg_path in config_files:
        cfg = load_config(cfg_path)
        if args.device is not None:
            cfg.device = args.device
        if args.quick:
            cfg.epochs = 2
        for seed in args.seeds:
            try:
                summary = train_one_run(cfg, seed=seed,
                                        download_data=not args.no_download)
                summary["config_path"] = str(cfg_path)
                all_summaries.append(summary)
            except Exception as e:
                print(f"\n[!!] Run failed: {cfg.name} seed={seed}: {e}",
                      file=sys.stderr)
                traceback.print_exc()
                if not args.continue_on_error:
                    raise
                all_summaries.append({
                    "config_path": str(cfg_path),
                    "name": cfg.name,
                    "seed": seed,
                    "error": str(e),
                })

    out_path = Path(cfg.output_dir) / "sweep_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nWrote {out_path} with {len(all_summaries)} runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
