"""Run one experiment from a YAML config, optionally over multiple seeds.

Usage:
    python -m scripts.run_experiment --config configs/mnist/adam.yaml
    python -m scripts.run_experiment --config configs/mnist/adam.yaml --seeds 1 2 3
    python -m scripts.run_experiment --config configs/mnist/adam.yaml --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from opt2order.harness import load_config, train_one_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="YAML config in configs/<task>/<optimizer>.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Seeds to run.  If omitted, uses cfg.seed.")
    parser.add_argument("--device", default=None,
                        help="Override cfg.device (e.g. 'cpu' or 'cuda:1').")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override cfg.epochs (smoke tests).")
    parser.add_argument("--no-download", action="store_true",
                        help="Don't auto-download datasets.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.epochs = args.epochs

    seeds = args.seeds if args.seeds is not None else [cfg.seed]
    summaries = []
    for seed in seeds:
        summary = train_one_run(cfg, seed=seed,
                                download_data=not args.no_download)
        summaries.append(summary)

    # Aggregate summary across seeds.
    out_dir = Path(cfg.output_dir) / cfg.name
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {out_dir / 'summary.json'} with {len(summaries)} runs.")


if __name__ == "__main__":
    main()
