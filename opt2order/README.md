# opt2order — Second-Order Optimizers for Deep Learning

A unified Python framework for benchmarking second-order optimization
methods against first-order baselines on standard deep learning tasks.

This codebase accompanies the M.Sc. thesis *Enhancing Deep Learning
Efficiency: A Numerical Optimization Approach using Second-Order Methods
in Python*. It implements every method described in Chapter 3 of the
thesis behind a uniform `torch.optim.Optimizer`-style interface, with
YAML-configured experiments and a reproducible analysis pipeline.

---

## What's inside

| Folder | What it contains |
|---|---|
| `opt2order/optimizers/` | `SGDMomentum`, `Adam`, `LBFGS`, `HessianFree`, `KFAC`, `DampedNewton` |
| `opt2order/curvature/` | `hvp`, `gnvp`, conjugate gradient with damping and best-iterate fallback |
| `opt2order/models/` | MNIST MLP (~530k params), CIFAR-10 VGG (~1M params), 2-layer LSTM (~270k params) |
| `opt2order/data/` | MNIST / CIFAR-10 loaders, the Hochreiter–Schmidhuber addition task, plus closed-form Rosenbrock and synthetic logistic regression |
| `opt2order/harness/` | YAML config loader, JSONL metrics logger, `train_one_run` |
| `configs/` | One YAML per (task, optimizer): MNIST × {sgdm, adam, lbfgs, hf, kfac}, CIFAR-10 × {sgdm, adam, lbfgs, hf, kfac}, RNN × {sgdm, adam, hf} |
| `scripts/` | `run_sanity.py` (Rosenbrock + logistic), `run_experiment.py` (one config), `run_sweep.py` (full Chapter 4 protocol) |
| `analysis/` | `regenerate.py` — rebuilds figures and tables from raw JSONL logs |
| `tests/` | Curvature-oracle correctness, CG convergence, per-optimizer smoke tests, harness wiring |

The optimizers correspond to thesis chapter content as follows:

| Optimizer | Algorithm | Thesis ref |
|---|---|---|
| `sgdm` | SGD with momentum | Algorithm 3.1 |
| `adam` | Adam | Algorithm 3.2 |
| `newton` | Damped Newton (full Hessian) | Algorithm 3.3 |
| `lbfgs` | Limited-memory BFGS, two-loop recursion + Wolfe line search | Algorithm 3.4 / §4.1.4 |
| `hf` | Hessian-Free truncated Newton with CG and Levenberg–Marquardt damping | Algorithm 3.5 / §4.1.3 / Listing A.1 |
| `kfac` | Kronecker-Factored Approximate Curvature | Algorithm 3.6 / §4.1.5 / Listing A.2 |

---

## 1. Install dependencies

The framework targets Python ≥ 3.10 and PyTorch ≥ 2.1. The thesis
pinned PyTorch 2.3.0 with CUDA 12.1; any reasonably recent build is
fine.

### Option A — pip with virtualenv (recommended)

```bash
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                         # install the opt2order package itself
```

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# python -m venv .venv  
# .\.venv\Scripts\activate.ps1
# python.exe -m pip install --upgrade pip
### Option B — conda

```bash
conda create -n opt2order python=3.11 -y
conda activate opt2order
pip install -r requirements.txt
pip install -e .
```

### Option C — CPU-only PyTorch (laptops / CI)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
pip install -e .
```

### Verify the install

```bash
pytest -q                                # ~30s on CPU
```

You should see all tests pass. If you don't have a GPU, also pass
`--device cpu` to every script below.

---

## 2. Sanity check (Rosenbrock + logistic regression)

This reproduces Section 4.2 of the thesis: closed-form Rosenbrock
minimization and small logistic regression. It runs in seconds and
exercises every optimizer on a problem where the answer is known.

```bash
python -m scripts.run_sanity
```

Sample output:

```
=== Rosenbrock (dim=2) ===
   newton: iters=  14  final_loss=4.872e-15  time=0.18s  converged=OK
    lbfgs: iters=  33  final_loss=1.022e-13  time=0.06s  converged=OK
       hf: iters=  22  final_loss=8.314e-12  time=0.31s  converged=OK
     sgdm: iters=5000  final_loss=2.870e-04  time=0.42s  converged=no
     adam: iters=5000  final_loss=8.451e-03  time=0.51s  converged=no
=== Logistic regression ===
   newton: iters=   8  final_loss=0.0078  time=0.42s  converged=OK
    lbfgs: iters=  15  final_loss=0.0089  time=0.06s  converged=OK
       hf: iters=  12  final_loss=0.0091  time=0.21s  converged=OK
     kfac: iters=  20  final_loss=0.0098  time=0.04s  converged=OK
     adam: iters= 350  final_loss=0.0097  time=0.18s  converged=OK
```

A JSON record of the run is written to `runs/sanity.json`.

---

## 3. Run a single experiment

Every experiment is described by a YAML config under `configs/`. The
schema is documented in `opt2order/harness/config.py`.

```bash
# MNIST with K-FAC, single seed, GPU
python -m scripts.run_experiment --config configs/mnist/kfac.yaml

# Multiple seeds (the thesis uses 5)
python -m scripts.run_experiment --config configs/mnist/kfac.yaml --seeds 1 2 3 4 5

# CPU run for a quick smoke test (5 epochs instead of 30)
python -m scripts.run_experiment --config configs/mnist/adam.yaml --device cpu --epochs 5

# Skip dataset auto-download (assumes data is already in ./data)
python -m scripts.run_experiment --config configs/mnist/adam.yaml --no-download
```

For each run, the harness writes:

```
runs/<experiment_name>/seed<N>/log.jsonl    # one JSON record per step / epoch / final summary
runs/<experiment_name>/summary.json         # aggregated summary across the seeds you ran
```

The JSONL log is the canonical artifact — every figure and table in
the analysis is regenerated from it. A typical record:

```json
{"kind":"epoch","epoch":5,"step":1955,"wall":52.4,"train_loss":0.118,"val_loss":0.084,"val_acc":0.9802}
{"kind":"summary","iters_to_target_acc":1955,"time_to_target_acc":52.4,"epochs_to_target_acc":5,
 "test_acc":0.9840,"test_loss":0.0521,"n_params":531242,"optimizer":"kfac","seed":1}
```

### Available configs

```
configs/mnist/{sgdm,adam,lbfgs,hf,kfac}.yaml
configs/cifar10/{sgdm,adam,lbfgs,hf,kfac}.yaml
configs/rnn/{sgdm,adam,hf}.yaml
```

The hyperparameter values in these configs are the *selected* values
from Table B.1 of the thesis (i.e. the winners of the grid search that
the thesis itself ran). You can override any field at the command line
by editing the YAML, or programmatically by editing the
`ExperimentConfig` returned by `load_config()`.

---







## 4. Run the full sweep

The full sweep reproduces Chapter 4: every (task × optimizer) config
across 5 random seeds.

```bash
# Everything: 13 configs × 5 seeds = 65 runs
python -m scripts.run_sweep


# Only MNIST experiments
python -m scripts.run_sweep --tasks mnist

# MNIST + CIFAR-10, 3 seeds
python -m scripts.run_sweep --tasks mnist cifar10 --seeds 1 2 3

# Smoke test: 2 epochs, 2 seeds, runs in a few minutes on CPU
python -m scripts.run_sweep --quick --device cpu

# Restrict to a single device
python -m scripts.run_sweep --device cuda:0
```

The sweep writes one log file per run and a top-level
`runs/sweep_summary.json` aggregating the summary records of every run.
Failed runs are logged but do not abort the sweep (`--continue-on-error`
is on by default).

### Resource expectations

These are rough estimates for the full 5-seed sweep. The thesis itself
reports ~45 GPU-hours total on an RTX 4090 (Section B.3). Per-run
times scale roughly linearly on weaker hardware.

| Task | Per-run time (RTX 4090) | Per-run memory | Total runs (× 5 seeds) |
|---|---|---|---|
| MNIST × 5 optimizers | 1–3 min | ≤ 1 GB | 25 |
| CIFAR-10 × 5 optimizers | 20–80 min | ≤ 2 GB | 25 |
| RNN × 3 optimizers | 5–15 min | ≤ 1 GB | 15 |

If you don't have a GPU, run with `--device cpu --quick`. The full
sweep is not practical on CPU.

---

## 5. Regenerate the analysis

Once you have at least one run logged, regenerate every figure and
table of Chapter 4:

```bash
python -m analysis.regenerate
```

This reads `runs/*/seed*/log.jsonl`, aggregates across seeds (median
and IQR, exactly as Section 3.7 prescribes), and writes:

```
analysis/figures/mnist_acc_vs_epoch.png         # Figure 4.2
analysis/figures/mnist_acc_vs_wallclock.png     # Figure 4.3
analysis/figures/cifar_acc_vs_epoch.png         # Figure 4.4
analysis/figures/cifar_acc_vs_wallclock.png
analysis/figures/addition_loss_vs_epoch.png     # Figure 4.6 (loss form)
analysis/tables/mnist_results.csv               # Table 4.1
analysis/tables/cifar_results.csv               # Table 4.2
analysis/tables/addition_results.csv
analysis/summary.json                           # machine-readable rollup
```

All paths can be overridden:

```bash
python -m analysis.regenerate --runs-dir runs --out-dir analysis
```

---

## 6. Reproduce the published numbers exactly

```bash
# 1. Install (Section 1).
pip install -e . && pip install -r requirements.txt

# 2. Sanity-check the curvature oracles (~1 min).
python -m scripts.run_sanity

# 3. Run unit tests (~30 s).
pytest -q

# 4. Run the full Chapter 4 protocol (~45 GPU-h on RTX 4090).
python -m scripts.run_sweep

# 5. Regenerate Figures 4.2–4.7 and Tables 4.1–4.2.
python -m analysis.regenerate
```

Reproducibility is enforced at multiple levels (Section 3.7):
- Python, NumPy, and PyTorch RNGs are seeded per-run.
- Dataset shuffling and CIFAR-10 augmentation use seeded generators.
- Every config and the exact code revision that produced a log
  is recoverable from the JSONL artifacts and `git rev-parse HEAD`.

Some sources of nondeterminism in cuDNN convolutional backward
passes are accepted, as discussed in the thesis. To eliminate them
at the cost of speed, set `torch.backends.cudnn.deterministic = True`
in the harness or set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in your
shell.

---

## 7. Adding a new optimizer

1. Create `opt2order/optimizers/my_opt.py` subclassing
   `opt2order.optimizers.base.CurvatureOptimizer`.
2. Register it in `opt2order/optimizers/__init__.py`'s
   `OPTIMIZER_REGISTRY`.
3. Pick the closure protocol it expects (simple / lbfgs / hf) and add
   a branch in `opt2order/harness/trainer.py` if needed.
4. Add a YAML config under `configs/<task>/my_opt.yaml`.
5. Run `python -m scripts.run_experiment --config configs/<task>/my_oy
pt.yaml`.

The closure protocols are:
- **simple** — one `closure()` call, runs forward + backward, returns
  loss tensor. Used by SGDM, Adam, K-FAC.
- **lbfgs** — re-evaluable `closure()` for the line search. Same body
  as `simple` but the optimizer may invoke it many times per step.
- **hf** — `closure(create_graph=True)` returns a graph that can be
  differentiated again, for second-order autograd. Used by
  Hessian-Free.

---

## 8. Repository layout

```
opt2order/
├── opt2order/                       # importable package
│   ├── curvature/                   # HVP, GNVP, CG
│   ├── optimizers/                  # 6 optimizers + base + registry
│   ├── models/                      # MnistMLP, CifarVGG, AdditionLSTM
│   ├── data/                        # MNIST, CIFAR-10, addition, synthetic
│   └── harness/                     # config, metrics logger, trainer
├── configs/
│   ├── mnist/                       # one YAML per optimizer
│   ├── cifar10/
│   ├── rnn/
│   └── sanity/                      # (placeholder for future sanity configs)
├── scripts/
│   ├── run_sanity.py
│   ├── run_experiment.py
│   └── run_sweep.py
├── analysis/
│   └── regenerate.py
├── tests/
│   ├── test_curvature.py
│   ├── test_optimizers.py
│   └── test_harness.py
├── README.md                        # this file
├── requirements.txt
└── pyproject.toml
```

---

## License

Code released under MIT for use as an educational and research artifact.

## Citation

If you use this codebase, please cite the accompanying thesis:

```
@mastersthesis{opt2order2026,
  title  = {Enhancing Deep Learning Efficiency: A Numerical Optimization
            Approach using Second-Order Methods in Python},
  author = {Abraham Akuei Makuac},
  superviser ={Mohamed H. Doda},
  school = {University of Juba, School of Mathematics Department of Applied Mahtematics },
  year   = {2026}
}
```
