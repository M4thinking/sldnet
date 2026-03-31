# SLDNet
This repository contains the centralized SLDNet model and benchmark runners for the ALeRCE and DiMMAD workflows.

Canonical data layout at repository root:

- `data/alerce/data/`
- `data/alerce/data_raw/features_BHRF_model.pkl`
- `data/dimmad/`

The commands below are written for the intended final root layout `./data/...`.

## Reproduce ALeRCE `conference3`

These commands reproduce the family-specific sweeps corresponding to the ALeRCE benchmark.

Transient:

```bash # -m sldnet.
python alerce_cli.py \
  --device cuda \
  --run_name alerce \
  --scheme Transient \
  --fold all \
  --all_outliers \
  --epochs 10000 \
  --lr 1e-4 \
  --batch_size 512 \
  --units 1024 512 \
  --sigma_low 0.001 \
  --sigma_high 3.0 \
  --beta 0.001 \
  --dropout 0.1 \
  --norm layernorm \
  --activation gelu \
  --L 0.001
```

Periodic:

```bash
python alerce_cli.py \
  --device cuda \
  --run_name alerce \
  --scheme Periodic \
  --fold all \
  --all_outliers \
  --epochs 10000 \
  --lr 1e-4 \
  --batch_size 512 \
  --units 1024 512 \
  --sigma_low 0.001 \
  --sigma_high 3.0 \
  --beta 0.001 \
  --dropout 0.1 \
  --norm layernorm \
  --activation gelu \
  --L 0.001
```

Stochastic:

```bash
python alerce_cli.py \
  --device cuda \
  --run_name alerce \
  --scheme Stochastic \
  --fold all \
  --all_outliers \
  --epochs 5000 \
  --lr 2e-5 \
  --batch_size 512 \
  --units 1024 512 \
  --sigma_low 0.001 \
  --sigma_high 3.0 \
  --beta 0.001 \
  --dropout 0.1 \
  --norm layernorm \
  --activation gelu \
  --L 0.001
```

## Reproduce DiMMAD SLDNet benchmark

This reproduces the SLDNet benchmark outputs used alongside the DiMMAD comparison workflow.

```bash
python dimmad_cli.py \
  --device cuda \
  --data_dir data/dimmad \
  --result_path runs/dimmad_benchmark \
  --dataset all \
  --scheme all \
  --num_runs 20 \
  --epochs 1000 \
  --standarize \
  --qft uniform
```
