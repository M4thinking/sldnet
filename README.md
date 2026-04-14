# SLDNet

This repository contains the centralized SLDNet model and benchmark runners for the ALeRCE and DiMMAD workflows.

## Installation

```bash
python -m pip install -U pip uv

# install into the currently active environment
uv sync --active

# Option A:
uv run <command> # to run directly without activating

# Option B:
source .venv/bin/activate # to activate the environment
```

## Data extraction references

Use the data extraction and preparation steps described in these repositories before running the commands below:

- ALeRCE data: [mperezcarrasco/AnomalyALeRCE](https://github.com/mperezcarrasco/AnomalyALeRCE)
- DiMMAD data: [sidchaini/dimmad](https://github.com/sidchaini/dimmad/)

After following those instructions, place the exported artifacts in the canonical layout expected by this repository.

Canonical data layout at repository root:

- `data/alerce/data/`
- `data/alerce/data_raw/features_BHRF_model.pkl`
- `data/dimmad/`

The commands below are written for the intended final root layout `./data/...`.

## Reproduce ALeRCE and DiMMAD benchmarks

```bash
# DiMMAD benchmark
python dimmad_cli.py --dataset ztf --scheme rid --num_runs 20 --epochs 1000
python dimmad_cli.py --dataset ztf --scheme ood --num_runs 20 --epochs 1000
python dimmad_cli.py --dataset elasticc --scheme rid --num_runs 20 --epochs 1000
python dimmad_cli.py --dataset elasticc --scheme ood --num_runs 20 --epochs 1000
```

```bash
# ALeRCE benchmark
python alerce_cli.py --epochs 10000 --patience -1 --fold all --scheme Transient  --all_outliers
python alerce_cli.py --epochs 10000 --patience -1 --fold all --scheme Periodic   --all_outliers
python alerce_cli.py --epochs  5000 --patience -1 --fold all --scheme Stochastic --all_outliers --lr 2e-5

```

Or, to run all benchmarks sequentially:

```bash
bash experiments.sh
```
