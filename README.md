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
python alerce_cli.py --epochs 10000 --fold all --scheme Transient  --all_outliers
python alerce_cli.py --epochs 10000 --fold all --scheme Periodic   --all_outliers 
python alerce_cli.py --epochs  5000 --fold all --scheme Stochastic --all_outliers --lr 2e-5
```

Or, to run all benchmarks sequentially:

```bash
bash experiments.sh
```

## References

The following works describe methods and benchmarks used in this repository:

```bibtex
@article{Perez-Carrasco_2023,
  author = {Manuel Perez-Carrasco and Guillermo Cabrera-Vives and Lorena Hernandez-García and F. Förster and Paula Sanchez-Saez and Alejandra M. Muñoz Arancibia and Javier Arredondo and Nicolás Astorga and Franz E. Bauer and Amelia Bayo and M. Catelan and Raya Dastidar and P. A. Estévez and Paulina Lira and Giuliano Pignata},
  title = {Alert Classification for the {ALeRCE} Broker System: The Anomaly Detector},
  journal = {The Astronomical Journal},
  year = {2023},
  volume = {166},
  number = {4},
  pages = {151},
  doi = {10.3847/1538-3881/ace0c1},
  publisher = {The American Astronomical Society}
}

@inproceedings{Chaini_2025,
  author = {Chaini, Siddharth and Bianco, Federica B. and Mahabal, Ashish},
  title = {In Search of the Unknown Unknowns: A Multi-Metric Distance Ensemble for Out of Distribution Anomaly Detection in Astronomical Surveys},
  booktitle = {Machine Learning and the Physical Sciences Workshop at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year = {2025},
  eprint = {2510.23702},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.IM}
}
```

## Cite us

If you use SLDNet, please cite:

```bibtex
@inproceedings{Guzman-Olave_2026,
  author = {Guzmán-Olave, Sebastián and Estévez, Pablo A.},
  title = {Score-Based Log-Density Estimation for Anomaly Detection in Astronomical Surveys},
  booktitle = {2026 International Joint Conference on Neural Networks (IJCNN)},
  year = {2026},
  month = {jun},
  address = {Maastricht, The Netherlands},
  publisher = {IEEE}
}
```
