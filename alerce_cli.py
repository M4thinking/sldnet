import argparse
from pathlib import Path

import torch

from alerce_benchmark import parse_L, parse_fold
from alerce_benchmark.runner import run, run_all


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="runs/alerce_benchmark")
    parser.add_argument("--local_dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default=None, choices=["step", "linear", "cosine", "tanh"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--units", nargs="+", default=[1024, 512], type=int)
    parser.add_argument("--sigma_low", type=float, default=1e-3)
    parser.add_argument("--sigma_high", type=float, default=3.0)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--in_dim", type=int, default=152)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--standarize", action="store_true")
    parser.add_argument("--gmm", action="store_true")
    parser.add_argument("--L", nargs="+", type=parse_L, default=[0.001])
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm", type=str, default="layernorm", choices=["layernorm", "batchnorm", "identity"])
    parser.add_argument("--gradient_clipping", type=float, default=None)
    parser.add_argument("--scheme", type=str, default="Transient")
    parser.add_argument("--outlier", type=str, default="SNIa")
    parser.add_argument("--fold", type=parse_fold, default="all")
    parser.add_argument("--feature_list_pt", type=str, default="data/alerce/data_raw/features_BHRF_model.pkl")
    parser.add_argument("--data_file", type=str, default="data/alerce/data")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--all_outliers", action="store_true")
    return parser


def _resolve_relative_to_package(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parent / path).resolve())


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.feature_list_pt = _resolve_relative_to_package(args.feature_list_pt)
    args.data_file = _resolve_relative_to_package(args.data_file)
    if args.device == "auto":
        args.device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if args.all_outliers:
        run_all(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
