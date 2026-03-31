import datetime
import glob
import os
import pathlib
from abc import ABC, abstractmethod
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from alerce_benchmark.utils import parse_L, parse_fold

banned_features = [
    "mean_mag_1",
    "mean_mag_2",
    "min_mag_1",
    "min_mag_2",
    "Mean_1",
    "Mean_2",
    "n_det_1",
    "n_det_2",
    "n_pos_1",
    "n_pos_2",
    "n_neg_1",
    "n_neg_2",
    "first_mag_1",
    "first_mag_2",
    "MHPS_non_zero_1",
    "MHPS_non_zero_2",
    "MHPS_PN_flag_1",
    "MHPS_PN_flag_2",
    "W1",
    "W2",
    "W3",
    "W4",
    "iqr_1",
    "iqr_2",
    "delta_mjd_fid_1",
    "delta_mjd_fid_2",
    "last_mjd_before_fid_1",
    "last_mjd_before_fid_2",
    "g-r_ml",
    "MHAOV_Period_1",
    "MHAOV_Period_2",
]


class FeaturePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, features):
        raise NotImplementedError


class QuantileFeaturePreprocessor(FeaturePreprocessor):
    def __init__(self, output_distribution="uniform"):
        self.banned_features = banned_features
        self.transformer = QuantileTransformer(output_distribution=output_distribution)
        self.output_distribution = output_distribution

    def fit(self, features):
        allowed_features = self._remove_banned_features(features)
        self.transformer.fit(allowed_features.values)

    def preprocess(self, features):
        allowed_features = self._remove_banned_features(features)
        x = self.transformer.transform(allowed_features.values)
        if self.output_distribution == "normal":
            x = np.nan_to_num(x, nan=-4.0)
        else:
            x = np.nan_to_num(x, nan=-1.0)
        return x

    def _remove_banned_features(self, features):
        allowed_features = self.get_allowed_features(features.columns.values)
        return features[allowed_features].copy()

    def get_allowed_features(self, feature_list):
        return [feature for feature in feature_list if feature not in self.banned_features]


class LossAccumulate:
    def __init__(self):
        self.losses = {}

    def __getitem__(self, key):
        if key not in self.losses:
            self.losses[key] = []
        return self.losses[key]

    def __setitem__(self, key, value):
        self.losses[key] = value

    def items(self):
        return self.losses.items()


def get_writer(root_dir_runs, run_name, scheme, outlier, fold, postfix=None, args=None):
    now = datetime.datetime.now()
    timestamp = "_".join(
        map(
            lambda value: str(value).zfill(2),
            [now.year, now.month, now.day, now.hour, now.minute, now.second],
        )
    )
    log_path = os.path.join(root_dir_runs, run_name, scheme, outlier, str(fold), timestamp)
    if postfix:
        log_path += f"_{postfix}"
    summary_writer = SummaryWriter(log_path)
    if args is not None:
        config_parameters = "_".join(
            [f"{arg}_{getattr(args, arg)}" for arg in vars(args)]
        )
        summary_writer.add_text("config parameters", config_parameters, 0)
    return log_path, summary_writer


def save_source_code(log_path):
    code_path = os.path.join(log_path, "code")
    for search_folder, file_extension in [
        ("", "*.py"),
        ("", "*.ipynb"),
        ("", "*.sh"),
        ("./src/", "*.py"),
    ]:
        file_names = glob.glob(os.path.join(search_folder, file_extension))
        if not file_names:
            continue
        pathlib.Path(os.path.join(code_path, search_folder)).mkdir(
            parents=True, exist_ok=True
        )
        for file_name in file_names:
            copyfile(file_name, f"{code_path}/{file_name}")


def sample_outliers(data, outlier, target_ratio=0.1):
    outlier_proportion = data[data.classALeRCE == outlier].shape[0] / data.shape[0]
    if outlier_proportion > target_ratio:
        n_remove = round(
            data[data.classALeRCE == outlier].shape[0] - data.shape[0] * target_ratio
        )
        subset = data[data.classALeRCE == outlier].sample(n_remove)
        return data.drop(subset.index)

    n_remove = round(
        (data.shape[0] * target_ratio - data[data.classALeRCE == outlier].shape[0])
        / target_ratio
    )
    subset = data[data.classALeRCE != outlier].sample(n_remove)
    return data.drop(subset.index)


def weighted_sampler(data, class_):
    class_sample_count = np.unique(class_, return_counts=True)[1]
    weights = 1.0 / torch.Tensor(class_sample_count)
    samples_weight = np.array([weights[target] for target in class_])
    samples_weight = torch.from_numpy(samples_weight)
    return WeightedRandomSampler(
        samples_weight.type(torch.DoubleTensor),
        len(samples_weight),
    )


def map2numerical(labels):
    labels_mapped = labels
    inv_map = {}
    for index, class_name in enumerate(np.unique(labels)):
        labels_mapped = np.where(labels == class_name, index, labels_mapped)
        inv_map[index] = class_name
    return inv_map, labels_mapped.astype("int8")


def get_scheme_outliers(scheme):
    scheme_mapping = {
        "Transient": ["SLSN", "SNII", "SNIa", "SNIbc"],
        "Stochastic": ["AGN", "Blazar", "CV/Nova", "QSO", "YSO"],
        "Periodic": ["CEP", "DSCT", "E", "RRL", "LPV"],
    }
    return scheme_mapping.get(scheme, [])


def get_data(args):
    feature_list = pd.read_pickle(args.feature_list_pt)
    train = pd.read_pickle(os.path.join(args.data_file, "train_data_filtered.pkl"))
    test = pd.read_pickle(os.path.join(args.data_file, "test_data_filtered.pkl"))

    train = train[train.hierClass == args.scheme]
    test = test[test.hierClass == args.scheme]

    test = pd.concat([test, train[train.classALeRCE == args.outlier]], sort=False)
    test = sample_outliers(test, args.outlier)
    train = train[train.classALeRCE != args.outlier]

    fold_ixs = pd.read_pickle(os.path.join(args.data_file, f"fold_{args.fold}_ixs.pkl"))
    val = train[train.index.isin(fold_ixs) == False]
    train = train[train.index.isin(fold_ixs)]

    _, train_labels = map2numerical(train.classALeRCE)
    sampler_train = weighted_sampler(train, train_labels)
    _, val_labels = map2numerical(val.classALeRCE)
    sampler_val = weighted_sampler(val, val_labels)

    feature_preprocessor = QuantileFeaturePreprocessor()
    feature_preprocessor.fit(train[feature_list])
    train_features = feature_preprocessor.preprocess(train[feature_list])
    val_features = feature_preprocessor.preprocess(val[feature_list])
    test_features = feature_preprocessor.preprocess(test[feature_list])

    train_labels = np.zeros(train.shape[0]).astype("int8")
    val_labels = np.zeros(val.shape[0]).astype("int8")
    test_labels = np.where(test["classALeRCE"] != args.outlier, 0, test["classALeRCE"])
    test_labels = np.where(test["classALeRCE"] == args.outlier, 1, test_labels)
    test_labels = test_labels.reshape(-1).astype("int8")

    return (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        sampler_train,
        sampler_val,
    )


__all__ = [
    "LossAccumulate",
    "QuantileFeaturePreprocessor",
    "banned_features",
    "get_data",
    "get_scheme_outliers",
    "get_writer",
    "map2numerical",
    "parse_L",
    "parse_fold",
    "sample_outliers",
    "save_source_code",
    "weighted_sampler",
]
