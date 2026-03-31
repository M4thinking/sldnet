import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import mixture
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from alerce_benchmark import (
    LossAccumulate,
    get_data,
    get_scheme_outliers,
    get_writer,
    save_source_code,
)
from alerce_benchmark.model import (
    MLPs,
    ScoreOrLogDensityNetwork,
    compute_sldnet_loss,
    sigma_values_from_L,
)
from alerce_benchmark.reporting import write_json

torch.cuda.empty_cache()


def get_tanh_scheduler(optimizer, n_epochs, start_lr=1e-4, end_lr=1e-5):
    def lr_lambda(epoch, steepness=3):
        x = steepness * (epoch - n_epochs / 4) / (n_epochs / 3)
        numerator = end_lr + (start_lr - end_lr) * (1 - torch.tanh(torch.tensor(x)).item()) / 2
        return numerator / start_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Autoencoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.enc1 = nn.Linear(args.in_dim, 512)
        self.encbn1 = nn.LayerNorm(512)
        self.enc2 = nn.Linear(512, 256)
        self.encbn2 = nn.LayerNorm(256)
        self.enc3 = nn.Linear(256, 128)
        self.encbn3 = nn.LayerNorm(128)
        self.enc4 = nn.Linear(128, args.z_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

        self.dec1 = nn.Linear(args.z_dim, 128)
        self.decbn1 = nn.LayerNorm(128)
        self.dec2 = nn.Linear(128, 256)
        self.decbn2 = nn.LayerNorm(256)
        self.dec3 = nn.Linear(256, 512)
        self.decbn3 = nn.LayerNorm(512)
        self.dec4 = nn.Linear(512, args.in_dim)

    def encode(self, x):
        h = F.leaky_relu(self.encbn1(self.enc1(x)))
        h = self.dropout(h)
        h = F.leaky_relu(self.encbn2(self.enc2(h)))
        h = self.dropout(h)
        h = F.leaky_relu(self.encbn3(self.enc3(h)))
        h = self.dropout(h)
        return self.enc4(h)

    def decode(self, x):
        h = F.leaky_relu(self.decbn1(self.dec1(x)))
        h = self.dropout(h)
        h = F.leaky_relu(self.decbn2(self.dec2(h)))
        h = self.dropout(h)
        h = F.leaky_relu(self.decbn3(self.dec3(h)))
        h = self.dropout(h)
        return torch.tanh(self.dec4(h))

    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)

    def compute_loss(self, x):
        _, x_hat = self.forward(x)
        return F.mse_loss(x_hat, x, reduction="mean")


class AnomalyScoreEvaluator:
    def __init__(self, model, device, summary_writer, args, log_path, autoencoder=None):
        self.model = model
        self.autoencoder = autoencoder
        self.device = device
        self.summary_writer = summary_writer
        self.args = args
        self.log_path = log_path
        self.metrics = {}
        self.metrics_path = os.path.join(self.log_path, "metrics.json")
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r") as handle:
                self.metrics = json.load(handle)

    def _save_metrics(self):
        with open(self.metrics_path, "w") as handle:
            json.dump(self.metrics, handle, indent=4)

    def save_args(self, args):
        with open(os.path.join(self.log_path, "args.json"), "w") as handle:
            json.dump(vars(args), handle, indent=4)

    def _add_metric(self, metric_path, value, epoch):
        self.summary_writer.add_scalar(metric_path, value, epoch)
        if str(epoch) not in self.metrics:
            self.metrics[str(epoch)] = {}

        path_parts = metric_path.split("/")
        current_dict = self.metrics[str(epoch)]
        for part in path_parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        current_dict[path_parts[-1]] = float(value)
        self._save_metrics()

    def _add_histogram(self, metric_path, scores, labels, epoch):
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        self.summary_writer.add_histogram(
            f"{metric_path}/normal", scores[labels == 0], epoch
        )
        if np.any(labels == 1):
            self.summary_writer.add_histogram(
                f"{metric_path}/anomaly", scores[labels == 1], epoch
            )

    def calculate_scores(self, dataloader, return_scores_by_sigma=True):
        del return_scores_by_sigma
        self.model.eval()
        scores_by_sigma = {}
        sigma_values = sigma_values_from_L(
            self.args.L, self.args.sigma_low, self.args.sigma_high
        )
        sigma_values = [float(f"{sigma:.6f}") for sigma in sigma_values.tolist()]

        for sigma in sigma_values:
            scores_by_sigma[sigma] = {
                "log_density": [],
                "score_norm": [],
                "score_norm_density_ratio": [],
            }

        for data, _ in dataloader:
            x = data.to(self.device)
            if self.args.pretrain:
                with torch.no_grad():
                    x = self.autoencoder.encode(x)
            x = x.reshape(x.shape[0], -1).requires_grad_()

            for sigma in sigma_values:
                self.model.zero_grad()
                score, log_density = self.model.score(
                    x,
                    sigma * torch.ones((x.shape[0], 1), device=x.device),
                    class_labels=None,
                    p_observe_label=0.0,
                    return_log_density=True,
                    create_graph=False,
                )
                score_squared_norms = (torch.norm(score, dim=1) ** 2).detach().cpu().numpy()
                log_density_np = log_density.detach().cpu().numpy().ravel()
                ratio = score_squared_norms / (np.abs(log_density_np) + 1e-8)
                scores_by_sigma[sigma]["log_density"].extend(log_density_np.tolist())
                scores_by_sigma[sigma]["score_norm"].extend(score_squared_norms.tolist())
                scores_by_sigma[sigma]["score_norm_density_ratio"].extend(ratio.tolist())

        for sigma in sigma_values:
            for score_type in scores_by_sigma[sigma]:
                scores_by_sigma[sigma][score_type] = np.array(scores_by_sigma[sigma][score_type])
        return scores_by_sigma

    def evaluate_aggregate(self, scores_train, scores_test, labels_test, epoch):
        anomaly_score_names = list(scores_train.values())[0].keys()
        test_sigmas = list(sorted(scores_train.keys()))
        auc_roc_aggregate = {}
        auc_pr_aggregate = {}

        for score_type in anomaly_score_names:
            multiscale_data_train = np.asarray(
                [scores_train[sigma][score_type] for sigma in test_sigmas]
            ).T
            multiscale_data_test = np.asarray(
                [scores_test[sigma][score_type] for sigma in test_sigmas]
            ).T
            ms_mean = multiscale_data_train.mean(axis=0)
            ms_std = multiscale_data_train.std(axis=0)
            multiscale_data_test_standarized = (multiscale_data_test - ms_mean) / (
                ms_std + 1e-8
            )

            auc_roc_aggregate[score_type] = {
                "max": roc_auc_score(labels_test, multiscale_data_test_standarized.max(axis=1)),
                "median": roc_auc_score(labels_test, np.median(multiscale_data_test_standarized, axis=1)),
                "mean": roc_auc_score(labels_test, multiscale_data_test_standarized.mean(axis=1)),
            }
            auc_pr_aggregate[score_type] = {
                "max": average_precision_score(labels_test, multiscale_data_test_standarized.max(axis=1)),
                "median": average_precision_score(labels_test, np.median(multiscale_data_test_standarized, axis=1)),
                "mean": average_precision_score(labels_test, multiscale_data_test_standarized.mean(axis=1)),
            }

            if self.args.gmm:
                for components in [1, 3, 5]:
                    gmm = mixture.GaussianMixture(
                        n_components=components, covariance_type="full"
                    ).fit(multiscale_data_train)
                    ll_scores = gmm.score_samples(multiscale_data_test)
                    auc_roc_aggregate[score_type][f"gmm({components})_nll"] = roc_auc_score(
                        labels_test, -ll_scores
                    )
                    auc_pr_aggregate[score_type][f"gmm({components})_nll"] = average_precision_score(
                        labels_test, -ll_scores
                    )

        self._log_aggregate_results(auc_roc_aggregate, auc_pr_aggregate, epoch)
        return auc_roc_aggregate, auc_pr_aggregate

    def evaluate_individual_sigma(self, scores_train, scores_test, labels_test, epoch):
        del scores_train
        results = {
            "log_density": {
                "best_auc_roc": 0,
                "best_auc_pr": 0,
                "best_sigma": 0,
                "all_aucs_roc": [],
                "all_aucs_pr": [],
            },
            "score_norm": {
                "best_auc_roc": 0,
                "best_auc_pr": 0,
                "best_sigma": 0,
                "all_aucs_roc": [],
                "all_aucs_pr": [],
            },
            "score_norm_density_ratio": {
                "best_auc_roc": 0,
                "best_auc_pr": 0,
                "best_sigma": 0,
                "all_aucs_roc": [],
                "all_aucs_pr": [],
            },
        }

        for sigma in sorted(scores_test.keys()):
            for score_type in ["score_norm", "log_density", "score_norm_density_ratio"]:
                scores = np.asarray(scores_test[sigma][score_type])
                auc_roc = roc_auc_score(labels_test, scores)
                auc_pr = average_precision_score(labels_test, scores)
                results[score_type]["all_aucs_roc"].append(auc_roc)
                results[score_type]["all_aucs_pr"].append(auc_pr)

                if auc_roc > results[score_type]["best_auc_roc"]:
                    results[score_type]["best_auc_roc"] = auc_roc
                    results[score_type]["best_sigma"] = sigma
                if auc_pr > results[score_type]["best_auc_pr"]:
                    results[score_type]["best_auc_pr"] = auc_pr

                self._add_metric(f"roc_auc_{score_type}_individual/sigma_{sigma}", auc_roc, epoch)
                self._add_metric(f"pr_auc_{score_type}_individual/sigma_{sigma}", auc_pr, epoch)
                self._add_histogram(
                    f"histogram_{score_type}_individual/sigma_{sigma}",
                    scores,
                    labels_test,
                    epoch,
                )
        self._log_individual_results(results, scores_test, epoch)
        return results

    def _log_aggregate_results(self, auc_roc_aggregate, auc_pr_aggregate, epoch):
        for score_type in auc_roc_aggregate.keys():
            best_auc_roc = max(auc_roc_aggregate[score_type].values())
            best_auc_pr = max(auc_pr_aggregate[score_type].values())
            for agg_type, current_auc in auc_roc_aggregate[score_type].items():
                self._add_metric(f"roc_auc_{score_type}_aggregate/{agg_type}", current_auc, epoch)
            for agg_type, current_auc in auc_pr_aggregate[score_type].items():
                self._add_metric(f"pr_auc_{score_type}_aggregate/{agg_type}", current_auc, epoch)
            self._add_metric(f"_roc_auc_best/_best_{score_type}_aggregate", best_auc_roc, epoch)
            self._add_metric(f"_pr_auc_best/_best_{score_type}_aggregate", best_auc_pr, epoch)

    def _log_individual_results(self, results, scores_test, epoch):
        del scores_test
        for score_type, values in results.items():
            self._add_metric(
                f"roc_auc_{score_type}_individual/best", values["best_auc_roc"], epoch
            )
            self._add_metric(
                f"pr_auc_{score_type}_individual/best", values["best_auc_pr"], epoch
            )


def initialize_model(args, input_dim):
    model = ScoreOrLogDensityNetwork(
        score_network=False,
        net=MLPs(
            input_dim=input_dim + 2,
            units=args.units,
            dropout=args.dropout,
            norm=args.norm,
            activation=args.activation,
            output_dim=args.out_dim,
        ),
    ).to(args.device)

    if args.gradient_clipping:
        clip_value = args.gradient_clipping
        for parameter in model.parameters():
            parameter.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=1e-5
    )
    if args.scheduler == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1 / 3,
            total_iters=args.warmup_epochs if args.warmup_epochs > 0 else 100,
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.9
        )
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )
    elif args.scheduler == "tanh":
        scheduler = get_tanh_scheduler(optimizer, args.epochs, start_lr=args.lr, end_lr=1e-5)
    else:
        scheduler = None

    log_path, writer = get_writer(
        root_dir_runs=os.path.join(args.local_dir, "runs"),
        run_name=args.run_name,
        scheme=args.scheme,
        outlier=args.outlier.replace("/", "-"),
        fold=args.fold,
        args=args,
    )
    save_source_code(log_path)
    return model, optimizer, scheduler, writer, log_path


def preprocess_data(
    args,
    data_train,
    data_val,
    data_test,
    labels_train,
    labels_val,
    labels_test,
    sampler_train,
    sampler_val,
):
    data_train = torch.Tensor(data_train)
    data_val = torch.Tensor(data_val)
    data_test = torch.Tensor(data_test)
    labels_train = torch.Tensor(labels_train)
    labels_val = torch.Tensor(labels_val)
    labels_test = torch.Tensor(labels_test)

    if args.standarize:
        data_train_mean = data_train.mean(dim=0)
        data_train_std = data_train.std(dim=0)
        data_train_std[data_train_std == 0] = 1
        data_train = (data_train - data_train_mean) / data_train_std
        data_val = (data_val - data_train_mean) / data_train_std
        data_test = (data_test - data_train_mean) / data_train_std

    dataset_train = TensorDataset(data_train, labels_train)
    dataset_val = TensorDataset(data_val, labels_val)
    dataset_test = TensorDataset(data_test, labels_test)

    common_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": str(args.device).startswith("cuda"),
    }
    if args.num_workers > 0:
        common_args["persistent_workers"] = True

    dataloader_train = DataLoader(dataset_train, sampler=sampler_train, **common_args)
    dataloader_val = DataLoader(dataset_val, sampler=sampler_val, **common_args)
    dataloader_test = DataLoader(dataset_test, shuffle=False, **common_args)

    return dataloader_train, dataloader_val, dataloader_test, data_train.shape[1]


def train_step(model, optimizer, scheduler, writer, dataloader, epoch, args, stage="train", autoencoder=None):
    model.train() if stage == "train" else model.eval()
    loss_accumulate = LossAccumulate()

    grad_context = torch.enable_grad() if stage == "train" else torch.enable_grad()
    with grad_context:
        for batch_idx, (data, labels) in enumerate(dataloader):
            x = data.to(args.device)
            labels = labels.to(args.device)

            if autoencoder is not None:
                x = autoencoder.encode(x)
            x = x.reshape(x.shape[0], -1)

            loss, metrics = compute_sldnet_loss(
                model,
                x,
                sigma_low=args.sigma_low,
                sigma_high=args.sigma_high,
                beta=args.beta,
                # labels=labels, # This can be commented out to condition over labels during training
                stage=stage,
            )

            loss_accumulate["loss_dsm"].append(metrics["loss_dsm"])
            loss_accumulate["loss_regularizer"].append(metrics["loss_regularizer"])
            loss_accumulate["score_norm"].append(metrics["score_norm"])
            loss_accumulate["log_density"].append(metrics["log_density"])
            loss_accumulate["loss_dsm_reg"].append(loss.item())

            if stage == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_grads = [
                torch.max(parameter.grad).item()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if all_grads:
                writer.add_scalar(
                    f"gradients/{stage}_max_gradient",
                    max(all_grads),
                    epoch * len(dataloader) + batch_idx,
                )
                writer.add_scalar(
                    f"gradients/{stage}_min_gradient",
                    min(all_grads),
                    epoch * len(dataloader) + batch_idx,
                )

    if stage == "train" and scheduler is not None:
        scheduler.step()
        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)

    for key, values in list(loss_accumulate.items()):
        mean_value = np.asarray(values).mean()
        loss_accumulate[key] = mean_value
        if "loss" in key:
            writer.add_scalar(f"loss_{stage}/{key}", mean_value, epoch)
        else:
            writer.add_scalar(f"values_{stage}/{key}", mean_value, epoch)
    return loss_accumulate


def pretrain_step(autoencoder, optimizer, writer, dataloader, epoch, stage="train", device="cpu"):
    autoencoder.train() if stage == "train" else autoencoder.eval()
    loss_accumulate = LossAccumulate()
    with torch.set_grad_enabled(stage == "train"):
        for data, _ in dataloader:
            x = data.to(device)
            loss = autoencoder.compute_loss(x)
            loss_accumulate["loss"].append(loss.item())
            if stage == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    if loss_accumulate["loss"]:
        writer.add_scalar(
            f"pretrain_loss_{stage}/loss",
            np.asarray(loss_accumulate["loss"]).mean(),
            epoch,
        )


def print_results(results_individual, auc_roc_aggregate):
    for outer_key, outer_value in results_individual.items():
        for inner_key, inner_value in outer_value.items():
            if isinstance(inner_value, list):
                print(f"{outer_key} - {inner_key}: {[round(x * 100, 4) for x in inner_value]}")
            elif inner_key == "best_sigma":
                print(f"{outer_key} - {inner_key}: {inner_value}")
            else:
                print(f"{outer_key} - {inner_key}: {inner_value * 100:.2f}")

    for outer_key, outer_value in auc_roc_aggregate.items():
        for inner_key, inner_value in outer_value.items():
            print(f"{outer_key} - {inner_key}: {inner_value * 100:.2f}")


def write_run_summary(log_path, args, results):
    summary = {
        "benchmark": "alerce",
        "scheme": args.scheme,
        "outlier": args.outlier,
        "fold": args.fold,
        "run_name": args.run_name,
        "last_epoch": results["last_epoch"],
        "best_individual": {
            score_type: {
                "best_auc_roc": float(values["best_auc_roc"]),
                "best_auc_pr": float(values["best_auc_pr"]),
                "best_sigma": float(values["best_sigma"]) if isinstance(values["best_sigma"], (int, float, np.floating)) else values["best_sigma"],
            }
            for score_type, values in results["individual"].items()
        },
        "aggregate_roc": {
            score_type: {agg: float(value) for agg, value in values.items()}
            for score_type, values in results["aggregate"].items()
        },
    }
    write_json(os.path.join(log_path, "run_summary.json"), summary)


def run_single(args):
    print(f"Running experiment for outlier type: {args.outlier} (fold {args.fold})")
    data = get_data(args)
    train_data, train_labels, val_data, val_labels, test_data, test_labels, sampler_train, sampler_val = data
    train_loader, val_loader, test_loader, n_features = preprocess_data(
        args,
        train_data,
        val_data,
        test_data,
        train_labels,
        val_labels,
        test_labels,
        sampler_train,
        sampler_val,
    )

    args.in_dim = n_features
    in_dim = args.z_dim if args.pretrain else n_features
    model, optimizer, scheduler, writer, log_path = initialize_model(args, in_dim)

    autoencoder = None
    if args.pretrain:
        autoencoder = Autoencoder(args).to(args.device)
        ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        for epoch in tqdm(range(args.pretrain_epochs), desc="Pretraining", unit="epochs", leave=False):
            pretrain_step(
                autoencoder,
                ae_optimizer,
                writer,
                train_loader,
                epoch,
                stage="train",
                device=args.device,
            )
            if epoch % 10 == 0:
                pretrain_step(
                    autoencoder,
                    ae_optimizer,
                    writer,
                    val_loader,
                    epoch,
                    stage="val",
                    device=args.device,
                )

    evaluator = AnomalyScoreEvaluator(model, args.device, writer, args, log_path, autoencoder)
    evaluator.save_args(args)

    best_model_state = copy.deepcopy(model.state_dict())
    patience = 100
    patience_counter = 0

    for epoch in tqdm(range(args.epochs + 1), desc="Training", unit="epochs", leave=False):
        train_loss = train_step(
            model, optimizer, scheduler, writer, train_loader, epoch, args, stage="train", autoencoder=autoencoder
        )
        val_loss = train_step(
            model, optimizer, scheduler, writer, val_loader, epoch, args, stage="val", autoencoder=autoencoder
        )
        writer.add_scalar(
            "loss_diff/log_density",
            abs(val_loss["log_density"] - train_loss["log_density"]),
            epoch,
        )

        if abs(val_loss["log_density"] - train_loss["log_density"]) < 1:
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            scores_test = evaluator.calculate_scores(test_loader, return_scores_by_sigma=True)
            scores_train = evaluator.calculate_scores(train_loader, return_scores_by_sigma=True)
            auc_roc_aggregate, auc_pr_aggregate = evaluator.evaluate_aggregate(
                scores_train, scores_test, test_labels, epoch
            )
            results_individual = evaluator.evaluate_individual_sigma(
                scores_train, scores_test, test_labels, epoch
            )
            if args.verbose:
                print("\nMétricas individuales:")
                print_results(results_individual, auc_roc_aggregate)
            if args.plot:
                plot_auc_roc_comparison(auc_roc_aggregate, writer, epoch)
                plot_individual_sigma(results_individual, writer, epoch)

        if patience_counter > patience:
            break

    results = {
        "aggregate": auc_roc_aggregate,
        "aggregate_pr": auc_pr_aggregate,
        "individual": results_individual,
        "metrics": evaluator.metrics,
        "last_epoch": epoch,
    }
    print_results(results_individual, auc_roc_aggregate)
    torch.save(best_model_state, os.path.join(log_path, "best_model.pth"))
    write_run_summary(log_path, args, results)
    writer.close()
    return results


def run(args):
    if args.fold == "all":
        fold_results = {}
        original_fold = args.fold
        for fold in range(5):
            args.fold = fold
            fold_results[fold] = run_single(args)
        args.fold = original_fold
        return fold_results
    return run_single(args)


def run_all(args):
    outliers = get_scheme_outliers(args.scheme)
    if not outliers:
        raise ValueError(f"Scheme '{args.scheme}' no válido o no tiene outliers definidos")

    results_by_outlier = {}
    original_outlier = args.outlier
    for current_outlier in outliers:
        if args.fold == "all":
            fold_results = {}
            original_fold = args.fold
            for fold in range(5):
                print(f"\nRunning {current_outlier} - Fold {fold}")
                args.fold = fold
                args.outlier = current_outlier
                fold_results[fold] = run_single(args)
            args.fold = original_fold
            results_by_outlier[current_outlier] = fold_results
        else:
            print(f"\nRunning {current_outlier} - Fold {args.fold}")
            args.outlier = current_outlier
            results_by_outlier[current_outlier] = run(args)

    args.outlier = original_outlier
    return results_by_outlier


def plot_auc_roc_comparison(auc_roc_aggregate, writer, epoch, figsize=(7, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    for score_type, values in auc_roc_aggregate.items():
        ax.bar(
            np.arange(len(values)) + 0.2 * list(auc_roc_aggregate.keys()).index(score_type),
            list(values.values()),
            width=0.2,
            label=score_type,
        )
    ax.set_xticks(np.arange(len(list(auc_roc_aggregate.values())[0])))
    ax.set_xticklabels(list(list(auc_roc_aggregate.values())[0].keys()), rotation=45)
    ax.set_ylim(0, 1)
    ax.legend()
    writer.add_figure("aggregate_auc_roc", fig, epoch)
    plt.close(fig)


def plot_individual_sigma(results_individual, writer, epoch, figsize=(7, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    for score_type, values in results_individual.items():
        ax.plot(values["all_aucs_roc"], label=f"{score_type} roc")
    ax.legend()
    writer.add_figure("individual_sigma_auc_roc", fig, epoch)
    plt.close(fig)
