import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer
from tqdm.auto import tqdm


class MLPs(nn.Module):
    """Simple MLP used by the score-based generative model."""

    def __init__(
        self,
        input_dim,
        output_dim=1,
        units=(1024, 512),
        norm="layernorm",
        activation="gelu",
        dropout=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        if activation == "gelu":
            act_layer = nn.GELU()
        elif activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "leaky_relu":
            act_layer = nn.LeakyReLU()
        else:
            act_layer = nn.Identity()

        if norm == "layernorm":
            norm_layer = nn.LayerNorm
        elif norm == "batchnorm":
            norm_layer = nn.BatchNorm1d
        else:
            norm_layer = nn.Identity

        in_dim = input_dim
        for out_dim in units:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    norm_layer(out_dim),
                    act_layer,
                    nn.Dropout(dropout) if dropout else nn.Identity(),
                )
            )
            in_dim = out_dim

        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        if x.shape[1] > 1:
            x = x.sum(dim=1, keepdim=True)
        return x


class ScoreOrLogDensityNetwork(nn.Module):
    """Noise-conditioned network that outputs log-density and its score."""

    def __init__(self, net):
        super().__init__()
        self.network = net

    def forward(self, x):
        return self.network(x)

    def score(
        self,
        x,
        sigma,
        class_labels=None,
        p_observe_label=0.0,
        return_log_density=False,
        create_graph=False,
    ):
        batch_size = x.shape[0]
        device = x.device

        if class_labels is not None:
            mask = torch.rand(batch_size, device=device) < p_observe_label
            labels = torch.where(mask, class_labels, torch.tensor(-1, device=device))
        else:
            labels = torch.full((batch_size,), -1, device=device)

        conditioned_input = torch.cat([x, sigma, labels.float().unsqueeze(1)], dim=1)
        log_density = self.network(conditioned_input)
        logp = -log_density.sum()
        score = torch.autograd.grad(
            logp, x, create_graph=create_graph, retain_graph=create_graph
        )[0]

        if return_log_density:
            return score, log_density
        return score


class SBGMAnomalyDetector(BaseEstimator, OutlierMixin):
    """Score-based generative model (SBGM) with sklearn-compatible API."""

    def __init__(
        self,
        epochs=1000,
        lr=1e-4,
        batch_size=256,
        sigma_low=1e-3,
        sigma_high=3.0,
        L=8,
        beta=1e-3,
        units=(1024, 512),
        dropout=0.1,
        norm="layernorm",
        activation="gelu",
        gradient_clipping=None,
        standarize=True,
        device="auto",
        betas=(0.5, 0.9),
        weight_decay=1e-5,
        gmm_component=0,
        random_state=42,
        metric_agg="mean",
        qft=None,
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.L = L
        self.beta = beta
        self.units = units
        self.dropout = dropout
        self.norm = norm
        self.activation = activation
        self.gradient_clipping = gradient_clipping
        self.standarize = standarize
        self.device = device
        self.betas = betas
        self.weight_decay = weight_decay
        self.gmm_component = gmm_component
        self.random_state = random_state
        self.metric_agg = metric_agg
        self.qft = qft

    def fit(self, X, y=None):
        self._set_random_state()
        X = np.asarray(X, dtype=np.float32)
        X = self._apply_qft(X, fit=True)
        self.mean_ = X.mean(axis=0) if self.standarize else None
        self.std_ = X.std(axis=0) if self.standarize else None
        if self.standarize:
            std = self.std_.copy()
            std[std == 0] = 1.0
            X = (X - self.mean_) / std

        device = self._resolve_device()
        x_tensor = torch.from_numpy(X)
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X.shape[1]
        model_net = MLPs(
            input_dim=input_dim + 2,
            output_dim=1,
            units=self.units,
            norm=self.norm,
            activation=self.activation,
            dropout=self.dropout,
        )
        self.model_ = ScoreOrLogDensityNetwork(model_net).to(device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay  
        )

        for _ in tqdm(range(self.epochs), leave=False, desc="Training SBGM"):
            self.model_.train()
            for batch in loader:
                x = batch[0].to(device)
                sigma = torch.exp(
                    torch.empty(x.size(0), 1, device=device).uniform_(
                        np.log(self.sigma_low), np.log(self.sigma_high)
                    )
                )
                noise = torch.randn_like(x) * sigma
                x_noisy = (x + noise).requires_grad_()
                x_clean = x.detach().requires_grad_()

                score, log_density = self.model_.score(
                    x_noisy,
                    sigma,
                    class_labels=None,
                    p_observe_label=0.0,
                    return_log_density=True,
                    create_graph=True,
                )

                loss_dsm = (
                    (sigma**2).ravel()
                    * torch.norm(score + noise / (sigma**2), dim=1) ** 2
                ).mean() / 2.0

                loss_reg = 0.0
                if self.beta:
                    _, log_density_nf = self.model_.score(
                        x_clean,
                        sigma,
                        class_labels=None,
                        p_observe_label=0.0,
                        return_log_density=True,
                        create_graph=True,
                    )
                    loss_reg = self.beta * (log_density_nf**2).mean() / 2.0

                loss = loss_dsm + loss_reg

                optimizer.zero_grad()
                loss.backward()
                if self.gradient_clipping is not None:
                    nn.utils.clip_grad_norm_(
                        self.model_.parameters(), self.gradient_clipping
                    )
                optimizer.step()

        self.model_.eval()
        self.sigma_values_ = self._sigma_values()
        train_scores = self._compute_scores(X, device)
        self.score_mean_ = train_scores.mean(axis=0)
        self.score_std_ = train_scores.std(axis=0) + 1e-8
        if self.gmm_component is not None and self.gmm_component >= 1:
            self.gmm_ = GaussianMixture(
                n_components=self.gmm_component,
                covariance_type="full",
                random_state=self.random_state,
            ).fit(train_scores)
        return self

    def decision_function(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("The SBGM model is not fitted.")

        X = np.asarray(X, dtype=np.float32)
        X = self._apply_qft(X, fit=False)
        if self.standarize and self.mean_ is not None and self.std_ is not None:
            std = self.std_.copy()
            std[std == 0] = 1.0
            X = (X - self.mean_) / std
        device = self._resolve_device()
        test_scores = self._compute_scores(X, device)
        if self.gmm_component is not None and self.gmm_component >= 1:
            if not hasattr(self, "gmm_"):
                raise RuntimeError("The SBGM GMM model is not fitted.")
            return -self.gmm_.score_samples(test_scores)
        standarized = (test_scores - self.score_mean_) / self.score_std_
        if self.metric_agg == "min":
            return np.min(standarized, axis=1)
        elif self.metric_agg == "median":
            return np.median(standarized, axis=1)
        elif self.metric_agg == "max":
            return np.max(standarized, axis=1)
        else: # mean
            return np.mean(standarized, axis=1)

    def _compute_scores(self, X, device):
        X_copy = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(X_copy).to(device)
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_scores = []
        for batch in loader:
            x = batch[0].to(device).requires_grad_()
            batch_scores = []
            for sigma_ in self.sigma_values_:
                sigma = torch.full((x.size(0), 1), sigma_, device=device)
                _, log_density = self.model_.score(
                    x,
                    sigma,
                    class_labels=None,
                    p_observe_label=0.0,
                    return_log_density=True,
                    create_graph=False,
                )
                log_density_np = log_density.detach().cpu().numpy().ravel()
                batch_scores.append(log_density_np)
            batch_scores = np.stack(batch_scores, axis=1) if len(batch_scores) > 1 else batch_scores[0][:, np.newaxis]
            all_scores.append(batch_scores)

        return np.vstack(all_scores)

    def _sigma_values(self):
        if isinstance(self.L, int):
            sigmas = np.linspace(self.sigma_low, self.sigma_high, self.L)
        elif isinstance(self.L, float) and 0 < self.L < 1: # use just 1 sigma
            sigmas = [self.L]
        else:
            sigmas = np.asarray(self.L, dtype=float)
        return np.asarray(sigmas, dtype=np.float32)

    def _apply_qft(self, X, fit=False):
        if self.qft is None:
            return X
        qft_mode = self.qft.lower()
        if qft_mode == "normal":
            output_distribution = "normal"
        elif qft_mode == "uniform":
            output_distribution = "uniform"
        else:
            raise ValueError("qft must be None, 'uniform', or 'normal'")

        X_clean = np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan)
        if fit or not hasattr(self, "qft_"):
            n_quantiles = min(1000, X_clean.shape[0])
            self.qft_ = QuantileTransformer(
                output_distribution=output_distribution,
                n_quantiles=n_quantiles,
                random_state=self.random_state,
                subsample=None,
            )
            self.qft_.fit(X_clean)
    
        X_transform = self.qft_.transform(X_clean)
        if self.qft.lower() == "normal":
            X_transform[np.isnan(X)] = -3.0
        else: # uniform
            X_transform[np.isnan(X)] = -1.0
            
        return X_transform

    def _resolve_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_random_state(self):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
