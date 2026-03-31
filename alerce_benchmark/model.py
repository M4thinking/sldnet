import torch
import torch.nn as nn

from alerce_benchmark import sample_sigmas


class MLPs(nn.Module):
    """Shared MLP backbone for SLDNet."""

    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        units=(1024, 512),
        norm="layernorm",
        activation="gelu",
        dropout=0.1,
        last_activation=nn.Identity(),
        skip_connections=False,
        init_weights=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.skip_connections = skip_connections
        self.norm = norm

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Identity()

        if norm == "layernorm":
            norm = nn.LayerNorm
        elif norm == "batchnorm":
            norm = nn.BatchNorm1d
        else:
            norm = nn.Identity

        def block(in_, out_):
            layers = [
                nn.Linear(in_, out_),
                norm(out_),
                self.activation,
                nn.Dropout(dropout) if dropout else nn.Identity(),
            ]
            return nn.Sequential(*layers)

        in_dim = input_dim
        for out_dim in units:
            self.layers.append(block(in_dim, out_dim))
            in_dim = out_dim

        self.output_layer = nn.Linear(in_dim, output_dim)
        self.last_activation = last_activation
        if init_weights:
            self._init_weights()

    def _init_weights(self):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if isinstance(self.activation, nn.GELU):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                elif isinstance(self.activation, nn.ReLU):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif isinstance(self.activation, nn.LeakyReLU):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        prev_x = x
        for layer in self.layers:
            x = layer(x)
            if self.skip_connections:
                if prev_x.shape[-1] == x.shape[-1]:
                    x = x + prev_x
                prev_x = x
        x = self.output_layer(x)
        x = self.last_activation(x)
        if x.shape[1] > 1:
            x = x.sum(dim=1, keepdim=True)
        return x


class ScoreOrLogDensityNetwork(nn.Module):
    """Noise-conditioned scalar potential whose gradient defines the score."""

    def __init__(self, net, score_network=False):
        super().__init__()
        self.network = net
        self.is_score_network = score_network

    def forward(self, x):
        return self.network(x)

    def score(
        self,
        x,
        sigma,
        class_labels=None,
        p_observe_label=0.5,
        return_log_density=False,
        create_graph=True,
    ):
        batch_size = x.shape[0]
        device = x.device

        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(1)

        if class_labels is not None:
            class_labels = class_labels.to(device)
            mask = torch.rand(batch_size, device=device) < p_observe_label
            masked_labels = torch.where(
                mask, class_labels, torch.tensor(-1, device=device)
            )
        else:
            masked_labels = torch.full((batch_size,), -1, device=device)

        conditioned_input = torch.cat(
            [x, sigma, masked_labels.float().unsqueeze(1)], dim=1
        )

        if self.is_score_network:
            score = self.network(conditioned_input)
            log_density = (
                torch.zeros_like(score[:, 0][:, None]) if return_log_density else None
            )
        else:
            log_density = self.network(conditioned_input)
            logp = -log_density.sum()
            score = torch.autograd.grad(
                logp,
                x,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]

        if return_log_density:
            return score, log_density
        return score


def compute_sldnet_loss(
    model,
    x,
    sigma_low,
    sigma_high,
    beta=0.0,
    labels=None,
    stage="train",
    label_observe_train=0.2,
    label_observe_clean_train=0.5,
):
    """Shared denoising score matching loss."""
    device = x.device
    sigma = sample_sigmas(x.size(0), sigma_low, sigma_high, device)
    noise = torch.randn_like(x, device=device) * sigma
    x = x.requires_grad_()
    x_noisy = x + noise

    create_graph = stage == "train"
    score, log_density = model.score(
        x_noisy,
        sigma,
        class_labels=labels,
        p_observe_label=label_observe_train if stage == "train" else 0.0,
        return_log_density=True,
        create_graph=create_graph,
    )

    lambda_factor = (sigma**2).squeeze(1)
    loss_dsm = 0.5 * (
        lambda_factor * torch.sum((score + noise / (sigma**2)) ** 2, dim=-1)
    ).mean()

    loss_regularizer = torch.tensor(0.0, device=device)
    if beta:
        _, log_density_clean = model.score(
            x,
            sigma,
            class_labels=labels,
            p_observe_label=label_observe_clean_train if stage == "train" else 0.0,
            return_log_density=True,
            create_graph=create_graph,
        )
        loss_regularizer = 0.5 * beta * (log_density_clean**2).mean()

    loss = loss_dsm + loss_regularizer
    return loss, {
        "loss_dsm": loss_dsm.detach().item(),
        "loss_regularizer": loss_regularizer.detach().item(),
        "score_norm": (
            (lambda_factor * (torch.norm(score, dim=1) ** 2)).mean().detach().item()
        ),
        "log_density": log_density.mean().detach().item(),
    }
