import random

import numpy as np
import torch


def parse_L(value):
    """Parse score-net legacy L values."""
    if isinstance(value, list):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    float_val = float(value)
    if float_val.is_integer():
        return int(float_val)
    return float(float_val)


def parse_fold(value):
    if value == "all":
        return "all"
    return int(value)


def sigma_values_from_L(L, sigma_low, sigma_high):
    """Resolve the multiscale inference sigmas."""
    if isinstance(L, list) and len(L) == 1:
        L = L[0]

    if isinstance(L, int):
        sigmas = np.linspace(sigma_low, sigma_high, L)
    elif isinstance(L, float) and 0 < L < 1:
        sigmas = np.asarray([L], dtype=np.float32)
    else:
        sigmas = np.asarray(L, dtype=float)
    return np.asarray(sigmas, dtype=np.float32)


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_random_state(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def sample_sigmas(batch_size, sigma_low, sigma_high, device):
    return torch.exp(
        torch.empty(batch_size, 1, device=device).uniform_(
            np.log(sigma_low), np.log(sigma_high)
        )
    )


def standarize_array(X, mean, std):
    safe_std = std.copy()
    safe_std[safe_std == 0] = 1.0
    return (X - mean) / safe_std
