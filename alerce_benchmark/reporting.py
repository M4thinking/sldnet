import json
import os

import numpy as np


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def write_json(path, data):
    ensure_parent_dir(path)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def maybe_float(value):
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return value
