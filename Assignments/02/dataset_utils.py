import os
from dataclasses import dataclass

import numpy as np


@dataclass
class ToyDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def lhs_sample(n_samples: int, n_dims: int, low: float, high: float, seed: int) -> np.ndarray:
    """Simple Latin Hypercube Sampling implementation without external deps."""
    rng = np.random.default_rng(seed)
    result = np.empty((n_samples, n_dims), dtype=np.float64)
    cut = np.linspace(0.0, 1.0, n_samples + 1)

    for d in range(n_dims):
        u = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u)
        result[:, d] = u

    return low + (high - low) * result


def additive_function(xyzt: np.ndarray) -> np.ndarray:
    x = xyzt[:, 0:1]
    y = xyzt[:, 1:2]
    t = xyzt[:, 2:3]
    z = xyzt[:, 3:4]
    return np.exp(-0.5 * x) + np.log1p(np.exp(0.4 * y)) + np.tanh(t) + np.sin(z) - 0.4


def multiplicative_function(xyzt: np.ndarray) -> np.ndarray:
    x = xyzt[:, 0:1]
    y = xyzt[:, 1:2]
    t = xyzt[:, 2:3]
    z = xyzt[:, 3:4]

    fx = np.exp(-0.3 * x)
    fy = (0.15 * y) ** 2
    ft = np.tanh(0.3 * t)
    fz = 0.2 * np.sin(0.5 * z + 2.0) + 0.5
    return fx * fy * fz * ft


def make_toy_dataset(problem: str, seed: int, n_train: int = 500, n_test: int = 5000) -> ToyDataset:
    if problem not in {"additive", "multiplicative"}:
        raise ValueError("problem must be either 'additive' or 'multiplicative'")

    train_x = lhs_sample(n_train, n_dims=4, low=0.0, high=4.0, seed=seed)

    if problem == "additive":
        test_high = 6.0
        fn = additive_function
    else:
        test_high = 10.0
        fn = multiplicative_function

    test_x = lhs_sample(n_test, n_dims=4, low=0.0, high=test_high, seed=seed + 1000)

    train_y = fn(train_x)
    test_y = fn(test_x)

    return ToyDataset(x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)


def save_dataset_csv(dataset: ToyDataset, out_dir: str, prefix: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    train = np.hstack([dataset.x_train, dataset.y_train])
    test = np.hstack([dataset.x_test, dataset.y_test])

    train_header = "x,y,t,z,target"
    test_header = "x,y,t,z,target"

    np.savetxt(os.path.join(out_dir, f"{prefix}_train.csv"), train, delimiter=",", header=train_header, comments="")
    np.savetxt(os.path.join(out_dir, f"{prefix}_test.csv"), test, delimiter=",", header=test_header, comments="")


def response_grid(max_value: float, n_points: int = 400) -> np.ndarray:
    values = np.linspace(0.0, max_value, n_points, dtype=np.float64)
    return np.stack([values, values, values, values], axis=1)
