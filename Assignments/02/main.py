import argparse
import json
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# Avoid torch._dynamo startup overhead/issues on some Windows setups.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
import torch

from dataset_utils import (
    additive_function,
    make_toy_dataset,
    multiplicative_function,
    response_grid,
    save_dataset_csv,
)
from isnn_numpy import ISNN1Numpy, ISNN2Numpy, default_numpy_configs
from isnn_torch import ISNN1Torch, ISNN2Torch, default_torch_configs


def mse_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def train_torch_model(model: torch.nn.Module, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, epochs: int, lr: float):
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train_t)
        loss = torch.mean((pred - y_train_t) ** 2)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_test = model(x_test_t)
            test_loss = torch.mean((pred_test - y_test_t) ** 2)

        train_losses.append(float(loss.item()))
        test_losses.append(float(test_loss.item()))

    return np.array(train_losses), np.array(test_losses)


def train_numpy_model(model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, epochs: int, lr: float):
    train_losses = []
    test_losses = []

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-8
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-8

    x_train_n = (x_train - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std

    n_train = x_train_n.shape[0]

    for _ in range(epochs):
        pred_n = model.forward(x_train_n)
        grad_pred = (2.0 / n_train) * (pred_n - y_train_n)
        model.backward(grad_pred)
        model.step(lr)

        pred_train = model.forward(x_train_n) * y_std + y_mean
        pred_test = model.forward(x_test_n) * y_std + y_mean

        loss = np.mean((pred_train - y_train) ** 2)
        test_loss = np.mean((pred_test - y_test) ** 2)

        train_losses.append(float(loss))
        test_losses.append(float(test_loss))

    norm = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    return np.array(train_losses), np.array(test_losses), norm


def plot_losses(stats: dict, out_path: str, title: str) -> None:
    epochs = np.arange(1, len(next(iter(stats.values()))["train_mean"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for model_name, model_stats in stats.items():
        axes[0].plot(epochs, model_stats["train_mean"], label=model_name.upper())
        axes[0].fill_between(
            epochs,
            model_stats["train_mean"] - model_stats["train_std"],
            model_stats["train_mean"] + model_stats["train_std"],
            alpha=0.2,
        )

        axes[1].plot(epochs, model_stats["test_mean"], label=model_name.upper())
        axes[1].fill_between(
            epochs,
            model_stats["test_mean"] - model_stats["test_std"],
            model_stats["test_mean"] + model_stats["test_std"],
            alpha=0.2,
        )

    axes[0].set_title("Epoch vs Training Loss")
    axes[1].set_title("Epoch vs Testing Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[1].set_ylabel("MSE Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_behavior(
    response_grid_xyzt: np.ndarray,
    y_true: np.ndarray,
    curve_stats: dict,
    train_max: float,
    out_path: str,
    title: str,
) -> None:
    values = response_grid_xyzt[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, model_name in zip(axes, ["isnn1", "isnn2"]):
        mean_curve = curve_stats[model_name]["curve_mean"]
        std_curve = curve_stats[model_name]["curve_std"]

        ax.plot(values, y_true[:, 0], color="black", linestyle="--", linewidth=2, label="Ground truth")
        ax.plot(values, mean_curve, linewidth=2, label=model_name.upper())
        ax.fill_between(values, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

        ax.axvspan(0.0, train_max, color="#b7e4c7", alpha=0.25, label="Interpolated region")
        ax.axvspan(train_max, values.max(), color="#f8d7da", alpha=0.25, label="Extrapolated region")
        ax.set_title(model_name.upper())
        ax.set_xlabel("x = y = t = z")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Model response")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_losses(all_train: list[np.ndarray], all_test: list[np.ndarray]) -> dict:
    train_arr = np.vstack(all_train)
    test_arr = np.vstack(all_test)
    return {
        "train_mean": train_arr.mean(axis=0),
        "train_std": train_arr.std(axis=0),
        "test_mean": test_arr.mean(axis=0),
        "test_std": test_arr.std(axis=0),
        "train_final_mean": float(train_arr[:, -1].mean()),
        "test_final_mean": float(test_arr[:, -1].mean()),
    }


def run_framework(
    framework: str,
    problem: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    response_x: np.ndarray,
    n_runs: int,
    epochs: int,
    lr: float,
):
    stats = {}
    curves = {}

    for model_name in ["isnn1", "isnn2"]:
        all_train = []
        all_test = []
        all_curves = []

        for run_seed in range(n_runs):
            seed = 123 + run_seed

            if framework == "torch":
                torch.manual_seed(seed)
                cfg = default_torch_configs()[model_name]
                model = ISNN1Torch(cfg) if model_name == "isnn1" else ISNN2Torch(cfg)
                train_loss, test_loss = train_torch_model(model, x_train, y_train, x_test, y_test, epochs, lr)

                with torch.no_grad():
                    pred_curve = model(torch.tensor(response_x, dtype=torch.float32)).cpu().numpy().reshape(-1)
            else:
                cfg = default_numpy_configs()[model_name]
                model = ISNN1Numpy(cfg, seed=seed) if model_name == "isnn1" else ISNN2Numpy(cfg, seed=seed)
                lr_numpy_model = lr * 0.2 if model_name == "isnn1" else lr
                train_loss, test_loss, norm = train_numpy_model(
                    model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    epochs,
                    lr_numpy_model,
                )
                response_x_n = (response_x - norm["x_mean"]) / norm["x_std"]
                pred_curve = (model.forward(response_x_n) * norm["y_std"] + norm["y_mean"]).reshape(-1)

            all_train.append(train_loss)
            all_test.append(test_loss)
            all_curves.append(pred_curve)

        model_stats = summarize_losses(all_train, all_test)
        stats[model_name] = model_stats

        curve_arr = np.vstack(all_curves)
        curves[model_name] = {
            "curve_mean": curve_arr.mean(axis=0),
            "curve_std": curve_arr.std(axis=0),
        }

    return stats, curves


def to_serializable_stats(stats: dict) -> dict:
    out = {}
    for model_name, vals in stats.items():
        out[model_name] = {
            "train_final_mean": vals["train_final_mean"],
            "test_final_mean": vals["test_final_mean"],
            "train_mean": vals["train_mean"].tolist(),
            "train_std": vals["train_std"].tolist(),
            "test_mean": vals["test_mean"].tolist(),
            "test_std": vals["test_std"].tolist(),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="ANN Assignment 2 - ISNN implementation in PyTorch and NumPy")
    parser.add_argument("--epochs_torch", type=int, default=400)
    parser.add_argument("--epochs_numpy", type=int, default=400)
    parser.add_argument("--lr_torch", type=float, default=1e-2)
    parser.add_argument("--lr_numpy", type=float, default=5e-5)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = "outputs"
    datasets_dir = os.path.join(out_root, "datasets")
    plots_dir = os.path.join(out_root, "plots")
    results_dir = os.path.join(out_root, "results")

    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    final_report = {}

    for problem in ["additive", "multiplicative"]:
        data = make_toy_dataset(problem=problem, seed=args.seed)
        save_dataset_csv(data, datasets_dir, prefix=problem)

        max_eval = 6.0 if problem == "additive" else 10.0
        response_x = response_grid(max_eval, n_points=400)
        y_true = additive_function(response_x) if problem == "additive" else multiplicative_function(response_x)

        torch_stats, torch_curves = run_framework(
            framework="torch",
            problem=problem,
            x_train=data.x_train,
            y_train=data.y_train,
            x_test=data.x_test,
            y_test=data.y_test,
            response_x=response_x,
            n_runs=args.n_runs,
            epochs=args.epochs_torch,
            lr=args.lr_torch,
        )

        numpy_stats, numpy_curves = run_framework(
            framework="numpy",
            problem=problem,
            x_train=data.x_train,
            y_train=data.y_train,
            x_test=data.x_test,
            y_test=data.y_test,
            response_x=response_x,
            n_runs=args.n_runs,
            epochs=args.epochs_numpy,
            lr=args.lr_numpy,
        )

        plot_losses(
            torch_stats,
            out_path=os.path.join(plots_dir, f"{problem}_torch_losses.png"),
            title=f"{problem.title()} toy problem - PyTorch ISNN losses",
        )
        plot_behavior(
            response_grid_xyzt=response_x,
            y_true=y_true,
            curve_stats=torch_curves,
            train_max=4.0,
            out_path=os.path.join(plots_dir, f"{problem}_torch_behavior.png"),
            title=f"{problem.title()} toy problem - PyTorch behavioral response",
        )

        plot_losses(
            numpy_stats,
            out_path=os.path.join(plots_dir, f"{problem}_numpy_losses.png"),
            title=f"{problem.title()} toy problem - NumPy ISNN losses",
        )
        plot_behavior(
            response_grid_xyzt=response_x,
            y_true=y_true,
            curve_stats=numpy_curves,
            train_max=4.0,
            out_path=os.path.join(plots_dir, f"{problem}_numpy_behavior.png"),
            title=f"{problem.title()} toy problem - NumPy behavioral response",
        )

        final_report[problem] = {
            "torch": to_serializable_stats(torch_stats),
            "numpy": to_serializable_stats(numpy_stats),
        }

    with open(os.path.join(results_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("Finished. Outputs written to ./outputs")
    for root, _, files in os.walk(out_root):
        for name in files:
            print(os.path.join(root, name))


if __name__ == "__main__":
    main()
