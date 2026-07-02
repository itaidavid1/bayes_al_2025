"""Compute and plot the average first derivative of a kernel matrix."""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))
import pycls.datasets.utils as ds_utils
# Reuse kernel construction helpers to avoid duplication.
from pycls.al.find_kernel_thresholds import get_rbf_kernel, load_dataset_features


def load_kernel_matrix(path: Path) -> np.ndarray:
    """Load a kernel matrix saved as .npy or .pt and return a float64 ndarray."""
    if not path.exists():
        raise FileNotFoundError(f"Kernel file not found: {path}")

    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    elif path.suffix.lower() == ".pt":
        tensor = torch.load(path, map_location="cpu")
        matrix = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .npy or .pt.")

    if matrix.ndim != 2:
        raise ValueError(f"Kernel matrix must be 2D, got shape {matrix.shape}")

    return matrix.astype(np.float64, copy=False)


def average_first_derivative(kernel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort each row, take the first discrete derivative, and average across rows.

    Returns:
        x: index positions for the derivative points.
        y: average first derivative over all rows.
    """
    sorted_rows = np.sort(kernel, axis=1)
    derivatives = np.diff(sorted_rows, axis=1)  # finite differences along sorted entries
    avg_derivative = derivatives.mean(axis=0)
    x = np.arange(1, derivatives.shape[1] + 1, dtype=np.int64)
    return x, avg_derivative


def plot_average_derivative(x: np.ndarray, y: np.ndarray, output_path: Path, show: bool) -> None:
    """Plot and optionally display the averaged first derivative."""
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="Average first derivative")
    plt.xlabel("Sorted index")
    plt.ylabel("Average first derivative")
    plt.title("Average first derivative of sorted kernel rows")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute an RBF kernel (like find_kernel_thresholds.py) or load one, "
        "then plot the average first derivative across sorted rows."
    )

    kernel_group = parser.add_mutually_exclusive_group(required=False)
    kernel_group.add_argument(
        "--kernel-path",
        type=Path,
        default=None,
        help="Path to an existing kernel matrix (.npy or .pt). If omitted, a kernel "
        "is computed from --dataset using the same flow as find_kernel_thresholds.py.",
    )
    kernel_group.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="Dataset name to load features from (used when --kernel-path is not provided).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="RBF bandwidth for on-the-fly kernel computation (ignored if --kernel-path is given).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=True,
        help="Use the training split when computing a kernel (matches find_kernel_thresholds.py).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to save the plot (PNG). Defaults to <kernel_name>_avg_first_derivative.png "
        "if --kernel-path is given, otherwise <dataset>_sigma<sigma>_avg_first_derivative.png.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot in addition to saving it.")
    return parser.parse_args()


def get_kernel_matrix(
    kernel_path: Optional[Path],
    dataset: str,
    sigma: float,
    train: bool,
) -> np.ndarray:
    """Return a kernel matrix either from disk or computed like find_kernel_thresholds.py."""
    if kernel_path is not None:
        return load_kernel_matrix(kernel_path)

    features = load_dataset_features(dataset, train=train, dtype=torch.float16)
    K = get_rbf_kernel(features, features, h=sigma)
    if isinstance(K, torch.Tensor):
        return K.detach().cpu().numpy()
    return np.asarray(K, dtype=np.float64)


def main() -> int:
    args = parse_args()
    output_path: Path
    if args.output_path is not None:
        output_path = args.output_path
    elif args.kernel_path is not None:
        output_path = args.kernel_path.with_name(f"{args.kernel_path.stem}_avg_first_derivative.png")
    else:
        output_path = Path(
            f"{args.dataset.lower()}_sigma{args.sigma}_avg_first_derivative.png"
        )

    kernel = get_kernel_matrix(args.kernel_path, args.dataset, args.sigma, args.train)
    x, avg_derivative = average_first_derivative(kernel)
    plot_average_derivative(x, avg_derivative, output_path, args.show)
    print(f"Saved average first-derivative plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
