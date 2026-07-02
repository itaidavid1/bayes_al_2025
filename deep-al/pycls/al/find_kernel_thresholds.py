from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Iterable, List, Sequence
import sys
import os
import torch
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))
import pycls.datasets.utils as ds_utils


@dataclass
class ThresholdResult:
    sigma: float
    sparsity: float
    threshold: float

def compute_norm(x1, x2,batch_size=512):
    x1, x2 = x1.unsqueeze(0), x2.unsqueeze(0) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        # distance comparisons are done in batches to reduce memory consumption
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset, p=2.0).to(dtype=torch.float16)

        dist_matrix.append(dist.cpu())
        del dist

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix

def get_rbf_kernel( x1, x2, h=1.0, batch_size=512):
    norm = compute_norm(x1, x2, batch_size=batch_size)
    k = torch.exp(-1.0 * (norm / h) ** 2)
    return k

def sample_features(
    features: torch.Tensor, sample_size: int, seed: int
) -> torch.Tensor:
    sample_size = min(sample_size, features.shape[0])
    if sample_size == features.shape[0]:
        return features.clone()

    gen = torch.Generator(device=features.device)
    gen.manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=gen, device=features.device)
    indices = permutation[:sample_size]
    return features[indices]


def compute_pairwise_distances(
    sample: torch.Tensor, device: torch.device
) -> torch.Tensor:
    sample = sample.to(device)
    return torch.cdist(sample, sample, p=2.0)


def prepare_upper_triangle_values(distances: torch.Tensor) -> torch.Tensor:
    size = distances.shape[0]
    triu_indices = torch.triu_indices(size, size, offset=1)
    return distances[triu_indices[0], triu_indices[1]]


def estimate_thresholds(
    K,
    sigma,
    sparsities: Iterable[float],
) -> List[ThresholdResult]:
    upper_vals = prepare_upper_triangle_values(K)
    upper_vals = upper_vals.to(torch.float32)
    results: List[ThresholdResult] = []
    sorted_values = torch.sort(upper_vals)
    for sparsity in sparsities:
        sparse_ind = int(sparsity*len(sorted_values[0]))
        threshold = sorted_values[0][sparse_ind]
        results.append(ThresholdResult(sigma=sigma, sparsity=sparsity, threshold=threshold))

    return results


def load_dataset_features(dataset: str, train: bool, dtype: torch.dtype) -> torch.Tensor:
    np_features = ds_utils.load_features(dataset, train=train)
    tensor = torch.from_numpy(np_features).to(dtype=dtype)
    return tensor


def print_results_table(results: Iterable[ThresholdResult]) -> None:
    header = f"{'Sigma':>8} | {'Sparsity':>8} | {'Threshold':>10}"
    separator = "-" * len(header)
    print(header)
    print(separator)
    for entry in results:
        print(f"{entry.sigma:8.4f} | {entry.sparsity:8.4f} | {entry.threshold:10.6f}")


def save_results_csv(results: Iterable[ThresholdResult], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(("sigma", "sparsity", "threshold"))
        for entry in results:
            writer.writerow((entry.sigma, entry.sparsity, float(entry.threshold)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate threshold values that induce a target sparsity "
        "for an RBF kernel computed on CIFAR10 features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR100",
        help="Dataset to pull features from (delegates to pycls.datasets.utils).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=True,
        help="Use training split of the dataset.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=1,
        help="Bandwidth(s) to try when computing the RBF kernel.",
    )
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=[0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999],
        help="Target sparsity values (fraction of zeros) after thresholding.",
    )
    print("start")
    args = parser.parse_args()
    output_path = Path(f"/cs/labs/daphna/itai.david/py_repos/{args.dataset.lower()}_data/rbf_K_sparse_info/{float(args.sigma)}_sigma_sparse_info.csv")
    features = load_dataset_features(args.dataset, args.train, dtype=torch.float16)
    device = 'cuda'
    K = get_rbf_kernel(features, features, h=args.sigma)

    results = estimate_thresholds(K,args.sigma, args.sparsities)
    print_results_table(results)

    if output_path is not None:
        save_results_csv(results, output_path)
        print(f"\nSaved thresholds to {output_path}")

    return 0


if __name__ == "__main__":
    main()

