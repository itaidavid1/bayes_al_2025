import csv
from pathlib import Path
from typing import List, Sequence, Tuple
import scipy.sparse as sp
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))
import pycls.datasets.utils as ds_utils
import pycls.datasets.utils as ds_utils


SIGMA = 1.0
TOPK = 300
CHUNK_SIZE = 512
KERNEL_BATCH = 512
NUM_CLASSES = 10
THRESHOLD_DIR = Path("/cs/labs/daphna/itai.david/py_repos/cifar10_data/rbf_K_sparse_info")
THRESHOLD_FILE = THRESHOLD_DIR / f"{float(SIGMA)}_sigma_sparse_info.csv"
LABELS_PATH = Path("/cs/labs/daphna/itai.david/py_repos/TypiClust/results/cifar-10/labels_seed0.npy")
PLOT_PATH = Path(__file__).with_name(f"sparse_runtime_sigma_{SIGMA}.png")


def load_sigma_thresholds(csv_path: Path, sigma: float) -> List[Tuple[float, float]]:
    """Return (sparsity, threshold) pairs for the requested sigma."""
    thresholds: List[Tuple[float, float]] = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_sigma = float(row["sigma"])
            if abs(row_sigma - sigma) > 1e-6:
                continue
            raw_threshold = row["threshold"]
            if raw_threshold.startswith("tensor("):
                raw_threshold = raw_threshold[len("tensor(") : -1]
            thresholds.append((float(row["sparsity"]), float(raw_threshold)))

    if not thresholds:
        raise ValueError(f"No thresholds found for sigma={sigma} in {csv_path}")

    thresholds.sort(key=lambda pair: pair[0])
    return thresholds


def compute_norm(x1: torch.Tensor, x2: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    x1, x2 = x1.unsqueeze(0), x2.unsqueeze(0)
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        x2_subset = x2[:, i * batch_size : (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset, p=2.0).to(dtype=torch.float32)
        dist_matrix.append(dist.cpu())
        del dist
    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix


def get_rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, h: float = 1.0, batch_size: int = 512) -> torch.Tensor:
    norm = compute_norm(x1, x2, batch_size=batch_size)
    kernel = torch.exp(-1.0 * (norm / h) ** 2)
    return kernel


def build_class_statistics(
    kernel: torch.Tensor,
    labels: Sequence[int],
    num_classes: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_points = kernel.shape[0]
    C = torch.full((num_points, num_classes), 0.1, dtype=kernel.dtype)
    selections = min(top_k, len(labels))

    for i in range(selections):
        label = int(labels[i])
        C[:, label] += kernel[:, i]

    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)
    sum_C[sum_C == 0] = 1.0

    norm_C = C / sum_C
    old_max = max_C / sum_C
    C_diff = C - max_C
    return norm_C, sum_C, max_C, C_diff, old_max


def threshold_kernel_to_csr(kernel: torch.Tensor, threshold: float) -> torch.Tensor:
    pruned = kernel.clone()
    pruned[pruned < threshold] = 0
    return pruned.to_sparse_csr()


def run_sparse_code_v2(
    K_csr: torch.Tensor,
    norm_C: torch.Tensor,
    sum_C: torch.Tensor,
    max_C: torch.Tensor,
    C_diff: torch.Tensor,
    old_max: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    dev = torch.device(device if isinstance(device, str) else device)
    crow = K_csr.crow_indices().to(dev)
    ccol = K_csr.col_indices().to(dev)
    cvals = K_csr.values().to(dev)

    num_rows = crow.numel() - 1
    classes = norm_C.shape[1]
    result = torch.empty((num_rows,), device=dev, dtype=norm_C.dtype)

    norm_C = norm_C.to(dev)
    sum_C = sum_C.to(dev)
    max_C = max_C.to(dev)
    C_diff = C_diff.to(dev)
    old_max = old_max.to(dev)

    cont_method = "positive"

    for row_start in range(0, num_rows, chunk_size):
        row_end = min(row_start + chunk_size, num_rows)
        b = row_end - row_start

        starts = crow[row_start:row_end]
        ends = crow[row_start + 1 : row_end + 1]
        lengths = (ends - starts).to(torch.long)

        total_nnz = int(lengths.sum().item())
        if total_nnz == 0:
            result[row_start:row_end] = 0
            continue

        slice_start = int(starts[0].item())
        slice_end = int(ends[-1].item())
        cols_all = ccol[slice_start:slice_end]
        vals_all = cvals[slice_start:slice_end]

        row_indices = torch.repeat_interleave(torch.arange(b, device=dev, dtype=torch.long), lengths)
        global_rows = torch.arange(row_start, row_end, device=dev, dtype=torch.long)
        global_row_ids = global_rows[row_indices]

        kvals = vals_all.unsqueeze(1)
        sumC_cols = sum_C[cols_all]
        maxC_cols = max_C[cols_all]
        old_max_cols = old_max[cols_all]
        Cdiff_cols = C_diff[cols_all]

        negk = -kvals
        new_state = torch.maximum(negk, Cdiff_cols)
        state_add = maxC_cols + kvals
        new_state = new_state + state_add
        future_sum = kvals + sumC_cols
        new_state = new_state / future_sum
        new_state = new_state - old_max_cols

        if cont_method == "positive":
            new_state.clamp_(min=0.0)

        weights_for_nnz = norm_C[global_row_ids]
        per_nnz_weighted = (new_state * weights_for_nnz).sum(dim=1)

        chunk_result = torch.zeros((b,), device=dev, dtype=norm_C.dtype)
        chunk_result.scatter_add_(0, row_indices, per_nnz_weighted)
        result[row_start:row_end] = chunk_result

    return result


def measure_sparse_runtime(
    K_csr: torch.Tensor,
    norm_C: torch.Tensor,
    sum_C: torch.Tensor,
    max_C: torch.Tensor,
    C_diff: torch.Tensor,
    old_max: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    run_sparse_code_v2(K_csr, norm_C, sum_C, max_C, C_diff, old_max, chunk_size, device)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def plot_runtime(sparsities: Sequence[float], runtimes_ms: Sequence[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(sparsities, runtimes_ms, marker="o")
    plt.xlabel("Target Sparsity")
    plt.ylabel("Sparse Runtime (ms)")
    plt.title(f"Sparse Runtime vs. Sparsity (sigma={SIGMA})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved runtime plot to {output_path}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")

    device = torch.device("cuda")
    print("Loading CIFAR-10 features...")
    features = torch.from_numpy(ds_utils.load_features("CIFAR10", train=True)).to(torch.float32)
    labels = np.load(str(LABELS_PATH))

    print("Computing dense RBF kernel (this can take a while)...")
    kernel = get_rbf_kernel(features, features, h=SIGMA, batch_size=KERNEL_BATCH)

    print("Building class statistics...")
    norm_C, sum_C, max_C, C_diff, old_max = build_class_statistics(
        kernel, labels, num_classes=NUM_CLASSES, top_k=TOPK
    )
    kernel = kernel.numpy()
    norm_C = norm_C.to(device)
    sum_C = sum_C.to(device)
    max_C = max_C.to(device)
    C_diff = C_diff.to(device)
    old_max = old_max.to(device)

    thresholds = load_sigma_thresholds(THRESHOLD_FILE, SIGMA)

    runtime_records: List[Tuple[float, float]] = []
    for sparsity, threshold in thresholds:
        print(f"Benchmarking sparsity={sparsity:.6f} threshold={threshold:.6f}")
        new_kernel = kernel.copy()
        new_kernel[new_kernel < threshold] = 0.0
        K_coo = sp.coo_matrix(new_kernel)
        K_csr = K_coo.tocsr()
        crow_indices = torch.from_numpy(K_csr.indptr).to(torch.int64)
        col_indices = torch.from_numpy(K_csr.indices).to(torch.int64)
        values = torch.from_numpy(K_csr.data).to(torch.float32)  # or .to(torch.float64)

        # 2. Construct the PyTorch sparse_csr_tensor
        K_sparse_torch = torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            size=K_csr.shape,
            dtype=values.dtype  # Use the same dtype as the values tensor
        )
        print("Finish creating K sparse for sparsity:", sparsity)
        runtime_ms = measure_sparse_runtime(
            K_sparse_torch, norm_C, sum_C, max_C, C_diff, old_max, CHUNK_SIZE, device
        )
        runtime_records.append((sparsity, runtime_ms))
        print(f"  Runtime: {runtime_ms:.3f} ms")

    sparsities, runtimes = zip(*runtime_records)
    plot_runtime(sparsities, runtimes, PLOT_PATH)


if __name__ == "__main__":
    main()

