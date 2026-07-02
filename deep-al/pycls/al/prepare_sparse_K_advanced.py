"""
Compute pairwise Euclidean distances in batches and save a boolean top-k mask.

For each row we keep the k smallest distances (nearest neighbours). The output
is a SciPy CSR boolean matrix where True indicates that column j is among the
top-k neighbours of row i. By default self-neighbours are removed; use
``--keep-self`` to retain them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
import torch
import sys
import os


def add_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(os.path.abspath(".."))
from pycls.al.find_kernel_thresholds import load_dataset_features  # noqa: E402


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {name}")
    return DTYPE_MAP[name]


def compute_topk_mask(
    features: torch.Tensor,
    *,
    k: int,
    batch_size: int = 512,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16,
    exclude_self: bool = True,
) -> sp.csr_matrix:
    """
    Return a CSR boolean matrix marking top-k nearest neighbours per row.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    dev = torch.device(device)
    feats = features.to(device=dev, dtype=dtype, non_blocking=True)
    N = feats.shape[0]

    target_k = k + (1 if exclude_self else 0)

    all_rows = []
    all_cols = []

    batch_rounds = N // batch_size + int(N % batch_size > 0)

    with torch.no_grad():
        for i in range(batch_rounds):
            start = i * batch_size
            end = min(start + batch_size, N)
            x1_batch = feats[start:end]

            dists = torch.cdist(x1_batch, feats, p=2.0)
            _, batch_inds = torch.topk(dists, k=target_k, dim=1, largest=False)

            rows = torch.arange(start, end, device=dev).view(-1, 1).expand(-1, target_k)

            if exclude_self:
                # self-distance is expected to be the smallest entry; drop it.
                batch_inds = batch_inds[:, 1:]
                rows = rows[:, 1:]
            else:
                batch_inds = batch_inds[:, :k]
                rows = rows[:, :k]

            all_rows.append(rows.reshape(-1))
            all_cols.append(batch_inds.reshape(-1))

            del dists, batch_inds

    if not all_rows:
        return sp.csr_matrix((N, N), dtype=bool)

    row_indices = torch.cat(all_rows).cpu().numpy()
    col_indices = torch.cat(all_cols).cpu().numpy()
    values = np.ones_like(row_indices, dtype=bool)

    coo = sp.coo_matrix((values, (row_indices, col_indices)), shape=(N, N), dtype=bool)
    return coo.tocsr()


def load_features(
    dataset: Optional[str],
    features_path: Optional[Path],
    train: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    if features_path is not None:
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        np_feats = np.load(features_path)
        return torch.from_numpy(np_feats).to(dtype=dtype)
    if dataset is None:
        raise ValueError("Either dataset or features_path must be provided.")
    return load_dataset_features(dataset, train=train, dtype=dtype)


def save_sparse_matrix(matrix: sp.csr_matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, matrix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a boolean top-k neighbour mask from Euclidean distances and save as CSR."
    )

    src_group = parser.add_mutually_exclusive_group(required=False)
    src_group.add_argument(
        "--dataset",
        type=str,
        default="CIFAR100",
        help="Dataset name to load features via pycls.datasets.utils.",
    )
    src_group.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Path to a .npy features file. When given, --dataset is ignored.",
    )

    parser.add_argument("--k", type=int, default=50, help="k nearest neighbours to keep.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for cdist.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device for distance calculation.",
    )
    parser.add_argument(
        "--dtype",
        type=parse_dtype,
        default="float16",
        help="Torch dtype used during computation.",
    )

    split_group = parser.add_mutually_exclusive_group(required=False)
    split_group.add_argument("--train", dest="train", action="store_true", default=True)
    split_group.add_argument("--eval", dest="train", action="store_false")

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <dataset>_topk_mask_k<k>.npz",
    )
    parser.add_argument(
        "--torch-output",
        type=Path,
        default=None,
        help="Optional path to also save a dense torch.bool tensor for fast loading.",
    )
    parser.add_argument(
        "--keep-self",
        action="store_true",
        help="Keep self-connections instead of dropping them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dtype = args.dtype if isinstance(args.dtype, torch.dtype) else parse_dtype(args.dtype)
    feats = load_features(args.dataset, args.features_path, args.train, dtype)

    mask = compute_topk_mask(
        feats,
        k=args.k,
        batch_size=args.batch_size,
        device=args.device,
        dtype=dtype,
        exclude_self=not args.keep_self,
    )

    if args.output is not None:
        output_path = args.output
    else:
        name = args.dataset if args.dataset is not None else "features"
        output_path = Path(f"{name}_topk_mask_k{args.k}.npz")

    save_sparse_matrix(mask, output_path)

    if args.torch_output is not None:
        # Save a dense boolean tensor for fastest torch load (warning: may be large).
        dense_mask = torch.from_numpy(mask.toarray()).to(torch.bool)
        torch.save(dense_mask, args.torch_output)

    nnz = mask.nnz
    density = nnz / (mask.shape[0] * mask.shape[1])
    print(f"Top-k mask saved to {output_path}")
    print(f"Shape: {mask.shape}, nnz: {nnz}, density: {density:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

