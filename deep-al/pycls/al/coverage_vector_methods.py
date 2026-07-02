import numpy as np
import pandas as pd
import torch
import gc
import pickle
import pycls.datasets.utils as ds_utils
import time
import os
from pycls.al.kernel_utils import build_K_general_matrix, RBFKernel, TopHatKernel

torch.cuda.empty_cache()


class CoverageVectorMethod:
    def __init__(self, cfg, budgetSize, train_labels, lset, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, train=True)
        self.debug = self.cfg.DEBUG
        self.use_sparse = self.cfg.SPARSE_K
        self.matrices_type = torch.float32 if self.use_sparse else torch.float16
        self.budgetSize = budgetSize
        self.K_sparsity_threshold = self.cfg.K_SPARSITY_THRESHOLD
        self.sigma = cfg.ACTIVE_LEARNING.INITIAL_SIGMA if 'INITIAL_SIGMA' in cfg.ACTIVE_LEARNING else 1.0


        self.alpha = 0

        self.delta = delta

        self.train_labels_general = np.array(train_labels)
        unique_labels = np.unique(self.train_labels_general)
        self.unique_labels = unique_labels
        self.num_of_classes = unique_labels.size

        self.kernel_build_batch_size = getattr(self.cfg, 'KERNEL_BUILD_BATCH_SIZE', 1024)


        self.kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if self.kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
            initial_threshold = self.delta
        else:
            self.kernel_fn = RBFKernel('cuda')
            initial_threshold = self.K_sparsity_threshold

        self.K_general = build_K_general_matrix(
            self.all_features,
            threshold=initial_threshold,
            use_sparse=self.use_sparse,
            kernel_type=self.kernel_type,
            delta=self.delta,
            sigma=self.sigma,
            kernel_build_batch_size=self.kernel_build_batch_size,
            matrices_type=self.matrices_type,
            kernel_fn=self.kernel_fn,
            zero_indices=None,
        )

        n_points = self.all_features.shape[0]
        # C_general[i] = max similarity of point i to any labeled point seen so far
        self.C_general = torch.full((n_points,), self.alpha, device='cuda', dtype=self.matrices_type)
        # label_coverage_general[i] = global dataset index of the labeled point that achieved
        # the max similarity for point i. -1 means "no labeled point yet".
        self.label_coverage_general = torch.full((n_points,), -1, device='cuda', dtype=torch.int64)


    def init_sampling_loop(self, lset, uset):
        torch.cuda.empty_cache()
        self.set_rel_features(lset, uset)
        self.activeSet = []
        if self.use_sparse:
            K_csr_shuffled = self.K_general[self.relevant_indices, :][:, self.relevant_indices]
            crow_indices = torch.from_numpy(K_csr_shuffled.indptr).to(torch.int64)
            col_indices = torch.from_numpy(K_csr_shuffled.indices).to(torch.int64)
            values = torch.from_numpy(K_csr_shuffled.data).to(torch.float32)

            self.K = torch.sparse_csr_tensor(
                crow_indices=crow_indices,
                col_indices=col_indices,
                values=values,
                size=K_csr_shuffled.shape,
                dtype=values.dtype
            )
            del K_csr_shuffled, values, col_indices, crow_indices
        else:
            self.K = self.K_general[self.relevant_indices, :][:, self.relevant_indices]
        self.C = self.C_general[self.relevant_indices].to('cuda')
        self.label_coverage = self.label_coverage_general[self.relevant_indices].to('cuda')
        self.train_labels = self.train_labels_general[self.relevant_indices]

    def set_rel_features(self, lset, uset):
        self.lSet = lset
        self.uSet = uset
        print(lset)
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        if isinstance(self.all_features, torch.Tensor):
            self.rel_features = self.all_features[self.relevant_indices]
        elif isinstance(self.all_features, np.ndarray):
            self.rel_features = torch.from_numpy(self.all_features[self.relevant_indices])

    def select_samples(self, lset, uset):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """

        self.init_sampling_loop(lset, uset)

        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        for i in range(self.budgetSize):
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)

            # C is already a vector of per-point max similarities — use it directly
            if self.use_sparse:
                point_total_contribution = batched_diffs_sparse(self.K, self.C, 0, self.num_of_classes,
                                                                diff_method="abs_diff")
            else:
                point_total_contribution = batched_diffs(self.K, self.C, 0, self.num_of_classes,
                                                         diff_method="abs_diff")

            point_total_contribution[curr_l_set] = -np.inf
            sampled_point = np.argsort(point_total_contribution.cpu().numpy(), kind='stable')[::-1][0].item()

            K_row_dense = self.K[sampled_point].to_dense().to('cuda').squeeze()

            # Update C and label_coverage only where the new labeled point improves coverage
            improved_mask = K_row_dense > self.C
            self.C[improved_mask] = K_row_dense[improved_mask]
            global_labeled_idx = int(self.relevant_indices[sampled_point])
            self.label_coverage[improved_mask] = global_labeled_idx

            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)
            del K_row_dense, improved_mask

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]

        self.C_general[self.relevant_indices] = self.C
        self.label_coverage_general[self.relevant_indices] = self.label_coverage
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')

        del self.K
        del self.C
        del self.label_coverage

        return activeSet, remainSet


def batched_diffs(K, C, alpha, number_of_classes, chunk_size=1024, diff_method="abs_diff"):
    """
    K: (D, N) dense matrix — rows are candidate points, columns are all points.
    C: (N,) vector — per-point max similarity to any labeled point so far.
    Returns: (D,) contribution score for each candidate point.
    """
    D, N = K.shape
    result = torch.empty(D, device=C.device)
    for start in range(0, D, chunk_size):
        end = min(start + chunk_size, D)
        K_batched = K[start:end].to('cuda')   # (chunk, N)
        if diff_method == "abs_diff":
            # For each candidate row, sum ReLU(K_row - C) over all points
            result[start:end] = torch.sum(torch.relu(K_batched - C), dim=1)
        elif diff_method == "max":
            normed = (K_batched + alpha) / torch.clamp(K_batched + alpha * number_of_classes, min=1e-8)
            result[start:end] = torch.sum(torch.relu(normed - C), dim=1)
        elif diff_method == 'margin':
            normed = K_batched / torch.clamp(K_batched + alpha * number_of_classes, min=1e-8)
            result[start:end] = torch.sum(torch.relu(normed - C), dim=1)
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
    return result


def batched_diffs_sparse(K, C, alpha=None, number_of_classes=None, chunk_size=1024, diff_method="abs_diff"):
    """
    Computes batched diffs for a Sparse CSR Tensor K and a dense coverage vector C.

    K: (D, N) sparse CSR tensor — rows are candidate points, columns are all points.
    C: (N,) dense vector — per-point max similarity to any labeled point so far.
    Returns: (D,) contribution score for each candidate point.

    Assumes:
    1. K is a torch.sparse_csr_tensor.
    2. C >= 0 (This allows us to ignore the zero-entries in K, keeping operations sparse).
    """
    D, N = K.shape
    device = C.device

    # Extract CSR components and move to device
    crow = K.crow_indices().to(device)  # shape (D+1,)
    ccol = K.col_indices().to(device)   # shape (nnz,)
    cvals = K.values().to(device)       # shape (nnz,)

    # Pre-allocate result on the correct device
    result = torch.zeros(D, device=device, dtype=cvals.dtype)

    for row_start in range(0, D, chunk_size):
        row_end = min(row_start + chunk_size, D)
        b = row_end - row_start

        # CSR pointers for the chunk
        starts = crow[row_start:row_end]           # shape (b,)
        ends = crow[row_start + 1: row_end + 1]    # shape (b,)
        lengths = (ends - starts).to(torch.long)   # (b,)

        total_nnz = int(lengths.sum().item())

        if total_nnz == 0:
            # nothing in this chunk, skip
            continue

        # global slice of indices/values for this chunk
        slice_start = int(starts[0].item())
        slice_end = int(ends[-1].item())

        cols_all = ccol[slice_start:slice_end]   # (total_nnz,)
        vals_all = cvals[slice_start:slice_end]  # (total_nnz,)

        # row index for each nnz entry within the chunk: 0..b-1 repeated by lengths
        row_indices = torch.repeat_interleave(
            torch.arange(b, device=device, dtype=torch.long),
            lengths
        )  # (total_nnz,)

        if diff_method == "abs_diff":
            # --- SPARSE OPTIMIZATION EXPLAINED ---
            # The operation is sum(max(K - C, 0)).
            # For sparse entries where K=0, this is max(0 - C, 0).
            # If C >= 0, this result is 0.
            # Therefore, we only need to work with the non-zero values.

            # Handle broadcasting for C
            if C.numel() == 1:
                # C is a scalar
                c_mapped = C
            elif C.ndim == 1 and C.shape[0] == N:
                # C is a vector matching the columns (feature dim).
                # We gather only the C values that align with the non-zero columns.
                c_mapped = C[cols_all]
            else:
                raise NotImplementedError("C shape not supported for sparse optimization")

            # Perform ReLU(Values - C)
            new_vals = torch.relu(vals_all - c_mapped)

            # Aggregate per row via scatter_add
            chunk_result = torch.zeros(b, device=device, dtype=cvals.dtype)
            chunk_result.scatter_add_(0, row_indices, new_vals)

            result[row_start:row_end] = chunk_result

    return result