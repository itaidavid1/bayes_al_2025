import numpy as np
import scipy.sparse as sp
import torch

torch.cuda.empty_cache()

def compute_norm(x1, x2, device, batch_size=512, matrices_type=torch.float16):
    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        # distance comparisons are done in batches to reduce memory consumption
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset).to(dtype=matrices_type)

        dist_matrix.append(dist.cpu())
        del dist

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix

class RBFKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, matrices_type=torch.float16):
        norm = compute_norm(x1, x2, self.device, batch_size=batch_size, matrices_type=matrices_type)
        k = torch.exp(-1.0 * (norm / h) ** 2)
        return k

    def compute_kernel_from_norm(self, norm_matrix, h, matrices_type=torch.float16):
        k = torch.exp(-1.0 * (norm_matrix / h) ** 2).to(dtype=matrices_type)
        return k


class TopHatKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512, matrices_type=torch.float16):
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device) # 1 x n x d, 1 x n' x d
        dist_matrix = []
        batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(x1, x2_subset)
            dist = (dist < h).to(dtype=matrices_type)
            dist_matrix.append(dist.cpu())
            del dist
        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        # k = (dist_matrix < h).to(dtype=torch.float16)
        return dist_matrix

    def compute_kernel_from_norm(self, norm_matrix, h, matrices_type=torch.float16):
        k = (norm_matrix < h).to(dtype=matrices_type)
        return k


def build_sparse_kernel_matrix(
        features,
        threshold,
        *,
        kernel_type,
        kernel_param,
        batch_size=1024,
        device='cuda',
        dtype=torch.float32,
        zero_indices=None,
        prev_threshold=None,
        capture_zero_contrib=False,
):
    """
    Build a symmetric sparse kernel matrix in CSR format without materializing the full dense matrix.

    Args:
        features (np.ndarray or torch.Tensor): Feature matrix of shape (N, D) on CPU.
        threshold (float): Value used to sparsify the kernel. For 'rbf' this is the minimum kernel value kept.
                           For 'tophat' this is interpreted as the distance cutoff (delta).
        kernel_type (str): 'rbf' or 'tophat'.
        kernel_param (float): Kernel-specific parameter. For 'rbf' this is sigma. For 'tophat' it is ignored.
        batch_size (int): Number of rows processed per chunk.
        device (str or torch.device): Device used for intermediate GPU computations.
        dtype (torch.dtype): Dtype for intermediate tensors (float32 recommended for sparse path).
        zero_indices (array-like, optional): Indices whose rows/cols should be zeroed in the returned matrix.
        prev_threshold (float, optional): Previous threshold value. Required when
            capture_zero_contrib=True to identify newly added connections.
        capture_zero_contrib (bool): When True, returns the contributions that
            originate from zeroed indices but were newly introduced by lowering
            the sparsity threshold.

    Returns:
        scipy.sparse.csr_matrix: Symmetric sparse kernel matrix (N x N).
        If capture_zero_contrib is True, returns a tuple containing the CSR
        matrix and a dictionary with the newly discovered connections.
    """
    torch_device = torch.device(device)
    if isinstance(features, torch.Tensor):
        features_tensor = features.to(device=torch_device, dtype=torch.float32)
    else:
        features_tensor = torch.from_numpy(features).to(device=torch_device, dtype=torch.float32)

    n_samples = features_tensor.shape[0]

    row_blocks = []
    col_blocks = []
    data_blocks = []

    thresh_val = threshold.item() if torch.is_tensor(threshold) else float(threshold)
    prev_thresh_val = None
    if prev_threshold is not None:
        prev_thresh_val = prev_threshold.item() if torch.is_tensor(prev_threshold) else float(prev_threshold)

    if kernel_type == 'rbf':
        kernel_param_val = kernel_param.item() if torch.is_tensor(kernel_param) else float(kernel_param)

    capture_zero_contrib = bool(capture_zero_contrib and prev_thresh_val is not None and
                                zero_indices is not None and len(zero_indices) > 0)
    zero_idx = None
    zero_mask_np = None
    removed_sources = []
    removed_targets = []
    removed_values = []
    if zero_indices is not None and len(zero_indices) > 0:
        zero_idx = np.asarray(zero_indices, dtype=np.int64)
        if capture_zero_contrib:
            zero_mask_np = np.zeros(n_samples, dtype=bool)
            zero_mask_np[zero_idx] = True

    with torch.no_grad():
        for start_i in range(0, n_samples, batch_size):
            end_i = min(start_i + batch_size, n_samples)
            chunk_i = features_tensor[start_i:end_i].to(dtype=dtype, non_blocking=True)

            for start_j in range(start_i, n_samples, batch_size):
                end_j = min(start_j + batch_size, n_samples)
                chunk_j = features_tensor[start_j:end_j].to(dtype=dtype, non_blocking=True)

                dist_block = torch.cdist(chunk_i, chunk_j)

                if kernel_type == 'tophat':
                    kernel_block = (dist_block < thresh_val).to(dtype=dtype)
                elif kernel_type == 'rbf':
                    kernel_block = torch.exp(-1.0 * (dist_block / kernel_param_val) ** 2)
                else:
                    raise ValueError(f"Unsupported kernel type: {kernel_type}")

                if kernel_type == 'rbf':
                    mask = kernel_block > thresh_val
                else:
                    mask = kernel_block > 0

                nz_rows, nz_cols = torch.nonzero(mask, as_tuple=True)
                if nz_rows.numel() == 0:
                    continue

                values = kernel_block[nz_rows, nz_cols]

                rows_cpu = (nz_rows + start_i).cpu().numpy()
                cols_cpu = (nz_cols + start_j).cpu().numpy()
                data_cpu = values.cpu().numpy().astype(np.float32, copy=False)

                row_blocks.append(rows_cpu)
                col_blocks.append(cols_cpu)
                data_blocks.append(data_cpu)

                if start_j != start_i:
                    row_blocks.append(cols_cpu)
                    col_blocks.append(rows_cpu)
                    data_blocks.append(data_cpu.copy())

                if not capture_zero_contrib:
                    continue

                if kernel_type == 'rbf':
                    prev_keep_mask = kernel_block > prev_thresh_val
                else:
                    prev_keep_mask = dist_block < prev_thresh_val

                new_entries_mask = mask & (~prev_keep_mask)
                new_rows, new_cols = torch.nonzero(new_entries_mask, as_tuple=True)
                if new_rows.numel() == 0:
                    continue

                contrib_rows = (new_rows + start_i).cpu().numpy()
                contrib_cols = (new_cols + start_j).cpu().numpy()
                contrib_vals = kernel_block[new_rows, new_cols].cpu().numpy().astype(np.float32, copy=False)

                zero_rows_mask = zero_mask_np[contrib_rows]
                zero_cols_mask = zero_mask_np[contrib_cols]

                if zero_rows_mask.any():
                    non_zero_cols = ~zero_mask_np[contrib_cols]
                    row_mask = zero_rows_mask & non_zero_cols
                    if row_mask.any():
                        removed_sources.append(contrib_rows[row_mask])
                        removed_targets.append(contrib_cols[row_mask])
                        removed_values.append(contrib_vals[row_mask])

                if zero_cols_mask.any():
                    non_zero_rows = ~zero_mask_np[contrib_rows]
                    col_mask = zero_cols_mask & non_zero_rows
                    if col_mask.any():
                        removed_sources.append(contrib_cols[col_mask])
                        removed_targets.append(contrib_rows[col_mask])
                        removed_values.append(contrib_vals[col_mask])

    if not row_blocks:
        csr = sp.csr_matrix((n_samples, n_samples), dtype=np.float32)
    else:
        rows = np.concatenate(row_blocks)
        cols = np.concatenate(col_blocks)
        data = np.concatenate(data_blocks)
        coo = sp.coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        csr = coo.tocsr()

    if zero_idx is not None and zero_idx.size > 0:
        csr[zero_idx, :] = 0
        csr[:, zero_idx] = 0

    if not capture_zero_contrib:
        return csr

    if removed_sources:
        zero_contrib = {
            "sources": np.concatenate(removed_sources).astype(np.int64, copy=False),
            "targets": np.concatenate(removed_targets).astype(np.int64, copy=False),
            "values": np.concatenate(removed_values).astype(np.float32, copy=False),
        }
    else:
        zero_contrib = {
            "sources": np.empty((0,), dtype=np.int64),
            "targets": np.empty((0,), dtype=np.int64),
            "values": np.empty((0,), dtype=np.float32),
        }

    return csr, zero_contrib


def build_K_general_matrix(
        features,
        threshold,
        *,
        use_sparse,
        kernel_type,
        delta,
        sigma,
        kernel_build_batch_size,
        matrices_type,
        kernel_fn,
        zero_indices=None,
        prev_threshold=None,
        # Parameters for update_C_with_label_connections (optional)
        C_general=None,
        train_labels_general=None,
        labeled_points_mask_general=None,
        diff_method=None,
        cum_labels_info=None,
):
    """
    Build a kernel matrix (sparse or dense) based on the provided parameters.

    Args:
        features: Feature matrix of shape (N, D).
        threshold: Sparsity threshold value.
        use_sparse: Whether to build a sparse matrix.
        kernel_type: 'rbf' or 'tophat'.
        delta: Delta parameter for tophat kernel.
        sigma: Sigma parameter for RBF kernel.
        kernel_build_batch_size: Batch size for kernel computation.
        matrices_type: Dtype for the matrices.
        kernel_fn: Kernel function object (RBFKernel or TopHatKernel instance).
        zero_indices: Indices whose rows/cols should be zeroed.
        prev_threshold: Previous threshold for capturing new connections.
        C_general: The C matrix to update (required for label connection updates).
        train_labels_general: Array of training labels (required for label connection updates).
        labeled_points_mask_general: Boolean mask for labeled points (required for label connection updates).
        diff_method: The differencing method (required for label connection updates).
        cum_labels_info: Cumulative labels info tensor (optional, for non-prob_cover/max_herding methods).

    Returns:
        K_matrix: The kernel matrix (sparse CSR or dense tensor).
    """
    thresh_val = threshold.item() if isinstance(threshold, torch.Tensor) else float(threshold)
    prev_thresh_val = None
    if prev_threshold is not None:
        prev_thresh_val = prev_threshold.item() if isinstance(prev_threshold, torch.Tensor) else float(
            prev_threshold)

    can_update_C = (
            C_general is not None and
            train_labels_general is not None and
            labeled_points_mask_general is not None and
            diff_method is not None
    )
    should_capture = (
            can_update_C and
            prev_thresh_val is not None and
            zero_indices is not None and
            len(zero_indices) > 0
    )

    if use_sparse:
        kernel_param = delta if kernel_type == 'tophat' else sigma
        kernel_param_val = kernel_param.item() if isinstance(kernel_param, torch.Tensor) else float(kernel_param)
        build_result = build_sparse_kernel_matrix(
            features,
            threshold=thresh_val,
            kernel_type=kernel_type,
            kernel_param=kernel_param_val,
            batch_size=kernel_build_batch_size,
            device='cuda',
            dtype=matrices_type,
            zero_indices=zero_indices,
            prev_threshold=prev_thresh_val,
            capture_zero_contrib=should_capture,
        )
        if should_capture:
            K_matrix, zero_contrib = build_result
            if zero_contrib and zero_contrib["sources"].size > 0:
                update_C_with_label_connections(
                    zero_contrib,
                    C_general=C_general,
                    train_labels_general=train_labels_general,
                    labeled_points_mask_general=labeled_points_mask_general,
                    diff_method=diff_method,
                    cum_labels_info=cum_labels_info,
                )
            return K_matrix
        return build_result

    if isinstance(features, torch.Tensor):
        features_tensor = features.to(torch.float32)
    else:
        features_tensor = torch.from_numpy(features).to(torch.float32)

    norm_matrix = compute_norm(
        features_tensor,
        features_tensor,
        'cuda',
        batch_size=kernel_build_batch_size,
        matrices_type=torch.float32
    ).to('cpu')

    if kernel_type == 'tophat':
        dense_K = kernel_fn.compute_kernel_from_norm(
            norm_matrix, thresh_val, matrices_type=matrices_type)
    else:
        dense_K = kernel_fn.compute_kernel_from_norm(
            norm_matrix, sigma, matrices_type=matrices_type)
        if thresh_val > 0:
            dense_K = torch.where(dense_K > thresh_val, dense_K, torch.zeros_like(dense_K))

    if zero_indices is not None and len(zero_indices) > 0:
        zero_idx = torch.as_tensor(np.asarray(zero_indices, dtype=np.int64))
        dense_K[zero_idx, :] = 0
        dense_K[:, zero_idx] = 0

    return dense_K

def update_C_with_label_connections(
        zero_contrib,
        C_general,
        train_labels_general,
        labeled_points_mask_general,
        diff_method,
        cum_labels_info=None,
):
    """
    Update C matrix with newly available connections originating from the labeled set.

    Args:
        zero_contrib (dict): Dictionary with 'sources', 'targets', and 'values' arrays.
        C_general (torch.Tensor): The C matrix to update.
        train_labels_general (np.ndarray): Array of training labels.
        labeled_points_mask_general (torch.Tensor): Boolean mask for labeled points.
        diff_method (str): The differencing method ('prob_cover', 'max_herding', or other).
        cum_labels_info (torch.Tensor, optional): Cumulative labels info tensor (required for non-prob_cover/max_herding methods).
    """
    if not zero_contrib:
        return

    sources = zero_contrib.get("sources")
    targets = zero_contrib.get("targets")
    values = zero_contrib.get("values")

    if sources is None or targets is None or values is None:
        return

    if len(sources) == 0:
        return

    sources_np = np.asarray(sources, dtype=np.int64)
    targets_np = np.asarray(targets, dtype=np.int64)
    values_np = np.asarray(values, dtype=np.float32)

    device = C_general.device

    labels_np = train_labels_general[sources_np]
    targets_t = torch.from_numpy(targets_np).to(device=device, dtype=torch.long)
    labels_t_full = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
    values_t_full = torch.from_numpy(values_np).to(device=device, dtype=C_general.dtype)

    unlabeled_mask = ~labeled_points_mask_general[targets_t]
    if not torch.any(unlabeled_mask):
        return

    targets_t = targets_t[unlabeled_mask]
    labels_t = labels_t_full[unlabeled_mask]
    values_t = values_t_full[unlabeled_mask]

    if values_t.numel() == 0:
        return

    if diff_method in ['prob_cover', 'max_herding']:
        targets_cpu = targets_t.cpu().numpy()
        labels_cpu = labels_t.cpu().numpy()
        values_cpu = values_t.cpu().numpy()

        pair_to_max = {}
        for tgt, lab, val in zip(targets_cpu, labels_cpu, values_cpu):
            key = (tgt, lab)
            current_val = pair_to_max.get(key)
            if current_val is None or val > current_val:
                pair_to_max[key] = val

        for (tgt, lab), val in pair_to_max.items():
            current_tensor = C_general[tgt, lab]
            if val > current_tensor.item():
                C_general[tgt, lab] = current_tensor.new_tensor(val)
    else:
        C_general.index_put_((targets_t, labels_t), values_t, accumulate=True)
        if cum_labels_info is not None:
            cum_labels_info.index_put_((labels_t,), values_t, accumulate=True)