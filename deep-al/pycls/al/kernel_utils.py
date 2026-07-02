"""
Kernel utility functions for building and manipulating kernel matrices.
Extracted from BAYES_MISP.py for reuse across different active learning methods.
"""
import numpy as np
import torch
import scipy.sparse as sp


def compute_norm(x1, x2, device, batch_size=512, matrices_type=torch.float16):
    """
    Compute pairwise distance matrix between two feature sets in batches.
    
    Args:
        x1: Feature matrix of shape (n, d)
        x2: Feature matrix of shape (n', d)
        device: Device to perform computation on
        batch_size: Batch size for processing
        matrices_type: Data type for computation
        
    Returns:
        Distance matrix of shape (n, n')
    """
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


def _apply_degree_normalization_sparse(csr_matrix, alpha, degree_normalization_method='none'):
    """
    Apply degree normalization to a sparse CSR matrix.
    - 'none': K_ij <- K_ij / (degree_i * degree_j)^(alpha/2)
    - 'sum' or 'max': K_ij <- K_ij / (normalized_degree_i * normalized_degree_j)
    - 'log': K_ij <- K_ij / log(degree_i * degree_j)
    
    Degrees are computed excluding diagonal elements.
    Degrees can be optionally normalized before applying the transformation.
    If degree_i or degree_j is 0, K_ij is set to 0.
    
    Args:
        csr_matrix: Sparse CSR matrix (N, N)
        alpha: Normalization parameter (only used for 'none' method)
        degree_normalization_method: Method to normalize degrees ('none', 'sum', 'max', 'log')
        
    Returns:
        Normalized sparse CSR matrix
    """
    csr_no_diag = csr_matrix.copy()
    csr_no_diag.setdiag(0)
    csr_no_diag.eliminate_zeros()
    
    degrees = np.asarray(csr_no_diag.sum(axis=1)).flatten()
    
    # Normalize degrees based on the specified method
    if degree_normalization_method == 'sum':
        degree_sum = degrees.sum()
        if degree_sum > 0:
            degrees = degrees / degree_sum
    elif degree_normalization_method == 'max':
        degree_max = degrees.max()
        if degree_max > 0:
            degrees = degrees / degree_max
    # else: 'none' or 'log' - no pre-normalization
    
    coo = csr_matrix.tocoo()
    
    row_degrees = degrees[coo.row]
    col_degrees = degrees[coo.col]
    
    degree_product = row_degrees * col_degrees
    
    zero_mask = (degree_product == 0)
    
    degree_product_safe = degree_product.copy()
    degree_product_safe[zero_mask] = 1.0
    
    # Apply transformation based on method
    if degree_normalization_method == 'none':
        normalization_factor = np.power(degree_product_safe, alpha / 2.0)
    elif degree_normalization_method == 'log':
        normalization_factor = np.power(np.log(1 + degree_product_safe), 0.5)
    else:  # 'sum' or 'max'
        normalization_factor = degree_product_safe
    
    normalized_data = coo.data / normalization_factor
    normalized_data[zero_mask] = 0.0
    
    normalized_coo = sp.coo_matrix(
        (normalized_data, (coo.row, coo.col)),
        shape=csr_matrix.shape
    )
    
    return normalized_coo.tocsr()


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
        knn_distances=None,
        alpha=0.5,
        degree_normalization_method='none',
):
    """
    Build a symmetric sparse kernel matrix in CSR format without materializing the full dense matrix.

    Args:
        features (np.ndarray or torch.Tensor): Feature matrix of shape (N, D) on CPU.
        threshold (float): Value used to sparsify the kernel. For 'rbf', 'local_rbf' and 'cknn' this is the
                           minimum kernel value kept. For 'tophat' it is the distance cutoff (delta).
        kernel_type (str): 'rbf', 'tophat', 'cknn', or 'local_rbf'.
        kernel_param (float or int): Kernel-specific parameter. For 'rbf' and 'local_rbf' this is sigma.
                                     For 'tophat' it is ignored. For 'cknn' this is k (int neighbors).
        batch_size (int): Number of rows processed per chunk.
        device (str or torch.device): Device used for intermediate GPU computations.
        dtype (torch.dtype): Dtype for intermediate tensors (float32 recommended for sparse path).
        zero_indices (array-like, optional): Indices whose rows/cols should be zeroed in the returned matrix.
        prev_threshold (float, optional): Previous threshold value. Required when
            capture_zero_contrib=True to identify newly added connections.
        capture_zero_contrib (bool): When True, returns the contributions that
            originate from zeroed indices but were newly introduced by lowering
            the sparsity threshold.
        knn_distances (np.ndarray, optional): Pre-computed k-th NN distances of shape (N,).
            Only used when kernel_type='cknn'. If None, computed automatically.
        alpha (float): Degree normalization parameter for 'local_rbf' kernel (used for 'none' method).
        degree_normalization_method (str): Method to normalize degrees ('none', 'sum', 'max', 'log').
            Only used when kernel_type='local_rbf'.

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

    if kernel_type in ('rbf', 'local_rbf'):
        kernel_param_val = kernel_param.item() if torch.is_tensor(kernel_param) else float(kernel_param)

    # Pre-compute k-NN distances for CkNN kernel (needed before the batch loop)
    cknn_r = None
    if kernel_type == 'cknn':
        k_neighbors = int(kernel_param.item() if torch.is_tensor(kernel_param) else kernel_param)
        if knn_distances is not None:
            cknn_r = np.asarray(knn_distances, dtype=np.float32)
        else:
            cknn_r = compute_knn_distances(
                features_tensor, k=k_neighbors, batch_size=batch_size, device=device
            )
        cknn_r_tensor = torch.from_numpy(cknn_r).to(device=torch_device, dtype=torch.float32)

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
                elif kernel_type in ('rbf', 'local_rbf'):
                    kernel_block = torch.exp(-1.0 * (dist_block / kernel_param_val) ** 2)
                elif kernel_type == 'cknn':
                    r_i = cknn_r_tensor[start_i:end_i]   # (B_i,)
                    r_j = cknn_r_tensor[start_j:end_j]   # (B_j,)
                    denom = torch.outer(r_i, r_j).clamp(min=1e-10)
                    kernel_block = torch.exp(-(dist_block ** 2) / denom)
                else:
                    raise ValueError(f"Unsupported kernel type: {kernel_type}")

                if kernel_type in ('rbf', 'cknn', 'local_rbf'):
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

                if kernel_type in ('rbf', 'cknn', 'local_rbf'):
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

    if kernel_type == 'local_rbf':
        csr = _apply_degree_normalization_sparse(csr, alpha, degree_normalization_method)

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


class RBFKernel(object):
    """RBF (Gaussian) kernel implementation."""
    
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, matrices_type=torch.float16):
        """Compute RBF kernel between two feature sets."""
        norm = compute_norm(x1, x2, self.device, batch_size=batch_size, matrices_type=matrices_type)
        k = torch.exp(-1.0 * (norm / h) ** 2)
        return k

    def compute_kernel_from_norm(self, norm_matrix, h, matrices_type=torch.float16):
        """Compute RBF kernel from precomputed distance matrix."""
        k = torch.exp(-1.0 * (norm_matrix / h) ** 2).to(dtype=matrices_type)
        return k


class TopHatKernel(object):
    """Top-hat (indicator) kernel implementation."""
    
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512, matrices_type=torch.float16):
        """Compute top-hat kernel between two feature sets."""
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
        return dist_matrix

    def compute_kernel_from_norm(self, norm_matrix, h, matrices_type=torch.float16):
        """Compute top-hat kernel from precomputed distance matrix."""
        k = (norm_matrix < h).to(dtype=matrices_type)
        return k


def compute_knn_distances(features, k, batch_size=512, device='cpu'):
    """
    Compute the k-th nearest neighbor distance for every point.

    Args:
        features (np.ndarray or torch.Tensor): Feature matrix of shape (N, D).
        k (int): Neighbor index (1-based; k=1 means the closest non-self neighbor).
        batch_size (int): Row chunk size for batched cdist computation.
        device (str or torch.device): Device for intermediate computation.

    Returns:
        np.ndarray: Shape (N,) — the k-th NN distance for each point.
    """
    torch_device = torch.device(device)
    if isinstance(features, torch.Tensor):
        feat = features.to(device=torch_device, dtype=torch.float32)
    else:
        feat = torch.from_numpy(features).to(device=torch_device, dtype=torch.float32)

    n_samples = feat.shape[0]
    # kth distance is at sorted index k (index 0 = self, distance 0)
    knn_idx = min(k, n_samples - 1)
    knn_dists = np.empty(n_samples, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            chunk = feat[start:end]                        # (B, D)
            dists = torch.cdist(chunk, feat)               # (B, N)
            # Sort each row and take the knn_idx-th distance
            sorted_dists, _ = torch.sort(dists, dim=1)    # (B, N) ascending
            knn_dists[start:end] = sorted_dists[:, knn_idx].cpu().numpy()

    return knn_dists


class CkNNKernel(object):
    """
    CkNN (Continuous k-Nearest Neighbors) normalized kernel.

    K(i, j) = exp( -||x_i - x_j||^2 / (r_i(k) * r_j(k)) )

    where r_i(k) is the distance from x_i to its k-th nearest neighbor.
    The bandwidth is locally adaptive — no global sigma required.
    """

    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, k=7, batch_size=512, matrices_type=torch.float16):
        """
        Compute the CkNN kernel matrix between x1 and x2.

        When x1 is x2 (square, self-similarity matrix), kNN distances are
        computed once on the joint set for consistency.

        Args:
            x1 (torch.Tensor): Feature matrix (N, D).
            x2 (torch.Tensor): Feature matrix (M, D).
            k (int): Neighbor count for bandwidth estimation.
            batch_size (int): Batch size for distance computation.
            matrices_type (torch.dtype): Output dtype.

        Returns:
            torch.Tensor: Kernel matrix of shape (N, M) on CPU.
        """
        x1_np = x1.cpu().numpy() if torch.is_tensor(x1) else x1
        x2_np = x2.cpu().numpy() if torch.is_tensor(x2) else x2

        r1 = compute_knn_distances(x1_np, k=k, batch_size=batch_size, device=self.device)
        r2 = compute_knn_distances(x2_np, k=k, batch_size=batch_size, device=self.device)

        r1_t = torch.from_numpy(r1).to(dtype=torch.float32)   # (N,)
        r2_t = torch.from_numpy(r2).to(dtype=torch.float32)   # (M,)

        # Denominator: outer product r_i * r_j  shape (N, M)
        denom = torch.outer(r1_t, r2_t).clamp(min=1e-10)

        dist_matrix = compute_norm(x1, x2, self.device, batch_size=batch_size, matrices_type=torch.float32)
        k_matrix = torch.exp(-(dist_matrix ** 2) / denom).to(dtype=matrices_type)
        return k_matrix

    def compute_kernel_from_norm(self, norm_matrix, knn_dists_1, knn_dists_2, matrices_type=torch.float16):
        """
        Compute CkNN kernel from a precomputed distance matrix.

        Args:
            norm_matrix (torch.Tensor): Pairwise distance matrix (N, M).
            knn_dists_1 (array-like): k-th NN distances for the row points, shape (N,).
            knn_dists_2 (array-like): k-th NN distances for the column points, shape (M,).
            matrices_type (torch.dtype): Output dtype.

        Returns:
            torch.Tensor: Kernel matrix of shape (N, M).
        """
        r1 = torch.as_tensor(knn_dists_1, dtype=torch.float32)
        r2 = torch.as_tensor(knn_dists_2, dtype=torch.float32)
        denom = torch.outer(r1, r2).clamp(min=1e-10)
        k_matrix = torch.exp(-(norm_matrix.float() ** 2) / denom).to(dtype=matrices_type)
        return k_matrix


class LocalRBFKernel(object):
    """
    Local RBF kernel with degree normalization.
    
    First computes standard RBF kernel, then applies transformation based on method:
    - 'none': K_ij <- K_ij / (degree_i * degree_j)^(alpha/2)
    - 'sum' or 'max': K_ij <- K_ij / (normalized_degree_i * normalized_degree_j)
    - 'log': K_ij <- K_ij / log(degree_i * degree_j)
    
    where degree_i = sum of K_i* excluding diagonal (self-similarity).
    When using 'sum' or 'max', degrees are normalized but NO power is applied.
    When using 'log', logarithm is used instead of power.
    If either degree is 0, the kernel value is set to 0.
    """
    
    def __init__(self, device, alpha=0.5, degree_normalization_method='none'):
        self.device = device
        self.alpha = alpha
        self.degree_normalization_method = degree_normalization_method
        self.rbf_kernel = RBFKernel(device)
    
    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, matrices_type=torch.float16):
        """
        Compute degree-normalized RBF kernel between two feature sets.
        
        Args:
            x1: Feature matrix (N, D)
            x2: Feature matrix (M, D)
            h: Bandwidth parameter (sigma)
            batch_size: Batch size for distance computation
            matrices_type: Output dtype
            
        Returns:
            Degree-normalized kernel matrix of shape (N, M)
        """
        K = self.rbf_kernel.compute_kernel(x1, x2, h=h, batch_size=batch_size, matrices_type=torch.float32)
        
        is_symmetric = (x1.shape == x2.shape) and torch.allclose(x1, x2)
        
        if is_symmetric:
            K_no_diag = K.clone()
            K_no_diag.fill_diagonal_(0)
            degrees = K_no_diag.sum(dim=1)
        else:
            degrees_x1 = self.rbf_kernel.compute_kernel(x1, x1, h=h, batch_size=batch_size, matrices_type=torch.float32)
            degrees_x1.fill_diagonal_(0)
            degrees_x1 = degrees_x1.sum(dim=1)
            
            degrees_x2 = self.rbf_kernel.compute_kernel(x2, x2, h=h, batch_size=batch_size, matrices_type=torch.float32)
            degrees_x2.fill_diagonal_(0)
            degrees_x2 = degrees_x2.sum(dim=1)
            
            degrees = (degrees_x1, degrees_x2)
        
        K_normalized = self._apply_degree_normalization(K, degrees, is_symmetric)
        
        return K_normalized.to(dtype=matrices_type)
    
    def _apply_degree_normalization(self, K, degrees, is_symmetric):
        """
        Apply degree normalization:
        - 'none': K_ij / (degree_i * degree_j)^(alpha/2)
        - 'sum' or 'max': K_ij / (normalized_degree_i * normalized_degree_j)
        - 'log': K_ij / log(degree_i * degree_j)
        
        If degree_i or degree_j is 0, set K_ij = 0.
        
        Args:
            K: Kernel matrix (N, M)
            degrees: Degree vector (N,) if symmetric, or tuple (degrees_x1, degrees_x2)
            is_symmetric: Whether the matrix is symmetric
            
        Returns:
            Normalized kernel matrix
        """
        # Normalize degrees based on the specified method
        if is_symmetric:
            normalized_degrees = self._normalize_degrees(degrees)
            degree_product = torch.outer(normalized_degrees, normalized_degrees)
        else:
            degrees_x1, degrees_x2 = degrees
            normalized_degrees_x1 = self._normalize_degrees(degrees_x1)
            normalized_degrees_x2 = self._normalize_degrees(degrees_x2)
            degree_product = torch.outer(normalized_degrees_x1, normalized_degrees_x2)
        
        zero_mask = (degree_product == 0)
        
        degree_product_safe = degree_product.clone()
        degree_product_safe[zero_mask] = 1.0
        
        # Apply transformation based on method
        if self.degree_normalization_method == 'none':
            normalization_factor = torch.pow(degree_product_safe, self.alpha / 2.0)
        elif self.degree_normalization_method == 'log':
            normalization_factor = torch.pow(torch.log(1 + degree_product_safe), 0.5)
        else:  # 'sum' or 'max'
            normalization_factor = degree_product_safe
        
        K_normalized = K / normalization_factor
        K_normalized[zero_mask] = 0.0
        
        return K_normalized
    
    def _normalize_degrees(self, degrees):
        """
        Normalize degree values based on the specified method.
        
        Args:
            degrees: Degree vector (N,)
            
        Returns:
            Normalized degree vector (N,)
        """
        if self.degree_normalization_method == 'sum':
            degree_sum = degrees.sum()
            if degree_sum > 0:
                return degrees / degree_sum
            else:
                return degrees
        elif self.degree_normalization_method == 'max':
            degree_max = degrees.max()
            if degree_max > 0:
                return degrees / degree_max
            else:
                return degrees
        else:  # 'none'
            return degrees
