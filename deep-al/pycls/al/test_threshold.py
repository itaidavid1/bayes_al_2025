import numpy as np
import torch
import sys
import os
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))
import pycls.datasets.utils as ds_utils

import torch
import time
import pickle


def compute_kernel(x1, x2, h, batch_size=512):
    x1, x2 = x1.unsqueeze(0).to('cpu'), x2.unsqueeze(0).to('cpu')  # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        # distance comparisons are done in batches to reduce memory consumption
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset)
        dist = (dist < h).to(dtype=torch.float16)
        dist_matrix.append(dist.cpu())
        del dist
    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    # k = (dist_matrix < h).to(dtype=torch.float16)
    return dist_matrix

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


# Use CUDA events for accurate GPU timing
def time_gpu(func, *args):
    """Times a function's execution on the GPU."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warm-up
    for _ in range(5):
        func(*args)

    torch.cuda.synchronize()
    start.record()

    # Main run
    result = func(*args)

    end.record()
    torch.cuda.synchronize()

    print(f"Time taken: {start.elapsed_time(end):.5f} ms")
    return result


# --- Data Generation ---

def generate_data(L, N, sparsity, device):
    """
    Generates data for both dense and sparse tests.
    K_dense is (D, N, N)
    K_list is a Python list of D sparse (N, N) matrices
    """
    print(f"Generating data for D={L}, N={N}, sparsity={sparsity}...")

    # 1. Create Dense K and then convert to a list of sparse matrices
    K_dense = torch.randn(N, N, device=device)
    # Apply sparsity
    mask = torch.rand(N, N, device=device) > sparsity
    K_dense[mask] = 0

    # This is the "correct" way to hold a batch of sparse matrices
    K_list = []
    # for i in range(L):
    #     # Convert each 2D slice to sparse CSR for fast math
    #     K_list.append(K_dense[i].to_sparse_csr().to(device))

    # 2. Create C and its derivatives (all dense)
    C = torch.rand(N, L, device=device)
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)

    # Handle divide-by-zero just in case
    sum_C[sum_C == 0] = 1.0

    norm_C = (C / sum_C)
    norm_C[norm_C < 0.01] = 0  # threshold

    old_max = (max_C / sum_C)
    C_diff = (C - max_C)  # Shape (D, N)

    # This will be (D, N)
    class_corr = torch.rand(L, N, device=device) > 0.5

    print("Data generation complete.")

    return K_dense, K_list, C, norm_C, sum_C, max_C, C_diff, old_max, class_corr


# --- Dense Version (Corrected) ---

def run_dense_code(K, norm_C, sum_C, max_C, C_diff, old_max, N, chunk_size, device):
    result = torch.empty((N,)).to(device=device)
    cont_method = "positive"  # Using 'fusion' as it's the most complex
    # K = K.unsqueeze(2)
    # sum_C = sum_C.unsqueeze(2)
    # max_C = max_C.unsqueeze(2)
    # C_diff = C_diff.unsqueeze(1)
    # old_max = old_max.unsqueeze(-1)
    for i in range(0, N, int(chunk_size)):
        end = min(i + chunk_size, N)

        # --- Get dense batches ---
        K_batched = K[i:end]  # (chunk, N, N)
        weights_batched = norm_C[i:end]  # (chunk, N)

        # --- Batched dense math ---
        future_sum = K_batched + sum_C
        state_add = max_C + K_batched

        new_state_vec = torch.maximum(-K_batched, C_diff)

        new_state_vec.add_(state_add)
        new_state_vec.div_(future_sum)
        new_state_vec.sub_(old_max)

        if cont_method == "positive":
            new_state_vec.clamp_(min=0)
        # --- The einsum replacement ---
        # 'ijk,ik->i' becomes a batched multiply + sum
        # (chunk, N, N) * (chunk, 1, N) -> (chunk, N, N)
        # .sum(dim=(1, 2)) -> (chunk,)
        einsum_result = torch.einsum('ijk,ik->i', new_state_vec, weights_batched)
        result[i:end] = einsum_result

    return result


# --- Sparse Version (Loop-based) ---

def run_sparse_code(K_list, norm_C, sum_C, max_C, C_diff, old_max, class_corr, D, N, chunk_size, device):
    result = torch.empty((D,)).to(device=device)
    cont_method = "fusion"

    for i in range(0, D, int(chunk_size)):
        end = min(i + chunk_size, D)

        # --- Get batches ---
        # K_list is a *Python list* of sparse matrices
        K_batched_list = K_list[i:end]
        weights_batched = norm_C[i:end]  # (chunk, N)
        class_corr_batched = class_corr[i:end]  # (chunk, N)
        sum_C_batched = sum_C[i:end]  # (chunk, 1)
        max_C_batched = max_C[i:end]  # (chunk, 1)
        C_diff_batched = C_diff[i:end]  # (chunk, N)
        old_max_batched = old_max[i:end]  # (chunk, 1)

        chunk_results = []
        # --- This is the "batched" sparse loop ---
        for j in range(len(K_batched_list)):

            # Get the j-th (single) sparse matrix and dense vectors
            K_j = K_batched_list[j]  # (N, N) sparse
            weights_j = weights_batched[j]  # (N,)
            class_corr_j = class_corr_batched[j].unsqueeze(0)  # (1, N)
            sum_C_j = sum_C_batched[j]  # (1,)
            max_C_j = max_C_batched[j]  # (1,)
            C_diff_j = C_diff_batched[j].unsqueeze(0)  # (1, N)
            old_max_j = old_max_batched[j]  # (1,)

            # --- 2D Sparse Math (all ops are supported) ---
            # sparse + scalar
            future_sum = K_j + sum_C_j
            state_add = K_j + max_C_j

            # sparse vs dense
            new_state_vec = torch.maximum(-K_j, C_diff_j)

            new_state_vec.add_(state_add)
            new_state_vec.div_(future_sum)
            new_state_vec.sub_(old_max_j)

            if cont_method == "fusion":
                is_neg = new_state_vec < 0  # sparse
                # sparse & dense -> sparse
                new_state_vec[is_neg & ~class_corr_j] = 0
                new_state_vec[is_neg & class_corr_j] *= -1

            # --- The 'einsum' replacement ---
            # 'jk,k->' (einsum for 2D)
            # (N, N) sparse * (1, N) dense -> (N, N) sparse
            # .sum() -> scalar
            einsum_result = (new_state_vec * weights_j.unsqueeze(0)).sum()
            chunk_results.append(einsum_result)

        result[i:end] = torch.stack(chunk_results)

    return result


import torch


def slice_csr_rows(csr_tensor, start_row, length):
    """
    Fast slicing for CSR tensors. Replaces .narrow(0, start, length).
    """
    end_row = start_row + length

    crow_indices = csr_tensor.crow_indices()
    col_indices = csr_tensor.col_indices()
    values = csr_tensor.values()

    # 1. Find the data range in the underlying 1D arrays
    # Data starts where row 'start_row' begins
    p_start = crow_indices[start_row]
    # Data ends where row 'end_row' begins
    p_end = crow_indices[end_row]

    # 2. Slice the values and column indices
    new_values = values[p_start:p_end]
    new_col_indices = col_indices[p_start:p_end]

    # 3. Slice and Shift Row Pointers
    # Extract pointers for the specific rows we want
    new_crow_indices = crow_indices[start_row: end_row + 1]
    # Shift them so the first row starts at index 0
    new_crow_indices = new_crow_indices - p_start

    # 4. Create the new CSR tensor
    return torch.sparse_csr_tensor(
        new_crow_indices,
        new_col_indices,
        new_values,
        size=(length, csr_tensor.size(1)),
        dtype=csr_tensor.dtype,
        device=csr_tensor.device
    )


def csr_weighted_sum_collapsed(csr_weights, dense_vec):
    """
    Computes row-wise weighted sums, collapsing the feature dimension.

    Input:
      csr_weights: (Batch, N) - Sparse weights
      dense_vec:   (Batch, N, D) - Dense features

    Output:
      result:      (Batch,) - Scalar result per batch item
    """
    # 1. Components
    crow_indices = csr_weights.crow_indices()
    col_indices = csr_weights.col_indices()
    values = csr_weights.values()  # (NNZ)

    # 2. Decompress Row Indices
    rows_per_value = torch.arange(csr_weights.size(0), device=csr_weights.device).repeat_interleave(
        crow_indices.diff()
    )

    # 3. Gather Dense Values -> Shape (NNZ, 1024)
    # We extract only the vectors that correspond to non-zero weights
    gathered_dense = dense_vec[rows_per_value, col_indices]

    # --- OPTIMIZATION: Sum features FIRST ---
    # Instead of multiplying (NNZ, 1024) * (NNZ, 1), we sum the 1024 features now.
    # This reduces the problem from Matrix math to Vector math.
    # Shape: (NNZ, 1024) -> (NNZ,)
    gathered_sum = gathered_dense.sum(dim=1)

    # 4. Multiply Weights * Summed_Features
    # Shape: (NNZ,) * (NNZ,) -> (NNZ,)
    products = values * gathered_sum

    # 5. Aggregate back to Batch rows
    # Shape: (Batch,)
    result = torch.zeros(csr_weights.size(0), device=csr_weights.device, dtype=csr_weights.dtype)
    result.index_add_(0, rows_per_value, products)

    return result


def apply_shared_mask_to_batch(template_csr, batch_dense):
    """
    Applies the sparsity pattern of a 2D CSR matrix to a 3D Dense Batch.

    Args:
        template_csr: Sparse CSR (Rows, Cols) - Your 'norm_C'
        batch_dense:  Dense (Batch, Rows, Cols) - Your 'other_matrix'

    Returns:
        Sparse Batched CSR Tensor (Batch, Rows, Cols)
    """
    # 1. Get coordinates
    rows_per_value = torch.arange(template_csr.size(0), device=template_csr.device).repeat_interleave(
        template_csr.crow_indices().diff()
    )
    col_indices = template_csr.col_indices()

    # 2. Extract values (The heavy lifting)
    # Shape: (Batch, NNZ)
    gathered_vals = batch_dense[:, rows_per_value, col_indices]

    # 3. Build Batched CSR
    B = batch_dense.size(0)

    # We expand the indices to match the batch size.
    # Note: We use .contiguous() on indices only if PyTorch throws a stride error,
    # but usually .expand() is sufficient for current versions.
    return torch.sparse_csr_tensor(
        template_csr.crow_indices().unsqueeze(0).expand(B, -1),
        template_csr.col_indices().unsqueeze(0).expand(B, -1),
        gathered_vals,
        size=batch_dense.shape,
        dtype=batch_dense.dtype,
        device=batch_dense.device
    )

def run_minimal_sparse_code(K, norm_C, sum_C, max_C, C_diff, old_max, N, chunk_size,threshold, device):
    result = torch.empty((N,)).to(device=device)
    cont_method = "positive"
    norm_C[norm_C < threshold] = 0
    norm_C = norm_C.to_sparse_csr()
    # K = K.unsqueeze(2)
    # sum_C = sum_C.unsqueeze(2)
    # max_C = max_C.unsqueeze(2)
    # C_diff = C_diff.unsqueeze(1)
    # old_max = old_max.unsqueeze(-1)
    for i in range(0, N, int(chunk_size)):
        end = min(i + chunk_size,N )

        K_batched = K[i:end]  # (chunk, N, N)
        weights_batched = slice_csr_rows(norm_C, i, min(chunk_size, N - i) ) # (chunk, N)

        # --- Batched dense math ---
        future_sum = K_batched + sum_C
        state_add = max_C + K_batched

        new_state_vec = torch.maximum(-K_batched, C_diff)

        new_state_vec.add_(state_add)
        new_state_vec.div_(future_sum)
        new_state_vec.sub_(old_max)

        if cont_method == "positive":
            new_state_vec.clamp_(min=0)
            new_state_vec_sparse = apply_shared_mask_to_batch(norm_C, new_state_vec)

        # einsum_result = csr_weighted_sum_collapsed(weights_batched, new_state_vec)

        w_values = weights_batched.values()  # Shape: (Total_NNZ_in_Chunk)

        # 2. Get State Values (The robust way)
        # Since new_state_vec is a Sparse Batched CSR, slicing it like [i:end] might
        # throw a stride error depending on PyTorch version.
        # The safest way is to slice the 1D values array directly.

        # Calculate where the values for this chunk start and end
        # We rely on the fact that every batch item has the same number of non-zeros
        # (because they were built from the same norm_C template)
        total_nnz = new_state_vec_sparse.values().size(0)
        nnz_per_row = total_nnz // new_state_vec_sparse.size(0)  # e.g., Total / 15000

        start_val_idx = norm_C.crow_indices()[i].item()
        end_val_idx = norm_C.crow_indices()[end].item()

        # NOW we slice the values.
        # This will return a slice of size ~5102, matching w_values.
        s_values = new_state_vec_sparse.values()[start_val_idx: end_val_idx]

        # 3. Multiply (The "Einsum")
        # Handle broadcasting if s_values has an embedding dim (D)
        if s_values.dim() > 1:
            w_values = w_values.unsqueeze(1)  # (NNZ, 1)

        products = w_values * s_values  # (NNZ, D)

        # 4. Aggregate (Scatter Sum)
        # We need to map these products back to their batch row (0..Chunk_Size)
        # We generate the row map for just this chunk
        rows_per_value = torch.arange(weights_batched.size(0), device=weights_batched.device).repeat_interleave(
            weights_batched.crow_indices().diff()
        )

        # Initialize result chunk
        chunk_res = torch.zeros((weights_batched.size(0), products.shape[-1]), device=weights_batched.device)

        # Sum
        chunk_res.index_add_(0, rows_per_value, products)
        result[i:end] = chunk_res

    return result

def run_sparse_code_v2(K_csr, norm_C, sum_C, max_C, C_diff, old_max, N, chunk_size, device):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    cont_method = "positive"
    # K_csr = K.to_sparse_csr()
    # Move CSR components to device
    crow = K_csr.crow_indices().to(dev)  # shape (D+1,)
    ccol = K_csr.col_indices().to(dev)  # shape (nnz,)
    cvals = K_csr.values().to(dev)  # shape (nnz,)
    D = crow.numel() - 1
    N = C.shape[0]
    classes = C.shape[1]
    assert D == N, "Expected D == N"

    result = torch.empty((N,)).to(device=device)

    for row_start in range(0, D, chunk_size):
        row_end = min(row_start + chunk_size, D)
        b = row_end - row_start

        # CSR pointers for the chunk
        starts = crow[row_start:row_end]  # shape (b,)
        ends = crow[row_start + 1: row_end + 1]  # shape (b,)
        lengths = (ends - starts).to(torch.long)  # (b,)

        total_nnz = int(lengths.sum().item())
        if total_nnz == 0:
            # nothing in this chunk, skip
            continue

            # global slice of indices/values for this chunk
        slice_start = int(starts[0].item())
        slice_end = int(ends[-1].item())

        cols_all = ccol[slice_start:slice_end]  # (total_nnz,)
        vals_all = cvals[slice_start:slice_end]  # (total_nnz,)

        # row index for each nnz entry within the chunk: 0..b-1 repeated by lengths
        row_indices = torch.repeat_interleave(torch.arange(b, device=dev, dtype=torch.long),
                                              lengths)  # (total_nnz,)

        # Map chunk row-local indices -> global row indices (if needed)
        global_rows = torch.arange(row_start, row_end, device=dev, dtype=torch.long)  # (b,)

        # Now compute per-nnz dense class arrays on GPU:
        # shapes:
        # - vals_all: (total_nnz,)
        # - cols_all: (total_nnz,)
        # - C_diff[cols_all]: (total_nnz, classes)
        kvals = vals_all  # (total_nnz, 1)
        sumC_cols = sum_C[cols_all]  # (total_nnz, 1)
        maxC_cols = max_C[cols_all]  # (total_nnz, 1)
        old_max_cols = old_max[cols_all]  # (total_nnz, 1)
        Cdiff_cols = C_diff[cols_all]

        negk = -kvals  # (total_nnz,1)
        # maximum between negk and Cdiff_cols: broadcast negk on classes dimension
        # torch.maximum requires same shape; expand negk to (total_nnz, classes)
        negk_expand = negk.expand(classes, -1).T  # (total_nnz, classes)
        new_state = torch.maximum(negk_expand, Cdiff_cols)  # (total_nnz, classes)

        state_add = maxC_cols + kvals  # (total_nnz,1)
        new_state = new_state + state_add.expand(classes, -1).T  # add per-row scalar across classes

        future_sum = (kvals + sumC_cols)  # (total_nnz,1)
        # divide
        new_state = new_state / future_sum.expand(classes, -1).T
        # subtract old_max (per column)
        new_state = new_state - old_max_cols.expand(classes, -1).T

        # Now apply continuation method
        if cont_method == "positive":
            new_state.clamp_(min=0.0)

        weights_chunk = norm_C[global_rows]  # (b, classes)
        # Now map per-nnz: weights_for_nnz = weights_chunk[row_indices]
        weights_for_nnz = weights_chunk[row_indices]  # (total_nnz, classes)

        # Multiply elementwise and sum over classes -> per-nnz scalar
        per_nnz_weighted = (new_state * weights_for_nnz).sum(dim=1)  # (total_nnz,)

        # Aggregate per row via scatter_add
        chunk_result = torch.zeros((b,), device=dev, dtype=C.dtype)
        chunk_result.scatter_add_(0, row_indices, per_nnz_weighted)

        result[row_start:row_end] = chunk_result

    return result



if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"Using device: {torch.cuda.get_device_name(DEVICE)}")

    # --- Parameters ---
    L = 10  # Batch size
    N = 50000  # Matrix size
    CHUNK_SIZE = 512
    SPARSITY = 0.98  # 95% zeros

    weighted_alpha_01_lset = [12763, 48804, 36863, 40453, 46313, 44436, 15302, 48657, 34025, 44459, 5536, 278, 11661, 42936, 44620, 30514, 21897, 34209, 14374, 19729, 45494, 23129, 12303, 39524, 6381, 971, 29506, 35385, 47457, 25190, 37991, 12620, 14227, 26671, 18481, 37487, 23582, 17159, 40860, 13783, 10375, 21955, 33774, 37243, 12075, 31548, 49642, 23006, 30316, 47047, 42842, 6492, 25987, 38057, 49081, 29507, 836, 29894, 16501, 33306, 32523, 4615, 4718, 6391, 45329, 19355, 11735, 46393, 8508, 26576, 43475, 5002, 4178, 21695, 13831, 31477, 17102, 42951, 25817, 47580, 22077, 942, 22400, 48950, 34340, 44556, 44760, 32249, 35672, 35688, 45349, 32294, 40769, 3447, 39305, 42148, 16266, 25465, 21245, 27115, 21088, 28212, 36044, 16824, 7040, 29979, 17419, 28528, 21577, 13730, 21689, 32578, 19567, 23176, 5168, 41172, 42545, 42030, 16264, 24582, 23265, 46926, 34390, 2689, 26789, 20615, 560, 39525, 24720, 11516, 39414, 43484, 34110, 36384, 39035, 36538, 47771, 40760, 33213, 49169, 11627, 5769, 38379, 44266, 103, 24536, 27364, 23530, 17917, 6354, 16229, 10425, 20810, 20942, 6302, 43469, 38674, 36555, 38894, 46390, 34467, 35294, 44647, 47657, 15379, 27150, 29887, 31513, 30498, 15988, 40402, 26206, 19545, 40355, 16485, 27571, 7302, 46539, 2703, 45283, 34739, 8157, 13943, 8371, 38822, 625, 16641, 18156, 15448, 22466, 21244, 15081, 46162, 42367, 4041, 26469, 26400, 31572, 31946, 39646, 7750, 24501, 3534, 26059, 23680, 7737, 31932, 27077, 15934, 28416, 14960, 12889, 21020, 13660, 22333, 42149, 49198, 27935, 33382, 6830, 22991, 27377, 22269, 43791, 12565, 15686, 1574, 2257, 4951, 13141, 19265, 27678, 7104, 18559, 7529, 25771, 14492, 36794, 12278, 28614, 46101, 21092, 14964, 32675, 31845, 11570, 48835, 21381, 43737, 28632, 21779, 8254, 40578, 26425, 22175, 27392, 10799, 48152, 37841, 25369, 782, 30237, 5759, 12536, 12553, 9644, 7074, 35950, 34131, 15849, 31833, 13259, 19027, 23465, 35600, 18832, 16053, 5437, 23513, 38139, 30835, 1712, 49909, 29914, 12603, 4529, 39445, 13619, 31643, 32714, 39197, 46525, 34155, 7896, 40443, 29389, 19678, 12861, 39361, 40956, 34509, 21679, 35005, 31518, 37443, 6688, 25596, 47608, 42998, 24105, 19196, 20095, 27129, 3690, 38512, 6011, 37737, 23470, 6666, 35245, 35703, 16213, 30846, 46764, 35228, 38927, 43370, 42923, 30734, 12698, 36836, 47077, 23724, 20463, 28994, 44144, 22710, 29151, 23607, 1528, 5247, 32600, 2125, 40801, 35471, 30574, 28194, 6925, 2903, 20264, 36196, 40141, 14129, 30048, 18136, 10901, 38026, 47937, 3515, 49859, 35076, 24472, 27465, 38850, 9692, 41854, 47784, 13391, 22594, 21170, 3818, 49821, 40534, 43371, 13018, 11210, 42107, 2206, 40169, 25732, 27315, 39288, 47719, 28716, 16486, 24552, 6309, 1079, 1400, 11125, 30853, 43332, 36234, 43798, 41160, 41726, 23245, 36451, 31843, 25482, 16734, 7776, 38010, 19937, 38360, 48467, 13462, 30959, 3990, 25592, 6591, 14604, 22595, 9971, 9908, 38558, 46936, 23765, 45714, 17397, 3657, 22909, 8093, 12661, 32331, 45073, 14567, 14713, 40478, 29052, 40442, 4114, 11433, 28189, 18910, 25991, 20655, 42653, 18813, 49674, 14681, 8714, 10410, 4710, 12022, 9821, 40616, 40157, 2640, 216, 24535, 3565, 30226, 42831, 30092, 16027, 32254, 13518, 45432, 35923, 44032, 29130, 3149, 2648, 45868, 7293, 14304, 42816, 17087, 16125, 40116, 25133, 25294, 30163, 42301, 26012, 24670, 47377, 30998, 29521, 49652, 29760, 8942, 27484, 26529, 41096, 24060, 19403, 17512, 24887, 23213, 7609, 5848, 10055, 45396, 31107, 48246, 11265, 40914, 18160, 22974, 48816, 38757, 8397, 7473, 32657, 43294, 41209, 19360, 19054, 16458, 28565, 39869, 23521, 16893, 7359, 5655, 44908, 42226, 18660, 18515, 36734, 25231, 31253, 16531, 24429, 2955, 34398, 47209, 21116, 12545, 32365, 45820, 45222, 5783, 33208, 23871, 41835, 5923, 2226, 3734, 16104, 39103, 8433, 14734, 18641, 30407, 6106, 45977, 25949, 9514, 8009, 22668, 27910, 9449, 3691, 18511, 18447, 47060, 21388, 4287, 37098, 36626, 49265, 13156, 39015, 36595, 41335, 27765, 28545, 47131, 43599, 28407, 9996, 26878, 13998, 3329, 3194, 46415, 17173, 31801, 13785, 31653, 29045, 316, 25250, 24651, 5230, 12651, 23184, 15755, 217, 16377, 551, 34771, 43113, 18169, 37449, 14000, 46151, 4726, 28682, 6300, 18235, 26051, 4627, 48877, 18377, 15765, 29001, 20699, 2240, 19327, 10909, 49820, 31250, 32491, 16507, 15636, 23388, 42137, 40194, 6204, 25199, 21529, 48775, 46898, 7502, 18850, 4540, 1917, 25791, 45606, 17260, 18935, 22660, 14049, 12870, 7233, 4379, 16105, 37434, 24631, 22782, 39896, 31095, 12386, 29060, 12265, 41898, 28512, 14199, 46131, 34966, 4706, 38671, 17213, 28981, 18471, 16906, 19585, 4692, 27151, 20823, 11411, 35888, 37609, 9753, 7409, 38854, 29008, 24609, 44371, 8733, 23444, 15591, 17915, 34181, 36352, 21884, 10610, 5840, 29794, 12691, 16646, 49692, 34005, 23089, 39773, 34798, 15885, 2524, 47799, 33577, 2167, 14190, 43664, 10076, 18427, 40920, 30318, 24882, 6360, 48205, 3910, 33532, 21631, 34, 41144, 39782, 16520, 43386, 9101, 20499, 45435, 43280, 32309, 40458, 30940, 21601, 25301, 41207, 33053, 47682, 36211, 40736, 20444, 22381, 23097, 21591, 37834, 2019, 19197, 9004, 49545, 1146, 18784, 41619, 34908, 30760, 45554, 13252, 24981, 27812, 5498, 7089, 33368, 25513, 10329, 42584, 34625, 15745, 40969, 37779, 4759, 49278, 10043, 40882, 34414, 12987, 2102, 47527, 36681, 33057, 20921, 35213, 19989, 42763, 10309, 19529, 6444, 43657, 9316, 15539, 6877, 37768, 33497, 5264, 29981, 8017, 4140, 32463, 35391, 12779, 34497, 16865, 772, 32065, 11468, 6293, 40682, 31644, 8961, 35493, 25565, 45114, 15630, 35852, 48316, 27035, 2305, 12962, 24621, 302, 49179, 44671, 20478, 42850, 7789, 47340, 38091, 24642, 39236, 15789, 6575, 45093, 19084, 18925, 47965, 26645, 35244, 28056, 5747, 6904, 39956, 44678, 28117, 40332, 49667, 13900, 29325, 48366, 37820, 24490, 1719, 46336, 33145, 21796, 643, 5266, 12111, 20392, 15953, 9507, 40314, 6121, 49238, 13535, 8611, 14671, 34720, 23648, 41680, 32042, 20512, 45019, 16559, 22624, 35528, 19995, 8317, 19957, 14635, 10176, 17514, 3118, 17219, 6838, 28813, 26473, 27184, 28837, 30863, 43561, 32520, 31828, 43550, 27098, 6033, 31779, 14169, 32803, 19800, 44856, 10974, 8844, 45811, 18480, 16898, 4592, 39940, 12868, 2933, 454, 38535, 20200, 44548, 47082, 1358, 41731, 35951, 13524, 2048, 25786, 13969, 42138, 26376, 5421, 3020, 23962, 17376, 3335, 39804, 48378, 44536, 26084, 12422, 3380, 35501, 36308, 42224, 11655, 3094, 36705, 21636, 33526, 26614, 25440, 7930, 9025, 45046, 6311, 21737, 17146, 994, 10993, 43912, 23028, 13875, 31134, 21816, 39898, 30563, 31902, 31452, 24496, 9541, 14477, 44413, 46758, 28785, 4626, 2892, 20062, 26668, 7411, 38906, 31409, 30462, 21868, 42359, 43228, 36568, 38917, 48081, 26777, 41785, 36833, 34794, 26161, 27979, 30719, 22841, 41243, 15345, 964, 503, 36035, 14615, 48214, 22532, 9925, 6328, 1805, 42866, 28225, 17650, 45253, 42280, 11920, 7028, 12089, 45229, 49385, 41730, 695, 49931, 23851, 17735, 25283, 18664, 46796, 28594, 37189, 23659, 19000, 16401, 14067, 21666, 23793, 16319, 1599, 10462, 40367, 15009, 13184, 45136, 25944, 19362, 26915, 6123, 16149, 32571, 23309, 16147, 26200, 12501, 49067, 40205, 520, 37605, 44322, 14980, 23517, 4449, 32307, 24676, 12563, 32462, 39453, 9870, 125, 2836, 13175, 8350, 46856, 45204, 44762, 8046, 45319, 33235, 10952, 26970, 46212, 220, 38531, 35291, 27331, 17220, 11757, 41302, 26759, 3484, 2829, 10945, 4833, 36764, 49387, 43906, 35920, 49272, 7673, 31342, 49609, 39802, 16578, 10977, 42257, 32416, 21132, 48840, 10951, 5844, 39790, 18302, 45498, 30239, 6394, 28544, 30549, 48886, 40104, 28384, 5574, 890, 25932, 45774, 28188, 36273, 11056, 16054, 10723, 23826, 40585, 24325, 45414, 9846, 46370, 24035, 18158, 16610, 41095, 25412, 46839, 11383, 14958, 9366, 28543, 46616, 26521, 27424, 32838, 49579, 40248, 14459, 169, 45839, 40088, 20513, 26028, 31038, 22194, 8185, 16279, 43963, 21572, 15202, 7504, 45934, 19227, 2133, 34472, 3963, 31515, 10890, 9811, 1659, 46705, 22372, 12866, 48965, 29787, 6664, 29725, 17007, 45952, 9862, 7952, 45343, 16047, 39489, 15986, 12711, 6900, 34106, 41675, 39880, 29771, 46422, 44866, 30590, 26721, 29518, 32470, 636, 26114, 28421, 39902, 15336, 20965, 2838, 38113, 7677, 4892, 39923, 6590, 22306, 35020, 15691, 4964, 32087, 25607, 1391, 24195, 26290, 38801, 25921, 25912, 43211, 25976, 9985, 38482, 29673, 17811, 16494, 47673, 14456, 7610, 47747, 47630, 16324, 22488, 19857, 37956, 21887, 21209, 43353, 40894, 20091, 36552, 39641, 28022, 16923, 40158, 34709, 48995, 34902, 9257, 21339, 44358, 24231, 38673, 23694, 34182, 23132, 35149, 18979, 16623, 13794, 47813, 10256, 38999, 23933, 3364, 25849, 48680, 12198, 43185, 33686, 11313, 38690, 26060, 38188, 42009, 37844, 27993, 9153, 25109, 32823, 16389, 40811, 37005, 35508, 17431, 17987, 13399, 11936, 38280, 46392, 42158, 4429, 39910, 39475, 35957, 14245, 1160, 27422, 13131, 9859, 29996, 36522, 18876, 2999, 43698, 32779, 9701, 17545, 14905, 9262, 8697, 45166, 41771, 10615, 43368, 29553, 28803, 49923, 37833, 7439, 28237, 7580, 36322, 16528, 10284, 38778, 6648, 4636, 28810, 3816, 43590, 24634, 26515, 25188, 21569, 34154, 7311, 25619, 36389, 47180, 42567, 19406, 7400, 38654, 37325, 17394, 46573, 4451, 16602, 12356, 43152, 43558, 25936, 15164, 14515, 16219, 24281, 18777, 27413, 22541, 10244, 4188, 42305, 15296, 25062, 49441, 29382, 23613, 22748, 35047, 12956, 33144, 3971, 34160, 40331, 43651, 46185, 31670, 21847, 39946, 42534, 30324, 28839, 35342, 20891, 25414, 3086, 32422, 5694, 22911, 34252, 46056, 49803, 43819, 5577, 23545, 6161, 3227, 1720, 37332, 22874, 24601, 4032, 18195, 42395, 39462, 25637, 21560, 45927, 2731, 17624, 4189, 24427, 15779, 15380, 27380, 1522, 20166, 16790, 12239, 21006, 26092, 35786, 45620, 38401, 26854, 42125, 29002, 12799, 31522, 30435, 47260, 35276, 42207, 39541, 31830, 27859, 11881, 26357, 18198, 19488, 2271, 41143, 9132, 6612, 18629, 8805, 23421, 11898, 47492, 28133, 27175, 16148, 4905, 16471, 47254, 38816, 20944, 17503, 1868, 29198, 47440, 31267, 8069, 6307, 49217, 17614, 24889, 9633, 46388, 3536, 2509, 34785, 12295, 32616, 23027, 20113, 26531, 4130, 27843, 10362, 19232, 1649, 39172, 28480, 38476, 21815, 4874, 7498, 2729, 15723, 10321, 33786, 46725, 45171, 27262, 11759, 34630, 10673, 48749, 6316, 47986, 24982, 10235, 27522, 8027, 31869, 37603, 20654, 45285, 26713, 47911, 4604, 3393, 41152, 44492, 32234, 8613, 25622, 41097, 18822, 46337, 35762, 23469, 10769, 44329, 42174, 7564, 48932, 36550, 1165, 47351, 32690, 24687, 30585, 28460, 12555, 11122, 18015, 32289, 11830, 46258, 2673, 31675, 21320, 18102, 3298, 27706, 35889, 6039, 43111, 1930, 40698, 27796, 28893, 22788, 49189, 37129, 45569, 16025, 16216, 254, 39980, 24674, 20773, 47596, 38609, 27585, 42538, 29173, 3800, 27946, 11826, 7309, 14126, 7115, 28324, 39488, 43910, 26751, 37008, 31160, 49704, 15351, 1852, 15032, 2930, 21982, 7114, 19587, 35107, 26250, 13982, 2773, 25717, 23111, 35163, 27493, 5800, 42270, 29544, 20161, 5366, 32472, 538, 28833, 34443, 30694, 42239, 40657, 37316, 30426, 49622, 33585, 22251, 18034, 47648, 9098, 19650, 45583, 34975, 42421, 19852, 30808, 15255, 692, 33506, 47540, 1362, 26447, 30484, 22292, 37803, 10800, 45214, 25679, 38393, 11023, 33264, 13084, 36989, 12598, 39355, 32727, 48879, 45575, 26556, 23272, 24680, 32796, 31161, 38951, 27065, 23376, 14720, 4054, 9282, 37907, 15751, 7196, 4634, 48170, 8085, 45971, 30487, 27538, 25268, 16134, 43936, 13960, 11747, 3148, 29269, 44777, 26783, 49738, 49635, 37059, 35120, 32672, 20449, 46295, 37444, 31736, 5277, 42436, 36856, 27918, 27793, 12797, 1153, 45363, 11342, 3, 27237, 43851, 1394, 10896, 10018, 41727, 2315, 3188, 33938, 48548, 27594, 2391, 839, 26880, 10718, 34951, 3358, 36821, 44311, 33318, 31995, 6905, 30673, 6303, 45759, 1292, 8056, 47846, 41610, 27141, 22335, 45418, 12737, 27345, 9271, 8664, 13617, 24098, 44626, 18700, 6499, 6494, 48390, 27562, 36904, 33638, 31066, 6896, 33179, 42187, 48690, 19654, 8037, 42828, 47524, 34917, 26046, 5267, 26850, 23011, 19195, 11189, 26210, 28523, 3882, 3845, 36227, 16214, 4531, 11639, 10668, 8630, 47509, 42068, 46149, 29714, 31538, 20256, 39550, 19691, 26502, 481, 43679, 30580, 22847, 34355, 5457, 4676, 48173, 45678, 38716, 35708, 31618, 13387, 25749, 20722, 41417, 929, 40841, 40318, 35659, 38499, 45388, 40665, 4795, 48404, 46418, 21720, 25207, 21230, 20918, 15601, 19462, 34077, 13873, 405, 48522, 21122, 49461, 42775, 1601, 20952, 31542, 37397, 48143, 1832, 45424, 14486, 31061, 3139, 19741, 24629, 31120, 19059, 46299, 34811, 3142, 25071, 39307, 25443, 2941, 30675, 21710, 4516, 5100, 46621, 33858, 30608, 14090, 26434, 23779, 18532, 17282, 10862, 48847, 39894, 37695, 3906, 37930, 30217, 27307, 26936, 7252, 45542, 27258, 16071, 30263, 22751, 20051, 6201, 41831, 35766, 8329, 47919, 34323, 32712, 934, 37123, 2415, 14410, 21615, 5363, 9994, 31154, 3374, 29441, 28403, 32992, 34376, 36566, 30552, 10519, 39806, 27207, 25104, 23685, 38314, 15728, 44791, 45279, 4741, 21726, 4475, 29088, 13325, 7791, 29288, 15773, 44975, 40323, 18787, 18537, 610, 22294, 41623, 37797, 6604, 36154, 6806, 1696, 390, 7383, 2717, 990, 48890, 28607, 18805, 16200, 5473, 32591, 10219, 6768, 22970, 45275, 21928, 30629, 14003, 12638, 33540, 31663, 24942, 12794, 9735, 5287, 3916, 43877, 3320, 48993, 47404, 45143, 31221, 25214, 18145, 1500, 44211, 2342, 43654, 39311, 33439, 12284, 43888, 23511, 18001, 38876, 38077, 19356, 7496, 22894, 17393, 16436, 9628, 28613, 25924, 7407, 16463, 20254, 48122, 17105, 7381, 34463, 30464, 3718, 28963, 17398, 16536, 48441, 30140, 24274, 21249, 48148, 38565, 20829, 7755, 41079, 28023, 11054, 41356, 16435, 2139, 402, 29177, 32900, 45556, 30589, 29186, 49575, 46490, 16840, 45918, 37206, 4876, 37124, 31089, 44363, 16166, 42748, 27949, 4486, 18243, 17728, 32330, 25863, 35176, 34420, 7405, 9199, 48353, 37661, 30990, 30340, 28501, 34527, 11021, 38836, 43598, 22060, 157, 24491, 8888, 13078, 43248, 41773, 32504, 25798, 13327, 27414, 26050, 22687, 20286, 23862, 38957, 26542, 23753, 9969, 19210, 15866, 44147, 46909, 21594, 36571, 7355, 48001, 22652, 13648, 31171, 2214, 1855, 45245, 37953, 42800, 40347, 31535, 10676, 9215, 10155, 38311, 12085, 4056, 47127, 14395, 10315, 1503, 16912, 18726, 37622, 31901, 42325, 31247, 36690, 44129, 12322, 3063, 29349, 24531, 29190, 38964, 32259, 46554, 28369, 4479, 49134, 44594, 21439, 12076, 35231, 3940, 2499, 11987, 19916, 32373, 41603, 23058, 6179, 16679, 12976, 47638, 7280, 21257, 12376, 16294, 31261, 10246, 23491, 12787, 831, 2433, 33161, 11593, 37115, 20537, 18918, 47703, 33989, 21215, 20897, 36708, 18603, 9962, 44798, 23352, 34816, 10912, 45134, 30543, 17511, 20893, 44265, 42731, 42394, 25906, 21681, 38703, 13963, 19884, 39358, 30593, 7703, 29603, 47022, 44690, 18170, 10593, 35944, 25985, 12943, 5298, 3749, 10761, 8561, 41995, 13705, 9376, 37130, 32474, 44774, 42402, 23628, 20181, 39346, 38710, 838, 36387, 20272, 14452, 1396, 37113, 17885, 948, 25752, 25016, 13774, 7248, 14112, 42904, 41282, 14040, 6578, 45698, 44948, 43307, 42233, 45120, 1006, 29719, 1526, 30449, 4782, 10542, 1766, 2446, 14493, 13575, 9586, 42147, 18376, 11578, 7875, 40316, 21392, 6133, 29730, 33357, 14548, 10752, 1158, 19902, 32115, 30451, 44452, 48354, 38611, 15690, 2032, 45021, 41910, 17680, 27178, 17346, 35012, 24301, 9237, 48874, 36985, 25691, 16690, 10001, 25468, 10204, 42603, 10365, 40943, 15191, 12308, 41659, 17608, 15506, 45441, 36974, 25488, 19152, 45061, 24661, 47323, 45523, 34634, 17907, 7935, 1992, 36212, 23795, 28284, 20828, 1711, 36937, 28733, 27870, 17792, 11158, 49792, 46432, 46346, 42647, 38238, 38096, 30151, 23718, 22192, 13473, 13413, 13294, 7489, 2416, 37840, 35302, 13890, 7523, 7023, 44261, 31509, 32489, 32177, 24393, 21044, 30502, 8175, 45683, 45145, 43417, 42551, 39532, 35820, 32387, 29662, 21851, 19081, 17994, 17555, 15831, 7227, 9163, 14566, 4392, 46971, 41374, 16867, 18525, 12607, 6312, 16479, 19713, 3431, 46715, 49323, 19176, 12751, 10981, 47352, 38715, 38241, 44468, 5067, 1995, 46845, 35031, 29472, 9618, 49838, 42733, 45155, 33586, 23240, 9099, 41745, 9848, 41817, 14312, 2682, 25156, 3602, 15034, 41142, 8426, 1468, 29525, 18073, 31050, 26749, 10899, 41034, 22039, 35216, 18773, 16049, 18554, 5586, 43930, 24724, 35513, 34379, 24265, 7163, 23323, 11083, 7338, 43281, 36430, 14412, 34145, 24851, 39133, 27459, 36251, 12321, 49582, 46797, 46419, 34749, 14724, 12601, 6601, 42625, 18721, 47869, 36559, 23291, 12924, 183, 7948, 17481, 7414, 13889, 45643, 25356, 47645, 10442, 2132, 31796, 4437, 957, 9303, 47966, 33564, 12654, 13388, 4408, 31224, 17966, 174, 29258, 45912, 24033, 49269, 48111, 25958, 31564, 46360, 22852, 21504, 17309, 40777, 15159, 8043, 49508, 31236, 27666, 24603, 26507, 22507, 39182, 21552, 44009, 43837, 42976, 39737, 49292, 38513, 34152, 14861, 25160, 49576, 27254, 23600, 10402, 46645, 37451, 46973, 40870, 12639, 24828, 17259, 48197, 47578, 8251, 44331, 22837, 15393, 735, 4918, 41588, 4852, 49547, 39177, 12816, 49356, 37672, 2964, 43182, 40710, 30659, 11841, 47183, 46175, 34867, 20391, 16667, 6721, 30222, 37917, 19202, 14286, 13369, 12722, 11101, 26402, 6617, 42337, 14008, 35839, 36135, 6645, 2265, 40263, 2954, 34783, 34353, 6608, 49608, 36931, 48336, 26438, 21953, 12505, 30228, 24546, 18659, 38133, 31016, 37209, 21333, 14630, 1378, 41363, 9029, 1558, 23431, 44429, 33499, 32170, 26117, 21606, 27192, 45522, 37375, 9825, 44164, 7595, 39933, 43025, 22631, 6518, 37527, 37071, 26286, 33434, 22968, 45368, 47690, 44178, 25204, 42598, 18124, 325, 16318, 13742, 27852, 45825, 35006, 33180, 46866, 19330, 18619, 25903, 13645, 5, 28598, 16415, 33985, 39552, 35714, 27544, 19818, 22673, 36283, 9131, 31882, 25571, 3989, 43885, 22363, 43820, 18209, 49467, 48485, 42791, 4052, 39155, 32221, 28907, 15890, 38253, 19719, 23850, 27005, 38699, 10355, 7767, 21141, 20599, 1131, 16849, 47925, 37369, 25512, 1683, 25243, 22520, 8671, 4575, 17864, 43103, 5692, 4548, 48678, 46941, 7119, 2763, 29444, 40231, 12145, 43931, 47761, 3015, 806, 33590, 45094, 21060, 24834, 4199, 43578, 15922, 8681, 3494, 34860, 26185, 21586, 10984, 19943, 23355, 7175, 15645, 42461, 25926, 24352, 20083, 15067, 48097, 19661, 43967, 38817, 44593, 3222, 24727, 44913, 36464, 48084, 47931, 38546, 30732, 18741, 740, 18797, 11770, 25048, 11398, 26422, 17, 29378, 33353, 39714, 21364, 8963, 46777, 26285, 26255, 6273, 43684, 32420, 4355, 46338, 40069, 37989, 6969, 43266, 41920, 10863, 29428, 9417, 2998, 35931, 32094, 2320, 34237, 34866, 18356, 33147, 5825, 21945, 14569, 8846, 7301, 6585, 23152, 11840, 48504, 3907, 30387, 23969, 38266, 15548, 6829, 4333, 31614, 19388, 9349, 34200, 27210, 19754, 49175, 7154, 9244, 27079, 21575, 18820, 19789, 16103, 13386, 39436, 690, 11925, 7894, 17508, 11045, 7044, 8048, 41176, 22744, 2115, 6368, 5960, 45870, 44453, 40788, 11859, 9370, 36010, 39740, 36373, 11714, 21685, 12571, 10687, 9130, 38971, 642, 48876, 47533, 46882, 45509, 42192, 40833, 40385, 38007, 36610, 35836, 24130, 18818, 14426, 10976, 8782, 7178, 3646, 32076, 26364, 10653, 18233, 49245, 29574, 24037, 32576, 11674, 7477, 48194, 33906, 18174, 16525, 4048, 40909, 37877, 23048, 17252, 13882, 43891, 42173, 37632, 34094, 13027, 3225, 34543, 11596, 37132, 27746, 49707, 47050, 44653, 40907, 39847, 39539, 38718, 37934, 33366, 33359, 29427, 16529, 16335, 13863, 8273, 6512, 5608, 3618, 2408, 1879, 1737, 11981, 11407, 38099, 40854, 24055, 49173, 48102, 18933, 13980, 38526, 32455, 47511, 21009, 31473, 44729, 38635, 172, 48892, 35325, 33597, 46793, 45702, 1788, 48532, 27326, 9548, 5814, 45924, 11588, 11382, 1916, 45472, 31519, 29406, 49408, 10704, 4208, 20979, 4443, 34244, 28389, 16693, 14825, 5776, 19078, 43119, 2573, 13284, 37400, 43687, 30883, 5159, 37782, 34529, 22418, 24870, 32134, 47551, 38755, 29921, 21687, 37320, 33696, 25500, 22707, 34437, 34275, 38376, 32440, 33287, 31076, 8535, 47604, 32784, 2695, 7545, 28085, 7814, 1567, 35460, 21825, 17953, 43696, 46726, 42103, 32598, 29835, 23887, 1835, 13025, 28777, 3570, 44414, 43902, 39813, 49741, 30041, 25678, 6208, 42636, 11327, 9261, 48335, 19118, 15733, 8702, 47016, 40849, 39616, 9642, 510, 9437, 42405, 35586, 31290, 20342, 7520, 5913, 2830, 2741, 22564, 20032, 32617, 26522, 1563, 21850, 1138, 45742, 41021, 30006, 6401, 43285, 177, 11541, 43415, 47654, 23350, 40448, 37933, 25295, 18425, 28982, 49751, 41479, 38453, 27348, 981, 876, 18457, 18112, 8094, 204, 46785, 32660, 18690, 3071, 42682, 7965, 39160, 23848, 32901, 12462, 45673, 28153, 23026, 16696, 44380, 26853, 30902, 42188, 10157, 2376, 20565, 3001, 42760, 39814, 36673, 14587, 38313, 35693, 9742, 39941, 43164, 30834, 15311, 45722, 29031, 37368, 12881, 20689, 4401, 41884, 42185, 40026, 5061, 43754, 31730, 16412, 2020, 47344, 35140, 35849, 33132, 30248, 6046, 757, 11086, 39124, 5525, 991, 47287, 36453, 32613, 20692, 43078, 39515, 32766, 17153, 22997, 14991, 6750, 6144, 39336, 24092, 48821, 20778, 45974, 44957, 22431, 25993, 12144, 4024, 15049, 41283, 5313, 1673, 33065, 39661, 23769, 21754, 11332, 11159, 42092, 30588, 20346, 37691, 49770, 48108, 36834, 28330, 5060, 36531, 36669, 34289, 2041, 44512, 400, 39530, 29380, 22769, 5460, 39495, 25281, 39831, 38494, 37719, 38488, 48218, 19564, 1993, 45286, 34188, 13049, 31480, 18253, 5362, 14897, 37308, 33419, 17730, 8336, 2483, 36674, 30527, 10898, 2641, 19100, 7843, 4981, 22923, 33918, 49534, 10068, 1531, 8156, 5300, 30715, 32855, 23394, 24169, 17047, 42127, 20564, 12030, 925, 21802, 45590, 21010, 20026, 8840, 38599, 27101, 17392, 29664, 24334, 20076, 17805, 5541, 32145, 13878, 10746, 8808, 14736, 48705, 4849, 47566, 45827, 36965, 23183, 10503, 21269, 18031, 17411, 11009, 19864, 6442, 36446, 31241, 10549, 2142, 31248, 48862, 18212, 41770, 41461, 7881, 38584, 19241, 10858, 45614, 29352, 28555, 28214, 25425, 32480, 18306, 9980, 9670, 15119, 1605, 13091, 48555, 20935, 15692, 11043, 10928, 43329, 19613, 46597, 38487, 24378, 207, 45812, 44042, 18002, 825, 27990, 25079, 43455, 35258, 31323, 20221, 36757, 12679, 4923, 48338, 42261, 39195, 12708, 9057, 41032, 36802, 1806, 31060, 30627, 20798, 38966, 6113, 14026, 8031, 42414, 33719, 28521, 26573, 25510, 8395, 37597, 47933, 15519, 47345, 16579, 42219, 32261, 21011, 7431, 491, 42434, 40469, 49719, 37076, 29484, 4126, 35482, 29477, 44997, 43705, 28525, 30614, 10541, 22890, 39913, 42818, 26437, 22873, 5737, 2762, 25988, 12524, 8617, 39549, 20044, 7283, 8382, 39611, 35196, 19615, 48400, 38937, 17996, 3844, 45682, 17122, 11379, 14064, 47150, 31358, 29593, 44450, 39013, 45568, 40683, 36766, 40687, 33771, 28392, 47877, 25799, 16860, 10207, 49805, 42697, 18724, 14004, 41751, 43467, 44554, 46132, 37470, 45910, 32410, 7140, 1392, 40480, 21145, 27683, 23966, 46754, 10220, 315, 46792, 38669, 27287, 13028, 14279, 48896, 42583, 8098, 21879, 11600, 908, 22166, 25877, 20845, 16068, 8282, 1761, 49574, 41526, 25751, 20531, 15659, 18159, 12663, 32137, 48643, 38474, 13848, 42533, 43275, 33477, 21291, 18211, 6631, 47884, 44457, 41471, 20130, 23593, 33044, 801, 23017, 21375, 44357, 37310, 14172, 46445, 17745, 3439, 49750, 22027, 8703, 36853, 24100, 2832, 22143, 26381, 3909, 1595, 49527, 37947, 14224, 29527, 17669, 41901, 38407, 7637, 43547, 10632, 20695, 11285, 21669, 13351, 47716, 49482, 22588, 19505, 9795, 1787, 46673, 44806, 9176, 46281, 34764, 15000, 2220, 206, 21352, 19168, 10325, 42681, 38097, 31464, 18638, 15386, 13765, 1753, 29221, 46841, 43775, 7272, 22502, 16391, 10900, 46663, 8542, 38795, 48134, 12530, 26937, 15482, 39761, 17123, 40975, 40151, 39964, 22796, 24471, 22464, 20140, 35386, 26106, 14700, 34997, 37729, 13249, 9190, 44628, 43915, 41220, 29759, 16504, 42747, 40555, 31706, 27745, 19914, 9458, 29284, 34748, 33999, 17532, 37128, 24925, 18336, 40096, 34848, 47157, 45186, 31591, 23049, 6622, 24039, 33688, 27411, 27241, 23286, 19384, 7168, 37463, 48412, 29399, 27036, 26198, 6694, 11234, 9863, 26616, 17600, 12399, 35692, 19900, 18224, 38913, 33068, 4725, 31359, 23623, 44148, 42929, 36527, 28564, 12888, 44874, 37087, 25695, 2672, 47194, 32151, 31208, 28885, 21303, 9137, 3567, 48150, 43949, 18756, 35525, 10956, 32269, 18758, 17080, 4026, 34058, 30801, 26630, 22711, 19375, 13500, 26189, 25923, 18882, 49700, 47447, 42151, 40662, 16468, 5138, 277, 227, 30537, 14171, 9605, 40608, 18412, 9039, 8907, 40623, 28296, 13013, 2575, 44658, 16814, 12766, 29405, 18620, 16666, 13975, 40377, 34038, 30405, 48551, 43513, 40793, 40217, 38254, 37756, 34793, 28424, 27663, 25832, 21470, 20625, 17189, 16526, 16280, 13814, 5808, 616, 26597, 23462, 7764, 18962, 34980, 26173, 32386, 38471, 5593, 21717, 25119, 16515, 40372, 39468, 25146, 1813, 1123, 49231, 43549, 37341, 31875, 29492, 24335, 2662, 39184, 28357, 19555, 18861, 17740, 15039, 11726, 975, 39754, 15753, 1647, 28363, 1388, 30500, 21313, 20833, 18229, 6671, 5575, 31839, 29336, 27677, 25629, 15996, 7886, 3210, 31816, 26301, 42401, 14510, 49743, 49226, 49190, 48991, 48327, 47938, 47330, 46426, 45453, 45347, 44084, 43577, 43067, 41754, 40825, 39739, 38806, 38581, 37425, 37287, 37134, 37055, 35949, 34579, 32427, 30698, 30188, 28052, 26803, 26242, 26135, 24852, 24348, 24218, 23783, 23677, 23590, 22851, 22197, 22025, 19374, 19287, 18964, 18959, 18622, 18199, 16920, 16273, 16121, 15958, 15775, 14283, 13825, 13231, 12935, 12518, 11550, 10861, 10736, 10470, 9497, 8918, 8462, 8196, 7539, 7360, 6532, 6414, 4563, 3725, 2755, 2565, 1338, 770, 38420, 2645, 32152, 4600, 13911, 2495, 39134, 48544, 46981, 49624, 32954, 30577, 28235, 14094, 39930, 28014, 13330, 18061, 17417, 46439, 39948, 35743, 24226, 15183, 22836, 36229, 2414, 41341, 18374, 4815, 37731, 34369, 39114, 13791, 33962, 40027, 33804, 10390, 45612, 19173, 43539, 30197, 20624, 18103, 44851, 43485, 41781, 34567, 15217, 2236, 1407, 43650, 8530, 43488, 17797, 46509, 22889, 10766, 9716, 36653, 31577, 22736, 11668, 45263, 5847, 8303, 34796, 16131, 39659, 10326, 42362, 40964, 4141, 36142, 15170, 10262, 49424, 15373, 26867, 20467, 41133, 750, 37594, 29557, 8422, 1372, 47097, 26388, 10543, 42183, 35116, 26080, 17475, 47770, 30571, 28259, 17023, 3189, 43358, 40799, 18262, 10586, 15284, 7217, 20012, 15155, 9546, 15619, 13382, 10696, 25224, 20930, 34071, 27809, 9053, 46959, 5622, 42121, 4658, 40934, 36108, 8075, 39405, 34328, 20886, 35744, 35549, 21234, 11025, 11573, 1572, 44021, 35720, 41600, 18682, 43379, 37769, 8083, 32875, 34887, 12542, 43129, 35316, 29243, 27917, 21153, 6637, 43034, 24874, 6599, 43813, 21617, 19413, 30103, 3239, 49002, 16965, 48598, 28219, 31448, 30080, 1639, 170, 32610, 28743, 48948, 44784, 32880, 29958, 18072, 16069, 43769, 40287, 36493, 6942, 23, 44506, 27279, 35370, 27904, 23860, 14113, 43433, 36480, 24770, 22740, 17045, 10352, 38885, 27336, 25386, 19771, 19629, 34259, 3305, 2260, 29665, 16753, 4898, 28026, 15408, 33300, 17824, 15404, 36093, 33571, 24523, 18059, 13440, 466, 41202, 12124, 26360, 38235, 49368, 48056, 14696, 11589, 37966, 37414, 43471, 6547, 31547, 48645, 2089, 1298, 33259, 38432, 35534, 31587, 28988, 12968, 2975, 38851, 3905, 13307, 42462, 35523, 24221, 5717, 8713, 49266, 9423, 8953, 11644, 2276, 44990, 14827, 32153, 17806, 11426, 2788, 43183, 34962, 25127, 16337, 5051, 43220, 26180, 12457, 48425, 16338, 18133, 11691, 45915, 33519, 30689, 25555, 12900, 8632, 24002, 9615, 1530, 40498, 39138, 13405, 21882, 8072, 32644, 28247, 43551, 26928, 19459, 48185, 31884, 18105, 31601, 16107, 9484, 33312, 23170, 41693, 37810, 29327, 36616, 24021, 40461, 37509, 30143, 14015, 4191, 8220, 45068, 49360, 25642, 16136, 5332, 29184, 24786, 23464, 27224, 26517, 21677, 10187, 49315, 36513, 44423, 43843, 39219, 36030, 8195, 49706, 45841, 49783, 20734, 8683, 18584, 11314, 49852, 33531, 46312, 41020, 49762, 37456, 13528, 21983, 45623, 16962, 42037, 33250, 2255, 18474, 2663, 606, 24343, 41150, 8774, 43649, 20681, 22538, 17741, 29858, 23975, 18613, 348, 20017, 48064, 39545, 30845, 9400, 40223, 24254, 14009, 12718, 34541, 32852, 23577, 19151, 9238, 44145, 9503, 25594, 5158, 47006, 20376, 6690, 18825, 10688, 6610, 30847, 28891, 10887, 1081, 44086, 20087, 35763, 33282, 18578, 20832, 15268, 18764, 10719, 48771, 12169, 46750, 42165, 33881, 29255, 48632, 28746, 5920, 2151, 42860, 19869, 12373, 3057, 1991, 34849, 9587, 47767, 37739, 36081, 326, 37149, 26667, 22828, 49412, 36930, 29372, 32253, 28535, 17350, 2198, 43835, 23187, 7616, 35018, 26968, 12409, 10933, 27533, 14607, 15621, 15079, 43613, 33762, 23266, 19177, 11765, 2985, 39691, 33458, 6730, 3705, 39514, 21176, 17276, 6157, 24832, 21498, 44318, 17564, 14321, 9905, 31575, 13977, 43472, 17565, 16942, 8551, 44848, 20595, 32338, 28062, 15917, 2527, 42724, 37665, 34948, 49372, 17443, 2368, 12999, 35060, 32181, 16658, 4549, 1454, 31505, 19699, 11558, 269, 49710, 21812, 10382, 44052, 27776, 22729, 45294, 47019, 13782, 13050, 49344, 32360, 371, 19303, 8100, 39749, 37148, 34119, 10051, 7900, 484, 30473, 29736, 11710, 4668, 19990, 7674, 6498, 3654, 43448, 38385, 35747, 13086, 2536, 2004, 16850, 9939, 1701, 419, 41016, 27799, 6660, 47824, 49685, 35529, 31895, 6879, 45966, 33316, 3698, 48435, 43998, 36303, 32247, 25901, 7479, 46282, 36345, 22809, 40755, 7998, 7107, 32276, 43128, 41989, 26048, 25665, 25289, 16030, 9172, 49894, 16784, 31402, 16146, 41179, 38564, 32018, 20536, 15828, 4762, 32080, 37812, 38652, 32997, 31274, 29784, 23247, 26417, 41539, 40475, 15557, 18752, 43881, 4756, 43501, 15552, 9730, 32485, 25626, 10281, 6937, 19475, 10065, 7109, 43639, 7901, 1750, 12, 45110, 41925, 37196, 25835, 42885, 10450, 46602, 35456, 26771, 16302, 34046, 21728, 9492, 49867, 27394, 41594, 29203, 8463, 49184, 35567, 21725, 41383, 41151, 28639, 19383, 2702, 34689, 6058, 45561, 27425, 24548, 8725, 44719, 39456, 35869, 32664, 13807, 40530, 40450, 3936, 19556, 13714, 47018, 1041, 33115, 19476, 32255, 17312, 4501, 9506, 35262, 31347, 15931, 6802, 4175, 250, 47356, 45220, 43548, 35304, 31621, 45618, 49612, 47306, 31680, 31299, 28950, 22777, 15208, 34993, 27057, 6139, 48834, 35967, 10919, 7254, 37804, 20280, 3991, 37928, 35749, 12818, 6111, 46733, 41523, 16782, 5669, 49466, 43987, 37096, 13520, 12616, 34961, 26326, 13981, 13811, 9181, 46412, 42378, 22661, 46813, 17701, 40057, 38942, 36858, 19234, 48128, 36740, 9430, 46293, 44670, 29688, 28709, 27912, 11396, 44614, 39214, 27391, 2796, 46137, 39082, 30203, 7985, 42043, 40119, 33350, 901, 22876, 7891, 48074, 2460, 1214, 35183, 34067, 18747, 16828, 42782, 31750, 38369, 33703, 45410, 25638, 21179, 16697, 37488, 33940, 31840, 26253, 41006, 41116, 43387, 27301, 40408, 17216, 12834, 36911, 10509, 21709, 40425]

    # Generate data
    labels = np.load(f"/cs/labs/daphna/itai.david/py_repos/TypiClust/results/cifar-10/labels_seed0.npy")
    all_features = torch.from_numpy(ds_utils.load_features('CIFAR10', train=True))
    K = compute_kernel(all_features, all_features, 0.75)

    # K_rbf = get_rbf_kernel(all_features, all_features, 1)
    # import matplotlib.pyplot as plt
    # K_f = K_rbf[1000].flatten()
    # plt.scatter( K_f, np.arange(K_f.size()[0]))
    # plt.show()
    threshold_map = {'0.95':0.3088, '0.96': 0.322, '0.97': 0.3406, '0.975':0.353, '0.98': 0.3679,'0.985':0.3882, '0.99': 0.417}

    C = torch.full((50000, 10), fill_value=0.1)

    for i in range(300):
        chosen_idx = weighted_alpha_01_lset[i]
        chosen_label = labels[chosen_idx]
        C[:, chosen_label] += K[:, chosen_idx]

    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)
    norm_C = (C / sum_C)
    old_max = (max_C / sum_C)
    C_diff = (C - max_C).unsqueeze(0)
    max_C.unsqueeze_(0)
    sum_C.unsqueeze_(0)
    old_max.unsqueeze_(0)
    K.unsqueeze_(2)
    # K_dense, K_list, C, norm_C, sum_C, max_C, C_diff, old_max, class_corr = generate_data(L, N, SPARSITY, DEVICE)

    # --- Run Tests ---
    print("\nTesting DENSE version...")
    # res_dense = time_gpu(run_dense_code, K, norm_C, sum_C, max_C, C_diff, old_max, N,
    #                      CHUNK_SIZE, DEVICE)

    print("\nTesting SPARSE version...")
    threshold = 0.01
    with open("/cs/labs/daphna/itai.david/py_repos/cifar10_data/cifar10_tophat_K_sparse_csr.pkl", "rb") as f:
        K_csr = pickle.load(f)
    start = time.time()
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
    print(time.time() - start)
    norm_C = norm_C.to('cuda')
    sum_C = sum_C.squeeze().to('cuda')
    max_C = max_C.squeeze().to('cuda')
    C_diff = C_diff.squeeze().to('cuda')
    old_max = old_max.squeeze().to('cuda')

    res_sparse = time_gpu(run_sparse_code_v2, K_sparse_torch, norm_C, sum_C, max_C, C_diff, old_max, N,
                          CHUNK_SIZE, DEVICE)

    # --- Verify Results ---
    print("\nVerifying results...")
    if torch.allclose(res_dense, res_sparse, atol=1e-5):
        print("SUCCESS: Dense and Sparse results are (approximately) equal.")
    else:
        print("FAILURE: Dense and Sparse results differ.")
        print("Dense:", res_dense[:10])
        print("Sparse:", res_sparse[:10])
        print("Difference:", torch.abs(res_dense - res_sparse).sum())

else:
    print("CUDA not available. Please run this on a GPU-enabled machine.")



