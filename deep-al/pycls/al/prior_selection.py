import math
import torch

# ============ Actual Prior Selection Functions ============ #

def get_temp_priors(rel_measures, lb, ub):
    """
    Create temporary prior values scaled between lb and ub based on reliability measures.
    """
    rel_min = rel_measures.min()
    rel_max = rel_measures.max()

    # Scale to [lb, ub] inversely: max reliability -> min prior (lb), min reliability -> max prior (ub)
    scaled_priors = ub - (rel_measures - rel_min) * (ub - lb) / (rel_max - rel_min)
    return scaled_priors

# ============== Clarity\Reliability Measures ============ #

def est_reliability(K, transformation, batch_size=8192, p=2, a=1.0):
    """
    Assumes K is a tensor on the CPU and calculates rel using GPU in batches.
    """
    eps = 1e-5
    if torch.is_tensor(K):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n = K.shape[0]
        scores = []
        for i in range(0, n, batch_size):
            end_idx = min(i + batch_size, n)
            K_batch = K[i:end_idx, :].to(device)  # (b, n)
            if transformation == 'reg':
                transformed_K = K_batch
            elif transformation == 'exp':
                pass
            elif transformation == 'pow':
                transformed_K = K_batch ** p
            else:
                raise ValueError(f"Unknown transformation: {transformation}")
            rel_batch = transformed_K.sum(dim=1)  # (b,)
            scores.append(rel_batch.cpu())
            
            del K_batch, transformed_K, rel_batch

        return torch.cat(scores)
    else:
        raise ValueError("K must be a torch tensor")
    
    
def calculate_cohesion(K_cpu, k, device='cuda', batch_size=128,):
    """
    Calculates cohesion for a kernel matrix K too large for VRAM.

    This function keeps the main kernel matrix K on the CPU. For each batch, it
    identifies only the unique columns of K required for the computation, 
    transfers just those slices to the GPU, performs the calculation, and then
    transfers the result back to the CPU.
    
    Returns:
        torch.Tensor: A (N,) tensor of cohesion scores, residing on the CPU.
    """
    n_points = K_cpu.shape[0]

    if k >= n_points:
        raise ValueError("k must be smaller than the number of points N.")

    cohesion_cpu = torch.zeros(n_points, device='cpu')
    
    print("Pre-calculating neighbor indices on CPU...")
    top_k_indices_all = torch.topk(K_cpu, k+1, dim=0).indices[1:, :]
    print("Done.")

    for i in range(0, n_points, batch_size):
        batch_end = min(i + batch_size, n_points)

        top_k_indices_batch_cpu = top_k_indices_all[:, i:batch_end]

        # --- Step 1: Identify and Slice Necessary Data on CPU ---
        
        # Create the full index arrays for the batch on the CPU first
        row_indices_cpu = torch.repeat_interleave(top_k_indices_batch_cpu, k, dim=0)
        col_indices_cpu = top_k_indices_batch_cpu.tile((k, 1))

        # Find the UNIQUE columns we need to fetch from K_cpu. This is the key step.
        unique_cols_cpu, inverse_indices = torch.unique(col_indices_cpu, return_inverse=True)

        # Fetch only these unique columns from the master K matrix on the CPU.
        # This is a much smaller slice of K.
        K_slice_gpu = K_cpu[:, unique_cols_cpu].to(device, dtype=torch.float32)

        # --- Step 2: Transfer Minimal Data to GPU ---
        
        row_indices_gpu = row_indices_cpu.to(device)
        # We don't need the original col_indices on the GPU. Instead, we use the
        # `inverse_indices` which map back to our small K_slice.
        col_indices_remapped_gpu = inverse_indices.to(device)
        
        # --- Step 3: Computation on GPU using Sliced Data ---
        
        # Index into the small, sliced K matrix. The `col_indices_remapped_gpu`
        # now correctly points to the columns within K_slice_gpu.
        submatrix_values = K_slice_gpu[row_indices_gpu, col_indices_remapped_gpu]
        
        # Reshape the flat values back into a (k*k, batch_size) structure
        submatrix_values = submatrix_values.view(k*k, -1)

        cohesion_batch_gpu = torch.sum(submatrix_values, dim=0) / (k * k)
        
        # --- Step 4: Data Transfer back to CPU ---
        
        cohesion_cpu[i:batch_end] = cohesion_batch_gpu.to('cpu')
        # Free GPU memory
        del K_slice_gpu, row_indices_gpu, col_indices_remapped_gpu, submatrix_values, cohesion_batch_gpu
        torch.cuda.empty_cache()

    return cohesion_cpu


def compute_reliability(K, y, batch_size=8192, normalized=False):

    if not torch.is_tensor(K):
        raise ValueError("K must be a torch tensor")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = K.shape[0]

    y_t = torch.as_tensor(y, device=device).view(-1)

    scores = []

    for i in range(0, n, batch_size):
        end_idx = min(i + batch_size, n)
        K_batch = K[i:end_idx, :].to(device)  # (b, N)
        y_batch = y_t[i:end_idx]

        # Row-wise same-class / diff-class masks
        agree_mask = (y_batch[:, None] == y_t[None, :])
        disagree_mask = ~agree_mask

        # Compute agreement / disagreement sums
        agree = (K_batch * agree_mask).sum(dim=1)
        disagree = (K_batch * disagree_mask).sum(dim=1)

        # Compute final reliability
        if normalized:
            same_class_counts = agree_mask.sum(dim=1)
            diff_class_counts = disagree_mask.sum(dim=1)
            rel = (agree / torch.clamp(same_class_counts, min=1.0) -
                   disagree / torch.clamp(diff_class_counts, min=1.0))
        else:
            rel = agree - disagree

        scores.append(rel.cpu())

        # Free memory
        del K_batch, y_batch, agree_mask, disagree_mask, agree, disagree, rel

    return torch.cat(scores)


def compute_matrix_entropy(A, normalize=True):
    """
    Computes entropy for each row of the matrix
    and returns the scores in order.
    
    input: A torch tensor (N, M)
    output: A torch tensor (N,)
    """
    entropy = -torch.sum(A * torch.log(A + 1e-9), dim=1)
    if normalize:
        max_entropy = math.log(A.size(1))
        entropy = entropy / max_entropy
    return entropy


def compute_clarity_kp(K, y, M, oracle_method, batch_size=8192):
    """
    For each point, calculate its vector in C after the whole
    run of the algorithm is done. Then compute clarity as the
    entropy of each the vector that represent each sample.
    
    M is the number of classes.
    K is a torch tensor on CPU.
    y is the labels array.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = K.shape[0]

    # Convert y to one-hot encoding on the device: (N, M)
    y_tensor = torch.as_tensor(y, device=device, dtype=torch.int64).view(-1)
    Y_onehot = torch.zeros((N, M), device=device, dtype=K.dtype)
    Y_onehot.scatter_(1, y_tensor.unsqueeze(1), 1.0)

    scores = []

    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        
        # Move slice of K to GPU
        K_batch = K[i:end_idx, :].to(device)  # (b, N)
        
        # C matrix for the batch: (b, N) x (N, M) -> (b, M)
        # Each row i of C_batch contains the sum of similarities of point i to all points in class c
        # NOTE: added 1 for the uniform prior to adhere mathematical formalization,
        #       anyway its negligible when N is large
        C_batch = torch.mm(K_batch, Y_onehot) + 0.1
        norm_C_batch = C_batch / C_batch.sum(dim=1, keepdim=True)
        
        # Calculate clarity via entropy
        if oracle_method == 'entropy':
            clarity_batch = 1 - compute_matrix_entropy(norm_C_batch, normalize=True)
        elif oracle_method == 'max':
            clarity_batch = torch.max(norm_C_batch, axis=1)[0]

        scores.append(clarity_batch.cpu())
        
        del K_batch, C_batch, norm_C_batch, clarity_batch

    return torch.cat(scores)