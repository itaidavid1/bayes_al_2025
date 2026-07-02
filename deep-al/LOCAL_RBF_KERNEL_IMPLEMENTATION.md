# Local RBF Kernel Implementation

## Overview

A new kernel type `local_rbf` has been added to the CK matrix manager. This kernel applies degree-based normalization to the standard RBF kernel values.

## Mathematical Transformation

The transformation depends on the degree normalization method:

### Method 'none' (default)
```
K_ij ← K_ij / (degree_i × degree_j)^(α/2)
```

### Methods 'sum' or 'max'
```
K_ij ← K_ij / (normalized_degree_i × normalized_degree_j)
```

**Important:** When using 'sum' or 'max', the power of α/2 is **NOT** applied!

### Method 'log'
```
K_ij ← K_ij / log(degree_i × degree_j)
```

**Important:** When using 'log', natural logarithm is used **instead** of power of α/2!

Where:
- `K_ij` is the original RBF kernel value between points i and j
- `degree_i = Σ_{k≠i} K_ik` (sum of kernel values excluding diagonal)
- `degree_j = Σ_{k≠j} K_jk` (sum of kernel values excluding diagonal)
- `normalized_degree_i = normalize(degree_i)` (normalization applied for 'sum' or 'max')
- `α` is the normalization parameter (from `cfg.LOCAL_RBF_ALPHA`, only used for 'none' method)

**Special handling:**
- Diagonal elements are **excluded** from degree calculation
- If either `degree_i = 0` or `degree_j = 0`, then `K_ij = 0` (no epsilon added)

**Degree Normalization Options:**
- `'none'` (default): No normalization, power IS applied: `K_ij / (degree_i × degree_j)^(α/2)`
- `'sum'`: Normalize by sum, power NOT applied: `K_ij / (normalized_degree_i × normalized_degree_j)`
- `'max'`: Normalize by max, power NOT applied: `K_ij / (normalized_degree_i × normalized_degree_j)`
- `'log'`: No normalization, logarithm IS applied: `K_ij / log(degree_i × degree_j)`

See `DEGREE_NORMALIZATION_IMPLEMENTATION.md` for details.

## Implementation Details

### Files Modified

1. **`pycls/al/kernel_utils.py`**
   - Added `LocalRBFKernel` class for dense kernel computation
   - Added `_apply_degree_normalization_sparse()` helper function
   - Updated `build_sparse_kernel_matrix()` to support 'local_rbf' kernel type
   - Added `alpha` parameter to sparse kernel builder

2. **`pycls/al/ck_matrix_manager.py`**
   - Imported `LocalRBFKernel` class
   - Added 'local_rbf' case in kernel selection logic
   - Updated `build_K_general_matrix()` to handle 'local_rbf' for both sparse and dense matrices

### Key Classes and Functions

#### `LocalRBFKernel` Class
```python
class LocalRBFKernel(object):
    def __init__(self, device, alpha=0.5):
        # Wraps RBFKernel and applies degree normalization
        
    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, matrices_type=torch.float16):
        # Computes RBF kernel then applies degree normalization
        # Handles symmetric and asymmetric cases
        # Returns normalized kernel matrix
```

#### `_apply_degree_normalization_sparse()` Function
- Applies degree normalization to sparse CSR matrices
- Excludes diagonal from degree calculation
- Sets K_ij = 0 when degree is 0 (no epsilon)

## Usage

### Configuration

Set the kernel type in your configuration file or via CLI:

#### YAML Configuration
```yaml
KERNEL_TYPE: 'local_rbf'
CK_SIGMA: 1.0                    # RBF bandwidth parameter (sigma)
CK_ALPHA: 0.5                    # C matrix initialization value
LOCAL_RBF_ALPHA: 1.0             # Degree normalization parameter for local_rbf kernel
DEGREE_NORMALIZATION_METHOD: 'none'  # Degree normalization: 'none', 'sum', or 'max'
CK_SPARSE_K: false               # Use sparse or dense matrices
CK_K_SPARSITY_THRESHOLD: 0.0     # Threshold for sparsifying kernel values
```

#### CLI Arguments (train_al.py)
```bash
python train_al.py \
  --kernel_type local_rbf \
  --ck_sigma 1.0 \
  --ck_alpha 0.5 \
  --local_rbf_alpha 1.0 \
  --degree_normalization_method none \
  --ck_sparse_K False \
  --ck_K_sparsity_threshold 0.0
```

### Example Configuration Options

```python
cfg.KERNEL_TYPE = 'local_rbf'              # Use local RBF kernel
cfg.CK_SIGMA = 1.0                         # RBF bandwidth
cfg.CK_ALPHA = 0.5                         # C matrix initialization (baseline coverage)
cfg.LOCAL_RBF_ALPHA = 1.0                  # Degree normalization strength
cfg.DEGREE_NORMALIZATION_METHOD = 'none'   # Degree normalization: 'none', 'sum', 'max'
cfg.CK_SPARSE_K = False                    # Dense matrix mode
cfg.CK_K_SPARSITY_THRESHOLD = 0.01         # Remove values < 0.01
```

### Important: Separate Alpha Parameters

The implementation now uses **two different alpha parameters**:

1. **`CK_ALPHA` (or `ALPHA`)** - Used for C matrix initialization
   - This is the baseline pseudo-coverage value for unlabeled points
   - Typical values: 0.1 - 0.5
   - Default: 0.5

2. **`LOCAL_RBF_ALPHA`** - Used for kernel degree normalization (local_rbf only)
   - This controls the strength of degree normalization: K_ij / (degree_i × degree_j)^(alpha/2)
   - Typical values: 0.5, 1.0, 2.0
   - Default: 1.0
   - Only used when `KERNEL_TYPE = 'local_rbf'`

### Supported Kernel Types

The CK matrix manager now supports:
- `'rbf'` - Standard RBF (Gaussian) kernel
- `'tophat'` - Top-hat (indicator) kernel
- `'cknn'` - Continuous k-NN kernel
- `'local_rbf'` - **NEW** Degree-normalized RBF kernel

## Technical Notes

### Dense Matrix Path
1. Compute full RBF kernel matrix
2. Calculate degrees (excluding diagonal)
3. Apply normalization transformation
4. Set values to 0 where degrees are 0
5. Apply sparsity threshold if specified

### Sparse Matrix Path
1. Build sparse RBF kernel using batched computation
2. Create copy without diagonal for degree calculation
3. Compute degrees from non-diagonal elements
4. Apply normalization to non-zero elements
5. Set values to 0 where degree product is 0

### Memory Considerations
- Dense mode: Requires O(N²) memory for full kernel matrix
- Sparse mode: More memory efficient for large datasets
- Degree calculation adds minimal overhead

## Testing

To verify the implementation works correctly:

```python
# In your training script
from pycls.al.ck_matrix_manager import CKMatrixManager

# Configure for local_rbf
cfg.KERNEL_TYPE = 'local_rbf'
cfg.CK_SIGMA = 1.0
cfg.CK_ALPHA = 0.5

# Create CK manager
ck_manager = CKMatrixManager(cfg, data_obj, train_labels, lset)

# Verify kernel matrix
K = ck_manager.get_K_general()
print(f"K shape: {K.shape}")
print(f"K sparsity: {(K == 0).sum() / K.numel():.2%}")
print(f"K min: {K.min():.6f}, K max: {K.max():.6f}")
```

## Comparison with Other Kernels

| Kernel Type | Bandwidth | Normalization | Use Case |
|------------|-----------|---------------|----------|
| `rbf` | Global σ | None | General purpose |
| `cknn` | Local (k-NN) | Adaptive | Variable density |
| `local_rbf` | Global σ | Degree-based | Graph-theoretic |

## References

This implementation is based on degree-normalized kernel techniques commonly used in:
- Spectral clustering
- Graph-based semi-supervised learning
- Normalized graph Laplacians

The transformation helps balance connectivity across regions of different density in feature space.
