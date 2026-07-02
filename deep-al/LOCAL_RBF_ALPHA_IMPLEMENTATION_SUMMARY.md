# LOCAL_RBF_ALPHA Implementation Summary

## Overview

Successfully separated the alpha parameters to avoid confusion between:
1. **C matrix initialization alpha** (`CK_ALPHA` or `ALPHA`)
2. **Local RBF kernel degree normalization alpha** (`LOCAL_RBF_ALPHA`)

## Changes Made

### 1. Updated `pycls/al/ck_matrix_manager.py`

**Added new parameter:**
```python
self.local_rbf_alpha = cfg.LOCAL_RBF_ALPHA if 'LOCAL_RBF_ALPHA' in cfg else 1.0
```

**Updated LocalRBFKernel initialization:**
```python
elif self.kernel_type == 'local_rbf':
    self.kernel_fn = LocalRBFKernel(self.device, alpha=self.local_rbf_alpha)
    initial_threshold = self.K_sparsity_threshold
```

**Updated sparse kernel matrix building:**
```python
K_general = build_sparse_kernel_matrix(
    ...
    alpha=self.local_rbf_alpha,  # Changed from self.alpha
)
```

**Updated logging:**
- Added `local_rbf_alpha={self.local_rbf_alpha}` to kernel info when using local_rbf

### 2. Updated `tools/train_al.py`

**Added CLI argument (line ~857):**
```python
parser.add_argument('--local_rbf_alpha', default=1.0, type=float,
                    help='Alpha parameter for degree normalization in local_rbf kernel: K_ij / (degree_i * degree_j)^(alpha/2). Default: 1.0')
```

**Added config assignment (line ~2617):**
```python
cfg.LOCAL_RBF_ALPHA = args.local_rbf_alpha
```

### 3. Updated `scripts/extract_stats_from_dir.py`

**Added config reading (line ~246):**
```python
local_rbf_alpha = config['LOCAL_RBF_ALPHA'] if 'LOCAL_RBF_ALPHA' in config else 1.0
```

**Added to records dictionary (line ~393):**
```python
"local_rbf_alpha": local_rbf_alpha,
```

**Added to backward compatibility defaults (line ~479):**
```python
ck_columns_defaults = {
    ...
    'local_rbf_alpha': 1.0,
    ...
}
```

## Usage Examples

### Command Line
```bash
python train_al.py \
  --cfg configs/CIFAR100.yaml \
  --exp-name local_rbf_test \
  --al bayes_misp \
  --budget 100 \
  --seed 0 \
  --kernel_type local_rbf \
  --ck_sigma 1.0 \
  --ck_alpha 0.3 \
  --local_rbf_alpha 1.0 \
  --ck_sparse_K False
```

### Configuration File
```yaml
# config.yaml
KERNEL_TYPE: 'local_rbf'
CK_SIGMA: 1.0
CK_ALPHA: 0.3           # For C matrix initialization
LOCAL_RBF_ALPHA: 1.0    # For kernel degree normalization
CK_SPARSE_K: false
CK_K_SPARSITY_THRESHOLD: 0.01
```

## Parameter Meanings

| Parameter | Purpose | Typical Values | Where Used |
|-----------|---------|----------------|------------|
| `ALPHA` or `CK_ALPHA` | Baseline coverage for unlabeled points in C matrix | 0.1 - 0.5 | C matrix initialization |
| `LOCAL_RBF_ALPHA` | Degree normalization strength in kernel | 0.5, 1.0, 2.0 | LocalRBFKernel transformation |

## Mathematical Details

### C Matrix Initialization
```
C[i, c] = alpha  (for all unlabeled points i, class c)
```

### Local RBF Kernel Transformation
```
K_ij ← K_ij / (degree_i × degree_j)^(LOCAL_RBF_ALPHA / 2)

where:
  degree_i = Σ_{k≠i} K_ik  (excluding diagonal)
  degree_j = Σ_{k≠j} K_jk  (excluding diagonal)
```

## Backward Compatibility

- **Default value:** `LOCAL_RBF_ALPHA = 1.0`
- **Old experiments:** Automatically assigned `1.0` when loading from parquet files
- **Config files without LOCAL_RBF_ALPHA:** Will use default value `1.0`

## Testing

To verify the implementation:

```python
# In train_al.py, after CKMatrixManager initialization
print(f"C matrix alpha: {cfg.CK_ALPHA}")
print(f"Local RBF alpha: {cfg.LOCAL_RBF_ALPHA}")
print(f"Kernel type: {cfg.KERNEL_TYPE}")
```

Expected output when using local_rbf:
```
[CKMatrixManager] Building K_general matrix with local_rbf kernel, sigma=1.0, local_rbf_alpha=1.0, threshold=0.0, sparse=False, device=cpu, alpha=0.3
```

## Files Modified

1. `TypiClust/deep-al/pycls/al/ck_matrix_manager.py`
2. `TypiClust/deep-al/tools/train_al.py`
3. `scripts/extract_stats_from_dir.py`
4. `TypiClust/deep-al/LOCAL_RBF_KERNEL_IMPLEMENTATION.md` (documentation updated)

## No Breaking Changes

- Existing experiments with `kernel_type='rbf'`, `'tophat'`, or `'cknn'` are unaffected
- Only affects experiments using `kernel_type='local_rbf'`
- Default value ensures consistent behavior
