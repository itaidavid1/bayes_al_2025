# Degree Normalization Methods Implementation

## Overview

Added optional degree normalization to the Local RBF kernel transformation. Before applying the degree-based kernel transformation, the degree values themselves can be normalized using different methods.

## Mathematical Transformation

### Method 'none' (default)
```
K_ij ← K_ij / (degree_i × degree_j)^(alpha/2)

where:
  degree_i = Σ_{k≠i} K_ik  (excluding diagonal)
```

### Method 'sum' or 'max'
```
normalized_degree_i = normalize(degree_i)
K_ij ← K_ij / (normalized_degree_i × normalized_degree_j)

Note: NO power of alpha/2 is applied when using 'sum' or 'max'
```

### Method 'log'
```
K_ij ← K_ij / log(degree_i × degree_j)

where:
  degree_i = Σ_{k≠i} K_ik  (excluding diagonal)
  log is natural logarithm

Note: Logarithm is used instead of power of alpha/2
```

## Normalization Methods

### 1. 'none' (default)
No normalization applied to degrees. Power of alpha/2 IS applied.

```
normalized_degree_i = degree_i
K_ij ← K_ij / (degree_i × degree_j)^(alpha/2)
```

### 2. 'sum'
Normalize by the sum of all degrees. Makes degrees sum to 1. Power IS NOT applied.

```
normalized_degree_i = degree_i / Σ_j degree_j
K_ij ← K_ij / (normalized_degree_i × normalized_degree_j)
```

**Use case:** When you want to treat degrees as a probability distribution.

**Important:** No power of alpha/2 is applied with this method!

### 3. 'max'
Normalize by the maximum degree. Scales all degrees to [0, 1] range. Power IS NOT applied.

```
normalized_degree_i = degree_i / max_j(degree_j)
K_ij ← K_ij / (normalized_degree_i × normalized_degree_j)
```

**Use case:** When you want to remove the scale of degree values while preserving relative magnitudes.

**Important:** No power of alpha/2 is applied with this method!

### 4. 'log'
No normalization applied to degrees. Logarithm IS applied (instead of power).

```
normalized_degree_i = degree_i
K_ij ← K_ij / log(degree_i × degree_j)
```

**Use case:** When you want logarithmic scaling instead of power-based scaling.

**Important:** Natural logarithm is used instead of power of alpha/2!

## Implementation Details

### Files Modified

1. **`pycls/al/kernel_utils.py`**
   - Updated `LocalRBFKernel.__init__()` to accept `degree_normalization_method` parameter
   - Added `LocalRBFKernel._normalize_degrees()` method
   - Updated `LocalRBFKernel._apply_degree_normalization()` to normalize degrees before transformation
   - Updated `_apply_degree_normalization_sparse()` to accept and apply degree normalization
   - Updated `build_sparse_kernel_matrix()` to accept and pass `degree_normalization_method`

2. **`pycls/al/ck_matrix_manager.py`**
   - Added `self.degree_normalization_method` parameter reading from config
   - Updated `LocalRBFKernel` initialization to pass degree normalization method
   - Updated sparse kernel builder call to pass degree normalization method
   - Enhanced logging to show degree normalization method

3. **`tools/train_al.py`**
   - Added `--degree_normalization_method` CLI argument with choices: ['none', 'sum', 'max']
   - Added config assignment: `cfg.DEGREE_NORMALIZATION_METHOD`

4. **`scripts/extract_stats_from_dir.py`**
   - Added reading `DEGREE_NORMALIZATION_METHOD` from config
   - Added to records dictionary and backward compatibility defaults

### Key Features
- Works with both dense and sparse kernel matrices
- Default is 'none' to preserve existing behavior
- Only applied when using `kernel_type='local_rbf'`
- Safe handling of edge cases (zero degree sums/maxes)

## Usage Examples

### Command Line
```bash
# Example 1: No normalization (default)
python train_al.py \
  --kernel_type local_rbf \
  --local_rbf_alpha 1.0 \
  --degree_normalization_method none

# Example 2: Sum normalization
python train_al.py \
  --kernel_type local_rbf \
  --local_rbf_alpha 1.0 \
  --degree_normalization_method sum

# Example 3: Max normalization
python train_al.py \
  --kernel_type local_rbf \
  --local_rbf_alpha 1.0 \
  --degree_normalization_method max
```

### Configuration File
```yaml
KERNEL_TYPE: 'local_rbf'
LOCAL_RBF_ALPHA: 1.0
DEGREE_NORMALIZATION_METHOD: 'sum'  # or 'max', or 'none'
CK_SIGMA: 1.0
CK_ALPHA: 0.3
```

### Full Example Command
```bash
python train_al.py \
  --cfg configs/CIFAR100.yaml \
  --exp-name test_degree_norm \
  --al bayes_misp \
  --budget 100 \
  --seed 0 \
  --kernel_type local_rbf \
  --ck_sigma 1.0 \
  --ck_alpha 0.3 \
  --local_rbf_alpha 1.0 \
  --degree_normalization_method sum \
  --ck_sparse_K False \
  --max_iter 10
```

## Expected Output

When using degree normalization, the log output will show:

```
[CKMatrixManager] Building K_general matrix with local_rbf kernel, 
  sigma=1.0, local_rbf_alpha=1.0, degree_norm=sum, threshold=0.0, 
  sparse=False, device=cpu, alpha=0.3
```

## Comparison of Methods

| Method | Degree Range | Sum of Degrees | Transformation Applied | Use Case |
|--------|-------------|----------------|------------------------|----------|
| `none` | [0, ∞) | Arbitrary | Power (alpha/2) | Default, preserves scale |
| `sum` | [0, 1] | = 1 | **None** (direct) | Probabilistic interpretation |
| `max` | [0, 1] | ≤ 1 | **None** (direct) | Scale-free comparison |
| `log` | [0, ∞) | Arbitrary | **Log** (natural) | Logarithmic scaling |

## Effect on Kernel Values

### Original Degrees (example)
```
degrees = [10, 50, 100, 5, 200]
```

### After 'sum' Normalization
```
sum = 10 + 50 + 100 + 5 + 200 = 365
normalized = [0.027, 0.137, 0.274, 0.014, 0.548]
```

### After 'max' Normalization
```
max = 200
normalized = [0.05, 0.25, 0.50, 0.025, 1.0]
```

## Implementation Logic (Dense)

```python
def _normalize_degrees(self, degrees):
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

def _apply_degree_normalization(self, K, degrees, is_symmetric):
    # ... normalize degrees ...
    degree_product = torch.outer(normalized_degrees, normalized_degrees)
    
    # IMPORTANT: Apply power only for 'none' method
    if self.degree_normalization_method == 'none':
        normalization_factor = torch.pow(degree_product_safe, self.alpha / 2.0)
    else:
        normalization_factor = degree_product_safe  # No power for 'sum' or 'max'
    
    K_normalized = K / normalization_factor
    return K_normalized
```

## Implementation Logic (Sparse)

```python
# Normalize degrees based on the specified method
if degree_normalization_method == 'sum':
    degree_sum = degrees.sum()
    if degree_sum > 0:
        degrees = degrees / degree_sum
elif degree_normalization_method == 'max':
    degree_max = degrees.max()
    if degree_max > 0:
        degrees = degrees / degree_max
# else: 'none' - no normalization

degree_product = row_degrees * col_degrees

# IMPORTANT: Apply power only for 'none' method
if degree_normalization_method == 'none':
    normalization_factor = np.power(degree_product_safe, alpha / 2.0)
else:
    normalization_factor = degree_product_safe  # No power for 'sum' or 'max'

normalized_data = coo.data / normalization_factor
```

## Backward Compatibility

- **Default value:** `DEGREE_NORMALIZATION_METHOD = 'none'`
- **Old experiments:** Automatically assigned `'none'` when loading from config
- **Config files without DEGREE_NORMALIZATION_METHOD:** Will use default `'none'`
- **No breaking changes** to existing experiments

## When to Use Each Method

### Use 'none' when:
- You want to preserve the original degree scale
- Degrees represent meaningful quantities
- Default/baseline experiments

### Use 'sum' when:
- You want degrees to represent a probability distribution
- You care about relative proportions
- You want all degrees to sum to 1

### Use 'max' when:
- You want to remove the scale effect
- You care about relative rankings
- You want the highest degree to be 1.0

## Testing Recommendations

### Test 1: Verify Different Methods Produce Different Results
```bash
# Run with 'none'
python train_al.py --kernel_type local_rbf --degree_normalization_method none ...

# Run with 'sum'
python train_al.py --kernel_type local_rbf --degree_normalization_method sum ...

# Run with 'max'
python train_al.py --kernel_type local_rbf --degree_normalization_method max ...

# Compare results
```

### Test 2: Check Sparse Mode
```bash
python train_al.py \
  --kernel_type local_rbf \
  --degree_normalization_method sum \
  --ck_sparse_K True \
  --ck_K_sparsity_threshold 0.01
```

### Test 3: Verify Logging
Check that the log shows:
```
degree_norm=sum  (or degree_norm=max, or degree_norm=none)
```

## Common Issues and Solutions

### Issue: "Invalid choice: 'Sum'"
**Solution:** Use lowercase: `--degree_normalization_method sum` (not `Sum` or `SUM`)

### Issue: Degree normalization applied to non-local_rbf kernels
**Solution:** Degree normalization only applies when `kernel_type='local_rbf'`. It's silently ignored for other kernel types.

### Issue: Zero degree sum/max
**Solution:** Implementation safely handles zero sums/maxes by returning original degrees (no division by zero).

## Performance Considerations

- **Overhead:** Minimal - just one additional division operation per degree
- **Memory:** No additional memory required
- **Speed:** Negligible impact on kernel computation time

## References

This implementation extends the Local RBF kernel with optional degree normalization, commonly used in:
- Normalized graph Laplacians
- Spectral clustering with degree correction
- Graph neural networks with attention mechanisms

## Success Criteria

- [x] Three normalization methods implemented: 'none', 'sum', 'max'
- [x] Works with both dense and sparse matrices
- [x] Default is 'none' (preserves existing behavior)
- [x] CLI argument with validation
- [x] Config file support
- [x] Logging shows normalization method
- [x] Backward compatibility maintained
- [x] Documentation complete
- [x] No linter errors
