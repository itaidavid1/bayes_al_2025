# Log Method Implementation

## Overview

Added 'log' as a new degree normalization method for the Local RBF kernel. This method uses natural logarithm instead of power transformation.

## Mathematical Formula

### Method 'log'
```
K_ij ← K_ij / log(degree_i × degree_j)

where:
  degree_i = Σ_{k≠i} K_ik  (excluding diagonal)
  log is natural logarithm (ln)
```

## Comparison with Other Methods

| Method | Pre-normalization | Transformation | Formula |
|--------|-------------------|----------------|---------|
| `none` | No | Power (α/2) | `K / (d_i × d_j)^(α/2)` |
| `sum` | Yes (by sum) | None | `K / (d_i' × d_j')` |
| `max` | Yes (by max) | None | `K / (d_i' × d_j')` |
| `log` | **No** | **Log** | `K / log(d_i × d_j)` |

## Key Features

- **No pre-normalization**: Like 'none', uses raw degree values
- **Logarithmic scaling**: Uses natural log instead of power
- **Alpha-independent**: Does NOT use the LOCAL_RBF_ALPHA parameter
- **Sublinear scaling**: log(x) grows slower than x^a for most values

## When to Use

Use the 'log' method when:
- You want sublinear scaling of the degree normalization
- You want the effect to be less sensitive to large degree differences
- You want a middle ground between 'none' (power) and 'sum'/'max' (direct)

## Usage

### Command Line
```bash
python train_al.py \
  --kernel_type local_rbf \
  --degree_normalization_method log \
  --ck_sigma 1.0
```

### Config File
```yaml
KERNEL_TYPE: 'local_rbf'
DEGREE_NORMALIZATION_METHOD: 'log'
CK_SIGMA: 1.0
```

## Example Values

### Original Degrees
```
degrees = [10, 50, 100, 200, 500]
```

### Normalization Factors

**Method 'none' (alpha=1.0):**
```
(10 × 50)^0.5 = 22.36
(50 × 100)^0.5 = 70.71
(100 × 200)^0.5 = 141.42
(200 × 500)^0.5 = 316.23
```

**Method 'log':**
```
log(10 × 50) = log(500) = 6.21
log(50 × 100) = log(5000) = 8.52
log(100 × 200) = log(20000) = 9.90
log(200 × 500) = log(100000) = 11.51
```

**Ratio (none / log):**
```
22.36 / 6.21 = 3.60
70.71 / 8.52 = 8.30
141.42 / 9.90 = 14.28
316.23 / 11.51 = 27.47
```

Notice: The log method produces smaller normalization factors, especially for large degree products. This means less aggressive normalization.

## Advantages

1. **Sublinear growth**: log(x) grows slower than x^a
2. **More stable**: Less sensitive to outlier degrees
3. **Well-defined**: Always positive for positive degrees
4. **Simple**: No additional parameter tuning (alpha not used)

## Considerations

- **Requires positive degrees**: log(0) is undefined (handled by zero mask)
- **Different scale**: Produces different value ranges than power method
- **Natural log**: Uses ln (base e), not log10 or log2

## Implementation Details

### Dense Matrix (kernel_utils.py)
```python
elif self.degree_normalization_method == 'log':
    normalization_factor = torch.log(degree_product_safe)
```

### Sparse Matrix (kernel_utils.py)
```python
elif degree_normalization_method == 'log':
    normalization_factor = np.log(degree_product_safe)
```

### Safety
- Zero degrees are masked before applying log
- Zero mask ensures K_ij = 0 when degree_i or degree_j is 0

## Comparison with 'none' Method

### Similarities
- Both use raw (unnormalized) degrees
- Both apply a mathematical transformation
- Both preserve original degree scale information

### Differences
- 'none': Uses power (α/2), controlled by LOCAL_RBF_ALPHA
- 'log': Uses natural log, no parameter needed
- 'none': Can be tuned via alpha
- 'log': Fixed transformation

## Code Changes

1. **`tools/train_al.py`**
   - Added 'log' to CLI choices

2. **`pycls/al/kernel_utils.py`**
   - Added 'log' case in `_apply_degree_normalization()`
   - Added 'log' case in `_apply_degree_normalization_sparse()`
   - Updated docstrings

3. **Documentation**
   - Updated all relevant documentation files

## Expected Log Output

```
[CKMatrixManager] Building K_general matrix with local_rbf kernel, 
  sigma=1.0, local_rbf_alpha=1.0, degree_norm=log, threshold=0.0, 
  sparse=False, device=cpu, alpha=0.3
```

## Testing

```bash
# Test log method
python train_al.py \
  --cfg configs/CIFAR100.yaml \
  --exp-name test_log_method \
  --al bayes_misp \
  --budget 100 \
  --seed 0 \
  --kernel_type local_rbf \
  --degree_normalization_method log \
  --ck_sigma 1.0 \
  --max_iter 10
```

## Backward Compatibility

✓ Fully backward compatible
✓ Existing experiments unaffected
✓ Default remains 'none'
✓ New option, no breaking changes

## Summary

The 'log' method provides a logarithmic alternative to the power-based normalization in the 'none' method. It offers sublinear scaling that can be more stable for datasets with highly variable degree distributions.
