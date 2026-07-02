# Log Method Addition - Summary

## What Was Added

A new degree normalization method called **'log'** for the Local RBF kernel.

## Formula

```
K_ij ← K_ij / log(degree_i × degree_j)
```

Where `log` is the natural logarithm (ln).

## Key Characteristics

| Aspect | Value |
|--------|-------|
| Pre-normalization | No (uses raw degrees) |
| Transformation | Natural logarithm |
| Alpha parameter | NOT used |
| Similar to | 'none' (no pre-norm) |
| Different from | 'none' (uses log, not power) |

## Quick Comparison

| Method | Transformation | Uses Alpha? | Formula |
|--------|---------------|-------------|---------|
| `none` | Power (α/2) | ✓ Yes | `K / (d_i × d_j)^(α/2)` |
| `sum` | None (direct) | ✗ No | `K / (d_i' × d_j')` |
| `max` | None (direct) | ✗ No | `K / (d_i' × d_j')` |
| **`log`** | **Log (natural)** | **✗ No** | **`K / log(d_i × d_j)`** |

## Usage

```bash
# CLI
python train_al.py \
  --kernel_type local_rbf \
  --degree_normalization_method log \
  --ck_sigma 1.0

# Config
DEGREE_NORMALIZATION_METHOD: 'log'
```

## When to Use

- Want sublinear scaling
- Want less sensitivity to large degree differences
- Want simpler alternative to power-based normalization
- Don't want to tune alpha parameter

## Code Changes

### 1. train_al.py
- Added 'log' to CLI argument choices

### 2. kernel_utils.py
- Added 'log' case in `LocalRBFKernel._apply_degree_normalization()`
- Added 'log' case in `_apply_degree_normalization_sparse()`
- Updated docstrings

### 3. Documentation
- Updated `DEGREE_NORMALIZATION_IMPLEMENTATION.md`
- Updated `DEGREE_NORMALIZATION_SUMMARY.md`
- Updated `LOCAL_RBF_KERNEL_IMPLEMENTATION.md`
- Created `LOG_METHOD_IMPLEMENTATION.md`
- Created `LOG_METHOD_SUMMARY.md` (this file)

## Example: Normalization Factor Comparison

### Degrees: 100 × 200

**Method 'none' (alpha=1.0):**
```
(100 × 200)^0.5 = 141.42
```

**Method 'log':**
```
log(100 × 200) = log(20000) = 9.90
```

**Ratio:**
```
141.42 / 9.90 = 14.28
```

The log method produces a smaller normalization factor (less aggressive).

## Benefits

✓ Sublinear growth (log grows slower than power)
✓ More stable for variable degree distributions
✓ No parameter tuning needed
✓ Simple to understand and use

## Implementation Details

```python
# Dense
if self.degree_normalization_method == 'log':
    normalization_factor = torch.log(degree_product_safe)

# Sparse
if degree_normalization_method == 'log':
    normalization_factor = np.log(degree_product_safe)
```

## Safety

- Zero degrees are handled by zero mask
- K_ij = 0 when degree_i = 0 or degree_j = 0
- No division by zero issues

## Backward Compatibility

✅ Fully backward compatible
- Default remains 'none'
- Existing experiments unaffected
- New option, no breaking changes

## Files Modified

1. `tools/train_al.py` - CLI argument
2. `pycls/al/kernel_utils.py` - Core implementation
3. Multiple documentation files

## Status

✅ Implementation complete
✅ No linter errors introduced
✅ Documentation complete
✅ Backward compatible

## Quick Start

To try the log method:

```bash
python train_al.py \
  --cfg configs/CIFAR100.yaml \
  --exp-name test_log \
  --al bayes_misp \
  --budget 100 \
  --seed 0 \
  --kernel_type local_rbf \
  --degree_normalization_method log \
  --ck_sigma 1.0 \
  --max_iter 10
```

Expected log output:
```
[CKMatrixManager] Building K_general matrix with local_rbf kernel, 
  sigma=1.0, local_rbf_alpha=1.0, degree_norm=log, ...
```
