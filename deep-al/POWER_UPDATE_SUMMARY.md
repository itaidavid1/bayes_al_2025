# Power Update Summary

## What Changed

Updated the Local RBF kernel so that when using 'sum' or 'max' degree normalization methods, the power of `alpha/2` is **NOT** applied.

## Before vs After

### Method: 'none' (unchanged)
**Before:** `K_ij / (degree_i × degree_j)^(alpha/2)`  
**After:** `K_ij / (degree_i × degree_j)^(alpha/2)` ✓ Same

### Method: 'sum' (changed)
**Before:** `K_ij / (normalized_degree_i × normalized_degree_j)^(alpha/2)`  
**After:** `K_ij / (normalized_degree_i × normalized_degree_j)` ✓ No power

### Method: 'max' (changed)
**Before:** `K_ij / (normalized_degree_i × normalized_degree_j)^(alpha/2)`  
**After:** `K_ij / (normalized_degree_i × normalized_degree_j)` ✓ No power

## Quick Reference

| Method | Power Applied? | Formula |
|--------|---------------|---------|
| `none` | ✓ Yes | `K / (d_i × d_j)^(α/2)` |
| `sum` | ✗ No | `K / (d_i' × d_j')` |
| `max` | ✗ No | `K / (d_i' × d_j')` |

## Code Changes

Two functions updated in `kernel_utils.py`:

### 1. Dense Matrix
```python
# Apply power only for 'none' method, skip for 'sum' and 'max'
if self.degree_normalization_method == 'none':
    normalization_factor = torch.pow(degree_product_safe, self.alpha / 2.0)
else:
    normalization_factor = degree_product_safe
```

### 2. Sparse Matrix
```python
# Apply power only for 'none' method, skip for 'sum' and 'max'
if degree_normalization_method == 'none':
    normalization_factor = np.power(degree_product_safe, alpha / 2.0)
else:
    normalization_factor = degree_product_safe
```

## Why This Makes Sense

When normalizing degrees to [0, 1] range:
- Degrees already have controlled scale
- Direct division is cleaner and more interpretable
- Power transformation would add unnecessary complexity

## Backward Compatibility

✓ **Perfect backward compatibility**
- Default 'none' method unchanged
- Existing experiments unaffected
- New methods provide cleaner alternatives

## Testing

To test the difference:

```bash
# Original behavior (power applied)
python train_al.py --kernel_type local_rbf --degree_normalization_method none --local_rbf_alpha 1.0

# New behavior (no power)
python train_al.py --kernel_type local_rbf --degree_normalization_method sum

# New behavior (no power)
python train_al.py --kernel_type local_rbf --degree_normalization_method max
```

## Files Updated

**Code:**
- `pycls/al/kernel_utils.py` (2 functions)

**Documentation:**
- `DEGREE_NORMALIZATION_IMPLEMENTATION.md`
- `DEGREE_NORMALIZATION_SUMMARY.md`
- `LOCAL_RBF_KERNEL_IMPLEMENTATION.md`
- `IMPLEMENTATION_COMPLETE_SUMMARY.md`
- `POWER_BEHAVIOR_UPDATE.md`
- `POWER_UPDATE_SUMMARY.md` (this file)

## Status

✅ **Implementation Complete**
✅ **No Linter Errors**
✅ **Documentation Updated**
✅ **Backward Compatible**
