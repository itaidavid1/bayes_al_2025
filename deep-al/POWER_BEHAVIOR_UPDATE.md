# Power Behavior Update for Degree Normalization

## Important Change

When using degree normalization methods 'sum' or 'max', the power of `alpha/2` is **NOT** applied to the degree product.

## Transformation Formulas

### Method: 'none' (default)
```
K_ij ← K_ij / (degree_i × degree_j)^(alpha/2)
```
**Power IS applied** - Original behavior preserved

### Method: 'sum'
```
normalized_degree_i = degree_i / Σ_j degree_j
K_ij ← K_ij / (normalized_degree_i × normalized_degree_j)
```
**Power is NOT applied** - Direct division only

### Method: 'max'
```
normalized_degree_i = degree_i / max_j(degree_j)
K_ij ← K_ij / (normalized_degree_i × normalized_degree_j)
```
**Power is NOT applied** - Direct division only

## Rationale

When normalizing degrees to [0, 1] range (via 'sum' or 'max'), the degrees already have a controlled scale. Applying the power of `alpha/2` on top of this normalization would introduce an additional non-linear transformation that may not be desired.

The 'none' method preserves the original behavior where the power is applied to the raw degree values.

## Code Implementation

### Dense Matrix (kernel_utils.py)
```python
# Apply power only for 'none' method, skip for 'sum' and 'max'
if self.degree_normalization_method == 'none':
    normalization_factor = torch.pow(degree_product_safe, self.alpha / 2.0)
else:
    normalization_factor = degree_product_safe
```

### Sparse Matrix (kernel_utils.py)
```python
# Apply power only for 'none' method, skip for 'sum' and 'max'
if degree_normalization_method == 'none':
    normalization_factor = np.power(degree_product_safe, alpha / 2.0)
else:
    normalization_factor = degree_product_safe
```

## Comparison Table

| Method | Normalization | Power Applied | Final Formula |
|--------|--------------|---------------|---------------|
| `none` | No | Yes (alpha/2) | `K / (d_i × d_j)^(α/2)` |
| `sum` | Yes (sum) | **No** | `K / (d_i' × d_j')` |
| `max` | Yes (max) | **No** | `K / (d_i' × d_j')` |

Where `d_i'` denotes normalized degree.

## Example

### Original Degrees
```
degrees = [100, 200, 300, 50, 400]
```

### Method: 'none' (alpha=1.0)
```
K_ij / (100 × 200)^0.5 = K_ij / 141.42
```

### Method: 'sum' (alpha=1.0, NOT used)
```
sum = 1050
normalized = [0.095, 0.190, 0.286, 0.048, 0.381]
K_ij / (0.095 × 0.190) = K_ij / 0.018
(NO power applied!)
```

### Method: 'max' (alpha=1.0, NOT used)
```
max = 400
normalized = [0.25, 0.50, 0.75, 0.125, 1.0]
K_ij / (0.25 × 0.50) = K_ij / 0.125
(NO power applied!)
```

## Impact

This change makes the 'sum' and 'max' methods behave more intuitively:
- Degrees are normalized to a controlled range
- Division is straightforward without additional power transformation
- The `LOCAL_RBF_ALPHA` parameter only affects the 'none' method

## When to Use

- **'none':** When you want the original power-based transformation with raw degrees
- **'sum':** When you want probabilistic interpretation with direct division
- **'max':** When you want scale-free comparison with direct division

## Backward Compatibility

✓ The 'none' method (default) behaves exactly as before
✓ Existing experiments using 'none' are unaffected
✓ New 'sum' and 'max' methods provide cleaner alternatives

## Files Updated

1. `pycls/al/kernel_utils.py`
   - `LocalRBFKernel._apply_degree_normalization()`
   - `_apply_degree_normalization_sparse()`
   - Updated docstrings

2. Documentation files:
   - `DEGREE_NORMALIZATION_IMPLEMENTATION.md`
   - `DEGREE_NORMALIZATION_SUMMARY.md`
   - `LOCAL_RBF_KERNEL_IMPLEMENTATION.md`
   - `IMPLEMENTATION_COMPLETE_SUMMARY.md`
   - `POWER_BEHAVIOR_UPDATE.md` (this file)

## Summary

**Key Point:** When using 'sum' or 'max' degree normalization, the transformation is simpler and more direct - just divide by the normalized degree product without applying any power.
