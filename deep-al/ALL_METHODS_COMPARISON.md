# Complete Comparison of All Degree Normalization Methods

## Overview

The Local RBF kernel supports **four degree normalization methods**, each with different characteristics and use cases.

## Complete Comparison Table

| Method | Pre-normalize Degrees? | Transformation | Uses Alpha? | Formula |
|--------|------------------------|----------------|-------------|---------|
| `none` | No | Power (α/2) | ✓ Yes | `K / (d_i × d_j)^(α/2)` |
| `sum` | ✓ Yes (by sum) | None (direct) | ✗ No | `K / (d_i' × d_j')` |
| `max` | ✓ Yes (by max) | None (direct) | ✗ No | `K / (d_i' × d_j')` |
| `log` | No | Log (natural) | ✗ No | `K / log(d_i × d_j)` |

## Detailed Formulas

### Method: 'none' (default)
```
degree_i = Σ_{k≠i} K_ik
K_ij ← K_ij / (degree_i × degree_j)^(LOCAL_RBF_ALPHA / 2)
```
- **Uses raw degrees**
- **Applies power transformation**
- **Controlled by LOCAL_RBF_ALPHA parameter**

### Method: 'sum'
```
degree_i = Σ_{k≠i} K_ik
normalized_degree_i = degree_i / Σ_j degree_j
K_ij ← K_ij / (normalized_degree_i × normalized_degree_j)
```
- **Normalizes degrees to sum to 1**
- **Direct division, no transformation**
- **Degrees become probability distribution**

### Method: 'max'
```
degree_i = Σ_{k≠i} K_ik
normalized_degree_i = degree_i / max_j(degree_j)
K_ij ← K_ij / (normalized_degree_i × normalized_degree_j)
```
- **Normalizes degrees to [0, 1] range**
- **Direct division, no transformation**
- **Largest degree becomes 1.0**

### Method: 'log'
```
degree_i = Σ_{k≠i} K_ik
K_ij ← K_ij / log(degree_i × degree_j)
```
- **Uses raw degrees**
- **Applies logarithmic transformation**
- **Natural log (ln), base e**

## Characteristic Comparison

| Aspect | none | sum | max | log |
|--------|------|-----|-----|-----|
| Degree range | [0, ∞) | [0, 1] | [0, 1] | [0, ∞) |
| Degree sum | Arbitrary | = 1 | ≤ 1 | Arbitrary |
| Transformation | Power | None | None | Log |
| Alpha parameter | Used | Not used | Not used | Not used |
| Tunable | Yes | No | No | No |
| Scale | Preserves | Normalizes | Normalizes | Preserves |

## When to Use Each Method

### Use 'none' when:
- ✓ You want the original behavior
- ✓ You want to control normalization strength via alpha
- ✓ Degree scale is meaningful
- ✓ Default/baseline experiments

### Use 'sum' when:
- ✓ You want probabilistic interpretation
- ✓ You want degrees to sum to 1
- ✓ You want all degrees treated as relative proportions
- ✓ You want simpler, direct division

### Use 'max' when:
- ✓ You want scale-free comparison
- ✓ You want to remove absolute scale effects
- ✓ You want largest degree normalized to 1
- ✓ You want ranking-based normalization

### Use 'log' when:
- ✓ You want sublinear scaling
- ✓ You want less sensitivity to large degrees
- ✓ You want simpler alternative to power
- ✓ You don't want to tune alpha

## Example Values

### Original Degrees
```
degrees = [10, 50, 100, 200, 500]
```

### Normalization Factors for pairs (10, 50)

**Method 'none' (alpha=1.0):**
```
(10 × 50)^0.5 = 22.36
```

**Method 'sum':**
```
sum = 860
norm_10 = 10/860 = 0.0116
norm_50 = 50/860 = 0.0581
product = 0.0116 × 0.0581 = 0.000674
```

**Method 'max':**
```
max = 500
norm_10 = 10/500 = 0.02
norm_50 = 50/500 = 0.10
product = 0.02 × 0.10 = 0.002
```

**Method 'log':**
```
log(10 × 50) = log(500) = 6.21
```

### Relative Magnitudes
For the pair (10, 50):
- none: 22.36
- log: 6.21 (3.6× smaller than none)
- max: 0.002 (11,180× smaller than none)
- sum: 0.000674 (33,175× smaller than none)

## Usage Examples

### CLI Usage
```bash
# None (default)
python train_al.py --kernel_type local_rbf --degree_normalization_method none --local_rbf_alpha 1.0

# Sum
python train_al.py --kernel_type local_rbf --degree_normalization_method sum

# Max
python train_al.py --kernel_type local_rbf --degree_normalization_method max

# Log
python train_al.py --kernel_type local_rbf --degree_normalization_method log
```

### Config File
```yaml
KERNEL_TYPE: 'local_rbf'
DEGREE_NORMALIZATION_METHOD: 'none'  # or 'sum', 'max', 'log'
LOCAL_RBF_ALPHA: 1.0  # only used for 'none' method
```

## Implementation Patterns

### Dense Matrix (PyTorch)
```python
if self.degree_normalization_method == 'none':
    normalization_factor = torch.pow(degree_product_safe, self.alpha / 2.0)
elif self.degree_normalization_method == 'log':
    normalization_factor = torch.log(degree_product_safe)
else:  # 'sum' or 'max'
    normalization_factor = degree_product_safe
```

### Sparse Matrix (NumPy)
```python
if degree_normalization_method == 'none':
    normalization_factor = np.power(degree_product_safe, alpha / 2.0)
elif degree_normalization_method == 'log':
    normalization_factor = np.log(degree_product_safe)
else:  # 'sum' or 'max'
    normalization_factor = degree_product_safe
```

## Decision Tree

```
Do you want to normalize degrees first?
├─ No
│  ├─ Do you want to tune the strength?
│  │  ├─ Yes → use 'none' (power with alpha)
│  │  └─ No → use 'log' (natural log)
│  
└─ Yes
   ├─ Want probabilistic interpretation? → use 'sum'
   └─ Want scale-free comparison? → use 'max'
```

## Performance Characteristics

| Method | Computation Cost | Memory | Tuning Required |
|--------|------------------|--------|-----------------|
| none | Medium (power) | Low | Yes (alpha) |
| sum | Low (division) | Low | No |
| max | Low (division) | Low | No |
| log | Medium (log) | Low | No |

## Mathematical Properties

### Growth Rates (as degree product increases)
- **Power (none):** Superlinear for α > 0
- **Direct (sum/max):** Linear
- **Log:** Sublinear

### Sensitivity to Outliers
- **none:** Medium (depends on alpha)
- **sum:** High (one large degree affects all)
- **max:** Low (only depends on maximum)
- **log:** Low (log dampens large values)

## Recommendations

### For Most Users
Start with **'none'** (default) as it provides good baseline behavior with tunable strength.

### For Exploration
Try all four methods:
1. 'none' with alpha=1.0 (baseline)
2. 'log' (sublinear alternative)
3. 'max' (scale-free)
4. 'sum' (probabilistic)

### For Production
Choose based on your data characteristics:
- Variable degree distribution → 'log' or 'max'
- Uniform degree distribution → 'none'
- Need interpretability → 'sum'

## Summary

Four methods provide flexibility for different use cases:
- **'none'**: Traditional power-based with tunable strength
- **'sum'**: Probabilistic normalization with direct division
- **'max'**: Scale-free normalization with direct division
- **'log'**: Sublinear scaling without tuning

All methods are fully implemented, documented, and backward compatible.
