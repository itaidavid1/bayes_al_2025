# C/K Matrix Universal Support - Implementation Summary

## Overview

Successfully implemented universal C and K matrix support for all active learning methods in the TypiClust framework. This enables any AL method (TypiClust, random, uncertainty, etc.) to benefit from pseudo-labeling and distillation features that previously only worked with Bayesian MISP.

## Implementation Details

### Files Created

1. **`pycls/al/kernel_utils.py`** - Shared kernel utilities
   - `build_sparse_kernel_matrix()` - Builds sparse or dense kernel matrices
   - `RBFKernel` class - RBF (Gaussian) kernel implementation
   - `TopHatKernel` class - Top-hat (indicator) kernel implementation
   - `compute_norm()` - Batch distance matrix computation

2. **`pycls/al/ck_matrix_manager.py`** - Universal C/K matrix manager
   - `CKMatrixManager` class - Manages C and K matrices for non-Bayesian AL methods
   - Automatic K matrix building (sparse or dense)
   - C matrix initialization with alpha values
   - C matrix updates after point selection

### Files Modified

1. **`pycls/al/ActiveLearning.py`**
   - Added automatic detection of pseudo-labeling/distillation needs
   - Creates `CKMatrixManager` when needed
   - Exposes `C_general` and `K_general` through `sampling_fn` attribute
   - Added `_has_ck_matrices()` helper method

2. **`tools/train_al.py`**
   - Updated `active_sampling_part()` to call C matrix updates after selection
   - Maintains C matrix state across AL rounds

3. **`pycls/al/BAYES_MISP.py`**
   - Updated to import kernel functions from `kernel_utils` module
   - Removed duplicate function definitions

4. **`pycls/al/BAYES_MISP_v1.py`**
   - Updated to import kernel functions from `kernel_utils` module
   - Removed duplicate function definitions

## How It Works

### Activation Logic

The C/K matrix manager is automatically created when:
- `--train_pseudo_labels` flag is set, OR
- `--distillation_training` flag is set

AND the selected AL method doesn't already have native C/K matrix support.

### Data Flow

```
1. AL Round Begins
   ↓
2. Check if pseudo-labeling or distillation enabled
   ↓
3. If yes, check if AL method has C_general/K_general
   ↓
4. If no, create CKMatrixManager
   - Build K_general from all training features
   - Initialize C_general with alpha values
   - Attach to al_obj.sampling_fn
   ↓
5. AL method selects points (using its own logic)
   ↓
6. Update C matrix with kernel similarities from newly labeled points
   C[unlabeled_points, label] += K[newly_labeled_point, unlabeled_points]
   ↓
7. Prepare loader with pseudo-labels or distillation targets from C matrix
   ↓
8. Train model
```

### C Matrix Update Logic

After each AL round, for each newly labeled point:
1. Get the kernel similarity row: `K[point_idx, :]`
2. Update C for unlabeled points: `C[unlabeled_mask, label] += K[point_idx, unlabeled_mask]`

This mirrors the update logic from Bayesian MISP (lines 744-745).

## Configuration Parameters

The following existing command-line arguments are reused:

- `--kernel_type` (rbf, tophat) - Kernel function type
- `--initial_sigma` - RBF kernel sigma parameter
- `--K_sparsity_threshold` - Minimum kernel value to keep
- `--sparse_K` - Use sparse matrix representation
- `--alpha` - Initial C matrix values
- `--train_pseudo_labels` - Enable pseudo-labeling
- `--distillation_training` - Enable distillation training

## Testing Instructions

### 1. Test with TypiClust + Pseudo-Labels

```bash
python tools/train_al.py \
    --cfg configs/cifar10/al/RESNET18.yaml \
    --al typiclust \
    --budget 100 \
    --initial_size 1000 \
    --seed 42 \
    --train_pseudo_labels \
    --pseudo_labels_threshold 0.7 \
    --kernel_type rbf \
    --initial_sigma 1.0 \
    --K_sparsity_threshold 0.01 \
    --alpha 0.5 \
    --exp-name typiclust_with_ck_matrices
```

**Expected behavior:**
- Log message: `[CKMatrixManager] Building K_general matrix...`
- Log message: `[CKMatrixManager] Initialized with K shape...`
- Log message: `[CKMatrixManager] Updated C matrix after labeling X points`
- Pseudo-labels should be generated from C matrix
- Training should use both labeled and pseudo-labeled data

### 2. Test with Random Sampling + Distillation

```bash
python tools/train_al.py \
    --cfg configs/cifar10/al/RESNET18.yaml \
    --al random \
    --budget 100 \
    --initial_size 1000 \
    --seed 42 \
    --distillation_training \
    --distillation_threshold 0.25 \
    --distillation_temperature 3.0 \
    --distill_factor 0.5 \
    --kernel_type rbf \
    --initial_sigma 1.0 \
    --K_sparsity_threshold 0.01 \
    --alpha 0.5 \
    --exp-name random_with_distillation
```

**Expected behavior:**
- Log message: `Creating universal C/K matrix manager for pseudo-labeling/distillation`
- Distillation targets should be prepared from C matrix
- Training should use both CE loss and KL distillation loss
- Gradient norms for CE and KL should be logged

### 3. Test Bayesian MISP (No Regression)

```bash
python tools/train_al.py \
    --cfg configs/cifar10/al/RESNET18.yaml \
    --al bayes_misp \
    --budget 100 \
    --initial_size 1000 \
    --seed 42 \
    --train_pseudo_labels \
    --pseudo_labels_threshold 0.7 \
    --kernel_type rbf \
    --initial_sigma 1.0 \
    --K_sparsity_threshold 0.01 \
    --alpha 0.5 \
    --exp-name bayes_misp_regression_test
```

**Expected behavior:**
- Should NOT create CKMatrixManager (uses native C/K matrices)
- Should work exactly as before
- Pseudo-labels should be generated from native C_general
- No performance degradation

## Verification Steps

1. **Syntax Checks** ✅
   - All files compile without errors
   - Python syntax validation passed

2. **Import Checks**
   - kernel_utils module imported successfully in BAYES_MISP.py
   - kernel_utils module imported successfully in BAYES_MISP_v1.py
   - ck_matrix_manager imports correctly in ActiveLearning.py

3. **Integration Points**
   - ActiveLearning.__init__() detects need for C/K matrices
   - active_sampling_part() updates C matrix after selection
   - get_lset_loader() reads C matrix for pseudo-labels/distillation

## Code Quality

- ✅ Reuses existing kernel building logic
- ✅ No code duplication (extracted to kernel_utils)
- ✅ Minimal changes to existing code
- ✅ Backward compatible (doesn't affect existing Bayesian MISP)
- ✅ Automatic activation (no manual setup needed)
- ✅ Uses existing configuration parameters

## Benefits

1. **Universal Approach**: Any AL method can now use pseudo-labeling and distillation
2. **Code Reuse**: Leverages proven Bayesian MISP kernel logic
3. **Minimal Impact**: Existing methods unchanged, new capability layered on
4. **Flexible**: Automatically activates only when needed
5. **Consistent**: Uses same update logic as Bayesian MISP

## Future Enhancements

1. **Adaptive Alpha**: Could implement per-point or per-class alpha values
2. **K Matrix Updates**: Could periodically rebuild K with newly labeled features
3. **Alternative Kernels**: Could add support for more kernel types
4. **Memory Optimization**: Could implement incremental K matrix building for very large datasets

## Notes

- The C matrix tracks "coverage" - how much each point has been influenced by labeled neighbors
- The K matrix is the pairwise kernel similarity between all training points
- Sparse K matrices are recommended for large datasets (>10k points)
- The alpha parameter controls the initial "prior" coverage for all points
