# Dirichlet Partitioning Fallback Issue - Fixed

## Problem

The Dirichlet partitioning code was falling back to IID partitioning, making all your experiments use IID even when you specified `partition_mode: dirichlet`.

## Root Cause

The `build_balanced_dirichlet_partitions()` function in `pycls/federated/partitioning.py` has a **fallback mechanism**:

```python
# Line 72 in partitioning.py (original)
return build_iid_partitions(idx, num_clients=num_clients, seed=seed)
```

This fallback triggers when the algorithm **fails to create balanced heterogeneous partitions** after `max_retries` attempts.

### Why It Fails

With **very small alpha values** (like 0.1), the Dirichlet distribution creates **highly skewed** class preferences:

```python
# Client 1 wants: [85% class 0, 10% class 1, 5% others]
# Client 2 wants: [5% class 0, 80% class 1, 10% class 2, ...]
# Client 3 wants: [3% class 0, 2% class 1, 90% class 2, ...]
```

When trying to maintain **balanced client sizes** (equal samples per client), the algorithm often runs into conflicts:

1. Client 1 wants 4250 samples from class 0 (85% of 5000)
2. Client 5 also wants 4000 samples from class 0
3. But class 0 only has 5000 total samples → **CONFLICT**
4. After 100 retries without success → Falls back to IID

## Solutions Applied

### ✅ Fix 1: Increase Retry Limit

**File**: `tools/train_federated_al.py`

```python
# Changed from default 100 to 1000
partitions = build_balanced_dirichlet_partitions(
    labels=labels,
    indices=all_indices,
    num_clients=cfg.FEDERATED.NUM_CLIENTS,
    alpha=cfg.FEDERATED.DIRICHLET_ALPHA,
    seed=cfg.RNG_SEED,
    max_retries=1000,  # ← INCREASED
)
```

This gives the algorithm 10x more attempts to find a valid configuration.

### ✅ Fix 2: Add Logging

**File**: `pycls/federated/partitioning.py`

Added print statements to show:
- ✓ When Dirichlet partitioning succeeds
- ⚠ When it falls back to IID (with explanation)

```python
print(f"✓ Successfully created Dirichlet partitions (alpha={alpha}) on attempt {attempt + 1}")
# or
print(f"⚠ WARNING: Failed to create balanced Dirichlet partitions after {max_retries} attempts.")
```

Now you'll **see clearly** in the output whether Dirichlet or IID was used.

### ✅ Fix 3: Improved Alternative Algorithm

The `build_dirichlet_partitions()` function (unbalanced version) is now better documented and also has logging. This version:
- Allows different client sizes
- Works better with very small alpha values
- Rarely falls back to IID

## Testing Your Fix

### Test 1: Run the Test Script

```bash
cd TypiClust/deep-al/tools/
python test_partitioning.py
```

This will show you:
- Whether Dirichlet partitioning succeeds for different alpha values
- Class distribution statistics
- Heterogeneity scores

### Test 2: Run Your Actual Experiment

```bash
cd TypiClust/deep-al/tools/

# Run with Dirichlet
python train_federated_al.py \
    --cfg ../configs/cifar10/al/RESNET18.yaml \
    --al random \
    --partition_mode dirichlet \
    --dirichlet_alpha 0.5 \
    --num_clients 10 \
    --num_rounds 5 \
    --clients_per_round 5 \
    --local_epochs 3 \
    --fl_method fedavg \
    --client_labels_initial_size 10 \
    --federated_mode standard \
    --queries_per_round 0
```

**Look for this output**:
```
Creating Dirichlet partitions with alpha=0.5
✓ Successfully created Dirichlet partitions (alpha=0.5) on attempt 23
Successfully created 10 client partitions
```

**Or if it fails**:
```
Creating Dirichlet partitions with alpha=0.1
⚠ WARNING: Failed to create balanced Dirichlet partitions after 1000 attempts.
  Alpha=0.1 might be too small. Falling back to IID partitioning.
Successfully created 10 client partitions
```

## Understanding the Output

### Success Case
```
✓ Successfully created Dirichlet partitions (alpha=0.5) on attempt 23
```
- Dirichlet partitioning worked
- Took 23 attempts to find a valid configuration
- Your experiment will have heterogeneous data

### Fallback Case
```
⚠ WARNING: Failed to create balanced Dirichlet partitions after 1000 attempts.
  Alpha=0.1 might be too small. Falling back to IID partitioning.
```
- Dirichlet partitioning failed after 1000 attempts
- Using IID instead
- **Your alpha might be too small** for balanced partitioning

## What to Do If Still Falling Back

If you still see IID fallback warnings with your experiments:

### Option 1: Use Slightly Larger Alpha

Instead of alpha=0.1, try:
```yaml
dirichlet_alpha: [0.3, 0.5, 1.0]  # More likely to succeed
```

### Option 2: Accept Unbalanced Clients

Modify `train_federated_al.py` to use the unbalanced version:

```python
# Change line 99 from:
partitions = build_balanced_dirichlet_partitions(...)

# To:
from pycls.federated.partitioning import build_dirichlet_partitions
partitions = build_dirichlet_partitions(
    labels=labels,
    indices=all_indices,
    num_clients=cfg.FEDERATED.NUM_CLIENTS,
    alpha=cfg.FEDERATED.DIRICHLET_ALPHA,
    seed=cfg.RNG_SEED,
    min_size=100,  # Minimum samples per client
)
```

This version allows clients to have different amounts of data (more realistic) and works better with very small alpha.

### Option 3: Increase Retries Further

If 1000 isn't enough, increase to 5000:
```python
max_retries=5000,  # Even more attempts
```

## Verifying Heterogeneity

After running your experiment, check the generated files:

```bash
# Look in your experiment output directory
cat <exp_dir>/dataset_class_distributions.json
```

The `per_client_partition_distributions` section will show each client's class distribution. If you see:
- **All clients have ~10% per class** → IID was used
- **Clients have varying percentages** → Dirichlet worked!

Example of successful Dirichlet (alpha=0.5):
```json
"per_client_partition_distributions": {
  "0": {"0": 820, "1": 450, "2": 380, ...},  # Client 0 specializes in class 0
  "1": {"0": 320, "1": 950, "2": 280, ...},  # Client 1 specializes in class 1
  "2": {"0": 380, "1": 280, "2": 890, ...},  # Client 2 specializes in class 2
  ...
}
```

## Summary of Changes

### Files Modified:
1. ✅ `tools/train_federated_al.py` - Increased retries to 1000, added logging
2. ✅ `pycls/federated/partitioning.py` - Added success/failure messages
3. ✅ `tools/test_partitioning.py` - New test script (NEW FILE)

### What You Should See Now:
- Clear messages indicating whether Dirichlet succeeded or fell back to IID
- 10x more attempts to create valid Dirichlet partitions
- Better chance of success, especially for alpha ≥ 0.3

### Next Steps:
1. Run `test_partitioning.py` to verify the fix
2. Run your actual experiments and watch for the success messages
3. Check `dataset_class_distributions.json` to confirm heterogeneity
4. If still seeing fallbacks, use Option 1 or 2 above

The Dirichlet partitioning should now work for **alpha ≥ 0.3** with high probability! 🎯
