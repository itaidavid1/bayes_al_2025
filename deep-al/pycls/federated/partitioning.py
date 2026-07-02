from typing import Dict

import numpy as np


def build_iid_partitions(indices, num_clients: int, seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    splits = np.array_split(shuffled, num_clients)
    return {client_id: split.astype(np.int64) for client_id, split in enumerate(splits)}


def build_balanced_dirichlet_partitions(
    labels,
    indices,
    num_clients: int,
    alpha: float,
    seed: int,
    max_retries: int = 100,
) -> Dict[int, np.ndarray]:
    """
    Create balanced non-IID partitions using the LDA (Latent Dirichlet Allocation) approach.
    
    This is the standard method used in federated learning research (e.g., FedProx paper).
    Each client samples a label distribution p ~ Dir(alpha), then samples data points
    according to p until they have enough samples.
    
    Reference: https://arxiv.org/abs/1812.06127 (Measuring the Effects of Non-Identical Data Distribution)
    """
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be > 0.")
    
    idx = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)[idx]
    classes = np.unique(y)
    num_classes = len(classes)
    rng = np.random.default_rng(seed)
    
    samples_per_client = len(idx) // num_clients
    
    # Group indices by class
    class_indices = {cls: idx[y == cls].tolist() for cls in classes}
    for cls in classes:
        rng.shuffle(class_indices[cls])
    
    # Each client draws a class distribution from Dirichlet
    client_class_probs = []
    for _ in range(num_clients):
        probs = rng.dirichlet(np.repeat(alpha, num_classes))
        client_class_probs.append(probs)
    
    # Assign samples to clients
    client_samples = [[] for _ in range(num_clients)]
    available_indices = {cls: class_indices[cls].copy() for cls in classes}
    
    # Iteratively assign samples
    for client_id in range(num_clients):
        target_size = samples_per_client
        probs = client_class_probs[client_id]
        
        while len(client_samples[client_id]) < target_size:
            # Sample a class according to this client's distribution
            # But only from classes that still have samples available
            available_classes = [cls for cls in classes if len(available_indices[cls]) > 0]
            
            if not available_classes:
                break
            
            # Renormalize probabilities over available classes
            available_probs = np.array([probs[cls] for cls in available_classes])
            if available_probs.sum() > 0:
                available_probs = available_probs / available_probs.sum()
            else:
                available_probs = np.ones(len(available_classes)) / len(available_classes)
            
            # Sample a class
            chosen_class = rng.choice(available_classes, p=available_probs)
            
            # Take a sample from that class
            if len(available_indices[chosen_class]) > 0:
                sample_idx = available_indices[chosen_class].pop(0)
                client_samples[client_id].append(sample_idx)
    
    # Convert to dict of numpy arrays
    partitions = {}
    for client_id in range(num_clients):
        partitions[client_id] = np.array(client_samples[client_id], dtype=np.int64)
    
    # Check if we got balanced partitions
    sizes = [len(partitions[cid]) for cid in range(num_clients)]
    min_size, max_size = min(sizes), max(sizes)
    
    if min_size == max_size == samples_per_client:
        # Perfect balance! Compute heterogeneity
        heterogeneity = 0.0
        for client_id in range(num_clients):
            client_labels = y[partitions[client_id]]
            class_counts = np.bincount(client_labels, minlength=num_classes)
            class_dist = class_counts / max(len(client_labels), 1)
            uniform_dist = np.ones(num_classes) / num_classes
            
            # KL divergence
            class_dist_smooth = (class_dist + 1e-10) / (1 + 1e-10 * num_classes)
            kl = np.sum(class_dist_smooth * np.log(class_dist_smooth / (uniform_dist + 1e-10)))
            heterogeneity += kl
        
        heterogeneity /= num_clients
        print(f"[SUCCESS] Created Dirichlet partitions (alpha={alpha}, het={heterogeneity:.3f})")
        return partitions
    
    # If not perfectly balanced, fall back to IID
    print(f"[WARNING] Created unbalanced partitions (min={min_size}, max={max_size}). Falling back to IID.")
    return build_iid_partitions(idx, num_clients=num_clients, seed=seed)


def build_dirichlet_partitions(
    labels,
    indices,
    num_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 1,
    max_retries: int = 100,
) -> Dict[int, np.ndarray]:
    """
    Create unbalanced non-IID partitions using Dirichlet distribution.
    
    This version allows clients to have different amounts of data,
    which makes it more robust for very small alpha values.
    
    Falls back to IID if unable to satisfy min_size constraint.
    """
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be > 0.")
    idx = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)[idx]
    classes = np.unique(y)
    rng = np.random.default_rng(seed)

    for attempt in range(max_retries):
        buckets = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = idx[y == cls]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            cls_splits = np.split(cls_idx, cuts)
            for cid, split in enumerate(cls_splits):
                if split.size:
                    buckets[cid].append(split)

        out = {}
        valid = True
        for cid, parts in enumerate(buckets):
            merged = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
            out[cid] = merged.astype(np.int64)
            if out[cid].size < min_size:
                valid = False
        
        if valid:
            print(f"[SUCCESS] Created unbalanced Dirichlet partitions (alpha={alpha}) on attempt {attempt + 1}")
            return out

    # Safe fallback: deterministic IID split.
    print(f"[WARNING] Failed to create Dirichlet partitions after {max_retries} attempts.")
    print(f"  Falling back to IID partitioning.")
    return build_iid_partitions(idx, num_clients=num_clients, seed=seed)
