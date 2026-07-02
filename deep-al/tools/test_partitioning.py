#!/usr/bin/env python3
"""
Test script to verify Dirichlet partitioning is working correctly.
Run from: TypiClust/deep-al/tools/
"""
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(".."))

from pycls.federated.partitioning import build_balanced_dirichlet_partitions, build_iid_partitions

def test_partitioning():
    """Test Dirichlet partitioning with different alpha values."""
    
    # Simulate CIFAR-10 dataset
    num_samples = 50000
    num_classes = 10
    num_clients = 10
    
    # Create balanced labels (5000 per class)
    labels = np.repeat(np.arange(num_classes), num_samples // num_classes)
    indices = np.arange(num_samples)
    
    alphas_to_test = [1.0, 0.5, 0.1]
    
    print("="*70)
    print("Testing Dirichlet Partitioning")
    print("="*70)
    print(f"Dataset: {num_samples} samples, {num_classes} classes")
    print(f"Clients: {num_clients}")
    print(f"Expected samples per client: {num_samples // num_clients}")
    print()
    
    for alpha in alphas_to_test:
        print(f"\n{'='*70}")
        print(f"Testing alpha = {alpha}")
        print(f"{'='*70}")
        
        partitions = build_balanced_dirichlet_partitions(
            labels=labels,
            indices=indices,
            num_clients=num_clients,
            alpha=alpha,
            seed=42,
            max_retries=1000,
        )
        
        # Analyze the partitions
        print(f"\nPartition Analysis:")
        print(f"  Total clients: {len(partitions)}")
        
        # Check if balanced
        sizes = [len(part) for part in partitions.values()]
        print(f"  Client sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
        
        if min(sizes) == max(sizes):
            print(f"  [SUCCESS] Balanced: All clients have exactly {sizes[0]} samples")
        else:
            print(f"  [WARNING] Unbalanced: Likely fell back to IID")
        
        # Check class distribution for first 3 clients
        print(f"\n  Class distribution for first 3 clients:")
        for cid in range(min(3, num_clients)):
            client_labels = labels[partitions[cid]]
            class_counts = np.bincount(client_labels, minlength=num_classes)
            class_pcts = (class_counts / len(client_labels) * 100).astype(int)
            
            # Find dominant classes
            top_classes = np.argsort(class_counts)[-3:][::-1]
            top_pcts = class_pcts[top_classes]
            
            print(f"    Client {cid}: Size={len(client_labels)}, "
                  f"Top 3 classes: {top_classes.tolist()} with {top_pcts.tolist()}%")
        
        # Compute heterogeneity metric (variance of class distributions)
        all_distributions = []
        for cid in range(num_clients):
            client_labels = labels[partitions[cid]]
            class_counts = np.bincount(client_labels, minlength=num_classes)
            class_dist = class_counts / len(client_labels)
            all_distributions.append(class_dist)
        
        all_distributions = np.array(all_distributions)
        heterogeneity = np.mean(np.var(all_distributions, axis=0))
        print(f"\n  Heterogeneity score: {heterogeneity:.4f} (higher = more heterogeneous)")
        
        if alpha == 1.0:
            expected = "moderate heterogeneity"
        elif alpha == 0.5:
            expected = "high heterogeneity"
        else:
            expected = "very high heterogeneity"
        print(f"  Expected: {expected}")
    
    print(f"\n{'='*70}")
    print("Testing Complete!")
    print(f"{'='*70}")
    print("\nIf you see warnings about falling back to IID,")
    print("it means the algorithm couldn't create balanced heterogeneous partitions.")
    print("This is normal for very small alpha values (< 0.1) with balanced constraints.")

if __name__ == "__main__":
    test_partitioning()
