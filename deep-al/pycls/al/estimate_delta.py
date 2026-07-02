import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

import pycls.datasets.utils as ds_utils
import argparse

# Copied from prob_cover
def construct_graph(delta, features, batch_size=500):
    """
    creates a directed graph where:
    x->y iff l2(x,y) < delta.

    represented by a list of edges (a sparse matrix).
    stored in a dataframe
    """
    xs, ys, ds = [], [], []
    print(f'Start constructing graph using delta={delta}')
    # distance computations are done in GPU
    cuda_feats = torch.tensor(features).cuda()
    for i in range(len(features) // batch_size):
        # distance comparisons are done in batches to reduce memory consumption
        cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(cur_feats, cuda_feats)
        mask = dist < delta
        # saving edges using indices list - saves memory.
        x, y = mask.nonzero().T
        xs.append(x.cpu() + batch_size * i)
        ys.append(y.cpu())
        ds.append(dist[mask].cpu())

    xs = torch.cat(xs).numpy()
    ys = torch.cat(ys).numpy()
    ds = torch.cat(ds).numpy()

    df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
    print(f'Finished constructing graph using delta={delta}')
    print(f'Graph contains {len(df)} edges.')
    return df


def calculate_purity(graph, labels):
    counter = torch.zeros(1, device='cuda')
    labels = torch.tensor(labels, device='cuda')
    graph_x = torch.tensor(graph.x.values, device='cuda')
    graph_y = torch.tensor(graph.y.values, device='cuda')

    for ind in tqdm(range(len(labels)), desc="Calculating purity"):
        ball_indices = (graph_x == ind).nonzero().flatten()
        if not torch.any(labels[graph_y[ball_indices]] != labels[ind]):
            counter += 1

    return counter.item() / len(labels)

# Copied from cov_vs_pur_kmeans_vs_norm.py
def max_curvature_try_2(delta, purity):
    """
    Knee point via *true* curvature with central differences that work on non‑uniform grids.

    For each interior index i:
        y'_i  = (y_{i+1} – y_{i-1}) / (x_{i+1} – x_{i-1})
        y''_i = 2 * [ (y_{i+1} – y_i)/(x_{i+1} – x_i) – (y_i – y_{i-1})/(x_i – x_{i-1}) ] / (x_{i+1} – x_{i-1})
    κ_i = |y''_i| / (1 + y'_i²)^{3/2}
    We then pick argmax κ inside purity ∈ [0.05, 0.95].
    """

    # --- sanitise input ---
    delta   = np.asarray(delta,   dtype=float)
    purity = np.asarray(purity, dtype=float)
    if delta.ndim != 1 or purity.ndim != 1 or delta.size != purity.size:
        raise ValueError("purity and coverage must be 1‑D arrays of equal length")
    n = delta.size
    if n < 3:
        raise ValueError("Need at least 3 points for curvature")

    # --- pre‑allocate derivatives ---
    y_prime  = np.zeros(n)
    y_second = np.zeros(n)

    # --- interior points ---
    for i in range(1, n - 1):
        x_im1,  x_i,  x_ip1  = delta[i - 1], delta[i], delta[i + 1]
        y_im1,  y_i,  y_ip1  = purity[i - 1], purity[i], purity[i + 1]

        dx_total = x_ip1 - x_im1
        if dx_total == 0:
            # duplicate x – skip this point later
            y_prime[i] = np.nan
            y_second[i] = np.nan
            continue

        # first derivative
        y_prime[i] = (y_ip1 - y_im1) / dx_total

        # second derivative (non‑uniform grid formula)
        dx_f = x_ip1 - x_i
        dx_b = x_i   - x_im1
        if dx_f == 0 or dx_b == 0:
            y_second[i] = np.nan
        else:
            y_second[i] = 2 * ( (y_ip1 - y_i) / dx_f - (y_i - y_im1) / dx_b ) / dx_total

    # --- edge points: copy nearest interior values to avoid NaN ---
    y_prime[0],  y_prime[-1]  = y_prime[1],  y_prime[-2]
    y_second[0], y_second[-1] = y_second[1], y_second[-2]

    # --- curvature ---
    # Normalize to curve length
    kappa = np.abs(y_second) / np.power(1 + y_prime ** 2, 1.5)
    kappa[~np.isfinite(kappa)] = -np.inf  # ignore bad points

    # --- restrict purity range ---
    mask = (delta >= 0.05) & (delta <= 0.95) & (purity >= 0.20)
    valid_idx = np.where(mask)[0]
    if valid_idx.size == 0:
        raise ValueError("No points within delta ∈ [0.15, 0.98]")

    idx_knee = valid_idx[np.argmax(kappa[valid_idx])]
    return float(delta[idx_knee]), float(purity[idx_knee])

def main():
    parser = argparse.ArgumentParser(description="Script for processing dataset with a specific embedding type.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset to use.", default='TINYIMAGENET')
    # parser.add_argument("--embedding", type=str, required=True, help="Type of embedding to apply.", nargs='+')
    # parser.add_argument('--embeddings_action', default='list', type=str, choices=['list', 'avg', 'concat', 'pca'])

    args = parser.parse_args()
    features = ds_utils.load_features(args.dataset, train=True)

    # features = ds_utils.load_features(args.dataset, embed_name=args.embedding, action=args.embeddings_action)

    num_classes = 100 if '100' in args.dataset else 10

    kmeans_model = KMeans(n_clusters=num_classes)
    clusters = kmeans_model.fit_predict(features)  # List of pseudo-labels

    print("Clusterd dataset {} into {} clusters.".format(args.dataset, num_classes))

    import matplotlib.pyplot as plt

    # Compute purity values for different delta values
    delta_values = np.arange(0, 1.01, 0.05)
    # if args.embedding == "solo_dino":
    #     delta_values = np.arange(0, 0.391, 0.015)
    purity_values = []

    try:
        for delta in delta_values:
            G = construct_graph(delta, features)
            purity_values.append(calculate_purity(G, clusters))
            print(f'delta: {delta}, purity: {purity_values[-1]}')
    finally:
        delta_values = delta_values[:len(purity_values)]

        # Find the largest delta where purity is greater than alpha=0.95
        alpha = 0.95
        max_delta = max(delta for delta, purity in zip(delta_values, purity_values) if purity >= alpha)

        x_knee, y_knee = max_curvature_try_2(delta_values, purity_values)

        # Plot the purity curve
        plt.figure(figsize=(6, 4))

        # Add the dataset label
        plt.title(f"{args.dataset}")
        plt.plot(delta_values, purity_values, 'bo-', label="Purity", zorder=1)  # Blue dots with line
        plt.axvline(x=max_delta, color='deepskyblue', linestyle='dashed', label=f"$\delta^*$ = {max_delta:.2f}", zorder=2)
        plt.scatter([x_knee], [y_knee], color='red', marker='X', s=50,
                    label=f'Max Curv $\delta$ = {x_knee:.2f}', zorder=3)

        # Formatting
        plt.xlabel(r"$\delta$", fontsize=14)
        plt.ylabel("Purity", fontsize=14)
        plt.ylim(0, 1.05)
        plt.xlim(0, 1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        # Save the figure
        plt.tight_layout()
        # plt.savefig(f"purity_vs_delta_{args.dataset}_{args.embeddings_action if args.embeddings_action != 'list' else ''}{args.embedding if len(args.embedding) > 1 else args.embedding[0]}_knee.png", dpi=300)


if __name__ == "__main__":
    main()