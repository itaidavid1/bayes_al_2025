import numpy as np
import pandas as pd
import torch
import gc
import pickle
import scipy.sparse as sp
import pycls.datasets.utils as ds_utils
from tools.utils import visualize_tsne
import time
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.profiler import profile, record_function, ProfilerActivity
from . import prior_selection
###MISP = maximum importance sampling points
torch.cuda.empty_cache()

def compute_norm(x1, x2, device, batch_size=512, matrices_type=torch.float16):
    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        # distance comparisons are done in batches to reduce memory consumption
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset).to(dtype=matrices_type)

        dist_matrix.append(dist.cpu())
        del dist

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix

def build_sparse_kernel_matrix(
        features,
        threshold,
        *,
        kernel_type,
        kernel_param,
        batch_size=1024,
        device='cuda',
        dtype=torch.float32,
        zero_indices=None,
        prev_threshold=None,
        capture_zero_contrib=False,
):
    """
    Build a symmetric sparse kernel matrix in CSR format without materializing the full dense matrix.

    Args:
        features (np.ndarray or torch.Tensor): Feature matrix of shape (N, D) on CPU.
        threshold (float): Value used to sparsify the kernel. For 'rbf' this is the minimum kernel value kept.
                           For 'tophat' this is interpreted as the distance cutoff (delta).
        kernel_type (str): 'rbf' or 'tophat'.
        kernel_param (float): Kernel-specific parameter. For 'rbf' this is sigma. For 'tophat' it is ignored.
        batch_size (int): Number of rows processed per chunk.
        device (str or torch.device): Device used for intermediate GPU computations.
        dtype (torch.dtype): Dtype for intermediate tensors (float32 recommended for sparse path).
        zero_indices (array-like, optional): Indices whose rows/cols should be zeroed in the returned matrix.
        prev_threshold (float, optional): Previous threshold value. Required when
            capture_zero_contrib=True to identify newly added connections.
        capture_zero_contrib (bool): When True, returns the contributions that
            originate from zeroed indices but were newly introduced by lowering
            the sparsity threshold.

    Returns:
        scipy.sparse.csr_matrix: Symmetric sparse kernel matrix (N x N).
        If capture_zero_contrib is True, returns a tuple containing the CSR
        matrix and a dictionary with the newly discovered connections.
    """
    torch_device = torch.device(device)
    if isinstance(features, torch.Tensor):
        features_tensor = features.to(device=torch_device, dtype=torch.float32)
    else:
        features_tensor = torch.from_numpy(features).to(device=torch_device, dtype=torch.float32)

    n_samples = features_tensor.shape[0]

    row_blocks = []
    col_blocks = []
    data_blocks = []

    thresh_val = threshold.item() if torch.is_tensor(threshold) else float(threshold)
    prev_thresh_val = None
    if prev_threshold is not None:
        prev_thresh_val = prev_threshold.item() if torch.is_tensor(prev_threshold) else float(prev_threshold)

    if kernel_type == 'rbf':
        kernel_param_val = kernel_param.item() if torch.is_tensor(kernel_param) else float(kernel_param)

    capture_zero_contrib = bool(capture_zero_contrib and prev_thresh_val is not None and
                                zero_indices is not None and len(zero_indices) > 0)
    zero_idx = None
    zero_mask_np = None
    removed_sources = []
    removed_targets = []
    removed_values = []
    if zero_indices is not None and len(zero_indices) > 0:
        zero_idx = np.asarray(zero_indices, dtype=np.int64)
        if capture_zero_contrib:
            zero_mask_np = np.zeros(n_samples, dtype=bool)
            zero_mask_np[zero_idx] = True

    with torch.no_grad():
        for start_i in range(0, n_samples, batch_size):
            end_i = min(start_i + batch_size, n_samples)
            chunk_i = features_tensor[start_i:end_i].to(dtype=dtype, non_blocking=True)

            for start_j in range(start_i, n_samples, batch_size):
                end_j = min(start_j + batch_size, n_samples)
                chunk_j = features_tensor[start_j:end_j].to(dtype=dtype, non_blocking=True)

                dist_block = torch.cdist(chunk_i, chunk_j)

                if kernel_type == 'tophat':
                    kernel_block = (dist_block < thresh_val).to(dtype=dtype)
                elif kernel_type == 'rbf':
                    kernel_block = torch.exp(-1.0 * (dist_block / kernel_param_val) ** 2)
                else:
                    raise ValueError(f"Unsupported kernel type: {kernel_type}")

                if kernel_type == 'rbf':
                    mask = kernel_block > thresh_val
                else:
                    mask = kernel_block > 0

                nz_rows, nz_cols = torch.nonzero(mask, as_tuple=True)
                if nz_rows.numel() == 0:
                    continue

                values = kernel_block[nz_rows, nz_cols]

                rows_cpu = (nz_rows + start_i).cpu().numpy()
                cols_cpu = (nz_cols + start_j).cpu().numpy()
                data_cpu = values.cpu().numpy().astype(np.float32, copy=False)

                row_blocks.append(rows_cpu)
                col_blocks.append(cols_cpu)
                data_blocks.append(data_cpu)

                if start_j != start_i:
                    row_blocks.append(cols_cpu)
                    col_blocks.append(rows_cpu)
                    data_blocks.append(data_cpu.copy())

                if not capture_zero_contrib:
                    continue

                if kernel_type == 'rbf':
                    prev_keep_mask = kernel_block > prev_thresh_val
                else:
                    prev_keep_mask = dist_block < prev_thresh_val

                new_entries_mask = mask & (~prev_keep_mask)
                new_rows, new_cols = torch.nonzero(new_entries_mask, as_tuple=True)
                if new_rows.numel() == 0:
                    continue

                contrib_rows = (new_rows + start_i).cpu().numpy()
                contrib_cols = (new_cols + start_j).cpu().numpy()
                contrib_vals = kernel_block[new_rows, new_cols].cpu().numpy().astype(np.float32, copy=False)

                zero_rows_mask = zero_mask_np[contrib_rows]
                zero_cols_mask = zero_mask_np[contrib_cols]

                if zero_rows_mask.any():
                    non_zero_cols = ~zero_mask_np[contrib_cols]
                    row_mask = zero_rows_mask & non_zero_cols
                    if row_mask.any():
                        removed_sources.append(contrib_rows[row_mask])
                        removed_targets.append(contrib_cols[row_mask])
                        removed_values.append(contrib_vals[row_mask])

                if zero_cols_mask.any():
                    non_zero_rows = ~zero_mask_np[contrib_rows]
                    col_mask = zero_cols_mask & non_zero_rows
                    if col_mask.any():
                        removed_sources.append(contrib_cols[col_mask])
                        removed_targets.append(contrib_rows[col_mask])
                        removed_values.append(contrib_vals[col_mask])

    if not row_blocks:
        csr = sp.csr_matrix((n_samples, n_samples), dtype=np.float32)
    else:
        rows = np.concatenate(row_blocks)
        cols = np.concatenate(col_blocks)
        data = np.concatenate(data_blocks)
        coo = sp.coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        csr = coo.tocsr()

    if zero_idx is not None and zero_idx.size > 0:
        csr[zero_idx, :] = 0
        csr[:, zero_idx] = 0

    if not capture_zero_contrib:
        return csr

    if removed_sources:
        zero_contrib = {
            "sources": np.concatenate(removed_sources).astype(np.int64, copy=False),
            "targets": np.concatenate(removed_targets).astype(np.int64, copy=False),
            "values": np.concatenate(removed_values).astype(np.float32, copy=False),
        }
    else:
        zero_contrib = {
            "sources": np.empty((0,), dtype=np.int64),
            "targets": np.empty((0,), dtype=np.int64),
            "values": np.empty((0,), dtype=np.float32),
        }

    return csr, zero_contrib


class RBFKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, matrices_type=torch.float16):
        norm = compute_norm(x1, x2, self.device, batch_size=batch_size, matrices_type=matrices_type)
        k = torch.exp(-1.0 * (norm / h) ** 2)
        return k

    def compute_kernel_from_norm(self, norm_matrix, h, matrices_type=torch.float16):
        k = torch.exp(-1.0 * (norm_matrix / h) ** 2).to(dtype=matrices_type)
        return k


class TopHatKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512, matrices_type=torch.float16):
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device) # 1 x n x d, 1 x n' x d
        dist_matrix = []
        batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(x1, x2_subset)
            dist = (dist < h).to(dtype=matrices_type)
            dist_matrix.append(dist.cpu())
            del dist
        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        # k = (dist_matrix < h).to(dtype=torch.float16)
        return dist_matrix

    def compute_kernel_from_norm(self, norm_matrix, h, matrices_type=torch.float16):
        k = (norm_matrix < h).to(dtype=matrices_type)
        return k


class BAYES_MISP:
    def __init__(self, cfg, budgetSize, train_labels, lset, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, train=True, )
        self.diff_method = self.cfg.DIFF_METHOD if 'DIFF_METHOD' in self.cfg else 'abs_diff'
        self.alpha = self.cfg.ALPHA if self.diff_method not in ['prob_cover', 'max_herding'] else 0
        self.debug = self.cfg.DEBUG
        self.use_sparse = self.cfg.SPARSE_K
        self.matrices_type = torch.float32 if self.use_sparse else torch.float16
        self.cont_method = self.cfg.CONT_METHOD if 'CONT_METHOD' in self.cfg else 'positive'
        self.distribution_cont_weight_method = self.cfg.DISTRIBUTION_CONT_WEIGHT_METHOD if 'DISTRIBUTION_CONT_WEIGHT_METHOD' in self.cfg else 'weighted'
        self.budgetSize = budgetSize
        self.K_sparsity_threshold = self.cfg.K_SPARSITY_THRESHOLD
        self.sigma = cfg.ACTIVE_LEARNING.INITIAL_SIGMA if 'INITIAL_SIGMA' in cfg.ACTIVE_LEARNING else 1.0
        self.update_K_matrix = self.cfg.UPDATE_K_MATRIX if 'UPDATE_K_MATRIX' in self.cfg else False
        self.class_weighting_method = self.cfg.CLASS_WEIGHTING_METHOD if 'CLASS_WEIGHTING_METHOD' in self.cfg else 'none'


        self.delta = delta

        self.train_labels_general = np.array(train_labels)
        unique_labels = np.unique(self.train_labels_general)
        self.unique_labels = unique_labels
        self.num_of_classes = unique_labels.size

        self.alpha_lower = cfg.ALPHA_LOWER_BOUND
        self.alpha_upper = cfg.ALPHA_UPPER_BOUND

        self.kernel_build_batch_size = getattr(self.cfg, 'KERNEL_BUILD_BATCH_SIZE', 1024)

        self.chosen_labels_num = torch.zeros(self.num_of_classes).to('cuda')
        self.cum_labels_info = torch.zeros(self.num_of_classes).to('cuda')
        self.labeled_points_mask_general = torch.zeros(self.all_features.shape[0], dtype=torch.bool).to('cuda')

        self.kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if self.kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
            initial_threshold = self.delta
        else:
            self.kernel_fn = RBFKernel('cuda')
            initial_threshold = self.K_sparsity_threshold

        self.K_general = self.build_K_general_matrix(
            self.all_features,
            threshold=initial_threshold,
            zero_indices=None
        )

        self.total_connections_chosen = 0

        if cfg.LOCAL_ALPHA:
            self.init_C(lset, self.K_general, cfg.LOCAL_ALPHA_ORACLE_METHOD)
        else:
            self.C_general = torch.full((self.all_features.shape[0], unique_labels.size), self.alpha, device='cuda',
                                        dtype=self.matrices_type)

        total_points = self.all_features.shape[0]
        if sp.issparse(self.K_general):
            kernel_sum = float((self.K_general > 0).sum())
        elif torch.is_tensor(self.K_general):
            kernel_sum = float((self.K_general > 0).sum().item())
        else:
            kernel_sum = float(np.sum((self.K_general > 0)))
        total_pairs = total_points ** 2
        sparsity_ratio = 1 - (kernel_sum / total_pairs)
        self.initial_sparse_index = max(
            (int(sparsity_ratio * total_pairs) - total_points) // 2,
            0
        )  ### calculate the general sparsity value, then calculate the general initial sparse index (by multiply by the total K size, then remove the self connections (-all_features.shape[0]) and divide by 2 (since the matrix is symmetric)

        if lset is not None and lset.size > 0 and not self.use_sparse:
            temp_K = self.K_general
            class_indices = {label: np.where(self.train_labels_general[lset.astype(int)] == label)[0] for label in unique_labels}

            for label in unique_labels:
                label_positions = class_indices[label]
                if label_positions.size == 0:
                    continue
                curr_labels_sim = temp_K[label_positions]
                self.C_general[:, label] = torch.max(curr_labels_sim, axis=0).values.to(device='cuda')
                del curr_labels_sim
            del temp_K, class_indices
        torch.cuda.empty_cache()

    def build_K_general_matrix(self, features, threshold, zero_indices=None, prev_threshold=None):
        thresh_val = threshold.item() if isinstance(threshold, torch.Tensor) else float(threshold)
        prev_thresh_val = None
        if prev_threshold is not None:
            prev_thresh_val = prev_threshold.item() if isinstance(prev_threshold, torch.Tensor) else float(prev_threshold)
        if self.use_sparse:
            kernel_param = self.delta if self.kernel_type == 'tophat' else self.sigma
            kernel_param_val = kernel_param.item() if isinstance(kernel_param, torch.Tensor) else float(kernel_param)
            capture_new_connections = (
                prev_thresh_val is not None and
                zero_indices is not None and
                len(zero_indices) > 0
            )
            build_result = build_sparse_kernel_matrix(
                features,
                threshold=thresh_val,
                kernel_type=self.kernel_type,
                kernel_param=kernel_param_val,
                batch_size=self.kernel_build_batch_size,
                device='cuda',
                dtype=self.matrices_type,
                zero_indices=zero_indices,
                prev_threshold=prev_thresh_val,
                capture_zero_contrib=capture_new_connections,
            )
            if capture_new_connections:
                K_matrix, zero_contrib = build_result
                if zero_contrib and zero_contrib["sources"].size > 0:
                    self._update_C_with_label_connections(zero_contrib)
                return K_matrix
            return build_result

        if isinstance(features, torch.Tensor):
            features_tensor = features.to(torch.float32)
        else:
            features_tensor = torch.from_numpy(features).to(torch.float32)

        norm_matrix = compute_norm(
            features_tensor,
            features_tensor,
            'cuda',
            batch_size=self.kernel_build_batch_size,
            matrices_type=torch.float32
        ).to('cpu')

        if self.kernel_type == 'tophat':
            dense_K = self.kernel_fn.compute_kernel_from_norm(
                norm_matrix, thresh_val, matrices_type=self.matrices_type)
        else:
            dense_K = self.kernel_fn.compute_kernel_from_norm(
                norm_matrix, self.sigma, matrices_type=self.matrices_type)
            if thresh_val > 0:
                dense_K = torch.where(dense_K > thresh_val, dense_K, torch.zeros_like(dense_K))

        if zero_indices is not None and len(zero_indices) > 0:
            zero_idx = torch.as_tensor(np.asarray(zero_indices, dtype=np.int64))
            dense_K[zero_idx, :] = 0
            dense_K[:, zero_idx] = 0

        return dense_K

    def _update_C_with_label_connections(self, zero_contrib):
        """
        Update C matrix with newly available connections originating from the labeled set.

        Args:
            zero_contrib (dict): Dictionary with 'sources', 'targets', and 'values' arrays.
        """
        if not zero_contrib:
            return

        sources = zero_contrib.get("sources")
        targets = zero_contrib.get("targets")
        values = zero_contrib.get("values")

        if sources is None or targets is None or values is None:
            return

        if len(sources) == 0:
            return

        sources_np = np.asarray(sources, dtype=np.int64)
        targets_np = np.asarray(targets, dtype=np.int64)
        values_np = np.asarray(values, dtype=np.float32)

        device = self.C_general.device

        labels_np = self.train_labels_general[sources_np]
        targets_t = torch.from_numpy(targets_np).to(device=device, dtype=torch.long)
        labels_t_full = torch.from_numpy(labels_np).to(device=device, dtype=torch.long)
        values_t_full = torch.from_numpy(values_np).to(device=device, dtype=self.C_general.dtype)

        unlabeled_mask = ~self.labeled_points_mask_general[targets_t]
        if not torch.any(unlabeled_mask):
            return

        targets_t = targets_t[unlabeled_mask]
        labels_t = labels_t_full[unlabeled_mask]
        values_t = values_t_full[unlabeled_mask]

        if values_t.numel() == 0:
            return

        if self.diff_method in ['prob_cover', 'max_herding']:
            targets_cpu = targets_t.cpu().numpy()
            labels_cpu = labels_t.cpu().numpy()
            values_cpu = values_t.cpu().numpy()

            pair_to_max = {}
            for tgt, lab, val in zip(targets_cpu, labels_cpu, values_cpu):
                key = (tgt, lab)
                current_val = pair_to_max.get(key)
                if current_val is None or val > current_val:
                    pair_to_max[key] = val

            for (tgt, lab), val in pair_to_max.items():
                current_tensor = self.C_general[tgt, lab]
                if val > current_tensor.item():
                    self.C_general[tgt, lab] = current_tensor.new_tensor(val)
        else:
            self.C_general.index_put_((targets_t, labels_t), values_t, accumulate=True)
            self.cum_labels_info.index_put_((labels_t,), values_t, accumulate=True)


    def get_priors(self, K, oracle_method):
        # Set sample specific prior
        # rel_measures = prior_selection.compute_reliability(self.K, self.train_labels, batch_size=8192, normalized=True)
        if sp.issparse(K):
            K_csr = K.tocsr().astype(np.float32, copy=False)
            n_points = K_csr.shape[0]
            y_indices = self.train_labels_general.astype(np.int64)
            one_hot = np.zeros((n_points, self.num_of_classes), dtype=np.float32)
            one_hot[np.arange(n_points), y_indices] = 1.0

            class_accum = K_csr @ one_hot
            class_accum += 0.1
            row_sums = class_accum.sum(axis=1, keepdims=True)
            np.maximum(row_sums, 1e-12, out=row_sums)
            norm_c = class_accum / row_sums

            if oracle_method == 'entropy':
                entropy = -np.sum(norm_c * np.log(norm_c + 1e-9), axis=1)
                clarity = 1.0 - entropy / np.log(self.num_of_classes)
            elif oracle_method == 'max':
                clarity = np.max(norm_c, axis=1)
            else:
                raise ValueError(f"Unknown oracle method: {oracle_method}")

            rel_measures = torch.from_numpy(clarity.astype(np.float32))
        else:
            rel_measures = prior_selection.compute_clarity_kp(K, self.train_labels_general, self.num_of_classes, oracle_method, batch_size=8192)

        priors = prior_selection.get_temp_priors(rel_measures, lb=self.alpha_lower, ub=self.alpha_upper) # (N,)
        return priors

    def init_C(self, lset, K, oracle_method):
        """
        Init the main matrix C with the priors. If lset != empty, Init C accordingly.
        NOTE: Attention! float16
        """
        if len(lset) > 0:
            print("Using a method which is not yet tested --- :(")
            self.load_C_from_lset(lset)
        self.priors = self.get_priors(K, oracle_method)  # This is (N,) tensor on CPU
        priors_tensor = torch.as_tensor(self.priors, device='cuda', dtype=torch.float16).unsqueeze(1)  # (N, 1)
        self.C_general = priors_tensor.repeat(1, self.num_of_classes).to(dtype=self.matrices_type)

        if len(lset) > 0:
            print("Using a method which is not yet tested --- :(")
            self.load_C_from_lset(lset)

    def load_C_from_lset(self, lset):
        """
        Load matrix C from lset as if lset was labeled in the algorithm process.

        Assumes self.K is already computed.
        Assumes K is on CPU to save GPU memory.
        Assumes C is on GPU.
        """
        # Since self.features (and K) are sorted as [lset, uset], the indices of lset are simply 0..len(lset)-1
        lset_indices = np.arange(len(lset))
        lset_labels = np.array(self.train_labels_general)[lset_indices]

        for label in self.unique_labels:
            is_label = (lset_labels == label)
            indices_to_slice = lset_indices[is_label]
            med_K = self.K[indices_to_slice].to('cuda')
            self.C[:, label] += torch.sum(med_K, axis=0)

            del med_K
        torch.cuda.empty_cache()


    def init_sampling_loop(self,lset, uset):
        torch.cuda.empty_cache()
        self.set_rel_features(lset, uset)
        self.activeSet = []
        if self.use_sparse:
            # now using scipy csr sparse matrix and convert it to torch csr sparse matrix
            K_csr_shuffled = self.K_general[self.relevant_indices, :][:, self.relevant_indices]
            crow_indices = torch.from_numpy(K_csr_shuffled.indptr).to(torch.int64)
            col_indices = torch.from_numpy(K_csr_shuffled.indices).to(torch.int64)
            values = torch.from_numpy(K_csr_shuffled.data).to(torch.float32)

            self.K = torch.sparse_csr_tensor(
                crow_indices=crow_indices,
                col_indices=col_indices,
                values=values,
                size=K_csr_shuffled.shape,
                dtype=values.dtype
            )
            del K_csr_shuffled, values, col_indices, crow_indices
        else:
            self.K =  self.K_general[self.relevant_indices, :][:, self.relevant_indices]
        self.C = self.C_general[self.relevant_indices].to('cuda')
        self.train_labels = self.train_labels_general[self.relevant_indices]
        self.labeled_points_mask = self.labeled_points_mask_general[self.relevant_indices]

    def set_rel_features(self, lset, uset):
        self.lSet = lset
        self.uSet = uset
        print(lset)
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        # self.relevant_indices = np.arange(self.lSet.size +self.uSet.size).astype(int)
        if isinstance(self.all_features, torch.Tensor):
            self.rel_features = self.all_features[self.relevant_indices]
        elif isinstance(self.all_features, np.ndarray):
            self.rel_features = torch.from_numpy(self.all_features[self.relevant_indices])

    def select_samples(self, lset, uset):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """

        ## update general K
        if len(lset)>0 and len(lset) % 10000 == 0 and self.use_sparse and self.update_K_matrix:
            self.update_K_general_matrix(lset)

        self.init_sampling_loop(lset, uset)

        # lset = np.array([12763, 48804, 36863, 40453, 46313, 44436, 15302, 48657, 34025, 44459])
        #
        # for i, l in enumerate(lset):
        #     label_idx = np.where(self.relevant_indices == l)[0][0]
        #     chosen_label = self.train_labels[label_idx]
        #     self.C[:, chosen_label] += self.K[label_idx].squeeze()
        # invalid_mask = np.isin(uset, lset)
        # uset = uset[~invalid_mask]
        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        for i in range(self.budgetSize):
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)

            class_gains_weights = self.get_class_gains_weighting_vector()

            if self.use_sparse:
                point_total_contribution = batched_diffs_efficient_weighted_sparse(self.K, self.C, cont_method=self.cont_method, weight_method=self.weight_method, class_gains_weights=class_gains_weights)
            else:
                if len(self.K.shape) == 2:
                    self.K.unsqueeze_(2)
                point_total_contribution = batched_diffs_efficient_weighted(self.K, self.C,
                                                      diff_method="efficient_full_weighted_max",cont_method=self.cont_method, weight_method=self.weight_method)

            point_total_contribution[curr_l_set] = -np.inf

            sampled_point = np.argsort(point_total_contribution.cpu().numpy(), kind='stable')[::-1][0].item()
            chosen_label = self.train_labels[sampled_point].item()

            self.chosen_labels_num[chosen_label] += 1

            K_row_dense = self.K[sampled_point].to_dense().to('cuda').squeeze()


            self.labeled_points_mask[sampled_point] = True
            self.C[sampled_point, :] = torch.zeros(self.num_of_classes).to('cuda')
            self.C[sampled_point, chosen_label] = 1.0

            self.C[~self.labeled_points_mask, chosen_label] += K_row_dense[~self.labeled_points_mask]
            self.cum_labels_info[chosen_label] += K_row_dense.sum()
            self.total_connections_chosen += torch.sum(K_row_dense > 0).item()

            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)
            del K_row_dense

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]

        self.C_general[self.relevant_indices] = self.C
        self.labeled_points_mask_general[self.relevant_indices] = self.labeled_points_mask
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')

        del self.K
        del self.C

        return activeSet, remainSet

    def get_class_gains_weighting_vector(self):
        if self.class_weighting_method == 'none':
            class_gains_weights = None
        elif self.class_weighting_method == 'linear':
            norm_class_gains_vector = self.cum_labels_info / (torch.sum(self.cum_labels_info) + 1e-8)
            class_gains_weights = 1 / (norm_class_gains_vector + 1)
        elif self.class_weighting_method == 'square':
            square_class_gains_vector = torch.sqrt(self.cum_labels_info + 1) - 1
            norm_class_gains_vector = square_class_gains_vector / (torch.sum(square_class_gains_vector) + 1e-8)
            class_gains_weights = 1 / (norm_class_gains_vector + 1)
        elif self.class_weighting_method == 'exp':
            square_class_gains_vector = torch.exp(self.cum_labels_info + 1) - 1
            norm_class_gains_vector = square_class_gains_vector / (torch.sum(square_class_gains_vector) + 1e-8)
            class_gains_weights = 1 / (norm_class_gains_vector + 1)
        else:
            raise ValueError(f"Unknown class weighting method: {self.class_weighting_method}")
        return class_gains_weights

    def update_K_general_matrix(self, lset):
        torch.cuda.empty_cache()
        sorted_values = np.load(
            "/cs/labs/daphna/itai.david/py_repos/TypiClust/results/K_sorted_values/cifar100/euclidean_dists_sorted.npy",
            mmap_mode='r')
        new_threshold_euclidean_dist = sorted_values[::-1][self.initial_sparse_index - self.total_connections_chosen]
        if self.kernel_type == 'tophat':
            old_threshold = self.delta
            new_threshold = float(new_threshold_euclidean_dist)
            self.delta = new_threshold
            rebuild_threshold = self.delta
        elif self.kernel_type == 'rbf':
            old_threshold = self.K_sparsity_threshold
            new_threshold = torch.exp(-1.0 * (torch.tensor(new_threshold_euclidean_dist) / self.sigma) ** 2)
            new_threshold = float(new_threshold.item()) if isinstance(new_threshold, torch.Tensor) else float(
                new_threshold)
            self.K_sparsity_threshold = new_threshold
            rebuild_threshold = self.K_sparsity_threshold
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
        zero_indices = np.asarray(lset, dtype=np.int64)
        self.K_general = self.build_K_general_matrix(
            self.all_features,
            threshold=rebuild_threshold,
            zero_indices=zero_indices,
            prev_threshold=old_threshold
        )


# @torch.compile(backend="inductor")
def batched_diffs_efficient_weighted(K: torch.Tensor, C: torch.Tensor, chunk_size: int = 1024, diff_method: str = "abs_diff", cont_method: str = "positive", weight_method: str = "weighted"):
    D, N, _ = K.shape
    result = torch.empty((D, )).to(device=C.device)
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)
    norm_C = (C / sum_C)
    old_max = (max_C / sum_C)
    C_diff = (C - max_C).unsqueeze(0)
    num_iterations = int(N)
    cont_method = cont_method
    max_C.unsqueeze_(0)
    for i in range(0, num_iterations, int(chunk_size)):
        end = min(i + chunk_size, D)
        K_batched = K[i:end]
        K_batched = K_batched.to('cuda')
        weights_batched = norm_C[i:end]


        future_sum = K_batched + sum_C
        state_add = max_C + K_batched

        new_state_vec = torch.maximum(-K_batched, C_diff)

        new_state_vec.add_(state_add)
        new_state_vec.div_(future_sum)
        new_state_vec.sub_(old_max)

        if cont_method == "positive": ### regular method
            new_state_vec.clamp_(min=0)
        elif cont_method == 'abs': ### take all contribution
            torch.abs(new_state_vec, out=new_state_vec)
        elif cont_method == "reg_sum_postive": ## take average contribution (not weighted by the prior)
            new_state_vec.clamp_(min=0)
            result[i:end] = torch.sum(new_state_vec, dim=(1, 2))

            del new_state_vec
            del K_batched
            del weights_batched

            continue
        elif cont_method == "reg_sum_abs":
            torch.abs(new_state_vec, out=new_state_vec)
            result[i:end] = torch.sum(new_state_vec, dim=(1, 2))

            del new_state_vec
            del K_batched
            del weights_batched

            continue

        # result[i:end] = torch.bmm(new_state_vec, weights_batched.unsqueeze_(2)).sum(dim=1).squeeze(1)
        if weight_method == "weighted":
            result[i:end] = torch.einsum('ijk,ik->i', new_state_vec, weights_batched)
        elif weight_method == "equal":
            result[i:end] = torch.sum(new_state_vec, dim=(1, 2))
        else:
            raise ValueError(f"Unknown weight_method: {weight_method}. Must be 'weighted' or 'equal'.")
        
        del new_state_vec
        del K_batched
        del weights_batched
        # result[i:end] = torch.einsum('ijk,ik->i',new_state_vec, weights_batched)
        # res = new_state_vec * weights_batched
    return result


# @torch.compile(backend="inductor")
def batched_diffs_efficient_weighted_sparse(K_csr: torch.Tensor, C: torch.Tensor, chunk_size: int = 2048, cont_method: str = "positive", weight_method: str = "weighted", class_gains_weights: torch.Tensor = None):
    D, N = K_csr.shape
    dev = C.device
    crow = K_csr.crow_indices().to(dev)  # shape (D+1,)
    ccol = K_csr.col_indices().to(dev)  # shape (nnz,)
    cvals = K_csr.values().to(dev)  # shape (nnz,)
    D = crow.numel() - 1
    classes = C.shape[1]
    uniform_default_val = 1.0 / classes



    result = torch.empty((D, )).to(device=C.device)
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)
    has_mass = (sum_C != 0)
    safe_sum = torch.where(has_mass, sum_C, torch.ones_like(sum_C))
    norm_C = torch.where(has_mass, C / safe_sum, torch.full_like(C, uniform_default_val))
    old_max = torch.where(has_mass, max_C / safe_sum, torch.zeros_like(max_C))

    # norm_C = (C / sum_C)
    # old_max = (max_C / sum_C)

    C_diff = (C - max_C)
    max_C = max_C.squeeze()
    sum_C = sum_C.squeeze()
    C_diff = C_diff.squeeze()
    old_max = old_max.squeeze()
    num_iterations = int(N)
    for row_start in range(0, D, chunk_size):
        row_end = min(row_start + chunk_size, D)
        b = row_end - row_start

        # CSR pointers for the chunk
        starts = crow[row_start:row_end]  # shape (b,)
        ends = crow[row_start + 1: row_end + 1]  # shape (b,)
        lengths = (ends - starts).to(torch.long)  # (b,)

        total_nnz = int(lengths.sum().item())

        if total_nnz == 0:
            # nothing in this chunk, skip
            continue

            # global slice of indices/values for this chunk
        slice_start = int(starts[0].item())
        slice_end = int(ends[-1].item())

        cols_all = ccol[slice_start:slice_end]  # (total_nnz,)
        vals_all = cvals[slice_start:slice_end]  # (total_nnz,)

        # row index for each nnz entry within the chunk: 0..b-1 repeated by lengths
        row_indices = torch.repeat_interleave(torch.arange(b, device=dev, dtype=torch.long),
                                              lengths)  # (total_nnz,)

        # Map chunk row-local indices -> global row indices (if needed)
        global_rows = torch.arange(row_start, row_end, device=dev, dtype=torch.long)  # (b,)

        kvals = vals_all  # (total_nnz, 1)
        sumC_cols = sum_C[cols_all]  # (total_nnz, 1)
        maxC_cols = max_C[cols_all]  # (total_nnz, 1)
        old_max_cols = old_max[cols_all]  # (total_nnz, 1)
        Cdiff_cols = C_diff[cols_all]

        negk = -kvals  # (total_nnz,1)
        # maximum between negk and Cdiff_cols: broadcast negk on classes dimension
        # torch.maximum requires same shape; expand negk to (total_nnz, classes)
        negk_expand = negk.expand(classes, -1).T  # (total_nnz, classes)
        new_state = torch.maximum(negk_expand, Cdiff_cols)  # (total_nnz, classes)

        del negk_expand, Cdiff_cols

        state_add = maxC_cols + kvals  # (total_nnz,1)
        new_state = new_state + state_add.expand(classes, -1).T  # add per-row scalar across classes

        future_sum = (kvals + sumC_cols)  # (total_nnz,1)

        valid_denom = (future_sum != 0)
        safe_future_sum = torch.where(valid_denom, future_sum, torch.ones_like(future_sum))
        safe_denom_expanded = safe_future_sum.expand(classes, -1).T
        mask_expanded = valid_denom.expand(classes, -1).T
        # divide
        new_state = new_state / safe_denom_expanded

        del safe_denom_expanded

        new_state = torch.where(mask_expanded, new_state, torch.zeros_like(new_state))
        # new_state = new_state / future_sum.expand(classes, -1).T
        # subtract old_max (per column)
        new_state = new_state - old_max_cols.expand(classes, -1).T

        # Now apply continuation method
        if cont_method == "positive":
            new_state.clamp_(min=0.0)

        if class_gains_weights is not None:
            new_state = new_state * class_gains_weights

        weights_chunk = norm_C[global_rows]  # (b, classes)
        # Now map per-nnz: weights_for_nnz = weights_chunk[row_indices]
        weights_for_nnz = weights_chunk[row_indices]  # (total_nnz, classes)

        # Multiply elementwise and sum over classes -> per-nnz scalar
        if weight_method == "weighted":
            per_nnz_weighted = (new_state * weights_for_nnz).sum(dim=1)  # (total_nnz,)
        elif weight_method == "equal":
            per_nnz_weighted = new_state.sum(dim=1)  # (total_nnz,) - equal contribution
        else:
            raise ValueError(f"Unknown weight_method: {weight_method}. Must be 'weighted' or 'equal'.")

        # Aggregate per row via scatter_add
        chunk_result = torch.zeros((b,), device=dev, dtype=C.dtype)
        chunk_result.scatter_add_(0, row_indices, per_nnz_weighted)



        # result[i:end] = torch.bmm(new_state_vec, weights_batched.unsqueeze_(2)).sum(dim=1).squeeze(1)
        result[row_start:row_end] = chunk_result
        torch.cuda.empty_cache()

    return result

