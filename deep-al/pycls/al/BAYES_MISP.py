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
from .kernel_utils import build_sparse_kernel_matrix, RBFKernel, TopHatKernel, compute_norm
###MISP = maximum importance sampling points
torch.cuda.empty_cache()


class BAYES_MISP:
    def __init__(self, cfg, budgetSize, train_labels, lset, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, train=True, )
        self.diff_method = self.cfg.DIFF_METHOD if 'DIFF_METHOD' in self.cfg else 'abs_diff'
        self.alpha_init_mode = self.cfg.ALPHA_INIT_MODE if 'ALPHA_INIT_MODE' in self.cfg else 'constant'
        self.alpha_vector_path = self.cfg.ALPHA_VECTOR_PATH if 'ALPHA_VECTOR_PATH' in self.cfg else ''
        self.alpha = self.cfg.ALPHA if self.diff_method not in ['prob_cover', 'max_herding'] else 0
        self.debug = self.cfg.DEBUG
        self.use_sparse = self.cfg.SPARSE_K
        self.matrices_type = torch.float32 if self.use_sparse else torch.float16
        self.cont_method = self.cfg.CONT_METHOD if 'CONT_METHOD' in self.cfg else 'positive'
        self.distribution_cont_weight_method = self.cfg.DISTRIBUTION_CONT_WEIGHT_METHOD if 'DISTRIBUTION_CONT_WEIGHT_METHOD' in self.cfg else 'weighted'
        self.c_normalization = self.cfg.C_NORMALIZATION if 'C_NORMALIZATION' in self.cfg else 'sum'
        self.budgetSize = budgetSize
        self.K_sparsity_threshold = self.cfg.K_SPARSITY_THRESHOLD
        self.sigma = cfg.ACTIVE_LEARNING.INITIAL_SIGMA if 'INITIAL_SIGMA' in cfg.ACTIVE_LEARNING else 1.0
        self.update_K_matrix = self.cfg.UPDATE_K_MATRIX if 'UPDATE_K_MATRIX' in self.cfg else False
        self.class_weighting_method = self.cfg.CLASS_WEIGHTING_METHOD if 'CLASS_WEIGHTING_METHOD' in self.cfg else 'none'
        self.switch_alpha_low_to_high = cfg.SWITCH_ALPHA_LOW_TO_HIGH
        self.switch_alpha_high_to_low = cfg.SWITCH_ALPHA_HIGH_TO_LOW
        self.switch_alpha_alltime = cfg.SWITCH_ALPHA_ALLTIME

        if self.switch_alpha_low_to_high or self.switch_alpha_alltime:
            self.alpha = 0.01
        if self.switch_alpha_high_to_low:
            self.alpha = 50

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

        self.alpha_decay_gamma = self.cfg.ALPHA_DECAY_GAMMA if 'ALPHA_DECAY_GAMMA' in self.cfg else 0.0
        self.alpha_base = None  # set after alpha is finalized (after alpha_init_mode logic)

        self.calc_method = self.cfg.CALC_METHOD if 'CALC_METHOD' in self.cfg else 'max'
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

        if self.alpha_init_mode == 'from_sparsity':
            sparsity_val = self.delta if self.kernel_type == 'tophat' else self.K_sparsity_threshold
            self.alpha = float(sparsity_val) / self.num_of_classes
            print(f"[alpha_init_mode=from_sparsity] alpha = {sparsity_val} / {self.num_of_classes} = {self.alpha}")
        elif self.alpha_init_mode == 'from_vector':
            k_sorted_vector = np.load(self.alpha_vector_path)
            self.alpha_per_point = k_sorted_vector / self.num_of_classes
            self.alpha = float(np.mean(self.alpha_per_point))
            print(f"[alpha_init_mode=from_vector] loaded vector from {self.alpha_vector_path}, "
                  f"shape={k_sorted_vector.shape}, alpha_per_point range=[{self.alpha_per_point.min():.4f}, {self.alpha_per_point.max():.4f}], mean={self.alpha:.4f}")

        self.alpha_base = self.alpha
        self._init_alpha_decay()

        if cfg.LOCAL_ALPHA:
            self.init_C(lset, self.K_general, cfg.LOCAL_ALPHA_ORACLE_METHOD)
        elif self.alpha_init_mode == 'from_vector':
            alpha_tensor = torch.from_numpy(self.alpha_per_point.astype(np.float32)).to(device='cuda', dtype=self.matrices_type)
            self.C_general = alpha_tensor.unsqueeze(1).expand(-1, unique_labels.size).contiguous()
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

    def _init_alpha_decay(self):
        """Initialize per-class decay factors beta_m and effective alpha vector."""
        if self.alpha_decay_gamma > 0:
            self.beta_per_class = torch.ones(self.num_of_classes, device='cuda')
            self.effective_alpha_per_class = torch.full((self.num_of_classes,), self.alpha_base, device='cuda')
            print(f"[alpha_decay] gamma={self.alpha_decay_gamma}, alpha_base={self.alpha_base}")
        else:
            self.beta_per_class = None
            self.effective_alpha_per_class = None

    def _compute_beta(self, class_idx):
        """Compute beta_m = 1 / (1 + gamma * I_m) for class m."""
        return 1.0 / (1.0 + self.alpha_decay_gamma * self.cum_labels_info[class_idx])

    def _update_alpha_decay(self, class_idx, unlabeled_mask):
        """
        Recompute beta and effective alpha for a class after its I_m changed.
        Adjusts C matrix for all unlabeled points to reflect the new effective alpha.

        Returns the delta (new_effective - old_effective) that was applied.
        """
        if self.beta_per_class is None:
            return 0.0

        old_effective = self.effective_alpha_per_class[class_idx].clone()
        new_beta = self._compute_beta(class_idx)
        new_effective = self.alpha_base * new_beta

        self.beta_per_class[class_idx] = new_beta
        self.effective_alpha_per_class[class_idx] = new_effective

        delta = new_effective - old_effective
        if delta != 0:
            self.C[unlabeled_mask, class_idx] += delta

        return delta.item()

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


    def _sync_alpha_decay_to_C_general(self):
        """
        Synchronize C_general with current effective alpha per class.
        Called at the start of each selection round to account for any
        cum_labels_info changes that happened between rounds (e.g. from
        K matrix rebuilding).
        """
        if self.beta_per_class is None:
            return

        unlabeled = ~self.labeled_points_mask_general
        for m in range(self.num_of_classes):
            new_beta = self._compute_beta(m)
            new_effective = self.alpha_base * new_beta
            old_effective = self.effective_alpha_per_class[m]
            delta = new_effective - old_effective
            if delta != 0:
                self.C_general[unlabeled, m] += delta
            self.beta_per_class[m] = new_beta
            self.effective_alpha_per_class[m] = new_effective

    def init_sampling_loop(self,lset, uset):
        torch.cuda.empty_cache()
        # self._sync_alpha_decay_to_C_general()
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



        is_iteration_5 = len(lset) == self.C_general.shape[1] * 5
        if self.switch_alpha_low_to_high and is_iteration_5:
            self.C_general[uset] += (50 - self.alpha)
            self.alpha = 50
        if self.switch_alpha_high_to_low and is_iteration_5:
            self.C_general[uset] += (0.01 - self.alpha)
            self.alpha = 0.01
        if self.switch_alpha_alltime and len(lset) > 0:
            if self.alpha == 50:
                self.alpha = 0.01
                self.C_general[uset] -= (50 - self.alpha)
            else: #self.alpha == 0.01
                self.alpha = 50
                self.C_general[uset] -= (0.01 - self.alpha)

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

        # Apply NN-C matrix fusion if enabled
        # self.C remains the clean kernel-based matrix
        # self.C_fused (if created) is used for selection decisions
        self.C_fused = None
        if hasattr(self, 'nn_fusion_enabled') and self.nn_fusion_enabled:
            self._apply_nn_fusion(uset)

        # Determine which C matrix to use for selection
        # Use C_fused if available, otherwise use clean C
        use_fused = self.C_fused is not None

        # lset = np.array([12763, 48804, 36863, 40453, 46313, 44436, 15302, 48657, 34025, 44459])
        #
        # for i, l in enumerate(lset):
        #     label_idx = np.where(self.relevant_indices == l)[0][0]
        #     chosen_label = self.train_labels[label_idx]
        #     self.C[:, chosen_label] += self.K[label_idx].squeeze()
        # invalid_mask = np.isin(uset, lset)
        # uset = uset[~invalid_mask]
        print(f'Start selecting {self.budgetSize} samples.')
        if use_fused:
            print("Using fused C matrix (kernel + NN) for selection decisions.")
        selected = []
        for i in range(self.budgetSize):
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)

            class_gains_weights = self.get_class_gains_weighting_vector()

            # Use C_fused for selection decisions if available
            C_for_selection = self.C_fused if use_fused else self.C
            if self.diff_method == 'max':
                C_sum = torch.sum(self.C, dim=1, keepdim=True)
                norm_C = self.C / C_sum
                max_vals, indices = torch.max(norm_C, dim=1)
                point_total_contribution = batched_diffs_sparse_max(self.K, max_vals, self.alpha, self.num_of_classes)
            elif self.use_sparse:
                point_total_contribution = batched_diffs_efficient_weighted_sparse(self.K, C_for_selection, cont_method=self.cont_method, weight_method=self.distribution_cont_weight_method, class_gains_weights=class_gains_weights, c_normalization=self.c_normalization, calc_method=self.calc_method)
            else:
                if len(self.K.shape) == 2:
                    self.K.unsqueeze_(2)
                point_total_contribution = batched_diffs_efficient_weighted(self.K, C_for_selection,
                                                      diff_method="efficient_full_weighted_max",cont_method=self.cont_method, weight_method=self.distribution_cont_weight_method, c_normalization=self.c_normalization)

            point_total_contribution[curr_l_set] = -np.inf

            sampled_point = np.argsort(point_total_contribution.cpu().numpy(), kind='stable')[::-1][0].item()
            chosen_label = self.train_labels[sampled_point].item()

            self.chosen_labels_num[chosen_label] += 1

            K_row_dense = self.K[sampled_point].to_dense().to('cuda').squeeze()

            # Update the clean C matrix (kernel-based only)
            self.labeled_points_mask[sampled_point] = True
            self.C[sampled_point, :] = torch.zeros(self.num_of_classes).to('cuda')
            self.C[sampled_point, chosen_label] = 1.0
            self.C[~self.labeled_points_mask, chosen_label] += K_row_dense[~self.labeled_points_mask]

            # Also update C_fused if it exists (to keep selection decisions consistent)
            if use_fused:
                self.C_fused[sampled_point, :] = torch.zeros(self.num_of_classes).to('cuda')
                self.C_fused[sampled_point, chosen_label] = 1.0
                self.C_fused[~self.labeled_points_mask, chosen_label] += K_row_dense[~self.labeled_points_mask]

            self.cum_labels_info[chosen_label] += K_row_dense.sum()
            self.total_connections_chosen += torch.sum(K_row_dense > 0).item()

            alpha_delta = self._update_alpha_decay(chosen_label, ~self.labeled_points_mask)
            if use_fused and alpha_delta != 0:
                self.C_fused[~self.labeled_points_mask, chosen_label] += alpha_delta

            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)
            del K_row_dense

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]

        # Save the clean C matrix (without NN fusion) to C_general for future rounds
        self.C_general[self.relevant_indices] = self.C
        self.labeled_points_mask_general[self.relevant_indices] = self.labeled_points_mask
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        self.activeSet = activeSet
        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')

        del self.K
        del self.C
        if self.C_fused is not None:
            del self.C_fused
            self.C_fused = None

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

    def set_nn_fusion_params(self, clf_model, data_obj, train_data, per_class_accuracy):
        """
        Set parameters for NN-C matrix fusion mode.
        
        Args:
            clf_model: Trained classifier model
            data_obj: Data object for creating data loaders
            train_data: Training dataset
            per_class_accuracy: Per-class accuracy vector of shape (num_classes,), values in [0, 1]
        """
        self.nn_fusion_model = clf_model
        self.nn_fusion_data_obj = data_obj
        self.nn_fusion_train_data = train_data
        self.nn_fusion_per_class_accuracy = torch.from_numpy(per_class_accuracy).to(
            device='cuda', dtype=self.matrices_type
        )
        self.nn_fusion_enabled = True
        print(f"NN fusion enabled with per-class accuracy: {per_class_accuracy}")

    @torch.no_grad()
    def _compute_nn_C_matrix(self, uset_indices):
        """
        Compute NN-based pseudo-C matrix from model softmax predictions on unlabeled set.
        
        Args:
            uset_indices: Indices of unlabeled samples in the original dataset
            
        Returns:
            torch.Tensor: NN-based C matrix of shape (len(uset_indices), num_classes)
        """
        if not hasattr(self, 'nn_fusion_enabled') or not self.nn_fusion_enabled:
            return None
        
        model = self.nn_fusion_model
        data_obj = self.nn_fusion_data_obj
        train_data = self.nn_fusion_train_data
        
        if model is None or data_obj is None or train_data is None:
            return None
        
        # Create data loader for unlabeled set
        uset_loader = data_obj.getSequentialDataLoader(
            indexes=uset_indices, 
            batch_size=self.cfg.TRAIN.BATCH_SIZE, 
            data=train_data
        )
        
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        
        all_softmax_preds = []
        for batch in uset_loader:
            if len(batch) == 3:
                inputs, _, _ = batch
            else:
                inputs, _ = batch
            
            inputs = inputs.cuda().type(torch.cuda.FloatTensor)
            logits = model(inputs)
            softmax_preds = torch.softmax(logits, dim=1)
            all_softmax_preds.append(softmax_preds)
        
        # Concatenate all predictions: shape (len(uset_indices), num_classes)
        nn_C_matrix = torch.cat(all_softmax_preds, dim=0).to(dtype=self.matrices_type)
        
        return nn_C_matrix

    def _scale_nn_matrix(self, nn_C_matrix, C_matrix_subset):
        """
        Scale NN predictions to match the magnitude of C matrix values.
        
        For each point i:
            info_gained[i] = sum(C[i, :]) - alpha * num_classes
            scaled_nn[i, :] = nn_C[i, :] * info_gained[i]
        
        Args:
            nn_C_matrix: NN-based C matrix of shape (N_unlabeled, num_classes)
            C_matrix_subset: Kernel-based C matrix subset for unlabeled points, shape (N_unlabeled, num_classes)
            
        Returns:
            torch.Tensor: Scaled NN matrix of same shape
        """
        # Compute information gained per point (sum of C values minus initial alpha contribution)
        sum_C = torch.sum(C_matrix_subset, dim=1, keepdim=True)  # (N, 1)
        initial_alpha_sum = self.alpha * self.num_of_classes
        info_gained = sum_C - initial_alpha_sum  # (N, 1)
        
        # Clamp to non-negative values (in case some points have less info than initial)
        info_gained = torch.clamp(info_gained, min=0)
        
        # Scale NN predictions by the information gained
        scaled_nn_matrix = nn_C_matrix * info_gained  # (N, num_classes)
        
        return scaled_nn_matrix

    def _fuse_C_matrices(self, C_matrix, scaled_nn_matrix, unlabeled_mask):
        """
        Fuse kernel-based C matrix with scaled NN matrix using per-class accuracy as weights.
        
        For each class c:
            C_fused[:, c] = C[:, c] + accuracy[c] * scaled_nn_matrix[:, c]
        
        Args:
            C_matrix: Kernel-based C matrix of shape (N_total, num_classes)
            scaled_nn_matrix: Scaled NN matrix for unlabeled points, shape (N_unlabeled, num_classes)
            unlabeled_mask: Boolean mask indicating unlabeled points in C_matrix
            
        Returns:
            torch.Tensor: Fused C matrix of shape (N_total, num_classes)
        """
        if not hasattr(self, 'nn_fusion_per_class_accuracy'):
            return C_matrix
        
        accuracy_weights = self.nn_fusion_per_class_accuracy  # (num_classes,)
        
        # Create a copy to avoid modifying the original
        C_fused = C_matrix.clone()
        
        # Apply weighted fusion only to unlabeled points
        # For each class, add accuracy[c] * scaled_nn[c] to C[c]
        weighted_nn = scaled_nn_matrix * accuracy_weights.unsqueeze(0)  # (N_unlabeled, num_classes)
        C_fused[unlabeled_mask] = C_fused[unlabeled_mask] + weighted_nn
        
        return C_fused

    def _apply_nn_fusion(self, uset):
        """
        Apply NN-C matrix fusion to create a fused C matrix for selection.
        
        This method orchestrates the full fusion pipeline:
        1. Compute NN-based C matrix from model predictions
        2. Scale NN predictions to match C matrix magnitude
        3. Fuse the matrices using per-class accuracy weights
        
        The original self.C (kernel-based) is preserved, and a new self.C_fused
        is created for use in sample selection. This allows future fusions to
        always start from the clean kernel-based C matrix.
        
        Args:
            uset: Unlabeled set indices
        """
        print("======== APPLYING NN-C MATRIX FUSION ========")
        
        # Compute NN-based C matrix for unlabeled set
        nn_C_matrix = self._compute_nn_C_matrix(uset)
        if nn_C_matrix is None:
            print("NN fusion skipped: could not compute NN C matrix")
            self.C_fused = None  # Use original C matrix
            return
        
        # Get the unlabeled portion of the current (clean) C matrix
        # self.C is indexed by relevant_indices which is [lset, uset]
        # The unlabeled mask corresponds to indices after len(lSet)
        unlabeled_local_mask = ~self.labeled_points_mask
        C_unlabeled = self.C[unlabeled_local_mask]
        
        # Scale NN predictions to match C matrix magnitude
        scaled_nn_matrix = self._scale_nn_matrix(nn_C_matrix, C_unlabeled)
        
        # Create fused matrix (self.C remains the clean kernel-based matrix)
        self.C_fused = self._fuse_C_matrices(self.C, scaled_nn_matrix, unlabeled_local_mask)
        
        print(f"NN fusion applied. Scaled NN matrix stats: min={scaled_nn_matrix.min():.4f}, "
              f"max={scaled_nn_matrix.max():.4f}, mean={scaled_nn_matrix.mean():.4f}")
        print(f"C_fused created. Original C preserved for future fusions.")
        
        # Reset fusion state after use (to avoid reusing stale data)
        self.nn_fusion_enabled = False


# @torch.compile(backend="inductor")
def batched_diffs_efficient_weighted(K: torch.Tensor, C: torch.Tensor, chunk_size: int = 1024, diff_method: str = "abs_diff", cont_method: str = "positive", weight_method: str = "weighted", c_normalization: str = "sum"):
    D, N, _ = K.shape
    result = torch.empty((D, )).to(device=C.device)
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)
    if c_normalization == 'softmax':
        norm_C = torch.softmax(C, dim=1)
    else:
        norm_C = (C / sum_C)
    old_max = norm_C.max(dim=1, keepdim=True).values
    C_diff = (C - max_C).unsqueeze(0)
    num_iterations = int(N)
    cont_method = cont_method
    max_C.unsqueeze_(0)
    use_softmax_norm = (c_normalization == 'softmax')
    if use_softmax_norm:
        exp_C = torch.exp(C - C.max(dim=1, keepdim=True).values)  # numerically stable exp (N, classes)
        Z_old = exp_C.sum(dim=1, keepdim=True)  # (N, 1)
        exp_C_max = exp_C.max(dim=1, keepdim=True).values  # (N, 1) — exp of the max class
    for i in range(0, num_iterations, int(chunk_size)):
        end = min(i + chunk_size, D)
        K_batched = K[i:end]
        K_batched = K_batched.to('cuda')
        weights_batched = norm_C[i:end]

        if use_softmax_norm:
            # For each class c, adding K[i,j] to C[j,c]:
            # Z_new[j,c] = Z_old[j] + exp_C[j,c] * (exp(K[i,j]) - 1)
            # new_max_softmax[j,c] = max(exp_C_max[j], exp_C[j,c]*exp(K[i,j])) / Z_new[j,c]
            exp_K = torch.exp(K_batched)  # (batch, N, 1)
            # exp_C[j,c] * (exp(K) - 1) for each class c
            Z_new = Z_old + exp_C.unsqueeze(0) * (exp_K - 1)  # (batch, N, classes)
            # numerator: max(exp_C_max, exp_C[c]*exp(K)) per class c
            boosted_exp = exp_C.unsqueeze(0) * exp_K  # (batch, N, classes) — exp(C[c]+K)
            new_max_num = torch.maximum(exp_C_max.unsqueeze(0).expand_as(boosted_exp), boosted_exp)
            new_state_vec = (new_max_num / Z_new) - old_max
            del exp_K, Z_new, boosted_exp, new_max_num
        else:
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
def batched_diffs_efficient_weighted_sparse(K_csr: torch.Tensor, C: torch.Tensor, chunk_size: int = 2048, cont_method: str = "positive", weight_method: str = "weighted", class_gains_weights: torch.Tensor = None, c_normalization: str = "sum", calc_method: str = "max"):
    D, N = K_csr.shape
    dev = C.device
    crow = K_csr.crow_indices().to(dev)  # shape (D+1,)
    ccol = K_csr.col_indices().to(dev)  # shape (nnz,)
    cvals = K_csr.values().to(dev)  # shape (nnz,)
    D = crow.numel() - 1
    classes = C.shape[1]
    uniform_default_val = 1.0 / classes

    use_softmax_norm = (c_normalization == 'softmax')

    result = torch.empty((D, )).to(device=C.device)
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)

    if use_softmax_norm:
        norm_C = torch.softmax(C, dim=1)
        old_max = norm_C.max(dim=1, keepdim=True).values
        C_stable = C - C.max(dim=1, keepdim=True).values  # for numerical stability
        exp_C = torch.exp(C_stable)  # (N, classes)
        Z_old = exp_C.sum(dim=1, keepdim=True)  # (N, 1)
        exp_C_max = exp_C.max(dim=1, keepdim=True).values  # (N, 1)
        exp_C = exp_C.squeeze()
        Z_old = Z_old.squeeze()
        exp_C_max = exp_C_max.squeeze()
    else:
        has_mass = (sum_C != 0)
        safe_sum = torch.where(has_mass, sum_C, torch.ones_like(sum_C))
        norm_C = torch.where(has_mass, C / safe_sum, torch.full_like(C, uniform_default_val))
        old_max = torch.where(has_mass, max_C / safe_sum, torch.zeros_like(max_C))

    C_diff = (C - max_C)
    max_C = max_C.squeeze()
    sum_C = sum_C.squeeze()
    C_diff = C_diff.squeeze()
    old_max = old_max.squeeze()


    if calc_method == "entropy":
        p_log2_p = norm_C * torch.log2(norm_C)
        h_before_global = -torch.sum(p_log2_p, dim=1) # (N,)
        h_before_global = torch.where(has_mass.squeeze(), h_before_global, torch.zeros_like(h_before_global))

    elif calc_method == "margin":
        top2 = torch.topk(C, 2, dim=1)
        max1_val_global = top2.values[:, 0]
        max2_val_global = top2.values[:, 1]
        max1_idx_global = top2.indices[:, 0]
        max2_idx_global = top2.indices[:, 1]

        m_before_global = (max1_val_global - max2_val_global) / safe_sum.squeeze()
        m_before_global = torch.where(has_mass.squeeze(), m_before_global, torch.zeros_like(m_before_global))

    elif calc_method == "margin_not_normalized":
        top2 = torch.topk(C, 2, dim=1)
        max1_val_global = top2.values[:, 0]
        max2_val_global = top2.values[:, 1]
        max1_idx_global = top2.indices[:, 0]
        max2_idx_global = top2.indices[:, 1]

        m_before_global = max1_val_global - max2_val_global

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
        kvals = vals_all  # (total_nnz,)

        if calc_method == "entropy":
            c_vecs = C[cols_all] 
            sum_C_cols = sum_C[cols_all].unsqueeze(1) 
            h_before_cols = h_before_global[cols_all].unsqueeze(1) 
            k_vals_2d = kvals.unsqueeze(1) 
            
            future_base_vec_sum = sum_C_cols + k_vals_2d
            safe_future_sum = torch.where(future_base_vec_sum != 0, future_base_vec_sum, torch.ones_like(future_base_vec_sum))
            
            future_norm_base_vec = c_vecs / safe_future_sum
            

            future_base_log_vec = torch.sum(future_norm_base_vec * torch.log2(future_norm_base_vec), dim=1, keepdim=True)
            
            vec_plus_k = c_vecs + k_vals_2d
            first_ele = k_vals_2d * torch.log2(safe_future_sum / (vec_plus_k))
            second_ele = c_vecs * torch.log2((c_vecs) / (vec_plus_k))
            
            delta_vec = first_ele + second_ele
            final_vec = -(future_base_log_vec - (delta_vec / safe_future_sum))
            
            new_state = final_vec - h_before_cols
            
            del c_vecs, future_norm_base_vec, vec_plus_k, first_ele, second_ele, delta_vec
        elif calc_method == "max":

            old_max_cols = old_max[cols_all]  # (total_nnz,) or (total_nnz, classes)

            if use_softmax_norm:
                # For each class c, adding kvals to C[col, c]:
                # Z_new = Z_old + exp_C[col, c] * (exp(kvals) - 1)
                # new softmax max = max(exp_C_max[col], exp_C[col,c]*exp(kvals)) / Z_new
                exp_K = torch.exp(kvals)  # (total_nnz,)
                exp_C_cols = exp_C[cols_all]  # (total_nnz, classes)
                Z_old_cols = Z_old[cols_all]  # (total_nnz,)
                exp_C_max_cols = exp_C_max[cols_all]  # (total_nnz,)

                # Z_new per class: Z_old + exp_C[c] * (exp(K) - 1)
                exp_K_minus1 = (exp_K - 1).unsqueeze(1).expand_as(exp_C_cols)  # (total_nnz, classes)
                Z_new = Z_old_cols.unsqueeze(1) + exp_C_cols * exp_K_minus1  # (total_nnz, classes)

                # boosted exp for the class we're adding to
                boosted = exp_C_cols * exp_K.unsqueeze(1)  # (total_nnz, classes)
                new_max_num = torch.maximum(exp_C_max_cols.unsqueeze(1).expand_as(boosted), boosted)

                safe_Z = torch.where(Z_new != 0, Z_new, torch.ones_like(Z_new))
                new_state = (new_max_num / safe_Z) - old_max_cols.unsqueeze(1)

                del exp_K, exp_C_cols, Z_old_cols, exp_C_max_cols, exp_K_minus1, Z_new, boosted, new_max_num, safe_Z
            else:
                sumC_cols = sum_C[cols_all]  # (total_nnz,)
                maxC_cols = max_C[cols_all]  # (total_nnz,)
                Cdiff_cols = C_diff[cols_all]

                negk = -kvals  # (total_nnz,)
                negk_expand = negk.expand(classes, -1).T  # (total_nnz, classes)
                new_state = torch.maximum(negk_expand, Cdiff_cols)  # (total_nnz, classes)

                del negk_expand, Cdiff_cols

                state_add = maxC_cols + kvals  # (total_nnz,)
                new_state = new_state + state_add.expand(classes, -1).T

                future_sum = (kvals + sumC_cols)  # (total_nnz,)

                valid_denom = (future_sum != 0)
                safe_future_sum = torch.where(valid_denom, future_sum, torch.ones_like(future_sum))
                safe_denom_expanded = safe_future_sum.expand(classes, -1).T
                mask_expanded = valid_denom.expand(classes, -1).T
                new_state = new_state / safe_denom_expanded

                del safe_denom_expanded

                new_state = torch.where(mask_expanded, new_state, torch.zeros_like(new_state))
                new_state = new_state - old_max_cols.expand(classes, -1).T
        elif calc_method in ["margin", "margin_not_normalized"]:
            max1_val = max1_val_global[cols_all].unsqueeze(1)  # (total_nnz, 1)
            max2_val = max2_val_global[cols_all].unsqueeze(1)
            max1_idx = max1_idx_global[cols_all].unsqueeze(1)
            max2_idx = max2_idx_global[cols_all].unsqueeze(1)
            m_before_cols = m_before_global[cols_all].unsqueeze(1)

            c_vecs = C[cols_all]
            sum_C_cols = sum_C[cols_all].unsqueeze(1)
            k_vals_2d = kvals.unsqueeze(1)

            future_base_vec_sum = sum_C_cols + k_vals_2d
            safe_future_sum = torch.where(future_base_vec_sum != 0, future_base_vec_sum,
                                          torch.ones_like(future_base_vec_sum))

            new_val = c_vecs + k_vals_2d

            # Base logic for classes that are neither max1 nor max2
            new_margin = torch.maximum(max1_val, new_val) - torch.maximum(max2_val, torch.minimum(new_val, max1_val))

            # Optimizing the absolute margin difference for max1 and max2 targets
            fixed_max1_margin = torch.abs((max1_val + k_vals_2d) - max2_val)
            fixed_max2_margin = torch.abs((max2_val + k_vals_2d) - max1_val)

            # Create fast boolean masks to apply the fixes exactly where needed
            class_indices = torch.arange(classes, device=dev).unsqueeze(0)  # (1, classes)
            is_max1 = (class_indices == max1_idx)  # (total_nnz, classes)
            is_max2 = (class_indices == max2_idx)

            new_margin = torch.where(is_max1, fixed_max1_margin, new_margin)
            new_margin = torch.where(is_max2, fixed_max2_margin, new_margin)

            final_margin = new_margin / safe_future_sum if calc_method == "margin" else new_margin
            new_state = final_margin - m_before_cols

            del c_vecs, new_val, new_margin, is_max1, is_max2, fixed_max1_margin, fixed_max2_margin
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
        chunk_result.scatter_add_(0, row_indices, torch.nan_to_num(per_nnz_weighted, nan=0.0))



        # result[i:end] = torch.bmm(new_state_vec, weights_batched.unsqueeze_(2)).sum(dim=1).squeeze(1)
        result[row_start:row_end] = chunk_result
        torch.cuda.empty_cache()

    return result

def batched_diffs(K, C, alpha, number_of_classes, chunk_size=1024, diff_method="abs_diff"):
    D, N = K.shape
    K_gpu = K.to(device=C.device)
    result = torch.empty(D).to(device=C.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if diff_method == "abs_diff":
            result[start:end] = torch.sum(torch.maximum(K[start:end] - C, torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        elif diff_method == "max":
            result[start:end] = torch.sum(
                torch.maximum(((K_gpu[start:end] + alpha) / (K_gpu[start:end] + alpha * number_of_classes)) - C, torch.zeros_like(K_gpu[start:end]).to(device=C.device)), dim=1)
        elif diff_method == 'margin':
            result[start:end] = torch.sum(
                torch.maximum((K[start:end] / (K[start:end] + alpha * number_of_classes)) - C, torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
    return result


def batched_diffs_sparse_max(
        K_csr: torch.Tensor,
        C: torch.Tensor,
        alpha: float,
        number_of_classes: int,
        chunk_size: int = 1024
):
    """
    Sparse equivalent of batched_diffs strictly for diff_method="max".
    Computes row-wise sums of max( ((K + alpha)/(K + alpha * classes)) - C, 0 ).
    """
    D, N = K_csr.shape
    dev = C.device

    # Extract CSR components and move to target device
    crow = K_csr.crow_indices().to(dev)  # shape (D+1,)
    ccol = K_csr.col_indices().to(dev)  # shape (nnz,)
    cvals = K_csr.values().to(dev)  # shape (nnz,)

    # Based on the original dense broadcasting (K[start:end] - C), C must be 1D
    C_flat = C.squeeze()

    # --- 1. Compute global base values for implicit zeros ---
    # In a sparse matrix, absent values are 0.
    # For K_ij = 0, the transformed value evaluates to: 1.0 / number_of_classes
    base_val = 1.0 / number_of_classes
    base_diffs = torch.maximum(base_val - C_flat, torch.zeros_like(C_flat))

    # The sum if an entire row were mathematically evaluated as all zeros
    base_row_sum = torch.sum(base_diffs)

    result = torch.empty(D, device=dev, dtype=C.dtype)

    for row_start in range(0, D, chunk_size):
        row_end = min(row_start + chunk_size, D)
        b = row_end - row_start

        # CSR pointers for the current chunk
        starts = crow[row_start:row_end]
        ends = crow[row_start + 1: row_end + 1]
        lengths = (ends - starts).to(torch.long)

        total_nnz = int(lengths.sum().item())

        if total_nnz == 0:
            # If the chunk is entirely empty, every row is just the base sum
            result[row_start:row_end] = base_row_sum
            continue

        # Global slice bounds for indices/values in this chunk
        slice_start = int(starts[0].item())
        slice_end = int(ends[-1].item())

        cols_all = ccol[slice_start:slice_end]
        vals_all = cvals[slice_start:slice_end]

        # Map nnz entries back to their local chunk row index (0 to b-1)
        row_indices = torch.repeat_interleave(
            torch.arange(b, device=dev, dtype=torch.long), lengths
        )

        # Look up C values and baseline diffs for the non-zero columns
        C_cols = C_flat[cols_all]
        base_diffs_cols = base_diffs[cols_all]

        # --- 2. Calculate the transformed K values for non-zeros ---
        transformed_K = (vals_all + alpha) / (vals_all + alpha * number_of_classes)
        active_diffs = torch.maximum(transformed_K - C_cols, torch.zeros_like(transformed_K))

        # --- 3. Compute the delta ---
        # Subtract the base difference (already included in base_row_sum) and add the active difference
        deltas = active_diffs - base_diffs_cols

        # Aggregate deltas per row using scatter_add
        chunk_deltas = torch.zeros(b, device=dev, dtype=C.dtype)
        chunk_deltas.scatter_add_(0, row_indices, torch.nan_to_num(deltas, nan=0.0))

        # Final result is the implicit zero sum + the exact deltas of the active non-zeros
        result[row_start:row_end] = base_row_sum + chunk_deltas

        # Free memory (avoids spikes on large batches)
        del cols_all, vals_all, row_indices, C_cols, base_diffs_cols, transformed_K, active_diffs, deltas
        torch.cuda.empty_cache()

    return result