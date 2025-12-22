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
import matplotlib.pyplot as plt
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

    def prepare_K_matrix_for_sparsity(self, K_cpu, lset, new_delta):
        tensor_lset = torch.from_numpy(lset.astype(int))
        K_cpu[tensor_lset, :] = 0
        K_cpu[:, tensor_lset] = 0
        return K_cpu

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

    def prepare_K_matrix_for_sparsity(self, norm_matrix, lset, new_delta):
        k = (norm_matrix < new_delta).to(dtype=torch.float16)
        tensor_lset = torch.from_numpy(lset.astype(int))
        k[tensor_lset, :] = 0
        k[:, tensor_lset] = 0
        return k


class BAYES_MISP:
    def __init__(self, cfg, budgetSize, train_labels, lset, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, train=True)
        self.diff_method = self.cfg.DIFF_METHOD if 'DIFF_METHOD' in self.cfg else 'abs_diff'
        self.alpha = self.cfg.ALPHA if self.diff_method not in ['prob_cover', 'max_herding'] else 0
        self.debug = self.cfg.DEBUG
        self.use_sparse = self.cfg.SPARSE_K
        self.matrices_type = torch.float32 if self.use_sparse else torch.float16
        self.confidence_method = self.cfg.CONFIDENCE_METHOD if 'CONFIDENCE_METHOD' in self.cfg else 'max'
        self.cont_method = self.cfg.CONT_METHOD if 'CONT_METHOD' in self.cfg else 'positive'
        self.decrease_alpha = self.cfg.DECREASING_ALPHA if 'DECREASING_ALPHA' in self.cfg else False
        self.budgetSize = budgetSize
        self.K_sparsity_threshold = self.cfg.K_SPARSITY_THRESHOLD
        self.sigma = cfg.ACTIVE_LEARNING.INITIAL_SIGMA if 'INITIAL_SIGMA' in cfg.ACTIVE_LEARNING else 1.0
        self.update_K_matrix = self.cfg.UPDATE_K_MATRIX if 'UPDATE_K_MATRIX' in self.cfg else False

        self.use_K_top50_mask = self.cfg.USE_K_TOP50_MASK

        self.delta = delta
        self.soft_border_val = self.cfg.SOFT_BORDER_VAL if 'SOFT_BORDER_VAL' in self.cfg else 0.15

        self.train_labels_general = np.array(train_labels)
        unique_labels = np.unique(self.train_labels_general)
        self.num_of_classes = np.unique(self.train_labels_general).size

        self.alpha_lower = cfg.ALPHA_LOWER_BOUND
        self.alpha_upper = cfg.ALPHA_UPPER_BOUND


        self.chosen_labels_num = torch.zeros(self.num_of_classes).to('cuda')
        self.cum_labels_info = torch.zeros(self.num_of_classes).to('cuda')
        self.labeled_points_mask_general = torch.zeros(self.all_features.shape[0], dtype=torch.bool).to('cuda')
        all_features_tensor = torch.from_numpy(self.all_features)
        all_features_dists_cpu = compute_norm(all_features_tensor, all_features_tensor, 'cuda', batch_size=1024,
                                                  matrices_type=torch.float32).to('cpu')
        # self.K_general_dense_gpu = self.kernel_fn.compute_kernel(
        #         all_features_tensor, all_features_tensor, self.delta, matrices_type=self.matrices_type).to('cpu')
        # self.K_general_dense_cpu = self.K_general_dense_gpu.to('cpu')

        self.kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if  self.kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
            self.K_general_dense_cpu = self.kernel_fn.compute_kernel_from_norm(
                all_features_dists_cpu, self.delta, matrices_type=self.matrices_type)
            self.K_general_backed = all_features_dists_cpu
        else:
            self.kernel_fn = RBFKernel('cuda')
            self.K_general_dense_cpu = self.kernel_fn.compute_kernel_from_norm(
                    all_features_dists_cpu, self.sigma , matrices_type=self.matrices_type)

            self.K_general_backed = self.K_general_dense_cpu
        self.total_connections_chosen = 0
        del all_features_tensor, all_features_dists_cpu

        # self.C_general = torch.full((self.all_features.shape[0], unique_labels.size), self.alpha, device='cuda', dtype=self.matrices_type)
        if cfg.LOCAL_ALPHA:
            self.init_C(lset, self.K_general_dense_cpu)
        else:
            self.C_general = torch.full((self.all_features.shape[0], unique_labels.size), self.alpha, device='cuda',
                                        dtype=self.matrices_type)

        self.K_general = self.build_K_general_matrix(self.K_general_dense_cpu)

        self.initial_sparse_index = (int((1 - self.K_general.sum() / (self.all_features.shape[0] ** 2)) *
                                         self.all_features.shape[0] ** 2) - self.all_features.shape[
                                         0]) // 2  ### calculate the general sparsity value, then calculate the general initial sparse index (by multiply by the total K size, then remove the self connections (-all_features.shape[0]) and divide by 2 (since the matrix is symmetric)

        if lset is not None and lset.size > 0:
            temp_K = self.kernel_fn.compute_kernel(
                torch.from_numpy(self.all_features), torch.from_numpy(self.all_features), self.delta).to('cuda')
            class_indices = {label: np.where(self.train_labels_general[lset.astype(int)] == label)[0] for label in unique_labels}

            for label in unique_labels:

                curr_labels_sim = temp_K[class_indices[label]]
                self.C_general[:, label] = torch.max(curr_labels_sim, axis=0).values
            del temp_K, curr_labels_sim, class_indices
        torch.cuda.empty_cache()

    def build_K_general_matrix(self, kernel_matrix_cpu):
        if self.use_sparse:
            if self.use_K_top50_mask:
                threshold_mask = kernel_matrix_cpu < self.K_sparsity_threshold
                K_top50_mask = torch.load(
                    "/cs/labs/daphna/itai.david/py_repos/TypiClust/results/K_topk_values/cifar100/top50_vals.npz")
                sparse_mask = K_top50_mask | (~threshold_mask)
                kernel_matrix_cpu[~sparse_mask] = 0.0
                del threshold_mask, K_top50_mask, sparse_mask
            else:
                # kernel_matrix_cpu[kernel_matrix_cpu < self.K_sparsity_threshold] = 0.0
                torch.nn.functional.threshold(kernel_matrix_cpu, self.K_sparsity_threshold, 0, inplace=True)
            K_coo = sp.coo_matrix(kernel_matrix_cpu)
            K_general = K_coo.tocsr()  # notice that K is now scipy csr sparse matrix
            del K_coo
        else:
            K_general = kernel_matrix_cpu
        del kernel_matrix_cpu

        return K_general


    def get_priors(self, K):
        # Set sample specific prior
        # rel_measures = prior_selection.compute_reliability(self.K, self.train_labels, batch_size=8192, normalized=True)
        rel_measures = prior_selection.compute_clarity_kp(K, self.train_labels_general, self.num_of_classes, batch_size=8192)

        priors = prior_selection.get_temp_priors(rel_measures, lb=self.alpha_lower, ub=self.alpha_upper) # (N,)
        return priors

    def init_C(self, lset, K):
        """
        Init the main matrix C with the priors. If lset != empty, Init C accordingly.
        NOTE: Attention! float16
        """
        if len(lset) > 0:
            print("Using a method which is not yet tested --- :(")
            self.load_C_from_lset(lset)
        self.priors = self.get_priors(K)  # This is (N,) tensor on CPU
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
            torch.cuda.empty_cache()
            sorted_values = np.load("/cs/labs/daphna/itai.david/py_repos/TypiClust/results/K_sorted_values/cifar100/euclidean_dists_sorted.npy", mmap_mode='r')
            new_threshold_euclidean_dist = sorted_values[::-1][self.initial_sparse_index-self.total_connections_chosen]
            if self.kernel_type =='tophat':
                new_threshold = new_threshold_euclidean_dist

            elif self.kernel_type == 'rbf':
                new_threshold = torch.exp(-1.0 * (torch.tensor(new_threshold_euclidean_dist) / self.sigma) ** 2)
            self.K_sparsity_threshold = new_threshold
            self.K_general_dense_cpu = self.kernel_fn.prepare_K_matrix_for_sparsity(self.K_general_backed, lset, new_threshold)
            tensor_lset = torch.from_numpy(lset.astype(int))
            self.K_general_dense_cpu[tensor_lset, :] = 0
            self.K_general_dense_cpu[:, tensor_lset] = 0
            del tensor_lset
            self.K_general = self.build_K_general_matrix(self.K_general_dense_cpu)

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
        # if self.decrease_alpha and len(lset) > 0:
        #     self.C -= self.alpha
        #     self.alpha /= 2
        #     self.C += self.alpha
        for i in range(self.budgetSize):
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)
            # curr_l_set = np.concatenate((self.lSet, selected)).astype(int)
            C_sum = torch.sum(self.C, dim=1, keepdim=True)
            norm_C =  C_sum
            # norm_C = self.C / C_sum
            class_corr = (self.C.T - self.alpha) @ (self.C -self.alpha)
            points_intres_class = (self.C - self.alpha) @ class_corr
            if self.diff_method == 'margin':
                vals, inds = torch.topk(self.C, k=2, dim=1)

                old_margin = vals[:, 0] - vals[:, 1]

                point_total_contribution = batched_diffs(self.K, old_margin, self.alpha, self.num_of_classes, diff_method="margin")
            elif self.diff_method == 'max':  ### old proxy with alphas vector without prior
                max_vals, indices = torch.max(norm_C, dim=1)
                point_total_contribution = batched_diffs(self.K, max_vals, self.alpha, self.num_of_classes, diff_method="max")
            elif self.diff_method in ['prob_cover', 'max_herding']:
                max_vals, indices = torch.max(self.C, dim=1)
                point_total_contribution = batched_diffs(self.K, max_vals, 0, self.num_of_classes,
                                                         diff_method="abs_diff")
            elif self.diff_method == 'top2_weighted_max':

                vals, inds = torch.topk(self.C, k=2, dim=1)
                point_total_contribution = batched_diffs_weighted(self.K, self.C, vals, inds, diff_method="weighted_max", cont_method=self.cont_method)
            elif self.diff_method == 'full_weighted_max': ### the method with the excepectation
                if self.use_sparse:
                    point_total_contribution = batched_diffs_efficient_weighted_sparse(self.K, self.C, cont_method=self.cont_method)
                else:
                    if len(self.K.shape) == 2:
                        self.K.unsqueeze_(2)
                    point_total_contribution = batched_diffs_efficient_weighted(self.K, self.C,
                                                          diff_method="efficient_full_weighted_max",cont_method=self.cont_method, class_corr=points_intres_class)
            else:
                point_total_contribution = batched_diffs(self.K, self.C, diff_method=self.diff_method)
            point_total_contribution[curr_l_set] = -np.inf
            # sampled_point = point_total_contribution.argmax().item()
            sampled_point = np.argsort(point_total_contribution.cpu().numpy(), kind='stable')[::-1][0].item()
            chosen_label = self.train_labels[sampled_point].item()

            self.chosen_labels_num[chosen_label] += 1

            K_row_dense = self.K[sampled_point].to_dense().to('cuda').squeeze()


            if self.diff_method in ['prob_cover', 'max_herding']:
                self.C[:, chosen_label] = torch.maximum(self.C[:, chosen_label],K_row_dense)
            else:
                self.labeled_points_mask[sampled_point] = True
                self.C[sampled_point, :] = torch.zeros(self.num_of_classes).to('cuda')
                self.C[sampled_point, chosen_label] = 1.0

                self.C[~self.labeled_points_mask, chosen_label] += K_row_dense[~self.labeled_points_mask]

            # self.C[:, chosen_label] = torch.maximum(self.C[:, chosen_label], self.K[sampled_point].squeeze())
            self.cum_labels_info[chosen_label] += K_row_dense.sum()
            self.total_connections_chosen += torch.sum(K_row_dense > 0).item()

            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)
            del K_row_dense

        if False:
            name = "prob_method_v1"
            np.save(f"/cs/labs/daphna/itai.david/py_repos/TypiClust/vectors_debug/0708/{name}.npy", self.K[selected].cpu())

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

    def plot_tsne(self):
        labeled_indices = np.array(self.lSet).astype(int)
        sampled_indices = np.array(self.activeSet).astype(int)
        visualize_tsne(labeled_indices, sampled_indices, algo_name='MISP')

# @torch.compile(backend="inductor")
def batched_diffs_efficient_weighted(K: torch.Tensor, C: torch.Tensor, chunk_size: int = 1024, diff_method: str = "abs_diff", cont_method: str = "positive", class_corr=None):
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
    class_corr = class_corr.unsqueeze(1).to(torch.bool)
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
        elif cont_method == "fusion":
            class_corr_batched = class_corr[i:end]
            is_neg = new_state_vec < 0
            new_state_vec[is_neg & ~class_corr_batched] = 0
            new_state_vec[is_neg & class_corr_batched] *= -1
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
        result[i:end] = torch.einsum('ijk,ik->i', new_state_vec, weights_batched)
        del new_state_vec
        del K_batched
        del weights_batched
        # result[i:end] = torch.einsum('ijk,ik->i',new_state_vec, weights_batched)
        # res = new_state_vec * weights_batched
    return result


# @torch.compile(backend="inductor")
def batched_diffs_efficient_weighted_sparse(K_csr: torch.Tensor, C: torch.Tensor, chunk_size: int = 5120, cont_method: str = "positive"):
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

        weights_chunk = norm_C[global_rows]  # (b, classes)
        # Now map per-nnz: weights_for_nnz = weights_chunk[row_indices]
        weights_for_nnz = weights_chunk[row_indices]  # (total_nnz, classes)

        # Multiply elementwise and sum over classes -> per-nnz scalar
        per_nnz_weighted = (new_state * weights_for_nnz).sum(dim=1)  # (total_nnz,)

        # Aggregate per row via scatter_add
        chunk_result = torch.zeros((b,), device=dev, dtype=C.dtype)
        chunk_result.scatter_add_(0, row_indices, per_nnz_weighted)



        # result[i:end] = torch.bmm(new_state_vec, weights_batched.unsqueeze_(2)).sum(dim=1).squeeze(1)
        result[row_start:row_end] = chunk_result
        torch.cuda.empty_cache()

    return result


def batched_diffs_efficient_weighted_v2(K: torch.Tensor, C: torch.Tensor, chunk_size: int = 256, diff_method: str = "abs_diff", cont_method: int = 0):
    D, N, _ = K.shape
    results_list = []
    max_C, _ = torch.max(C, dim=1, keepdim=True)
    sum_C = torch.sum(C, dim=1, keepdim=True)
    norm_C = (C / sum_C)
    old_max = (max_C / sum_C)
    C_diff = (C - max_C).unsqueeze(0)
    num_iterations = int(N)
    cont_method = int(cont_method)
    max_C = max_C.unsqueeze(0)
    n_labels = C_diff.shape[-1]

    s_C_diff = C_diff * sum_C
    s_min_c1 = sum_C - max_C
    s_square = sum_C * sum_C

    for i in range(0, num_iterations, int(chunk_size)):
            end = min(i + chunk_size, D)
            K_batch = K[i:end]
            p1 = s_min_c1 * K_batch
            p2 = p1 + s_C_diff
            nom = s_square + sum_C * K_batch
            new_state_vec = p2 / nom
            if cont_method == 0:
                new_state_vec.clamp_(min=0)
            elif cont_method == 1:
                 torch.abs(new_state_vec, out=new_state_vec)

            weighted_point_diff = torch.einsum('ijk,ik->i',new_state_vec, norm_C[i:end])
            results_list.append(weighted_point_diff)
    result = torch.cat(results_list)
    return result



def batched_diffs(K, C, alpha, number_of_classes, chunk_size=1024, diff_method="abs_diff"):
    D, N = K.shape
    result = torch.empty(D).to(device=C.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if diff_method == "abs_diff":
            K_batched = K[start:end]
            K_batched = K_batched.to('cuda')
            result[start:end] = torch.sum(torch.maximum(K_batched - C, torch.zeros_like(K_batched).to(device=C.device)), dim=1)
        elif diff_method == "max":
            K_batched = K[start:end]
            K_batched = K_batched.to('cuda')
            result[start:end] = torch.sum(
                torch.maximum(((K_batched + alpha) / (torch.maximum( K_batched+ alpha * number_of_classes, torch.full_like(K_batched, 1e-8)))) - C, torch.zeros_like(K_batched).to(device=C.device)), dim=1)
        elif diff_method == 'margin':
            result[start:end] = torch.sum(
                torch.maximum((K[start:end] / (K[start:end] + alpha * number_of_classes)) - C, torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
    return result

def batched_diffs_sparse(K, C, alpha, number_of_classes, chunk_size=1024, diff_method="abs_diff"):
    D, N = K.shape
    result = torch.empty(D).to(device=C.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if diff_method == "abs_diff":
            K_batched = K[start:end]
            K_batched = K_batched.to('cuda')
            result[start:end] = torch.sum(torch.maximum(K_batched - C, torch.zeros_like(K_batched).to(device=C.device)), dim=1)
        elif diff_method == "max":
            K_batched = K[start:end]
            K_batched = K_batched.to('cuda')
            result[start:end] = torch.sum(
                torch.maximum(((K_batched + alpha) / (torch.maximum( K_batched+ alpha * number_of_classes, torch.full_like(K_batched, 1e-8)))) - C, torch.zeros_like(K_batched).to(device=C.device)), dim=1)
        elif diff_method == 'margin':
            result[start:end] = torch.sum(
                torch.maximum((K[start:end] / (K[start:end] + alpha * number_of_classes)) - C, torch.zeros_like(K[start:end]).to(device=C.device)), dim=1)
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
# @torch.compile(backend="cudagraphs")
def batched_diffs_weighted(K, C, vals, inds, chunk_size=1024, diff_method="abs_diff", cont_method="positive"):
    D, N = K.shape
    result = torch.empty((D, )).to(device=C.device)
    sum_C = torch.sum(C, axis=1)
    norm_C = (C / sum_C.unsqueeze(1))
    num_iterations = N
    C_max_diff = vals[:, 1] - vals[:, 0]
    partial_sum = torch.sum(vals, dim=1)
    weights = torch.gather(norm_C, 1, inds).unsqueeze(1)
    # old_max = (max_C.squeeze() / sum_C.squeeze())
    old_max = vals[:, 0] / partial_sum
    # timing each iteration
    for i in range(0, num_iterations, chunk_size):
        if diff_method == "weighted_max":
            end = i + chunk_size
            K_batched = K[i:end]
            K_batched = K_batched.to('cuda')
            weights_batched = norm_C[i:end, inds]
            future_sum = K_batched + partial_sum
            new_state_vec = torch.stack([torch.zeros_like(K_batched), torch.maximum(-K_batched, C_max_diff)], dim=0).to(device=C.device) + vals[:, 0] + K_batched
            cont_vec = (new_state_vec / future_sum) - old_max
            if cont_method == "positive":
                cont_vec.clamp_(min=0)
            elif cont_method == "abs":
                cont_vec = torch.abs(cont_vec)
            # weighted_point_diff = weights[i:end] @ cont_vec.permute(1, 0, 2)
            # result[i:end] = torch.nansum(weighted_point_diff, dim=2)
            result[i:end] = torch.einsum('ijk,jki->j', cont_vec, weights_batched)

            del new_state_vec
            del K_batched
        else:
            raise ValueError(f"Unknown diff method: {diff_method}")
    return result



def slice_csr_rows(csr_tensor, start_row, length):
    """
    Fast slicing for CSR tensors. Replaces .narrow(0, start, length).
    """
    end_row = start_row + length

    crow_indices = csr_tensor.crow_indices()
    col_indices = csr_tensor.col_indices()
    values = csr_tensor.values()

    # 1. Find the data range in the underlying 1D arrays
    # Data starts where row 'start_row' begins
    p_start = crow_indices[start_row]
    # Data ends where row 'end_row' begins
    p_end = crow_indices[end_row]

    # 2. Slice the values and column indices
    new_values = values[p_start:p_end]
    new_col_indices = col_indices[p_start:p_end]

    # 3. Slice and Shift Row Pointers
    # Extract pointers for the specific rows we want
    new_crow_indices = crow_indices[start_row: end_row + 1]
    # Shift them so the first row starts at index 0
    new_crow_indices = new_crow_indices - p_start

    # 4. Create the new CSR tensor
    return torch.sparse_csr_tensor(
        new_crow_indices,
        new_col_indices,
        new_values,
        size=(length, csr_tensor.size(1)),
        dtype=csr_tensor.dtype,
        device=csr_tensor.device
    )


def csr_weighted_sum_collapsed(csr_weights, dense_vec):
    """
    Computes row-wise weighted sums, collapsing the feature dimension.

    Input:
      csr_weights: (Batch, N) - Sparse weights
      dense_vec:   (Batch, N, D) - Dense features

    Output:
      result:      (Batch,) - Scalar result per batch item
    """
    # 1. Components
    crow_indices = csr_weights.crow_indices()
    col_indices = csr_weights.col_indices()
    values = csr_weights.values()  # (NNZ)

    # 2. Decompress Row Indices
    rows_per_value = torch.arange(csr_weights.size(0), device=csr_weights.device).repeat_interleave(
        crow_indices.diff()
    )

    # 3. Gather Dense Values -> Shape (NNZ, 1024)
    # We extract only the vectors that correspond to non-zero weights
    gathered_dense = dense_vec[rows_per_value, col_indices]

    # --- OPTIMIZATION: Sum features FIRST ---
    # Instead of multiplying (NNZ, 1024) * (NNZ, 1), we sum the 1024 features now.
    # This reduces the problem from Matrix math to Vector math.
    # Shape: (NNZ, 1024) -> (NNZ,)
    gathered_sum = gathered_dense.sum(dim=1)

    # 4. Multiply Weights * Summed_Features
    # Shape: (NNZ,) * (NNZ,) -> (NNZ,)
    products = values * gathered_sum

    # 5. Aggregate back to Batch rows
    # Shape: (Batch,)
    result = torch.zeros(csr_weights.size(0), device=csr_weights.device, dtype=csr_weights.dtype)
    result.index_add_(0, rows_per_value, products)

    return result


def apply_shared_mask_to_batch(template_csr, batch_dense):
    """
    Applies the sparsity pattern of a 2D CSR matrix to a 3D Dense Batch.

    Args:
        template_csr: Sparse CSR (Rows, Cols) - Your 'norm_C'
        batch_dense:  Dense (Batch, Rows, Cols) - Your 'other_matrix'

    Returns:
        Sparse Batched CSR Tensor (Batch, Rows, Cols)
    """
    # 1. Get coordinates
    rows_per_value = torch.arange(template_csr.size(0), device=template_csr.device).repeat_interleave(
        template_csr.crow_indices().diff()
    )
    col_indices = template_csr.col_indices()

    # 2. Extract values (The heavy lifting)
    # Shape: (Batch, NNZ)
    gathered_vals = batch_dense[:, rows_per_value, col_indices]

    # 3. Build Batched CSR
    B = batch_dense.size(0)

    # We expand the indices to match the batch size.
    # Note: We use .contiguous() on indices only if PyTorch throws a stride error,
    # but usually .expand() is sufficient for current versions.
    return torch.sparse_csr_tensor(
        template_csr.crow_indices().unsqueeze(0).expand(B, -1),
        template_csr.col_indices().unsqueeze(0).expand(B, -1),
        gathered_vals,
        size=batch_dense.shape,
        dtype=batch_dense.dtype,
        device=batch_dense.device
    )

