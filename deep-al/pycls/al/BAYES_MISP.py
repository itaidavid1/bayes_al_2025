import numpy as np
import pandas as pd
import torch
import gc
import pycls.datasets.utils as ds_utils
from tools.utils import visualize_tsne
import time
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
###MISP = maximum importance sampling points
torch.cuda.empty_cache()

def compute_norm(x1, x2, device, batch_size=512):
    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        # distance comparisons are done in batches to reduce memory consumption
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset, p=2.0).to(dtype=torch.float16)

        dist_matrix.append(dist.cpu())
        del dist

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix

class RBFKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        norm = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = torch.exp(-1.0 * (norm / h) ** 2)
        return k

class TopHatKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512):
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device) # 1 x n x d, 1 x n' x d
        dist_matrix = []
        batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(x1, x2_subset)
            dist = (dist < h).to(dtype=torch.float16)
            dist_matrix.append(dist.cpu())
            del dist
        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        # k = (dist_matrix < h).to(dtype=torch.float16)
        return dist_matrix


class BAYES_MISP:
    def __init__(self, cfg, budgetSize, train_labels, lset, delta=1):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.all_features = ds_utils.load_features(self.ds_name, train=True)
        self.diff_method = self.cfg.DIFF_METHOD if 'DIFF_METHOD' in self.cfg else 'abs_diff'
        self.alpha = self.cfg.ALPHA if self.diff_method not in ['prob_cover', 'max_herding'] else 0
        self.debug = self.cfg.DEBUG
        self.norm_importance = self.cfg.NORM_IMPORTANCE
        self.confidence_method = self.cfg.CONFIDENCE_METHOD if 'CONFIDENCE_METHOD' in self.cfg else 'max'
        self.cont_method = self.cfg.CONT_METHOD if 'CONT_METHOD' in self.cfg else 'positive'
        self.decrease_alpha = self.cfg.DECREASING_ALPHA if 'DECREASING_ALPHA' in self.cfg else False
        self.budgetSize = budgetSize
        self.delta = delta
        self.soft_border_val = self.cfg.SOFT_BORDER_VAL if 'SOFT_BORDER_VAL' in self.cfg else 0.15

        kernel_type = self.cfg.KERNEL_TYPE if 'KERNEL_TYPE' in self.cfg else 'rbf'
        if kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel('cuda')
        else:
            self.kernel_fn = RBFKernel('cuda')

        self.train_labels_general = np.array(train_labels)
        unique_labels = np.unique(self.train_labels_general)
        self.C_general = torch.full((self.all_features.shape[0], unique_labels.size), self.alpha, device='cuda', dtype=torch.float16)
        self.num_of_classes = np.unique(self.train_labels_general).size
        self.chosen_labels_num = torch.zeros(self.num_of_classes).to('cuda')
        self.cum_labels_info = torch.zeros(self.num_of_classes).to('cuda')
        if lset is not None and lset.size > 0:
            temp_K = self.kernel_fn.compute_kernel(
                torch.from_numpy(self.all_features), torch.from_numpy(self.all_features), self.delta).to('cuda')
            class_indices = {label: np.where(self.train_labels_general[lset.astype(int)] == label)[0] for label in unique_labels}

            for label in unique_labels:

                curr_labels_sim = temp_K[class_indices[label]]
                self.C_general[:, label] = torch.max(curr_labels_sim, axis=0).values
            del temp_K, curr_labels_sim, class_indices
        torch.cuda.empty_cache()

    def init_sampling_loop(self,lset, uset):
        torch.cuda.empty_cache()
        self.set_rel_features(lset, uset)
        self.activeSet = []
        self.K = self.kernel_fn.compute_kernel(
            self.rel_features, self.rel_features, self.delta)
        self.C = self.C_general[self.relevant_indices].to('cuda')
        self.train_labels = self.train_labels_general[self.relevant_indices]

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
                if len(self.K.shape)==2:
                    self.K.unsqueeze_(2)
                use_thersh = False
                if use_thersh:

                    point_total_contribution = batched_diffs_efficient_weighted_with_threshold(self.K, self.C,
                                                                                threshold = 0.1,
                                                                                cont_method=self.cont_method,
                                                                                class_corr=points_intres_class)
                else:
                    point_total_contribution = batched_diffs_efficient_weighted(self.K, self.C,
                                                          diff_method="efficient_full_weighted_max",cont_method=self.cont_method, class_corr=points_intres_class)
            else:
                point_total_contribution = batched_diffs(self.K, self.C, diff_method=self.diff_method)
            point_total_contribution[curr_l_set] = -np.inf
            # sampled_point = point_total_contribution.argmax().item()
            sampled_point = np.argsort(point_total_contribution.cpu().numpy(), kind='stable')[::-1][0].item()
            chosen_label = self.train_labels[sampled_point].item()

            self.chosen_labels_num[chosen_label] += 1

            if self.diff_method in ['prob_cover', 'max_herding']:
                self.C[:, chosen_label] = torch.maximum(self.C[:, chosen_label], self.K[sampled_point].to('cuda').squeeze())
            else:
                self.C[:, chosen_label] += self.K[sampled_point].squeeze().to('cuda')
            # self.C[:, chosen_label] = torch.maximum(self.C[:, chosen_label], self.K[sampled_point].squeeze())
            self.cum_labels_info[chosen_label] += torch.sum(self.K[sampled_point].to('cuda'))

            assert sampled_point not in selected, 'sample was already selected'
            selected.append(sampled_point)



        if False:
            name = "prob_method_v1"
            np.save(f"/cs/labs/daphna/itai.david/py_repos/TypiClust/vectors_debug/0708/{name}.npy", self.K[selected].cpu())

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]

        self.C_general[self.relevant_indices] = self.C
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

def batched_diffs_efficient_weighted_with_threshold(K: torch.Tensor, C: torch.Tensor, chunk_size: int = 1024, threshold:float = 0.001, cont_method: str = "positive", class_corr=None):
    """
    Optimized version using sparse matrices for thresholded weights.
    All calculations leverage sparse matrix operations for efficiency.
    
    K: (D, N, 1) kernel matrix where D=N
    C: (D, num_classes) confidence matrix
    """
    D, N, _ = K.shape
    result = torch.empty((D, )).to(device=C.device)
    max_C, _ = torch.max(C, dim=1, keepdim=True)  # (D, 1)
    sum_C = torch.sum(C, dim=1, keepdim=True)     # (D, 1)
    norm_C = (C / sum_C)                          # (D, num_classes)
    
    # Apply threshold and convert to sparse (this is the key optimization)
    norm_C[norm_C < threshold] = 0
    norm_C_sparse = norm_C.to_sparse_csr()  # Use CSR format for efficient row access

    old_max = (max_C / sum_C)          # (D, 1)
    C_diff = (C - max_C).unsqueeze(0)  # (1, D, num_classes)
    num_iterations = int(N)
    
    if class_corr is not None:
        class_corr = class_corr.unsqueeze(1).to(torch.bool)  # (D, 1, num_classes)

    K_dense = K.squeeze(-1).to(device=C.device)           # (D, N)
    K_dense[K_dense < threshold] = 0
    K_sparse = K_dense.to_sparse_csr()
    
    for i in range(num_iterations):
            # Get the i-th row of K (kernel similarities for point i)
            weights_sparse = norm_C_sparse[i]

            if weights_sparse._nnz() == 0:
                result[i] = 0
                continue

            weights_coo = weights_sparse.to_sparse_coo()
            nonzero_indices = weights_coo.indices().squeeze(0)
            nonzero_weights = weights_coo.values()
            
            K_row_sparse = K_sparse[i]
            if K_row_sparse._nnz() == 0:
                result[i] = 0
                continue

            active_points = K_row_sparse.indices().squeeze(0)
            K_values = K_row_sparse.values().unsqueeze(1)  # (num_active, 1)

            sum_C_active = sum_C[active_points]
            max_C_active = max_C[active_points]
            old_max_active = old_max[active_points]

            future_sum = sum_C_active.unsqueeze(0) + K_values.unsqueeze(0)
            state_add = max_C_active.unsqueeze(0) + K_values.unsqueeze(0)
            old_max_batched = old_max_active.unsqueeze(0)

            C_diff_relevant = C_diff[:, active_points][:, :, nonzero_indices]  # (1, num_active, num_nonzero)

            new_state_vec = torch.maximum(-K_values.unsqueeze(0), C_diff_relevant)
            new_state_vec.add_(state_add)
            new_state_vec.div_(future_sum)
            new_state_vec.sub_(old_max_batched)

            # Apply contribution method
            if cont_method == "positive": ### regular method
                new_state_vec.clamp_(min=0)
            elif cont_method == 'abs': ### take all contribution
                 torch.abs(new_state_vec, out=new_state_vec)
            elif cont_method == "fusion":
                 if class_corr is not None:
                     class_corr_batched = class_corr[i, :, nonzero_indices].unsqueeze(1)
                     is_neg = new_state_vec < 0
                     new_state_vec[is_neg & ~class_corr_batched] = 0
                     new_state_vec[is_neg & class_corr_batched] *= -1
            elif cont_method == "reg_sum_postive": ## take average contribution (not weighted by the prior)
                 new_state_vec.clamp_(min=0)
                 result[i] = torch.sum(new_state_vec)

                 del new_state_vec
                 del K_values
                 del active_points
                 del sum_C_active
                 del max_C_active
                 del old_max_active

                 continue
            elif cont_method == "reg_sum_abs":
                 torch.abs(new_state_vec, out=new_state_vec)
                 result[i] = torch.sum(new_state_vec)

                 del new_state_vec
                 del K_values
                 del active_points
                 del sum_C_active
                 del max_C_active
                 del old_max_active

                 continue

            # Compute weighted contribution using sparse operations
            # new_state_vec: (1, D, num_nonzero_classes), nonzero_weights: (num_nonzero_classes,)
            # Only sum over non-zero weight classes for maximum efficiency
            new_state_vec_squeezed = new_state_vec.squeeze(0)  # (D, num_nonzero_classes)
            
            # Efficient sparse dot product: only multiply with non-zero weights
            result[i] = torch.einsum('jk,k->', new_state_vec_squeezed, nonzero_weights)
            
            del new_state_vec
            del new_state_vec_squeezed
            del nonzero_indices
            del nonzero_weights

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



