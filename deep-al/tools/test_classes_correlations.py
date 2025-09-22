import TypiClust.deep-al.pycls.datasets.utils as ds_utils
import numpy as np
import torch

def compute_norm(x1, x2, device, batch_size=512):
    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        # distance comparisons are done in batches to reduce memory consumption
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset, p=2.0) #.to(dtype=torch.float16)

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



all_features = ds_utils.load_features(self.ds_name, train=True)
train_labels = np.load("")
kernel_fn = RBFKernel('cuda')
K = kernel_fn.compute_kernel(all_features, all_features, 1).to('cuda')

exp_path = "/cs/labs/daphna/itai.david/py_repos/TypiClust/output/CIFAR100/resnet18/2025_8_14/CIFAR100_all_misp_from_features_2025_8_14_104059_635817/"

episodes_num = 32

def get_max_mean_classes_corr(activeset):
    num_classes = 100
    class_sim_max = torch.zeros((num_classes, num_classes), device='cuda')
    class_sim_mean = torch.zeros((num_classes, num_classes), device='cuda')
    class_indices = {label: np.where(train_labels[curr_l_set] == label)[0] for label in range(num_classes)}

    for c1 in range(num_classes):
        for c2 in range(c1, num_classes):
            indices_c1 = class_indices.get(c1, [])
            indices_c2 = class_indices.get(c2, [])

            if not len(indices_c1) or not len(indices_c2):
                continue

            sim_submatrix = K[indices_c1, :][:, indices_c2]

            if c1 == c2:
                if len(indices_c1) > 1:
                    # Exclude diagonal for intra-class similarity
                    non_diagonal_mask = ~torch.eye(len(indices_c1), dtype=torch.bool,
                                                   device=sim_submatrix.device)
                    if non_diagonal_mask.any():
                        class_sim_mean[c1, c1] = sim_submatrix[non_diagonal_mask].mean()
                        class_sim_max[c1, c1] = sim_submatrix[non_diagonal_mask].max()
            else:
                mean_val = sim_submatrix.mean()
                max_val = sim_submatrix.max()
                class_sim_mean[c1, c2] = mean_val
                class_sim_mean[c2, c1] = mean_val
                class_sim_max[c1, c2] = max_val
                class_sim_max[c2, c1] = max_val


for ep in range(episodes_num):
    cur_path = exp_path + f"episode_{ep}/" + 'activeSet.npy'
