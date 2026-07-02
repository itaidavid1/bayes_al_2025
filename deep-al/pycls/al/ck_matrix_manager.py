"""
CKMatrixManager: Universal C and K matrix manager for active learning methods
that don't natively support these matrices.

Provides kernel-based similarity tracking for pseudo-labeling and distillation.
"""
import numpy as np
import torch
import scipy.sparse as sp
import pycls.datasets.utils as ds_utils
from .kernel_utils import build_sparse_kernel_matrix, RBFKernel, TopHatKernel, CkNNKernel, LocalRBFKernel


class CKMatrixManager:
    """
    Manages C and K matrices for AL methods that don't natively support them.
    Provides kernel-based similarity tracking for pseudo-labeling/distillation.
    
    This class mimics the K/C matrix functionality from BAYES_MISP but in a
    standalone, reusable way that works with any active learning method.
    """
    
    def __init__(self, cfg, data_obj, train_labels, lset):
        """
        Initialize the C/K matrix manager.
        
        Args:
            cfg: Configuration object with kernel parameters
            data_obj: Data object for loading features
            train_labels: Array of labels for all training data
            lset: Initial labeled set indices
        """
        self.cfg = cfg
        self.data_obj = data_obj
        self.ds_name = cfg['DATASET']['NAME']
        
        # Load features for the entire training set
        self.all_features = ds_utils.load_features(self.ds_name, train=True)
        
        # Get configuration parameters - use separate CK parameters if available
        self.alpha = cfg.CK_ALPHA if 'CK_ALPHA' in cfg else (cfg.ALPHA if 'ALPHA' in cfg else 0.5)
        self.use_sparse = cfg.CK_SPARSE_K if 'CK_SPARSE_K' in cfg else False
        self.matrices_type = torch.float32 if self.use_sparse else torch.float16
        self.K_sparsity_threshold = cfg.CK_K_SPARSITY_THRESHOLD if 'CK_K_SPARSITY_THRESHOLD' in cfg else 0.0
        self.kernel_type = cfg.KERNEL_TYPE if 'KERNEL_TYPE' in cfg else 'rbf'
        self.sigma = cfg.CK_SIGMA if 'CK_SIGMA' in cfg else (cfg.ACTIVE_LEARNING.INITIAL_SIGMA if 'INITIAL_SIGMA' in cfg.ACTIVE_LEARNING else 1.0)
        self.delta = cfg.ACTIVE_LEARNING.INITIAL_DELTA if 'INITIAL_DELTA' in cfg.ACTIVE_LEARNING else 0.6
        self.ck_k = cfg.CK_K if 'CK_K' in cfg else 7
        self.kernel_build_batch_size = getattr(cfg, 'KERNEL_BUILD_BATCH_SIZE', 1024)
        self.local_rbf_alpha = cfg.LOCAL_RBF_ALPHA if 'LOCAL_RBF_ALPHA' in cfg else 1.0
        self.degree_normalization_method = cfg.DEGREE_NORMALIZATION_METHOD if 'DEGREE_NORMALIZATION_METHOD' in cfg else 'none'
        
        # Force CPU device for CK manager matrices
        self.device = 'cpu'
        
        # Store training labels
        self.train_labels = np.array(train_labels)
        self.unique_labels = np.unique(self.train_labels)
        self.num_of_classes = self.unique_labels.size
        
        # Track which points are labeled
        self.labeled_mask = np.zeros(self.all_features.shape[0], dtype=bool)
        if lset is not None and len(lset) > 0:
            self.labeled_mask[lset.astype(int)] = True
        
        # Select kernel function and threshold (use CPU device)
        if self.kernel_type == 'tophat':
            self.kernel_fn = TopHatKernel(self.device)
            initial_threshold = self.delta
        elif self.kernel_type == 'cknn':
            self.kernel_fn = CkNNKernel(self.device)
            initial_threshold = self.K_sparsity_threshold
        elif self.kernel_type == 'local_rbf':
            self.kernel_fn = LocalRBFKernel(self.device, alpha=self.local_rbf_alpha, 
                                           degree_normalization_method=self.degree_normalization_method)
            initial_threshold = self.K_sparsity_threshold
        else:  # rbf (default)
            self.kernel_fn = RBFKernel(self.device)
            initial_threshold = self.K_sparsity_threshold
        
        # Build K_general matrix
        kernel_info = f"k={self.ck_k}" if self.kernel_type == 'cknn' else f"sigma={self.sigma}"
        if self.kernel_type == 'local_rbf':
            kernel_info += f", local_rbf_alpha={self.local_rbf_alpha}, degree_norm={self.degree_normalization_method}"
        print(f"[CKMatrixManager] Building K_general matrix with {self.kernel_type} kernel, "
              f"{kernel_info}, threshold={initial_threshold}, sparse={self.use_sparse}, device={self.device}, alpha={self.alpha}")
        self.K_general = self.build_K_general_matrix(
            self.all_features,
            threshold=initial_threshold
        )
        
        # Initialize C_general matrix
        self.C_general = self.initialize_C_general(lset)
        
        print(f"[CKMatrixManager] Initialized with K shape: {self.K_general.shape}, "
              f"C shape: {self.C_general.shape}, alpha={self.alpha}, sigma={self.sigma}, device={self.device}")
    
    def build_K_general_matrix(self, features, threshold):
        """
        Build the K_general kernel matrix.
        
        Args:
            features: Feature matrix (N, D)
            threshold: Sparsity threshold
            
        Returns:
            Kernel matrix (sparse CSR or dense tensor)
        """
        if self.use_sparse:
            kernel_param = self.ck_k if self.kernel_type == 'cknn' else self.sigma
            K_general = build_sparse_kernel_matrix(
                features,
                threshold=threshold,
                kernel_type=self.kernel_type,
                kernel_param=kernel_param,
                batch_size=self.kernel_build_batch_size,
                device=self.device,
                dtype=torch.float32,
                zero_indices=None,
                prev_threshold=None,
                capture_zero_contrib=False,
                alpha=self.local_rbf_alpha,
                degree_normalization_method=self.degree_normalization_method,
            )
            return K_general
        else:
            # Build dense kernel matrix on CPU
            features_tensor = torch.from_numpy(features).to(self.device)

            if self.kernel_type == 'tophat':
                K_general = self.kernel_fn.compute_kernel(
                    features_tensor,
                    features_tensor,
                    h=self.delta,
                    batch_size=self.kernel_build_batch_size,
                    matrices_type=self.matrices_type
                )
            elif self.kernel_type == 'cknn':
                K_general = self.kernel_fn.compute_kernel(
                    features_tensor,
                    features_tensor,
                    k=self.ck_k,
                    batch_size=self.kernel_build_batch_size,
                    matrices_type=self.matrices_type
                )
                K_general[K_general < self.K_sparsity_threshold] = 0
            elif self.kernel_type == 'local_rbf':
                K_general = self.kernel_fn.compute_kernel(
                    features_tensor,
                    features_tensor,
                    h=self.sigma,
                    batch_size=self.kernel_build_batch_size,
                    matrices_type=self.matrices_type
                )
                K_general[K_general < self.K_sparsity_threshold] = 0
            else:  # rbf
                K_general = self.kernel_fn.compute_kernel(
                    features_tensor,
                    features_tensor,
                    h=self.sigma,
                    batch_size=self.kernel_build_batch_size,
                    matrices_type=self.matrices_type
                )
                K_general[K_general < self.K_sparsity_threshold] = 0

            return K_general
    
    def initialize_C_general(self, lset):
        """
        Initialize C_general matrix with alpha values and initial coverage from lset.
        
        Args:
            lset: Initial labeled set indices (may be None or empty)
            
        Returns:
            C_general matrix of shape (N, num_classes)
        """
        n_samples = self.all_features.shape[0]
        
        # Initialize with alpha values on CPU
        C_general = torch.full(
            (n_samples, self.num_of_classes),
            self.alpha,
            device=self.device,
            dtype=self.matrices_type
        )
        
        # If there are initial labeled points, update C with their kernel similarities
        if lset is not None and len(lset) > 0 and not self.use_sparse:
            # For each class, find labeled points and compute max kernel similarity
            for label_idx, label in enumerate(self.unique_labels):
                label_mask = self.train_labels[lset.astype(int)] == label
                if not label_mask.any():
                    continue
                
                # Get indices of labeled points for this class
                labeled_indices_for_class = lset[label_mask]
                
                # Get kernel similarities from these labeled points
                if torch.is_tensor(self.K_general):
                    K_slice = self.K_general[labeled_indices_for_class]
                    max_similarities = torch.max(K_slice, dim=0).values
                    C_general[:, label_idx] = max_similarities.to(device=self.device)
                else:
                    # Handle numpy array case
                    K_slice = self.K_general[labeled_indices_for_class]
                    max_similarities = np.max(K_slice, axis=0)
                    C_general[:, label_idx] = torch.from_numpy(max_similarities).to(
                        device=self.device, dtype=self.matrices_type
                    )
        
        return C_general
    
    def update_C_after_selection(self, newly_labeled_indices, labels):
        """
        Update C matrix with kernel similarities from newly labeled points.
        Similar to BAYES_MISP update logic (lines 744-745).
        
        Args:
            newly_labeled_indices: Array of indices that were just labeled
            labels: Corresponding labels for the newly labeled points
        """
        newly_labeled_indices = np.asarray(newly_labeled_indices, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
        
        for idx, label in zip(newly_labeled_indices, labels):
            # Mark as labeled
            self.labeled_mask[idx] = True
            
            # Get kernel similarities (K row) for this point
            if sp.issparse(self.K_general):
                # Sparse matrix case
                K_row = self.K_general[idx].toarray().flatten().astype(np.float32)
                K_row_tensor = torch.from_numpy(K_row).to(device=self.device, dtype=self.matrices_type)
            elif torch.is_tensor(self.K_general):
                # Dense tensor case
                K_row_tensor = self.K_general[idx].to(self.device).squeeze()
                if K_row_tensor.dim() == 0:
                    K_row_tensor = K_row_tensor.unsqueeze(0)
            else:
                # Numpy array case
                K_row = self.K_general[idx].flatten().astype(np.float32)
                K_row_tensor = torch.from_numpy(K_row).to(device=self.device, dtype=self.matrices_type)
            
            # Update C for unlabeled points only
            unlabeled_mask_tensor = torch.from_numpy(~self.labeled_mask).to(self.device)
            
            # Add kernel similarities to the C matrix for this label's column
            self.C_general[unlabeled_mask_tensor, label] += K_row_tensor[unlabeled_mask_tensor]
        
        print(f"[CKMatrixManager] Updated C matrix after labeling {len(newly_labeled_indices)} points")
    
    def get_C_general(self):
        """Return the current C_general matrix."""
        return self.C_general
    
    def get_K_general(self):
        """Return the K_general matrix."""
        return self.K_general
