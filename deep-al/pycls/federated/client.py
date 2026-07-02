import os
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from pycls.al.ActiveLearning import ActiveLearning
from pycls.al.ck_matrix_manager import CKMatrixManager
from pycls.al.kernel_utils import RBFKernel, TopHatKernel
from pycls.federated.aggregation import fedprox_step_loss
from pycls.federated.types import ClientUpdate
from pycls.utils.meters import TestMeter
import pycls.core.losses as losses
import pycls.core.optimizer as optim
import pycls.utils.metrics as mu
import pycls.utils.model_handler as mh


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        cfg,
        data_obj,
        train_data,
        test_loader,
        client_indices: np.ndarray,
        exp_dir: str,
        local_seed: int,
        kernel_type: str = "rbf",
    ):
        self.client_id = client_id
        self.cfg = deepcopy(cfg)
        self.data_obj = data_obj
        self.train_data = train_data
        self.test_loader = test_loader
        self.kernel_type = kernel_type
        self.client_indices = np.asarray(client_indices, dtype=np.int64)
        self.rng = np.random.default_rng(local_seed)
        self.exp_dir = exp_dir

        self.client_dir = os.path.join(exp_dir, f"client_{client_id}")
        os.makedirs(self.client_dir, exist_ok=True)

        self._init_local_partitions()
        self._init_active_learning()
        self.veracity_targets: Dict[int, Tuple[np.ndarray, float]] = {}
        self.veracity_filtered_count = 0  # Track points filtered by threshold
        self.kernel_function = RBFKernel("cpu") if self.kernel_type == "rbf" else TopHatKernel("cpu")
        self.labeled_features = np.empty((0, 0), dtype=np.float32)
        self.labeled_labels = np.empty((0,), dtype=np.int64)
        self._refresh_labeled_cache()

    def clear_veracity_targets(self):
        """Clear all veracity feedback (for computing baseline)."""
        self.veracity_targets = {}

    def _init_local_partitions(self):
        n_local = self.client_indices.size
        # Use direct count instead of ratio for federated clients
        init_count = int(self.cfg.FEDERATED.CLIENT_LABELS_INITIAL_SIZE)
        init_count = max(0, min(init_count, n_local))  # Clamp to [0, n_local]
        
        shuffled = self.client_indices.copy()
        self.rng.shuffle(shuffled)
        self.lSet = np.asarray(shuffled[:init_count], dtype=np.int64)
        self.uSet = np.asarray(shuffled[init_count:], dtype=np.int64)
        self.valSet = np.asarray([], dtype=np.int64)

        if self.lSet.size == 0 and self.uSet.size > 0:
            bootstrap = self.rng.choice(
                self.uSet, size=min(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, self.uSet.size), replace=False
            )
            self.lSet = np.asarray(bootstrap, dtype=np.int64)
            self.uSet = np.asarray(sorted(list(set(self.uSet.tolist()) - set(self.lSet.tolist()))), dtype=np.int64)

    def _init_active_learning(self):
        self.train_data.return_index = True
        train_labels = np.asarray(self.train_data.targets)
        self.al_obj = ActiveLearning(self.data_obj, self.cfg, train_labels=train_labels, lset=self.lSet)
        needs_ck = self.cfg.TRAIN_PSEUDO_LABELS or getattr(self.cfg, "DISTILLATION_TRAINING", False)
        if needs_ck:
            ck = CKMatrixManager(self.cfg, self.data_obj, train_labels, self.lSet)
            self.al_obj.attach_ck_manager(ck)

    def load_global_model(self, global_state: Dict[str, torch.Tensor]):
        self.model = mh.get_model(self.cfg)
        if isinstance(self.model, torch.nn.Module):
            self.model.load_state_dict(global_state, strict=True)
        else:
            raise ValueError("Federated training currently supports torch.nn.Module models only.")

    def run_local_al_round(self):
        if self.uSet.size <= self.cfg.ACTIVE_LEARNING.BUDGET_SIZE:
            return
        active_set, new_u_set = self.al_obj.sample_from_uSet(
            self.model, self.lSet, self.uSet, self.train_data, data_obj=self.data_obj
        )
        self.lSet = np.append(self.lSet, active_set).astype(np.int64)
        self.uSet = np.asarray(new_u_set, dtype=np.int64)
        self._refresh_labeled_cache()

    def _refresh_labeled_cache(self):
        self.labeled_features, self.labeled_labels = self._get_labeled_embeddings_and_labels()

    def propose_query_point(self, num_queries: int = 1) -> Optional[np.ndarray]:
        """
        Propose multiple query points from the unlabeled set.
        
        Args:
            num_queries: Number of query points to propose (default: 1)
            
        Returns:
            Array of point indices, or None if no points available
        """
        if self.uSet.size == 0:
            return None
        if self.uSet.size == 1:
            return int(self.uSet[0])

        active_set, _ = self.al_obj.sample_from_uSet(
            self.model,
            self.lSet.copy(),
            self.uSet.copy(),
            self.train_data,
            data_obj=self.data_obj,
        )
        if active_set is not None and len(active_set) > 0:
            return np.asarray(active_set[:num_queries], dtype=np.int64)
        
        return None

    def get_point_embedding(self, point_idx: int) -> np.ndarray:
        """Get embedding for a single point."""
        if hasattr(self.train_data, "features") and self.train_data.features is not None:
            emb = np.asarray(self.train_data.features[int(point_idx)], dtype=np.float32)
            # Normalize embedding for federated similarity computation
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        loader = self.data_obj.getSequentialDataLoader(
            indexes=np.asarray([int(point_idx)], dtype=np.int64),
            batch_size=1,
            data=self.train_data,
        )
        batch = next(iter(loader))
        x = batch[0]
        if isinstance(x, torch.Tensor):
            emb = x[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            emb = np.asarray(x[0], dtype=np.float32).reshape(-1)
        # Normalize embedding for federated similarity computation
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    
    def get_points_embeddings(self, point_indices: np.ndarray) -> np.ndarray:
        """
        Get embeddings for multiple points at once.
        
        Args:
            point_indices: Array of point indices
            
        Returns:
            2D array of shape (num_points, embedding_dim) with normalized embeddings
        """
        if len(point_indices) == 0:
            return np.empty((0, 0), dtype=np.float32)
        
        point_indices = np.asarray(point_indices, dtype=np.int64)
        
        if hasattr(self.train_data, "features") and self.train_data.features is not None:
            embeddings = np.asarray(self.train_data.features[point_indices], dtype=np.float32)
            # Normalize embeddings for federated similarity computation
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)
            return embeddings
        
        loader = self.data_obj.getSequentialDataLoader(
            indexes=point_indices,
            batch_size=min(self.cfg.TRAIN.BATCH_SIZE, len(point_indices)),
            data=self.train_data,
        )
        
        embeddings_list = []
        for batch in loader:
            x = batch[0]
            if isinstance(x, torch.Tensor):
                batch_emb = x.detach().cpu().numpy().astype(np.float32).reshape(x.shape[0], -1)
            else:
                x_np = np.asarray(x, dtype=np.float32)
                batch_emb = x_np.reshape(x_np.shape[0], -1)
            embeddings_list.append(batch_emb)
        
        embeddings = np.concatenate(embeddings_list, axis=0)[:len(point_indices)]
        # Normalize embeddings for federated similarity computation
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)
        return embeddings

    def _get_labeled_embeddings_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.lSet.size == 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        labels = np.asarray(self.train_data.targets, dtype=np.int64)[self.lSet.astype(np.int64)]
        if hasattr(self.train_data, "features") and self.train_data.features is not None:
            feats = np.asarray(self.train_data.features[self.lSet.astype(np.int64)], dtype=np.float32)
            # Normalize embeddings for federated similarity computation
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = np.divide(feats, norms, out=np.zeros_like(feats), where=norms > 0)
            return feats, labels
        loader = self.data_obj.getSequentialDataLoader(
            indexes=self.lSet.astype(np.int64),
            batch_size=min(self.cfg.TRAIN.BATCH_SIZE, max(1, self.lSet.size)),
            data=self.train_data,
        )
        feat_batches = []
        for batch in loader:
            x = batch[0]
            if isinstance(x, torch.Tensor):
                feat_batches.append(x.detach().cpu().numpy().astype(np.float32).reshape(x.shape[0], -1))
            else:
                x_np = np.asarray(x, dtype=np.float32)
                feat_batches.append(x_np.reshape(x_np.shape[0], -1))
            if sum(b.shape[0] for b in feat_batches) >= self.lSet.size:
                break
        feats = np.concatenate(feat_batches, axis=0)[: self.lSet.size]
        # Normalize embeddings for federated similarity computation
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = np.divide(feats, norms, out=np.zeros_like(feats), where=norms > 0)
        return feats, labels

    @torch.no_grad()
    def predict_veracity_vector(self, embedding: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict veracity vector for a single embedding."""
        num_classes = int(self.cfg.MODEL.NUM_CLASSES)
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        labeled_feats, labeled_labels = self.labeled_features, self.labeled_labels

        if labeled_feats.size == 0:
            uniform = np.full((num_classes,), 1.0 / num_classes, dtype=np.float32)
            return uniform, float(1.0 / num_classes)

        emb_t = torch.as_tensor(emb, dtype=torch.float32)
        lfeats_t = torch.as_tensor(labeled_feats, dtype=torch.float32)

        if self.kernel_type == "tophat":
            delta = float(getattr(self.cfg.ACTIVE_LEARNING, "INITIAL_DELTA", 1.0))
            sims = self.kernel_function.compute_kernel(
                emb_t, lfeats_t, h=delta, batch_size=self.cfg.TRAIN.BATCH_SIZE, matrices_type=torch.float32
            ).squeeze(0)
        else:
            sigma = float(getattr(self.cfg.ACTIVE_LEARNING, "INITIAL_SIGMA", 1.0))
            sims = self.kernel_function.compute_kernel(
                emb_t, lfeats_t, h=max(sigma, 1e-8), batch_size=self.cfg.TRAIN.BATCH_SIZE, matrices_type=torch.float32
            ).squeeze(0)
            sparsity_thresh = float(getattr(self.cfg, "K_SPARSITY_THRESHOLD", 0.0))
            if sparsity_thresh > 0:
                sims = torch.where(sims > sparsity_thresh, sims, torch.zeros_like(sims))

        c_like = torch.zeros(num_classes, dtype=torch.float32)
        labels_t = torch.as_tensor(labeled_labels, dtype=torch.long)
        c_like.index_add_(0, labels_t, sims)

        alpha_prior = float(getattr(self.cfg, "ALPHA", 0.0))
        if alpha_prior > 0:
            c_like = c_like + alpha_prior

        total = torch.sum(c_like)
        if total <= 0:
            probs = torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32)
        else:
            probs = c_like / total
        confidence = float(torch.max(probs).item())
        return c_like.detach().cpu().numpy().astype(np.float32), confidence

    @torch.no_grad()
    def predict_veracity_vectors(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict veracity vectors for multiple embeddings at once.
        
        Args:
            embeddings: 2D array of shape (num_embeddings, embedding_dim)
            
        Returns:
            Tuple of (veracity_vectors, confidences)
            - veracity_vectors: shape (num_embeddings, num_classes)
            - confidences: shape (num_embeddings,)
        """
        num_classes = int(self.cfg.MODEL.NUM_CLASSES)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        num_embeddings = embeddings.shape[0]
        labeled_feats, labeled_labels = self.labeled_features, self.labeled_labels

        if labeled_feats.size == 0:
            uniform = np.full((num_embeddings, num_classes), 1.0 / num_classes, dtype=np.float32)
            confidences = np.full((num_embeddings,), 1.0 / num_classes, dtype=np.float32)
            return uniform, confidences

        emb_t = torch.as_tensor(embeddings, dtype=torch.float32)
        lfeats_t = torch.as_tensor(labeled_feats, dtype=torch.float32)

        if self.kernel_type == "tophat":
            delta = float(getattr(self.cfg.ACTIVE_LEARNING, "INITIAL_DELTA", 1.0))
            sims = self.kernel_function.compute_kernel(
                emb_t, lfeats_t, h=delta, batch_size=self.cfg.TRAIN.BATCH_SIZE, matrices_type=torch.float32
            )
        else:
            sigma = float(getattr(self.cfg.ACTIVE_LEARNING, "INITIAL_SIGMA", 1.0))
            sims = self.kernel_function.compute_kernel(
                emb_t, lfeats_t, h=max(sigma, 1e-8), batch_size=self.cfg.TRAIN.BATCH_SIZE, matrices_type=torch.float32
            )
            sparsity_thresh = float(getattr(self.cfg, "K_SPARSITY_THRESHOLD", 0.0))
            if sparsity_thresh > 0:
                sims = torch.where(sims > sparsity_thresh, sims, torch.zeros_like(sims))

        # sims shape: (num_embeddings, num_labeled_points)
        # Compute veracity vectors for all embeddings
        c_likes = torch.zeros(num_embeddings, num_classes, dtype=torch.float32)
        labels_t = torch.as_tensor(labeled_labels, dtype=torch.long)
        
        for i in range(num_embeddings):
            c_likes[i].index_add_(0, labels_t, sims[i])

        alpha_prior = float(getattr(self.cfg, "ALPHA", 0.0))
        if alpha_prior > 0:
            c_likes = c_likes + alpha_prior

        totals = torch.sum(c_likes, dim=1, keepdim=True)
        probs = torch.where(totals > 0, c_likes / totals, torch.full_like(c_likes, 1.0 / num_classes))
        confidences = torch.max(probs, dim=1)[0]
        
        return c_likes.detach().cpu().numpy().astype(np.float32), confidences.detach().cpu().numpy().astype(np.float32)

    def consume_veracity_feedback(self, point_idx: int, veracity_vector: np.ndarray, confidence: float):
        # Apply confidence threshold (similar to distillation_threshold in train_al.py)
        veracity_threshold = getattr(self.cfg.FEDERATED, 'VERACITY_THRESHOLD', 0.0)
        if float(confidence) >= veracity_threshold:
            self.veracity_targets[int(point_idx)] = (np.asarray(veracity_vector, dtype=np.float32), float(confidence))
            self.uSet = np.asarray([u for u in self.uSet if int(u) != int(point_idx)], dtype=np.int64)
        else:
            # Track filtered veracity points
            self.veracity_filtered_count += 1

    def _train_one_epoch(self, lset_loader, veracity_loader, optimizer, loss_fun, global_params=None):
        self.model.train()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()

        total_loss = 0.0
        num_batches = 0
        
        # Train on labeled samples (lSet) with real labels
        if lset_loader is not None:
            for batch in lset_loader:
                if len(batch) == 3:
                    inputs, labels, _ = batch
                else:
                    inputs, labels = batch

                if use_cuda:
                    inputs = inputs.type(torch.cuda.FloatTensor)
                    inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
                else:
                    inputs = inputs.float()
                
                preds = self.model(inputs)
                base_loss = loss_fun(preds, labels).mean()
                prox = fedprox_step_loss(self.model, global_params, self.cfg.FEDPROX_MU) if global_params is not None else 0.0
                loss = base_loss + prox
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                num_batches += 1
        
        # Train on veracity samples with soft labels
        if veracity_loader is not None:
            veracity_weight = float(self.cfg.FEDERATED.VERACITY_LOSS_WEIGHT)
            for batch in veracity_loader:
                inputs, _, batch_idx = batch  # labels are dummy, we use veracity targets
                
                if use_cuda:
                    inputs = inputs.type(torch.cuda.FloatTensor)
                    inputs = inputs.cuda()
                else:
                    inputs = inputs.float()
                
                preds = self.model(inputs)
                
                # Get veracity targets for this batch
                idx_np = batch_idx.detach().cpu().numpy().astype(np.int64)
                veracity_targets_list = []
                veracity_weights_list = []
                for global_idx in idx_np:
                    if int(global_idx) in self.veracity_targets:
                        target_vec, conf = self.veracity_targets[int(global_idx)]
                        veracity_targets_list.append(target_vec)
                        veracity_weights_list.append(conf)
                
                if veracity_targets_list:
                    tgt_t = torch.as_tensor(np.stack(veracity_targets_list), device=preds.device, dtype=preds.dtype)
                    w_t = torch.as_tensor(veracity_weights_list, device=preds.device, dtype=preds.dtype)
                    log_probs = torch.log_softmax(preds, dim=1)
                    kl = torch.sum(tgt_t * (torch.log(tgt_t + 1e-8) - log_probs), dim=1)
                    veracity_loss = (kl * w_t).mean() * veracity_weight
                    
                    prox = fedprox_step_loss(self.model, global_params, self.cfg.FEDPROX_MU) if global_params is not None else 0.0
                    loss = veracity_loss + prox
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())
                    num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        use_cuda = torch.cuda.is_available()
        meter = TestMeter(len(self.test_loader))
        misclassifications = 0.0
        total = 0.0
        
        num_classes = int(self.cfg.MODEL.NUM_CLASSES)
        per_class_correct = np.zeros(num_classes, dtype=np.int64)
        per_class_total = np.zeros(num_classes, dtype=np.int64)
        
        for batch in self.test_loader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
                inputs = inputs.type(torch.cuda.FloatTensor)
            else:
                inputs = inputs.float()
            preds = self.model(inputs)
            top1_err, _ = mu.topk_errors(preds, labels, [1, 5])
            err = top1_err.item()
            misclassifications += err * inputs.size(0) * self.cfg.NUM_GPUS
            total += inputs.size(0) * self.cfg.NUM_GPUS
            meter.update_stats(top1_err=err, mb_size=inputs.size(0) * self.cfg.NUM_GPUS)
            
            pred_classes = torch.argmax(preds, dim=1)
            labels_cpu = labels.cpu().numpy()
            pred_cpu = pred_classes.cpu().numpy()
            
            for cls in range(num_classes):
                cls_mask = (labels_cpu == cls)
                per_class_total[cls] += np.sum(cls_mask)
                per_class_correct[cls] += np.sum((pred_cpu == labels_cpu) & cls_mask)
        
        meter.reset()
        mean_acc = 100.0 - (misclassifications / max(total, 1.0))
        
        per_class_acc = {}
        for cls in range(num_classes):
            if per_class_total[cls] > 0:
                per_class_acc[str(cls)] = float(100.0 * per_class_correct[cls] / per_class_total[cls])
            else:
                per_class_acc[str(cls)] = None
        
        return mean_acc, per_class_acc

    def run_local_train(self, local_epochs: int, fl_method: str):
        # Create separate loaders for labeled and veracity samples
        lset_loader = None
        veracity_loader = None
        
        if self.lSet.size > 0:
            lset_loader = self.data_obj.getIndexesDataLoader(self.lSet, self.cfg.TRAIN.BATCH_SIZE, self.train_data)
        
        soft_idx = np.asarray(list(self.veracity_targets.keys()), dtype=np.int64)
        if soft_idx.size > 0:
            veracity_loader = self.data_obj.getIndexesDataLoader(soft_idx, self.cfg.TRAIN.BATCH_SIZE, self.train_data)
        
        if lset_loader is None and veracity_loader is None:
            return {"train_loss": 0.0, "test_acc": float("nan")}
        
        optimizer = optim.construct_optimizer(self.cfg, self.model)
        loss_fun = losses.get_loss_fun(reduction="none")
        global_params = None
        if fl_method.lower() == "fedprox":
            global_params = {k: v.detach().clone().to("cuda") for k, v in self.model.state_dict().items()}
        
        train_loss = 0.0
        for _ in range(local_epochs):
            train_loss = self._train_one_epoch(lset_loader, veracity_loader, optimizer, loss_fun, global_params=global_params)
        test_acc, per_class_acc = self._evaluate()
        
        # Include veracity statistics in metrics
        num_veracity_points = len(self.veracity_targets)
        return {
            "train_loss": train_loss, 
            "test_acc": float(test_acc),
            "test_acc_per_class": per_class_acc,
            "num_labeled": int(self.lSet.size),
            "num_veracity_used": num_veracity_points,
            "num_veracity_filtered": self.veracity_filtered_count,
        }

    def export_update(self, metrics: Optional[Dict[str, float]] = None) -> ClientUpdate:
        return ClientUpdate(
            client_id=self.client_id,
            state_dict={k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            num_samples=int(self.lSet.size),
            metrics=metrics or {},
        )
