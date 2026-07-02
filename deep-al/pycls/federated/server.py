import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from pycls.federated.aggregation import fedavg
from pycls.federated.client import FederatedClient
from pycls.federated.types import ClientUpdate
import pycls.utils.model_handler as mh
import pycls.utils.metrics as mu


def compute_class_distribution(labels: np.ndarray, num_classes: int) -> Dict[str, int]:
    """Compute the number of samples per class.
    
    Args:
        labels: Array of labels
        num_classes: Total number of classes
        
    Returns:
        Dictionary mapping class index (as string) to count
    """
    distribution = {}
    for cls in range(num_classes):
        count = np.sum(labels == cls)
        distribution[str(cls)] = int(count)
    return distribution


class FederatedServer:
    def __init__(
        self,
        cfg,
        data_obj,
        train_data,
        test_loader,
        client_partitions: Dict[int, np.ndarray],
        exp_dir: str,
    ):
        self.cfg = cfg
        self.data_obj = data_obj
        self.train_data = train_data
        self.test_loader = test_loader
        self.client_partitions = client_partitions
        self.exp_dir = exp_dir
        self.num_rounds = int(cfg.FEDERATED.NUM_ROUNDS)
        self.clients_per_round = int(cfg.FEDERATED.CLIENTS_PER_ROUND)
        self.local_epochs = int(cfg.FEDERATED.LOCAL_EPOCHS)
        self.fl_method = str(cfg.FEDERATED.METHOD).lower()
        self.mode = str(cfg.FEDERATED.MODE).lower()
        self.queries_per_round = int(cfg.FEDERATED.QUERIES_PER_ROUND)
        self.global_model = mh.get_model(cfg)
        
        # Store initial clean weights to reset each round
        self.initial_weights = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        
        self.global_metrics: List[Dict[str, float]] = []
        
        # Compute global class distributions for train and test
        num_classes = int(cfg.MODEL.NUM_CLASSES)
        train_labels = np.asarray(train_data.targets)
        self.train_class_distribution = compute_class_distribution(train_labels, num_classes)
        
        # Compute test class distribution
        test_labels = []
        for batch in test_loader:
            if len(batch) == 3:
                _, labels, _ = batch
            else:
                _, labels = batch
            test_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
        test_labels = np.concatenate(test_labels)
        self.test_class_distribution = compute_class_distribution(test_labels, num_classes)

        self.clients = []
        for client_id, part in sorted(client_partitions.items()):
            self.clients.append(
                FederatedClient(
                    client_id=client_id,
                    cfg=cfg,
                    data_obj=data_obj,
                    train_data=train_data,
                    test_loader=test_loader,
                    client_indices=part,
                    exp_dir=exp_dir,
                    local_seed=int(cfg.RNG_SEED) + int(client_id),
                )
            )

    def _sample_clients(self, round_id: int):
        rng = np.random.default_rng(int(self.cfg.RNG_SEED) + int(round_id))
        all_ids = np.arange(len(self.clients))
        k = min(self.clients_per_round, len(all_ids))
        picked = rng.choice(all_ids, size=k, replace=False)
        return [self.clients[int(i)] for i in picked]

    def _aggregate(self, updates: List[ClientUpdate]):
        states = [u.state_dict for u in updates]
        weights = [u.num_samples for u in updates]
        new_state = fedavg(states, weights)
        self.global_model.load_state_dict(new_state, strict=True)

    def _aggregate_veracity(self, vectors: List[np.ndarray], confidences: List[float]):
        if not vectors:
            raise ValueError("No peer veracity vectors supplied.")
        
        # Get confidence threshold from config (same threshold used for consuming veracity)
        conf_threshold = getattr(self.cfg.FEDERATED, 'VERACITY_THRESHOLD', 0.0)
        
        # Filter out predictions below confidence threshold
        filtered_vecs = []
        filtered_conf = []
        for vec, conf in zip(vectors, confidences):
            if conf >= conf_threshold:
                filtered_vecs.append(vec)
                filtered_conf.append(conf)
        
        # If all predictions filtered out, return None to signal no aggregation
        if not filtered_vecs:
            return None, None
        
        # Aggregate only confident predictions
        vecs = np.stack(filtered_vecs, axis=0).astype(np.float32)
        conf = np.asarray(filtered_conf, dtype=np.float32)
        conf = np.clip(conf, 1e-8, None)
        w = conf / np.sum(conf)
        return np.sum(vecs * w[:, None], axis=0).astype(np.float32), float(np.mean(filtered_conf))

    def _run_veracity_query_round(self, selected_clients):
        # First loop: get proposed points and embeddings from each client
        for client in selected_clients:
            point_indices = client.propose_query_point(self.queries_per_round)
            if point_indices is None or len(point_indices) == 0:
                continue
            
            embeddings = client.get_points_embeddings(point_indices)
            
            # Collect batch predictions from all peers
            all_peer_vectors = []
            all_peer_conf = []
            
            # Second loop: send all embeddings to each peer and get batch predictions
            for peer in self.clients:
                if peer.client_id == client.client_id:
                    continue
                peer_vectors, peer_conf = peer.predict_veracity_vectors(embeddings)
                all_peer_vectors.append(peer_vectors)
                all_peer_conf.append(peer_conf)
            
            if not all_peer_vectors:
                continue
            
            # Aggregate and send feedback for each query point
            for i, point_idx in enumerate(point_indices):
                peer_vectors_for_point = [peer_vecs[i] for peer_vecs in all_peer_vectors]
                peer_conf_for_point = [peer_confs[i] for peer_confs in all_peer_conf]
                
                agg_vec, agg_conf = self._aggregate_veracity(peer_vectors_for_point, peer_conf_for_point)
                
                # Skip this point if all peers were filtered out due to low confidence
                if agg_vec is None:
                    continue

                client.consume_veracity_feedback(point_idx, agg_vec, float(agg_conf))

    @torch.no_grad()
    def _evaluate_global_model(self):
        self.global_model.eval()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.global_model.cuda()
        misclassifications = 0.0
        total = 0.0
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
            preds = self.global_model(inputs)
            top1_err, _ = mu.topk_errors(preds, labels, [1, 5])
            err = float(top1_err.item())
            misclassifications += err * inputs.size(0) * self.cfg.NUM_GPUS
            total += inputs.size(0) * self.cfg.NUM_GPUS
        return 100.0 - (misclassifications / max(total, 1.0))

    def run(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        
        num_classes = int(self.cfg.MODEL.NUM_CLASSES)
        train_labels = np.asarray(self.train_data.targets)
        
        # Compute baseline ONCE at the start (only for veracity_query mode)
        baseline_metrics = {}
        if self.mode == "veracity_query":
            print("Computing baseline (no veracity feedback)...")
            # Use initial clean weights for baseline
            for client in self.clients:
                client.load_global_model(self.initial_weights)
                client.clear_veracity_targets()
                # Train with only initial labeled data
                metrics = client.run_local_train(
                    local_epochs=self.local_epochs, 
                    fl_method=self.fl_method
                )
                baseline_metrics[str(client.client_id)] = metrics
            
            avg_baseline_acc = float(np.nanmean([m.get("test_acc", np.nan) for m in baseline_metrics.values()]))
            print(f"Baseline (no veracity): Avg Test Acc = {avg_baseline_acc:.2f}%\n")
            
            # Save baseline metrics
            with open(os.path.join(self.exp_dir, "baseline_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "avg_baseline_acc": avg_baseline_acc,
                    "client_baseline_metrics": baseline_metrics
                }, f, indent=2)
        
        # Compute per-client initial partition distributions (only once)
        per_client_partition_distributions = {}
        for client in self.clients:
            client_partition_indices = client.client_indices
            client_partition_labels = train_labels[client_partition_indices]
            client_dist = compute_class_distribution(client_partition_labels, num_classes)
            per_client_partition_distributions[str(client.client_id)] = client_dist
        
        # Save global dataset distributions and client partitions
        with open(os.path.join(self.exp_dir, "dataset_class_distributions.json"), "w", encoding="utf-8") as f:
            json.dump({
                "train_class_distribution": self.train_class_distribution,
                "test_class_distribution": self.test_class_distribution,
                "per_client_partition_distributions": per_client_partition_distributions,
            }, f, indent=2)
        
        # Main federated training rounds
        for round_id in range(self.num_rounds):
            print(f"\n{'='*60}")
            print(f"Round {round_id}: Starting from clean initial weights")
            print(f"{'='*60}")
            
            round_dir = os.path.join(self.exp_dir, f"round_{round_id}")
            os.makedirs(round_dir, exist_ok=True)
            selected_clients = self.clients if self.mode == "veracity_query" else self._sample_clients(round_id)
            
            # Start each round from clean initial weights
            updates = []
            per_client_metrics = {}

            # Load clean initial weights for all selected clients
            for client in selected_clients:
                client.load_global_model(self.initial_weights)
            
            if self.mode == "veracity_query":
                # Add veracity feedback
                self._run_veracity_query_round(selected_clients)
                
                # Print veracity statistics
                veracity_threshold = getattr(self.cfg.FEDERATED, 'VERACITY_THRESHOLD', 0.0)
                total_veracity_used = sum(len(c.veracity_targets) for c in selected_clients)
                total_veracity_filtered = sum(c.veracity_filtered_count for c in selected_clients)
                print(f"\nVeracity Query Round {round_id}:")
                print(f"  Threshold: {veracity_threshold}")
                print(f"  Veracity points used: {total_veracity_used}")
                print(f"  Veracity points filtered: {total_veracity_filtered}")
                print(f"  Soft labels loss scale: {self.cfg.FEDERATED.VERACITY_LOSS_WEIGHT}")
            else:
                # Standard mode: just run AL rounds
                for client in selected_clients:
                    client.run_local_al_round()
            
            # Train
            for client in selected_clients:
                metrics = client.run_local_train(
                    local_epochs=self.local_epochs, 
                    fl_method=self.fl_method
                )
                per_client_metrics[str(client.client_id)] = metrics
                updates.append(client.export_update(metrics=metrics))

            # Aggregate for global model evaluation (but won't be used in next round)
            self._aggregate(updates)
            
            round_metrics = {
                "round": round_id,
                "num_selected_clients": len(selected_clients),
                "avg_client_acc": float(np.nanmean([u.metrics.get("test_acc", np.nan) for u in updates])),
                "avg_client_loss": float(np.nanmean([u.metrics.get("train_loss", np.nan) for u in updates])),
                "global_test_acc": float(self._evaluate_global_model()),
                "avg_num_labeled": float(np.mean([u.metrics.get("num_labeled", 0) for u in updates])),
                "avg_num_veracity_used": float(np.mean([u.metrics.get("num_veracity_used", 0) for u in updates])),
                "avg_num_veracity_filtered": float(np.mean([u.metrics.get("num_veracity_filtered", 0) for u in updates])),
                "train_class_distribution": self.train_class_distribution,
                "test_class_distribution": self.test_class_distribution,
                "client_metrics": per_client_metrics,
            }
            
            # Print summary with baseline comparison
            if self.mode == "veracity_query" and baseline_metrics:
                avg_baseline_acc = float(np.nanmean([m.get("test_acc", np.nan) for m in baseline_metrics.values()]))
                print(f"\nRound {round_id} Summary:")
                print(f"  Baseline Acc (no veracity): {avg_baseline_acc:.2f}%")
                print(f"  Current Acc (with veracity): {round_metrics['avg_client_acc']:.2f}%")
                print(f"  Improvement over baseline: {round_metrics['avg_client_acc'] - avg_baseline_acc:.2f}%")
            
            self.global_metrics.append(round_metrics)
            with open(os.path.join(round_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(round_metrics, f, indent=2)

        with open(os.path.join(self.exp_dir, "global_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.global_metrics, f, indent=2)
