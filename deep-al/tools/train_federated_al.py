import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(os.path.abspath(".."))

from pycls.core.config import cfg, dump_cfg
from pycls.datasets.data import Data
import pycls.utils.logging as lu
from pycls.federated.partitioning import build_balanced_dirichlet_partitions, build_iid_partitions
from pycls.federated.server import FederatedServer
from tools.al_runtime import apply_train_al_args
from tools.train_al import argparser as base_al_argparser


def argparser():
    parser = base_al_argparser()
    parser.description = "Federated Active Learning - Image Classification"
    parser.add_argument("--num_clients", default=10, type=int)
    parser.add_argument("--num_rounds", default=10, type=int)
    parser.add_argument("--clients_per_round", default=5, type=int)
    parser.add_argument("--local_epochs", default=200, type=int)
    parser.add_argument("--fl_method", default="fedavg", choices=["fedavg", "fedprox"], type=str)
    parser.add_argument("--fedprox_mu", default=0.01, type=float)
    parser.add_argument("--partition_mode", default="iid", choices=["iid", "dirichlet"], type=str)
    parser.add_argument("--dirichlet_alpha", default=0.5, type=float)
    parser.add_argument("--min_client_size", default=10, type=int)
    parser.add_argument("--federated_mode", default="standard", choices=["standard", "veracity_query"], type=str)
    parser.add_argument("--queries_per_round", default=1, type=int)
    parser.add_argument("--veracity_agg", default="confidence_mean", choices=["confidence_mean"], type=str)
    parser.add_argument("--veracity_loss_weight", default=1.0, type=float, 
                        help="Scale factor for veracity soft labels loss (similar to distill_factor in train_al.py)")
    parser.add_argument("--veracity_threshold", default=0.0, type=float,
                        help="Minimum confidence threshold for using veracity points (similar to distillation_threshold in train_al.py)")
    parser.add_argument("--client_labels_initial_size", default=100, type=int,
                        help="Number of initial labeled samples per client (overrides --initial_size for federated)")
    return parser


def build_federated_exp_dir():
    cfg.OUT_DIR = os.path.join(os.path.abspath("../.."), cfg.OUT_DIR)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    now = datetime.now()
    date_folder = f"{now.year}_{now.month:02}_{now.day:02}"
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE, "federated", date_folder)
    os.makedirs(dataset_out_dir, exist_ok=True)
    exp_name = f"{cfg.EXP_NAME}_{cfg.DATASET.NAME}_{cfg.ACTIVE_LEARNING.SAMPLING_FN}_{now.hour:02}{now.minute:02}{now.second:02}"
    cfg.EXP_DIR = os.path.join(dataset_out_dir, exp_name)
    os.makedirs(cfg.EXP_DIR, exist_ok=True)


def configure_federated_args(args):
    cfg.FEDERATED.NUM_CLIENTS = args.num_clients
    cfg.FEDERATED.NUM_ROUNDS = args.num_rounds
    cfg.FEDERATED.CLIENTS_PER_ROUND = args.clients_per_round
    cfg.FEDERATED.LOCAL_EPOCHS = args.local_epochs
    cfg.FEDERATED.METHOD = args.fl_method
    cfg.FEDERATED.PARTITION_MODE = args.partition_mode
    cfg.FEDERATED.DIRICHLET_ALPHA = args.dirichlet_alpha
    cfg.FEDERATED.MIN_CLIENT_SIZE = args.min_client_size
    cfg.FEDERATED.MODE = args.federated_mode
    cfg.FEDERATED.QUERIES_PER_ROUND = args.queries_per_round
    cfg.FEDERATED.VERACITY_AGG = args.veracity_agg
    cfg.FEDERATED.VERACITY_LOSS_WEIGHT = args.veracity_loss_weight
    cfg.FEDERATED.VERACITY_THRESHOLD = args.veracity_threshold
    cfg.FEDERATED.CLIENT_LABELS_INITIAL_SIZE = args.client_labels_initial_size
    cfg.FEDPROX_MU = args.fedprox_mu


def save_experiment_config(args, exp_dir):
    """Save experiment configuration as JSON for later analysis and filtering."""
    config = {
        # Dataset and model
        "dataset": cfg.DATASET.NAME,
        "model_type": cfg.MODEL.TYPE,
        
        # Active learning
        "al_method": cfg.ACTIVE_LEARNING.SAMPLING_FN,
        "budget_per_round": cfg.ACTIVE_LEARNING.BUDGET_SIZE,
        "eval_model": getattr(args, 'eval_model', None),
        "diff_method": getattr(args, 'diff_method', None),
        "cont_method": getattr(args, 'cont_method', None),
        "kernel_type": getattr(args, 'kernel_type', None),
        
        # Federated learning
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "clients_per_round": args.clients_per_round,
        "local_epochs": args.local_epochs,
        "fl_method": args.fl_method,
        "fedprox_mu": args.fedprox_mu if args.fl_method == "fedprox" else None,
        
        # Data partitioning
        "partition_mode": args.partition_mode,
        "dirichlet_alpha": args.dirichlet_alpha if args.partition_mode == "dirichlet" else None,
        "min_client_size": args.min_client_size,
        
        # Veracity query
        "federated_mode": args.federated_mode,
        "queries_per_round": args.queries_per_round,
        "veracity_agg": args.veracity_agg if args.federated_mode == "veracity_query" else None,
        "veracity_loss_weight": args.veracity_loss_weight if args.federated_mode == "veracity_query" else None,
        "veracity_threshold": args.veracity_threshold if args.federated_mode == "veracity_query" else None,
        
        # Initial settings
        "client_labels_initial_size": args.client_labels_initial_size,
        "rng_seed": cfg.RNG_SEED,
        "exp_name": cfg.EXP_NAME,
    }
    
    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}
    
    # Save to file
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved experiment config to: {config_path}")


def main():
    args = argparser().parse_args()
    cfg.merge_from_file(args.cfg_file)
    apply_train_al_args(args)
    configure_federated_args(args)
    build_federated_exp_dir()
    dump_cfg(cfg)
    save_experiment_config(args, cfg.EXP_DIR)
    lu.setup_logging(cfg)

    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath("../.."), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, _ = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)

    all_indices = np.arange(train_size, dtype=np.int64)
    if cfg.FEDERATED.PARTITION_MODE == "iid":
        partitions = build_iid_partitions(all_indices, cfg.FEDERATED.NUM_CLIENTS, cfg.RNG_SEED)
    else:
        labels = np.asarray(train_data.targets)
        print(f"Creating Dirichlet partitions with alpha={cfg.FEDERATED.DIRICHLET_ALPHA}")
        partitions = build_balanced_dirichlet_partitions(
            labels=labels,
            indices=all_indices,
            num_clients=cfg.FEDERATED.NUM_CLIENTS,
            alpha=cfg.FEDERATED.DIRICHLET_ALPHA,
            seed=cfg.RNG_SEED,
            max_retries=1000,  # Increased from default 100
        )
        print(f"Successfully created {len(partitions)} client partitions")

    server = FederatedServer(
        cfg=cfg,
        data_obj=data_obj,
        train_data=train_data,
        test_loader=test_loader,
        client_partitions=partitions,
        exp_dir=cfg.EXP_DIR,
    )
    server.run()


if __name__ == "__main__":
    main()
