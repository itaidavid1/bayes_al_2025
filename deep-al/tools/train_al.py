import os
import sys
from datetime import datetime
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from copy import deepcopy

from sklearn.exceptions import NotFittedError


# local

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

from pycls.al.ActiveLearning import ActiveLearning
import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
import pycls.core.losses as losses
import pycls.core.optimizer as optim
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.net as nu
import pycls.utils.model_handler as mh
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from pycls.utils.meters import ValMeter

logger = lu.get_logger(__name__)

plot_episode_xvalues = []
plot_episode_yvalues = []

plot_epoch_xvalues = []
plot_epoch_yvalues = []

plot_it_x_values = []
plot_it_y_values = []

delta_avg_lst = []
delta_std_lst = []


def plot_arrays(x_vals, y_vals, x_name, y_name, dataset_name, out_dir, isDebug=False):
    # if not du.is_master_proc():
    #     return

    import matplotlib.pyplot as plt
    temp_name = "{}_vs_{}".format(x_name, y_name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title("Dataset: {}; {}".format(dataset_name, temp_name))
    plt.plot(x_vals, y_vals)

    if isDebug: print("plot_saved at : {}".format(os.path.join(out_dir, temp_name + '.png')))

    plt.savefig(os.path.join(out_dir, temp_name + ".png"))
    plt.close()


def save_plot_values(temp_arrays, temp_names, out_dir, isParallel=True, saveInTextFormat=False, isDebug=True):
    """ Saves arrays provided in the list in npy format """
    # Return if not master process
    # if isParallel:
    #     if not du.is_master_proc():
    #         return

    for i in range(len(temp_arrays)):
        temp_arrays[i] = np.array(temp_arrays[i])
        temp_dir = out_dir
        # if cfg.TRAIN.TRANSFER_EXP:
        #     temp_dir += os.path.join("transfer_experiment",cfg.MODEL.TRANSFER_MODEL_TYPE+"_depth_"+str(cfg.MODEL.TRANSFER_MODEL_DEPTH))+"/"

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if saveInTextFormat:
            # if isDebug: print(f"Saving {temp_names[i]} at {temp_dir+temp_names[i]}.txt in text format!!")
            np.savetxt(temp_dir + '/' + temp_names[i] + ".txt", temp_arrays[i], fmt="%d")
        else:
            # if isDebug: print(f"Saving {temp_names[i]} at {temp_dir+temp_names[i]}.npy in numpy format!!")
            np.save(temp_dir + '/' + temp_names[i] + ".npy", temp_arrays[i])

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ─── Propagation utilities ────────────────────────────────────────────────────

def normalize_C(C, normalization='sum'):
    """Row-normalize a (N, num_classes) numpy confidence matrix.

    Args:
        C: numpy array (N, num_classes)
        normalization: 'sum' — divide each row by its sum; 'softmax' — torch softmax
    Returns:
        numpy float array (N, num_classes); zero rows map to zero
    """
    if normalization == 'softmax':
        return F.softmax(torch.from_numpy(C.astype(np.float32)), dim=1).numpy()
    row_sums = C.sum(axis=1, keepdims=True)
    return np.divide(C, row_sums, out=np.zeros_like(C, dtype=float), where=row_sums != 0)


def compute_sender_mask(C, labeled_mask, mode='all', threshold=None,
                        al_indices=None, al_resample_fn=None, normalization='sum', sender_budget=None):
    """Which unlabeled points may propagate (send) information this iteration.

    Labeled points are always excluded.  Sender modes:
        'all'             – every unlabeled point sends
        'high_confidence' – unlabeled with norm_C.max > threshold  (threshold #1)
                            If sender_budget is specified, select top K points by max(C) instead
        'al_constant'     – the AL picks from the last round, fixed for all iterations
        'al_updated'      – call al_resample_fn(C) each iteration to re-select via bayes_misp

    Note: 'al_constant' and 'al_updated' are equivalent when num_iterations=1, since
    sender selection happens only once regardless.

    Args:
        C: confidence matrix (N, num_classes)
        labeled_mask: boolean mask (N,) indicating labeled points
        mode: sender mode ('all', 'high_confidence', 'al_constant', 'al_updated')
        threshold: confidence threshold for 'high_confidence' mode
        al_indices: pre-selected indices for 'al_constant' mode
        al_resample_fn: callable for 'al_updated' mode
        normalization: 'sum' or 'softmax'
        sender_budget: optional int, number of senders to select (for 'high_confidence' mode)

    Returns boolean mask (N,).
    """
    sender_mask = ~labeled_mask  # base: all unlabeled

    if mode == 'high_confidence':
        norm = normalize_C(C, normalization)
        max_conf = norm.max(axis=1)
        
        if sender_budget is not None:
            # Top K by confidence
            unlabeled_indices = np.where(sender_mask)[0]
            if len(unlabeled_indices) > sender_budget:
                unlabeled_conf = max_conf[unlabeled_indices]
                top_k_local = np.argsort(unlabeled_conf)[-sender_budget:]
                top_k_global = unlabeled_indices[top_k_local]
                new_mask = np.zeros_like(sender_mask)
                new_mask[top_k_global] = True
                sender_mask = new_mask
            # else: use all unlabeled (budget >= available points)
        else:
            # Original threshold behavior
            sender_mask = sender_mask & (max_conf > threshold)

    elif mode == 'al_constant':
        if al_indices is not None and len(al_indices) > 0:
            al_mask = np.zeros(len(labeled_mask), dtype=bool)
            al_mask[np.asarray(al_indices, dtype=int)] = True
            sender_mask = sender_mask & al_mask
        else:
            print('[Propagation] Warning: al_constant mode but no al_indices — falling back to all unlabeled')

    elif mode == 'al_updated':
        if al_resample_fn is not None:
            new_indices = np.asarray(al_resample_fn(C), dtype=int)
            al_mask = np.zeros(len(labeled_mask), dtype=bool)
            al_mask[new_indices] = True
            sender_mask = sender_mask & al_mask
        else:
            print('[Propagation] Warning: al_updated mode but no al_resample_fn — falling back to all unlabeled')

    return sender_mask


def compute_receiver_mask(C, labeled_mask, mode='all', threshold=None, prev_mask=None, 
                         al_indices=None, al_resample_fn=None, normalization='sum', receiver_budget=None):
    """Which points may receive propagated information this iteration.

    Labeled points never receive.  Receiver modes:
        'all'             – every unlabeled point can receive (default)
        'high_confidence' – unlabeled with norm_C.max <= threshold can receive (low confidence)
                            Once a point crosses threshold it is frozen for remaining iterations
                            If receiver_budget is specified, select bottom K points by max(C) instead
        'al_constant'     – the AL picks from the last round, fixed for all iterations
        'al_updated'      – call al_resample_fn(C) each iteration to re-select via bayes_misp

    Args:
        C:         current C matrix (N, num_classes) numpy array
        labeled_mask: boolean (N,)
        mode: receiver mode ('all', 'high_confidence', 'al_constant', 'al_updated')
        threshold: threshold #2; norm_C.max > threshold → point is frozen (for 'high_confidence' mode)
        prev_mask: receiver mask from the previous iteration (None = initialise fresh)
        al_indices: pre-selected indices for 'al_constant' mode
        al_resample_fn: callable for 'al_updated' mode
        normalization: passed to normalize_C
        receiver_budget: optional int, number of receivers to select (for 'high_confidence' mode)

    Returns boolean mask (N,).
    """
    if prev_mask is None:
        prev_mask = ~labeled_mask  # initial: all unlabeled can receive
    
    if mode == 'all':
        # No filtering, all unlabeled can receive
        return prev_mask
    
    elif mode == 'high_confidence':
        # Points with LOW confidence can receive (below threshold)
        # Once a point crosses threshold it is frozen
        if threshold is None:
            return prev_mask
        
        norm = normalize_C(C, normalization)
        max_conf = norm.max(axis=1)
        
        if receiver_budget is not None:
            # Select bottom K points by confidence (most uncertain)
            unlabeled_indices = np.where(prev_mask)[0]
            if len(unlabeled_indices) > receiver_budget:
                unlabeled_conf = max_conf[unlabeled_indices]
                bottom_k_local = np.argsort(unlabeled_conf)[:receiver_budget]
                bottom_k_global = unlabeled_indices[bottom_k_local]
                new_mask = np.zeros_like(prev_mask)
                new_mask[bottom_k_global] = True
                return new_mask
            else:
                return prev_mask  # budget >= available points
        else:
            # Original threshold behavior: freeze points that exceed threshold
            exceeded = max_conf > threshold
            return prev_mask & ~exceeded  # once frozen, always frozen
    
    elif mode == 'al_constant':
        # Only AL-selected points can receive
        if al_indices is not None and len(al_indices) > 0:
            al_mask = np.zeros(len(labeled_mask), dtype=bool)
            al_mask[np.asarray(al_indices, dtype=int)] = True
            return prev_mask & al_mask
        else:
            print('[Propagation] Warning: al_constant receiver mode but no al_indices — falling back to all unlabeled')
            return prev_mask
    
    elif mode == 'al_updated':
        # Re-run AL selection each iteration
        if al_resample_fn is not None:
            new_indices = np.asarray(al_resample_fn(C), dtype=int)
            al_mask = np.zeros(len(labeled_mask), dtype=bool)
            al_mask[new_indices] = True
            return prev_mask & al_mask
        else:
            print('[Propagation] Warning: al_updated receiver mode but no al_resample_fn — falling back to all unlabeled')
            return prev_mask
    
    else:
        print(f'[Propagation] Warning: unknown receiver mode {mode!r} — falling back to all unlabeled')
        return prev_mask


def propagate_labels(C_matrix, K_matrix, labeled_mask, source_indices, source_labels, alpha,
                     num_iterations=1, sender_mode='all', sender_threshold=None,
                     receiver_threshold=None, receiver_mode='all', receiver_al_indices=None, 
                     receiver_al_resample_fn=None, receiver_budget=None,
                     al_indices=None, al_resample_fn=None,
                     normalization='sum', sender_budget=None, low_rank=None, decay_rate=1.0):
    """Propagate label information from source points to their K-weighted neighbors.

    Update rule per iteration, per point j, per class m:
        V_new[j, m] = α
                    + Σ_{s∈sources} K[s,j] · 1_{y_s=m}       (labeled contribution)
                    + Σ_{u∈can_send} K[u,j] · (C[u,m] − α)   (unlabeled propagation)

    At every iteration both masks are recomputed from current C:
      • can_send_mask    – determined by sender_mode / sender_threshold / sender_budget
      • can_receive_mask – determined by receiver_mode / receiver_threshold / receiver_budget

    Source (labeled / oracle) rows are always restored to their original values.

    Args:
        C_matrix:           (N, num_classes) confidence matrix — torch tensor or numpy
        K_matrix:           (N, N) kernel — sparse CSR, torch tensor, or numpy
        labeled_mask:       boolean (N,) — which points are labeled
        source_indices:     int array — fixed information sources (labeled or oracle)
        source_labels:      int array — class label per source point
        alpha:              baseline scalar
        num_iterations:     number of propagation rounds
        sender_mode:        'all' | 'high_confidence' | 'al_constant' | 'al_updated'
        sender_threshold:   threshold #1 for 'high_confidence' sender mode
        receiver_threshold: threshold #2 for 'high_confidence' receiver mode
        receiver_mode:      'all' | 'high_confidence' | 'al_constant' | 'al_updated'
        receiver_al_indices:     pre-selected indices for 'al_constant' receiver mode
        receiver_al_resample_fn: callable(C) -> indices for 'al_updated' receiver mode
        receiver_budget:    optional int, number of receivers (K) for 'high_confidence' mode
        al_indices:         pre-selected indices for 'al_constant' sender mode
        al_resample_fn:     callable(C) -> indices for 'al_updated' sender mode
        normalization:      'sum' | 'softmax'
        sender_budget:      optional int, number of senders (K) for 'high_confidence' mode
        low_rank:           optional int — when K is dense and GPU is unavailable, approximate
                            K ≈ Q·diag(λ)·Qᵀ using the top-`low_rank` eigenvectors (precomputed
                            once via scipy eigsh).  Each multiply becomes O(N·r·C) instead of
                            O(N²·C).  Ignored when K is sparse or CUDA is available.
        decay_rate:         float in [0, 1] — decay factor per iteration for unlabeled propagation.
                            At iteration i, the unlabeled propagation is scaled by decay_rate^i.
                            Default 1.0 (no decay). Values < 1.0 make later iterations propagate
                            less information from unlabeled points.

    Returns:
        C matrix (same type and device as input) after propagation.
    """
    import scipy.sparse as sp

    # --- type bookkeeping (round-trip tensor ↔ numpy) -------------------------
    is_tensor = torch.is_tensor(C_matrix)
    original_device = C_matrix.device if is_tensor else 'cpu'
    C_current = C_matrix.cpu().numpy() if is_tensor else np.array(C_matrix, copy=True)

    labeled_mask = np.asarray(
        labeled_mask.cpu() if torch.is_tensor(labeled_mask) else labeled_mask, dtype=bool)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    source_labels = np.asarray(source_labels, dtype=np.int64)

    n_samples, num_classes = C_current.shape
    labeled_indices = np.where(labeled_mask)[0]

    # --- K: build unified multiply and row-sum helpers ------------------------
    if sp.issparse(K_matrix):
        K_obj = K_matrix
        def _K_mul(X, row_mask=None): return K_obj @ X
        def _K_row_sum(idx): return K_obj[idx, :].sum(axis=0).A1
    else:
        K_obj = K_matrix.cpu().numpy() if torch.is_tensor(K_matrix) else np.asarray(K_matrix)
        K_obj = np.asarray(K_obj, dtype=np.float32)  # float32 BLAS is ~2x faster than float64

        # K is too large to fit in VRAM entirely; multiply in row-chunks so only
        # one tile (chunk_size × |active|) lives on GPU at a time.
        _K_gpu_chunk_size = 512   # rows per tile; tune down if still OOM
        print(f'[Propagation] Dense K backend: GPU chunked (chunk={_K_gpu_chunk_size}, '
              f'K={K_obj.nbytes / 1e9:.2f} GB on CPU)')
        logger.info(f'[Propagation] Dense K backend: GPU chunked (chunk={_K_gpu_chunk_size})')

        def _K_mul(X, row_mask=None):
            active = np.where(row_mask)[0] if (row_mask is not None and row_mask.sum() < len(row_mask)) else None
            X_src = X[active] if active is not None else X          # (|active|, C) or (N, C)
            K_cols = K_obj[:, active] if active is not None else K_obj   # (N, |active|) cpu
            X_t = torch.from_numpy(X_src.astype(np.float32)).cuda()     # (|active|, C) on GPU

            out = np.empty((K_cols.shape[0], X_src.shape[1]), dtype=np.float32)
            for start in range(0, K_cols.shape[0], _K_gpu_chunk_size):
                end = min(start + _K_gpu_chunk_size, K_cols.shape[0])
                K_tile = torch.from_numpy(K_cols[start:end]).cuda()     # (chunk, |active|)
                out[start:end] = (K_tile @ X_t).cpu().numpy()
            return out

        def _K_row_sum(idx): return K_obj[idx, :].sum(axis=0)

    # --- precompute labeled contribution (constant across iterations) ---------
    # labeled_contribution[j, m] = Σ_{s : y_s=m} K[s, j]
    labeled_contribution = np.zeros((n_samples, num_classes), dtype=np.float32)
    for cls in range(num_classes):
        src_cls = source_indices[source_labels == cls]
        if len(src_cls) > 0:
            labeled_contribution[:, cls] = _K_row_sum(src_cls)

    original_source_values = C_current[labeled_indices].copy()

    print(f'[Propagation] sources={len(source_indices)}, sender_mode={sender_mode!r}, '
          f'receiver_mode={receiver_mode!r}, iterations={num_iterations}, '
          f'sender_threshold={sender_threshold}, sender_budget={sender_budget}, '
          f'receiver_threshold={receiver_threshold}, receiver_budget={receiver_budget}, '
          f'decay_rate={decay_rate}')
    logger.info(f'[Propagation] sources={len(source_indices)}, sender_mode={sender_mode!r}, '
                f'receiver_mode={receiver_mode!r}, iterations={num_iterations}, '
                f'sender_budget={sender_budget}, receiver_budget={receiver_budget}, '
                f'decay_rate={decay_rate}')

    # --- propagation loop -----------------------------------------------------
    can_receive = None  # initialised on first call to compute_receiver_mask

    for i in range(num_iterations):
        can_send = compute_sender_mask(
            C_current, labeled_mask,
            mode=sender_mode, threshold=sender_threshold,
            al_indices=al_indices, al_resample_fn=al_resample_fn,
            normalization=normalization,
            sender_budget=sender_budget,
        )
        can_receive = compute_receiver_mask(
            C_current, labeled_mask,
            mode=receiver_mode, threshold=receiver_threshold, prev_mask=can_receive,
            al_indices=receiver_al_indices, al_resample_fn=receiver_al_resample_fn,
            normalization=normalization,
            receiver_budget=receiver_budget,
        )

        # Compute decay factor for this iteration (decay_rate^i)
        decay_factor = decay_rate ** i
        
        print(f'[Propagation] iter {i + 1}/{num_iterations}: '
              f'senders={can_send.sum()}, receivers={can_receive.sum()}, decay_factor={decay_factor:.4f}')
        logger.info(f'[Propagation] iter {i + 1}/{num_iterations}: '
                    f'senders={can_send.sum()}, receivers={can_receive.sum()}, decay_factor={decay_factor:.4f}')

        V_excess = (C_current - alpha) * can_send[:, None]
        unlabeled_propagation = _K_mul(V_excess, row_mask=can_send)
        V_new = alpha + labeled_contribution + decay_factor * unlabeled_propagation

        V_new[~can_receive] = C_current[~can_receive]   # keep frozen values
        V_new[labeled_indices] = original_source_values  # restore sources

        C_current = V_new

    if is_tensor:
        return torch.from_numpy(C_current).to(original_device)
    return C_current


def _build_propagation_cfg(cfg, al_obj, labeled_mask, train_labels, lSet=None, uSet=None, decay_rate=None):
    """Build kwargs dict for propagate_labels from cfg and al_obj state.

    Determines sources, sender mode, receiver mode, and all thresholds.
    lSet / uSet are required for al_updated sender/receiver modes (passed from get_lset_loader).
    
    Args:
        decay_rate: Optional float to override cfg.PROPAGATION_DECAY_RATE. If None, reads from config.
    
    Returns None with a warning if required data is missing.
    """
    if train_labels is None:
        print('[Propagation] Warning: train_labels not found, skipping propagation')
        return None

    labeled_mask_np = np.asarray(
        labeled_mask.cpu() if torch.is_tensor(labeled_mask) else labeled_mask, dtype=bool)

    normalization = getattr(cfg, 'CK_C_NORMALIZATION', 'sum')

    # --- sources: current labeled set -----------------------------------------
    source_indices = np.where(labeled_mask_np)[0]
    source_labels = train_labels[source_indices]

    # --- sender mode ----------------------------------------------------------
    sender_mode = getattr(cfg, 'PROPAGATION_SENDER_MODE', None) or 'all'
    sender_threshold = getattr(cfg, 'PROPAGATION_SENDER_THRESHOLD', 0.5)
    sender_budget = getattr(cfg, 'PROPAGATION_SENDER_BUDGET', None)

    # AL indices for 'al_constant' mode: always use the most recent round's picks
    al_indices = None
    if sender_mode == 'al_constant':
        al_indices = getattr(al_obj, 'propagation_sender_al_indices', None)
        if al_indices is None:
            print('[Propagation] Warning: al_constant mode but propagation_sender_al_indices not set — falling back to all unlabeled')

    # Resample fn for 'al_updated' mode:
    # After each propagation step, run the actual AL algorithm (e.g. bayes_misp) on the
    # updated C to pick the most informative unlabeled points as the next senders.
    # Those points pass their C values — not their labels — to their neighbors.
    al_resample_fn = None
    if sender_mode == 'al_updated':
        # Capture lSet and uSet once at build time — they are constant within this propagation run
        resample_lSet = np.asarray(lSet if lSet is not None else [], dtype=np.int64)
        resample_uSet = np.asarray(uSet if uSet is not None else np.where(~labeled_mask_np)[0], dtype=np.int64)

        def _resample_from_C(C, lSet=resample_lSet, uSet=resample_uSet):
            sf = al_obj.sampling_fn
            
            # Save original budget
            saved_budget_cfg = cfg.ACTIVE_LEARNING.BUDGET_SIZE
            saved_budget_sf = None
            if hasattr(sf, 'budgetSize'):
                saved_budget_sf = sf.budgetSize
            
            # Override budget if propagation_sender_budget is specified
            if sender_budget is not None:
                cfg.ACTIVE_LEARNING.BUDGET_SIZE = sender_budget
                if hasattr(sf, 'budgetSize'):
                    sf.budgetSize = sender_budget

            # Collect every object that holds a C_general (each may be on a different device)
            holders = {}
            if hasattr(sf, 'C_general'):
                holders['sampling_fn'] = sf
            if hasattr(al_obj, 'ck_manager') and al_obj.ck_manager is not None:
                holders['ck_manager'] = al_obj.ck_manager

            C_np = C.astype(np.float32)

            # Save all state that select_samples may mutate (fake AL must not affect the real round)
            originals_C = {name: obj.C_general for name, obj in holders.items()}
            saved = {
                'alpha': getattr(sf, 'alpha', None),
                'labeled_points_mask_general': (
                    sf.labeled_points_mask_general.clone()
                    if hasattr(sf, 'labeled_points_mask_general') else None
                ),
                'cum_labels_info': (
                    sf.cum_labels_info.clone()
                    if hasattr(sf, 'cum_labels_info') else None
                ),
                'beta_per_class': (
                    sf.beta_per_class.clone()
                    if getattr(sf, 'beta_per_class', None) is not None else None
                ),
                'effective_alpha_per_class': (
                    sf.effective_alpha_per_class.clone()
                    if getattr(sf, 'effective_alpha_per_class', None) is not None else None
                ),
            }

            for obj in holders.values():
                obj.C_general = torch.from_numpy(C_np).to(obj.C_general.device)

            try:
                activeSet, _ = al_obj.sample_from_uSet(None, lSet, uSet, None)
                indices = np.asarray(activeSet, dtype=int)
            except Exception as e:
                print(f'[Propagation] al_updated: sample_from_uSet failed ({e}), falling back to all unlabeled')
                indices = uSet
            finally:
                # Restore budget
                cfg.ACTIVE_LEARNING.BUDGET_SIZE = saved_budget_cfg
                if saved_budget_sf is not None:
                    sf.budgetSize = saved_budget_sf
                # Restore C matrices and other state
                for name, obj in holders.items():
                    obj.C_general = originals_C[name]
                if saved['labeled_points_mask_general'] is not None:
                    sf.labeled_points_mask_general = saved['labeled_points_mask_general']
                if saved['cum_labels_info'] is not None:
                    sf.cum_labels_info = saved['cum_labels_info']
                if saved['beta_per_class'] is not None:
                    sf.beta_per_class = saved['beta_per_class']
                if saved['effective_alpha_per_class'] is not None:
                    sf.effective_alpha_per_class = saved['effective_alpha_per_class']
                if saved['alpha'] is not None:
                    sf.alpha = saved['alpha']

            return indices

        al_resample_fn = _resample_from_C

    # --- receiver mode --------------------------------------------------------
    receiver_mode = getattr(cfg, 'PROPAGATION_RECEIVER_MODE', None) or 'all'
    receiver_threshold = getattr(cfg, 'PROPAGATION_RECEIVER_THRESHOLD', None)
    
    # Backward compatibility: fall back to legacy PROPAGATION_CONFIDENCE_THRESHOLD
    if receiver_threshold is None:
        legacy_threshold = getattr(cfg, 'PROPAGATION_CONFIDENCE_THRESHOLD', None)
        if legacy_threshold is not None:
            receiver_threshold = legacy_threshold
            # If using legacy threshold with 'all' mode, switch to 'high_confidence' for correct behavior
            if receiver_mode == 'all':
                receiver_mode = 'high_confidence'
                print(f'[Propagation] Using legacy PROPAGATION_CONFIDENCE_THRESHOLD={legacy_threshold} '
                      f'(auto-switched receiver_mode to high_confidence for backward compatibility)')
    
    receiver_budget = getattr(cfg, 'PROPAGATION_RECEIVER_BUDGET', None)

    # AL indices for 'al_constant' receiver mode: use the most recent round's picks
    receiver_al_indices = None
    if receiver_mode == 'al_constant':
        receiver_al_indices = getattr(al_obj, 'propagation_receiver_al_indices', None)
        if receiver_al_indices is None:
            # Fall back to sender AL indices if receiver indices not set
            receiver_al_indices = getattr(al_obj, 'propagation_sender_al_indices', None)
            if receiver_al_indices is None:
                print('[Propagation] Warning: al_constant receiver mode but propagation_receiver_al_indices not set — falling back to all unlabeled')

    # Resample fn for 'al_updated' receiver mode:
    # After each propagation step, run the actual AL algorithm (e.g. bayes_misp) on the
    # updated C to pick the most informative unlabeled points as receivers.
    receiver_al_resample_fn = None
    if receiver_mode == 'al_updated':
        # Capture lSet and uSet once at build time — they are constant within this propagation run
        resample_lSet = np.asarray(lSet if lSet is not None else [], dtype=np.int64)
        resample_uSet = np.asarray(uSet if uSet is not None else np.where(~labeled_mask_np)[0], dtype=np.int64)

        def _resample_from_C_receiver(C, lSet=resample_lSet, uSet=resample_uSet):
            sf = al_obj.sampling_fn
            
            # Save original budget
            saved_budget_cfg = cfg.ACTIVE_LEARNING.BUDGET_SIZE
            saved_budget_sf = None
            if hasattr(sf, 'budgetSize'):
                saved_budget_sf = sf.budgetSize
            
            # Override budget if propagation_receiver_budget is specified
            if receiver_budget is not None:
                cfg.ACTIVE_LEARNING.BUDGET_SIZE = receiver_budget
                if hasattr(sf, 'budgetSize'):
                    sf.budgetSize = receiver_budget

            # Collect every object that holds a C_general (each may be on a different device)
            holders = {}
            if hasattr(sf, 'C_general'):
                holders['sampling_fn'] = sf
            if hasattr(al_obj, 'ck_manager') and al_obj.ck_manager is not None:
                holders['ck_manager'] = al_obj.ck_manager

            C_np = C.astype(np.float32)

            # Save all state that select_samples may mutate (fake AL must not affect the real round)
            originals_C = {name: obj.C_general for name, obj in holders.items()}
            saved = {
                'alpha': getattr(sf, 'alpha', None),
                'labeled_points_mask_general': (
                    sf.labeled_points_mask_general.clone()
                    if hasattr(sf, 'labeled_points_mask_general') else None
                ),
                'cum_labels_info': (
                    sf.cum_labels_info.clone()
                    if hasattr(sf, 'cum_labels_info') else None
                ),
                'beta_per_class': (
                    sf.beta_per_class.clone()
                    if getattr(sf, 'beta_per_class', None) is not None else None
                ),
                'effective_alpha_per_class': (
                    sf.effective_alpha_per_class.clone()
                    if getattr(sf, 'effective_alpha_per_class', None) is not None else None
                ),
            }

            for obj in holders.values():
                obj.C_general = torch.from_numpy(C_np).to(obj.C_general.device)

            try:
                activeSet, _ = al_obj.sample_from_uSet(None, lSet, uSet, None)
                indices = np.asarray(activeSet, dtype=int)
            except Exception as e:
                print(f'[Propagation] al_updated receiver: sample_from_uSet failed ({e}), falling back to all unlabeled')
                indices = uSet
            finally:
                # Restore budget
                cfg.ACTIVE_LEARNING.BUDGET_SIZE = saved_budget_cfg
                if saved_budget_sf is not None:
                    sf.budgetSize = saved_budget_sf
                # Restore C matrices and other state
                for name, obj in holders.items():
                    obj.C_general = originals_C[name]
                if saved['labeled_points_mask_general'] is not None:
                    sf.labeled_points_mask_general = saved['labeled_points_mask_general']
                if saved['cum_labels_info'] is not None:
                    sf.cum_labels_info = saved['cum_labels_info']
                if saved['beta_per_class'] is not None:
                    sf.beta_per_class = saved['beta_per_class']
                if saved['effective_alpha_per_class'] is not None:
                    sf.effective_alpha_per_class = saved['effective_alpha_per_class']
                if saved['alpha'] is not None:
                    sf.alpha = saved['alpha']

            return indices

        receiver_al_resample_fn = _resample_from_C_receiver

    # --- alpha ----------------------------------------------------------------
    if hasattr(al_obj, 'ck_manager') and al_obj.ck_manager is not None:
        alpha = al_obj.ck_manager.alpha
    else:
        alpha = getattr(getattr(al_obj, 'sampling_fn', None), 'alpha', getattr(cfg, 'ALPHA', 1.0))

    # --- decay rate -----------------------------------------------------------
    # Use passed argument if provided, otherwise read from config
    if decay_rate is None:
        decay_rate = getattr(cfg, 'PROPAGATION_DECAY_RATE', 1.0)

    return dict(
        labeled_mask=labeled_mask,
        source_indices=source_indices,
        source_labels=source_labels,
        alpha=alpha,
        num_iterations=getattr(cfg, 'PROPAGATION_ITERATIONS', 1),
        sender_mode=sender_mode,
        sender_threshold=sender_threshold,
        sender_budget=sender_budget,
        receiver_mode=receiver_mode,
        receiver_threshold=receiver_threshold,
        receiver_budget=receiver_budget,
        receiver_al_indices=receiver_al_indices,
        receiver_al_resample_fn=receiver_al_resample_fn,
        al_indices=al_indices,
        al_resample_fn=al_resample_fn,
        normalization=normalization,
        decay_rate=decay_rate,
    )


def _otsu_threshold(values):
    """
    Otsu's method: find the threshold that maximally separates a 1D distribution
    into two populations by maximizing inter-class variance. Zero parameters.

    Args:
        values: 1D numpy array of non-negative scores

    Returns:
        threshold: optimal split point
    """
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0

    best_thresh = 0.0
    best_variance = -1.0

    # Use 256 evenly spaced candidate thresholds for efficiency
    num_bins = min(256, n)
    candidates = np.linspace(sorted_vals[0], sorted_vals[-1], num_bins + 2)[1:-1]

    for t in candidates:
        fg = sorted_vals[sorted_vals > t]
        bg = sorted_vals[sorted_vals <= t]
        if len(fg) == 0 or len(bg) == 0:
            continue
        w_fg = len(fg) / n
        w_bg = len(bg) / n
        inter_var = w_fg * w_bg * (fg.mean() - bg.mean()) ** 2
        if inter_var > best_variance:
            best_variance = inter_var
            best_thresh = t

    return best_thresh


def compute_bayesian_distill_mask(C_matrix, ck_alpha):
    """
    Compute distillation mask and weights from pure kernel evidence, decoupled
    from the Veracity Matrix prior (ck_alpha).

    Includes ALL points with nonzero evidence — no threshold needed.
    Confidence weights ensure high-evidence points dominate the distillation
    loss while low-evidence points contribute proportionally to their
    evidence strength. The effective distillation set size emerges from
    the weight distribution, not a hard cutoff.

    Args:
        C_matrix: Raw Veracity Matrix, shape (N, M), rows are Dirichlet concentrations
                  where each entry is ck_alpha + accumulated kernel evidence
        ck_alpha: The Veracity Matrix prior value (CK_ALPHA) to strip out

    Returns:
        distill_mask: Boolean mask — True for all points with nonzero evidence
        confidence: Per-point weights normalized to [0, 1], shape (N,)
    """
    # Strip the CK prior to get pure kernel evidence
    evidence = (C_matrix.float() - ck_alpha).clamp(min=0.0)

    # Evidence peakedness: max class evidence relative to total evidence
    total_evidence = evidence.sum(dim=1)
    max_evidence = evidence.max(dim=1).values
    peakedness = torch.where(
        total_evidence > 0,
        max_evidence / total_evidence,
        torch.zeros_like(total_evidence)
    )

    # Confidence = peakedness * max_evidence, normalized to [0, 1]
    confidence = peakedness * max_evidence
    conf_max = confidence.max()
    if conf_max > 0:
        confidence = confidence / conf_max

    # Include all points with any evidence — weights handle the rest
    distill_mask = total_evidence > 0

    num_selected = distill_mask.sum().item()
    total_points = C_matrix.shape[0]
    effective_size = confidence.sum().item()
    print(f"[Bayesian Distillation] ck_alpha={ck_alpha} stripped, no threshold")
    print(f"[Bayesian Distillation] {num_selected} / {total_points} points have evidence "
          f"({100.0 * num_selected / total_points:.1f}%)")
    print(f"[Bayesian Distillation] weight stats: mean={confidence[distill_mask].mean().item():.4f}, "
          f"median={confidence[distill_mask].median().item():.4f}, "
          f"effective set size (sum of weights)={effective_size:.0f}")

    return distill_mask, confidence


def prepare_distillation_targets(al_obj, threshold, cfg, lSet=None, uSet=None, decay_rate=None):
    """
    Prepare soft targets and distillation mask from the C matrix.

    Args:
        al_obj: ActiveLearning object containing the sampling function with C_general
        threshold: Minimum max-probability to use soft targets for distillation (normalized)
        cfg: Configuration object (used to check PSEUDO_LABEL_CLASS_THRESHOLD, CK_C_NORMALIZATION, and RAW_DISTILLATION_THRESHOLD)
        lSet: current labeled set indices (passed through to propagation for al_updated sender mode)
        uSet: current unlabeled set indices (passed through to propagation for al_updated sender mode)
        decay_rate: Optional float to override cfg.PROPAGATION_DECAY_RATE for propagation. If None, reads from config.

    Returns:
        raw_scores: Raw C matrix values (logit-like scores) as torch tensor on GPU, shape (N_total, num_classes).
                    Temperature-scaled softmax normalization is applied in distillation_loss.
        distill_mask: Boolean mask where both normalized and raw thresholds are passed, shape (N_total,) as torch tensor on GPU
        distill_weights: Per-sample weights based on max probability (for loss weighting), shape (N_total,) as torch tensor on GPU
    """
    # Use CK manager's C_general if it exists (for separate matrices), otherwise use sampling_fn's
    # Keep on GPU for efficiency
    if hasattr(al_obj, 'ck_manager') and al_obj.ck_manager is not None:
        C_matrix = al_obj.ck_manager.C_general
        K_matrix = al_obj.ck_manager.K_general
        labeled_mask = al_obj.ck_manager.labeled_mask
        use_ck_manager = True
    else:
        C_matrix = al_obj.sampling_fn.C_general
        K_matrix = getattr(al_obj.sampling_fn, 'K_general', None)
        labeled_mask = getattr(al_obj.sampling_fn, 'labeled_points_mask_general', None)
        use_ck_manager = False

    # Apply label propagation if enabled
    if getattr(cfg, 'PROPAGATE_C_LABELS', False):
        if K_matrix is not None and labeled_mask is not None:
            train_labels = np.array(al_obj.train_labels) if hasattr(al_obj, 'train_labels') else None
            prop_kwargs = _build_propagation_cfg(cfg, al_obj, labeled_mask, train_labels, lSet=lSet, uSet=uSet, decay_rate=decay_rate)
            if prop_kwargs is not None:
                print(f'[Distillation] Applying label propagation: {cfg.PROPAGATION_ITERATIONS} iterations')
                C_matrix = propagate_labels(C_matrix, K_matrix, **prop_kwargs)
                print('[Distillation] Label propagation complete.')
        else:
            print('[Distillation] Warning: K_matrix or labeled_mask not found, skipping propagation')

    # --- Bayesian distillation path ---
    if getattr(cfg, 'BAYESIAN_DISTILLATION', False):
        # Retrieve the CK alpha (Veracity prior) to strip from C_matrix
        if hasattr(al_obj, 'ck_manager') and al_obj.ck_manager is not None:
            ck_alpha = al_obj.ck_manager.alpha
        else:
            ck_alpha = getattr(al_obj.sampling_fn, 'alpha', getattr(cfg, 'ALPHA', 0.5))

        distill_mask, bayesian_confidence = compute_bayesian_distill_mask(
            C_matrix, ck_alpha=ck_alpha)

        num_distill_points = distill_mask.sum().item()
        total_points = distill_mask.shape[0]
        print(f"[Distillation] Bayesian: {num_distill_points} / {total_points} points "
              f"({100.0 * num_distill_points / total_points:.1f}%)")
        logger.info(f"[Distillation] Bayesian: {num_distill_points} / {total_points} points "
                     f"({100.0 * num_distill_points / total_points:.1f}%)")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return C_matrix.to(device), distill_mask.to(device), bayesian_confidence.to(device)

    # --- Legacy distillation path ---
    # Now continue with original logic using the (possibly propagated) C_matrix
    if hasattr(al_obj, 'ck_manager') and al_obj.ck_manager is not None:
        use_ck_manager = True
    else:
        use_ck_manager = False

    # Normalize C matrix based on configuration
    ck_c_normalization = getattr(cfg, 'CK_C_NORMALIZATION', 'sum')
    if use_ck_manager and ck_c_normalization == 'softmax':
        # Softmax normalization: apply temperature scaling with alpha
        soft_probs = F.softmax(C_matrix, dim=1)
    else:
        # Sum normalization: divide by row sum (default)
        soft_probs = C_matrix / torch.sum(C_matrix, dim=1, keepdim=True)

    max_probs = soft_probs.max(dim=1).values
    max_classes = soft_probs.argmax(dim=1)

    # Apply per-class threshold on normalized values if enabled
    if cfg.PSEUDO_LABEL_CLASS_THRESHOLD:
        # Compute per-class mean probabilities
        if cfg.PSEUDO_LABEL_CLASS_THRESHOLD_TOPK:
            # Use top-k mean per class (k = num_points / num_classes)
            num_points = soft_probs.shape[0]
            num_classes = soft_probs.shape[1]
            k = num_points // num_classes

            # For each class, select top-k points and compute mean
            soft_probs_per_class = torch.zeros(num_classes, device=soft_probs.device)
            for c in range(num_classes):
                class_probs = soft_probs[:, c]
                top_k_values = torch.topk(class_probs, k=min(k, num_points)).values
                soft_probs_per_class[c] = top_k_values.mean()
        else:
            # Use mean over all points
            soft_probs_per_class = soft_probs.mean(dim=0)

        # Normalize by max to get relative class weights
        soft_probs_per_class_norm = soft_probs_per_class / (soft_probs_per_class.max() + 1e-8)
        # Create per-class threshold vector
        class_threshold_vec = threshold * soft_probs_per_class_norm
        # Apply appropriate threshold based on max class for each point
        point_thresholds = class_threshold_vec[max_classes]
        distill_mask_normalized = max_probs > point_thresholds
    else:
        # Use scalar threshold for all points
        distill_mask_normalized = max_probs > threshold

    # Apply raw C matrix threshold (before normalization)
    raw_threshold = getattr(cfg, 'RAW_DISTILLATION_THRESHOLD', 0.0)
    if raw_threshold > 0.0:
        # Get max values from raw C matrix
        raw_max_values = C_matrix.max(dim=1).values
        distill_mask_raw = raw_max_values > raw_threshold
        # Combine both thresholds: both must pass
        distill_mask = distill_mask_normalized & distill_mask_raw
        print(f"Distillation filtering: {distill_mask_normalized.sum().item()} passed normalized threshold, "
              f"{distill_mask_raw.sum().item()} passed raw threshold, "
              f"{distill_mask.sum().item()} passed both")
    else:
        # Only use normalized threshold
        distill_mask = distill_mask_normalized

    # Log distillation statistics
    total_points = distill_mask.shape[0]
    num_distill_points = distill_mask.sum().item()
    print(f"[Distillation] {num_distill_points} / {total_points} points will participate in distillation ({100.0 * num_distill_points / total_points:.1f}%)")
    logger.info(f"[Distillation] {num_distill_points} / {total_points} points for distillation ({100.0 * num_distill_points / total_points:.1f}%)")

    # Return raw C values, mask, and max probabilities for per-sample weighting as GPU tensors
    # Normalization with temperature happens in distillation_loss
    # Keep on GPU to avoid CPU-GPU transfer overhead during training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return C_matrix.to(device), distill_mask.to(device), max_probs.to(device)


def _compute_grad_norm(model):
    """Compute the total L2 norm of gradients across all model parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--exp-name', help='Experiment Name', required=True, type=str)
    parser.add_argument('--al', help='AL Method', required=True, type=str) ## active learning type -> write TypiClust_rp\_dc
    parser.add_argument('--budget', help='Budget Per Round', required=True, type=int) ## in each iteration how much labels can I get
    parser.add_argument('--initial_size', help='Size of the initial random labeled set', default=0, type=int)
    parser.add_argument('--seed', help='Random seed', default=1, type=int)
    parser.add_argument('--finetune', help='Whether to continue with existing model between rounds', type=str2bool, default=False) # train the eval model from last ckpt and not from scratch (between iters)
    parser.add_argument('--eval_model_type', help='eval_model_type', type=str) # train the eval model from features not images
    # parser.add_argument('--linear_from_features', help='Whether to use a linear layer from self-supervised features', action='store_true') # train the eval model from features not images
    parser.add_argument('--initial_delta', help='Relevant only for ProbCover and DCoM', default=0.6, type=float)
    parser.add_argument('--initial_sigma', default=1.0, type=float)
    parser.add_argument('--kernel_type', default='rbf', type=str)
    parser.add_argument('--diff_method', default='abs_diff', type=str)
    parser.add_argument('--confidence_method', default='margin', type=str)
    parser.add_argument('--cont_method', default='positive', type=str)
    parser.add_argument('--distribution_cont_weight_method', default='weighted', type=str)
    parser.add_argument('--c_normalization', default='sum', type=str, choices=['sum', 'softmax'],
                        help='Normalization method for C matrix: sum (divide by row sum) or softmax')
    parser.add_argument('--class_weighting_method', default='none', type=str)
    parser.add_argument('--k_logistic', default=50, type=int)
    parser.add_argument('--a_logistic', default=0.8, type=float)
    parser.add_argument('--K_sparsity_threshold', default=0.0, type=float)
    parser.add_argument('--pseudo_labels_threshold', default=1.0, type=float)
    parser.add_argument('--train_pseudo_labels', action='store_true')
    parser.add_argument('--alpha_upper_bound', default=1.0, type=float)
    parser.add_argument('--alpha_lower_bound', default=1.0, type=float)
    parser.add_argument('--local_alpha', action='store_true')
    parser.add_argument('--local_alpha_oracle_method', default='entropy', type=str)
    parser.add_argument('--use_k_top50_mask', action='store_true')
    parser.add_argument('--update_k_matrix', action='store_true')
    parser.add_argument('--update_k_matrix_factor', default=1.0, type=float)
    parser.add_argument('--update_c_matrix', action='store_true')
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--ck_alpha', default=None, type=float, 
                        help='Alpha value for CK manager (separate from query method alpha). If not provided, uses --alpha value')
    parser.add_argument('--ck_c_normalization', default='sum', type=str, choices=['sum', 'softmax'],
                        help='Normalization method for CK manager C matrix: sum (divide by row sum) or softmax')
    parser.add_argument('--ck_K_sparsity_threshold', default=0.0, type=float,
                        help='K sparsity threshold for CK manager (separate from query method). If not provided, uses --K_sparsity_threshold value')
    parser.add_argument('--ck_sparse_K', default=None, type=str2bool, nargs='?', const=True,
                        help='Use sparse K matrix for CK manager (separate from query method). If not provided, uses --sparse_K value')
    parser.add_argument('--ck_sigma', default=None, type=float,
                        help='Sigma value for CK manager K matrix (for RBF kernel). If not provided, uses --initial_sigma value')
    parser.add_argument('--ck_k', default=7, type=int,
                        help='Number of neighbors for CkNN kernel bandwidth (k-th NN distance). Only used when --kernel_type=cknn')
    parser.add_argument('--local_rbf_alpha', default=1.0, type=float,
                        help='Alpha parameter for degree normalization in local_rbf kernel: K_ij / (degree_i * degree_j)^(alpha/2). Default: 1.0')
    parser.add_argument('--degree_normalization_method', default='none', type=str, choices=['none', 'sum', 'max', 'log'],
                        help='Method to normalize degree values in local_rbf kernel before applying transformation. '
                             'none: no normalization + power (default), sum: sum normalization + no power, '
                             'max: max normalization + no power, log: no normalization + log (instead of power)')
    parser.add_argument('--sparse_K', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--high_budget', action='store_true')
    parser.add_argument('--norm_importance', action='store_true')
    parser.add_argument('--sparse_ds', action='store_true')
    parser.add_argument('--decrease_alpha', action='store_true')
    parser.add_argument('--max_iter', default=32, type=int)
    parser.add_argument('--max_epoch', default=None, type=int,
                        help='Override training epochs per AL round (default: set by eval_model_type)')
    parser.add_argument('--start_iter', default=0, type=int)
    parser.add_argument('--eval_frequency', default=1, type=int)
    parser.add_argument('--pseudo_label_weight', default=0.5, type=float)
    parser.add_argument('--pseudo_label_weighting_func', choices=['constant', 'dynamic'], default='constant', type=str)
    parser.add_argument('--pseudo_label_class_threshold', action='store_true')
    parser.add_argument('--pseudo_label_class_threshold_topk', action='store_true',
                        help='Use top-k mean per class (k = num_points / num_classes) instead of mean over all points')
    parser.add_argument('--switch_alpha_low_to_high', action='store_true')
    parser.add_argument('--switch_alpha_high_to_low', action='store_true')
    parser.add_argument('--switch_alpha_alltime', action='store_true')
    parser.add_argument('--alpha_init_mode', default='constant', type=str,
                        choices=['constant', 'from_sparsity', 'from_vector'],
                        help='Alpha initialization mode: '
                             'constant (use --alpha value), '
                             'from_sparsity (alpha = K_sparsity_threshold / num_classes), '
                             'from_vector (load per-point K sorted vector and divide by num_classes)')
    parser.add_argument('--alpha_vector_path', default="/cs/labs/daphna/itai.david/py_repos/TypiClust/data/K_sorted_by_class_size/cifar100_rbf_kernel_sigma_1.npy", type=str,
                        help='Path to .npy file with per-point K sorted values (used with --alpha_init_mode from_vector)')
    parser.add_argument('--nn_fusion', action='store_true', help='Enable NN-C matrix fusion mode')
    parser.add_argument('--nn_fusion_start_round', default=10, type=int, help='Round to start NN fusion (default: 10)')
    parser.add_argument('--nn_fusion_interval', default=10, type=int, help='Interval between NN fusion rounds (default: 10)')
    parser.add_argument('--distillation_training', action='store_true', help='Enable distillation training from C matrix')
    parser.add_argument('--bayesian_distillation', action='store_true',
                        help='Use Bayesian credible interval criterion instead of fixed threshold. '
                             'Eliminates --distillation_threshold; uses Beta posterior on Dirichlet concentrations.')
    parser.add_argument('--bayesian_delta', default=0.05, type=float,
                        help='Confidence level for Bayesian credible interval (default: 0.05 = 95%% CI)')
    parser.add_argument('--bayesian_ref_prior', default=None, type=float,
                        help='Per-class reference prior for Bayesian distillation criterion. '
                             'Default: Perks prior (1/M). Higher values = stricter criterion (fewer points).')
    parser.add_argument('--distillation_threshold', default=0.25, type=float, help='Min softmax max-probability of C matrix to use soft targets for distillation')
    parser.add_argument('--raw_distillation_threshold', default=0.0, type=float, help='Min raw C matrix max value (before normalization) to use soft targets for distillation')
    parser.add_argument('--distillation_temperature', default=1.0, type=float, help='Temperature for softening distributions in distillation (higher = softer)')
    parser.add_argument('--distill_factor', default=8.0, type=float, help='Scaling factor for the distillation loss: loss = CE + distill_factor * KL')
    parser.add_argument('--distill_soft_target_normalization', default='power', type=str, choices=['power', 'softmax'],
                        help='Normalization method for soft targets in distillation: power (power-based temperature scaling) or softmax (softmax with temperature)')
    parser.add_argument('--alpha_decay_gamma', default=0.0, type=float,
                        help='Decay rate for per-class alpha: beta_m = 1/(1 + gamma * I_m). '
                             '0 disables decay (constant alpha).')
    parser.add_argument('--calc_method', default='max', type=str,
                        help='Calculation method for point total contribution: '
                             'max (use max C value), '
                             'entropy (use entropy of C values).')
    parser.add_argument('--switch_method_at_round', default=None, type=int,
                        help='Round number at which to switch AL methods (e.g., 10 to switch at episode 10)')
    parser.add_argument('--switch_to_method', default=None, type=str,
                        help='AL method to switch to (e.g., margin, entropy, bayes_misp)')
    
    # Label propagation arguments
    parser.add_argument('--propagate_c_labels', action='store_true',
                        help='Enable label propagation preprocessing for C matrix before distillation')
    parser.add_argument('--propagation_iterations', default=1, type=int,
                        help='Number of label propagation iterations (default: 1)')
    parser.add_argument('--propagation_confidence_threshold', default=None, type=float,
                        help='[LEGACY] Confidence threshold for receiver freezing (use --propagation_receiver_threshold instead). '
                             'For backward compatibility: automatically uses high_confidence receiver mode.')
    
    # Propagation sender filtering arguments
    parser.add_argument('--propagation_sender_mode', default=None, type=str,
                        choices=['high_confidence', 'al_constant', 'al_updated'],
                        help=(
                            'Sender filtering mode: '
                            'high_confidence — unlabeled points with norm_C.max > sender_threshold; '
                            'al_constant — last round\'s AL picks, fixed for all iterations; '
                            'al_updated — re-run bayes_misp on updated C at each iteration '
                            '(al_constant and al_updated are equivalent when propagation_iterations=1)'
                        ))
    parser.add_argument('--propagation_sender_threshold', default=0.5, type=float,
                        help='Confidence threshold for high_confidence sender mode (default: 0.5)')
    parser.add_argument('--propagation_sender_budget', default=None, type=int,
                        help='Number of sender points (K) for propagation. '
                             'For high_confidence mode: select top K points by max(C) instead of threshold. '
                             'For al_updated mode: budget for AL selection during propagation.')
    
    # Propagation receiver filtering arguments
    parser.add_argument('--propagation_receiver_mode', default=None, type=str,
                        choices=['all', 'high_confidence', 'al_constant', 'al_updated'],
                        help=(
                            'Receiver filtering mode: '
                            'all — all unlabeled points can receive (default); '
                            'high_confidence — unlabeled points with norm_C.max <= receiver_threshold can receive; '
                            'al_constant — last round\'s AL picks, fixed for all iterations; '
                            'al_updated — re-run bayes_misp on updated C at each iteration '
                            '(al_constant and al_updated are equivalent when propagation_iterations=1)'
                        ))
    parser.add_argument('--propagation_receiver_threshold', default=None, type=float,
                        help='Confidence threshold for high_confidence receiver mode (points exceeding this are frozen)')
    parser.add_argument('--propagation_receiver_budget', default=None, type=int,
                        help='Number of receiver points (K) for propagation. '
                             'For high_confidence mode: select bottom K points by max(C) (most uncertain). '
                             'For al_updated mode: budget for AL selection during propagation.')
    parser.add_argument('--propagation_decay_rate', default=1.0, type=float,
                        help='Decay rate for propagation strength per iteration. '
                             'At iteration i, unlabeled propagation is scaled by decay_rate^i. '
                             'Default 1.0 (no decay). Values < 1.0 make later iterations propagate less information.')
    
    parser.add_argument('--fixed_propagation_v1', action='store_true',
                        help='Experiment tag only (stored in cfg/stats; does not change training behavior)')

    parser.add_argument('--fixed_initial_set_dir', default=None, type=str,
                        help='Directory containing pre-generated initial labeled-set files '
                             '(random_1c_<dataset>_seed<N>.npy). When --al random_1c and '
                             '--seed is in [0,1,2,3] and the matching file exists, it is '
                             'loaded instead of running random_1c. '
                             'Default: auto-detected as data/fixed_initial_sets/ '
                             'relative to the repository root.')

    parser.add_argument('--fixed_distill_mlp', action='store_true', help='fixed_distill_mlp')
    parser.add_argument('--monitor_grad_norms', action='store_true', 
                        help='Enable gradient norm monitoring during distillation training (adds overhead)')
    parser.add_argument('--grad_norm_log_freq', default=500, type=int,
                        help='Frequency (in iterations) to log gradient norms when monitoring is enabled (default: 500)')


    return parser


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
        (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )


def main(cfg):
    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': cfg.DATA_LOADER.NUM_WORKERS, 'pin_memory': cfg.DATA_LOADER.PIN_MEMORY} if use_cuda else {}

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)

    # Using specific GPU
    # os.environ['NVIDIA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # print("Using GPU : {}.\n".format(cfg.GPU_ID))

    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = os.path.join(os.path.abspath('../..'), cfg.OUT_DIR)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)
    # Create "DATASET/MODEL TYPE" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir)
    # Creating the experiment directory inside the dataset specific directory 
    # all logs, labeled, unlabeled, validation sets are stroed here 
    # E.g., output/CIFAR10/resnet18/{timestamp or cfg.EXP_NAME based on arguments passed}
    if cfg.EXP_NAME == 'auto':
        # now = datetime.now()
        # exp_suffix = f'{now.year}_{now.month}_{now.day}_{now.hour:02}{now.minute:02}{now.second:02}_{now.microsecond}'
        # exp_dir = f'{cfg.DATASET.NAME}_{cfg.MODEL.TYPE}_{exp_suffix}'
        exp_prefix = ""
    else:
        exp_prefix = f"{cfg.EXP_NAME}_"

    now = datetime.now()
    day_dir = f'{now.year}_{now.month}_{now.day}'
    exp_suffix = f'{now.year}_{now.month}_{now.day}_{now.hour:02}{now.minute:02}{now.second:02}_{now.microsecond}'
    exp_name = f'{exp_prefix}{cfg.DATASET.NAME}_{cfg.ACTIVE_LEARNING.SAMPLING_FN}_{cfg.EVAL_MODEL_TYPE}_{exp_suffix}'

    full_day_dir = os.path.join(dataset_out_dir, day_dir)
    if not os.path.exists(full_day_dir):
        os.mkdir(full_day_dir)

    exp_dir = os.path.join(full_day_dir, exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        print("Experiment Directory is {}.\n".format(exp_dir))
    else:
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING DATA AND MODEL ========\n")
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)

    # train_data.targets = list(np.load("/cs/labs/daphna/itai.david/py_repos/TypiClust/new_embeddings/self_label/train_labels.npy"))
    # test_data.targets = list(np.load("/cs/labs/daphna/itai.david/py_repos/TypiClust/new_embeddings/self_label/test_labels.npy"))


    if cfg.SPARSE_DS:
        train_spare_inds = np.sort(np.load(f"/cs/labs/daphna/itai.david/py_repos/TypiClust/results/t-sne/{cfg['DATASET']['NAME']}_train_sparse_indices.npy"))
        test_spare_inds = np.sort(np.load(f"/cs/labs/daphna/itai.david/py_repos/TypiClust/results/t-sne/{cfg['DATASET']['NAME']}_test_sparse_indices.npy"))

        train_data.features = train_data.features[train_spare_inds.astype(int)]
        train_data.targets = np.array(train_data.targets)[train_spare_inds].tolist()
        train_data.data = train_data.data[train_spare_inds.astype(int)]

        test_data.features = test_data.features[test_spare_inds]
        test_data.targets = np.array(test_data.targets)[test_spare_inds].tolist()
        test_data.data = test_data.data[test_spare_inds.astype(int)]
    if cfg.TRAIN_PSEUDO_LABELS:
        pseudo_train_data, pseudo_train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    else:
        pseudo_train_data, pseudo_train_size = None, None
    cfg.ACTIVE_LEARNING.INIT_L_RATIO = args.initial_size / train_size
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))

    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(train_split_ratio=cfg.ACTIVE_LEARNING.INIT_L_RATIO, \
        val_split_ratio=cfg.DATASET.VAL_RATIO, data=train_data, seed_id=cfg.RNG_SEED, save_dir=cfg.EXP_DIR)

    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path

    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, \
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH)
    # model = model_builder.build_model(cfg).cuda()
    model = mh.get_model(cfg)
    if cfg.HIGH_BUDGET:
        seed = 42
        rng = np.random.default_rng(seed)
        lset_indices_from_uset = rng.choice(uSet.size, size=10000, replace=False)
        lset_mask = np.zeros_like(uSet, dtype=bool)
        lset_mask[lset_indices_from_uset] = True
        lSet = uSet[lset_mask]
        uSet = uSet[~lset_mask]

    train_labels = train_data.targets
    al_obj = ActiveLearning(data_obj, cfg, train_labels, lSet)
    
    al_obj.propagation_sender_al_indices = None
    al_obj.propagation_receiver_al_indices = None
    
    # Initialize CK manager if needed for pseudo-labeling or distillation
    needs_ck_matrices = (
        cfg.TRAIN_PSEUDO_LABELS or 
        getattr(cfg, 'DISTILLATION_TRAINING', False)
    )
    if needs_ck_matrices:
        print("======== INITIALIZING CK MANAGER ========")
        logger.info("Initializing CK matrix manager for pseudo-labeling/distillation")
        from pycls.al.ck_matrix_manager import CKMatrixManager
        ck_manager = CKMatrixManager(cfg, data_obj, train_labels, lSet)
        al_obj.attach_ck_manager(ck_manager)
        print(f"CK Manager initialized with alpha={ck_manager.alpha}, device={ck_manager.device}")
        logger.info(f"CK Manager initialized with alpha={ck_manager.alpha}, device={ck_manager.device}")
        print("=========================================\n")
    
    if len(lSet) == 0:
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ['dcom']:
            print('Labeled Set is Empty - Create and save the first delta values list')
            lSet_deltas = [str(cfg.ACTIVE_LEARNING.INITIAL_DELTA)] * cfg.ACTIVE_LEARNING.BUDGET_SIZE
            cfg.ACTIVE_LEARNING.DELTA_LST = lSet_deltas
            delta_avg_lst.append(cfg.ACTIVE_LEARNING.INITIAL_DELTA)

        print('Labeled Set is Empty - Sampling an Initial Pool')
        _fixed_dir = getattr(cfg, 'FIXED_INITIAL_SET_DIR', None)
        if _fixed_dir is None:
            # Default: data/fixed_initial_sets/ relative to repository root
            _fixed_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'fixed_initial_sets')
            )
        _fixed_path = os.path.join(_fixed_dir, f"random_1c_{cfg.DATASET.NAME}_seed{cfg.RNG_SEED}.npy")
        _use_fixed = (
            cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() == 'random_1c' and
            cfg.RNG_SEED in [0, 1, 2, 3, 4] and
            os.path.exists(_fixed_path)
        )
        if _use_fixed:
            print(f'[random_1c] Loading fixed initial labeled set from: {_fixed_path}')
            logger.info(f'[random_1c] Loading fixed initial labeled set from: {_fixed_path}')
            activeSet = np.load(_fixed_path)
            new_uSet = uSet[~np.isin(uSet, activeSet)]
        else:
            activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data)

        al_obj.propagation_sender_al_indices = activeSet.copy()
        al_obj.propagation_receiver_al_indices = activeSet.copy()

        # Update C matrix if using CK manager
        if hasattr(al_obj, 'ck_manager') and al_obj.ck_manager is not None:
            newly_labeled_indices = activeSet
            labels = np.array(train_data.targets)[newly_labeled_indices.astype(int)]
            al_obj.ck_manager.update_C_after_selection(newly_labeled_indices, labels)
            
            # Only update reference in sampling_fn if method doesn't have native C_general
            # (i.e., the sampling_fn.C_general is actually pointing to ck_manager.C_general)
            if (hasattr(al_obj.sampling_fn, 'C_general') and 
                al_obj.sampling_fn.C_general is al_obj.ck_manager.C_general):
                # Already pointing to the same object, no need to update
                pass
            elif not hasattr(al_obj.sampling_fn, 'C_general'):
                # Method doesn't have C_general at all, expose it
                al_obj.sampling_fn.C_general = al_obj.ck_manager.C_general

        print(f'Initial Pool is {activeSet}')
        # Save current lSet, new_uSet and activeSet in the episode directory
        # data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)
        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet)
        uSet = new_uSet

    print("Data Partitioning Complete. \nLabeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))
    logger.info("Labeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))

    # Preparing dataloaders for initial training
    lSet_loader = get_lset_loader(al_obj, cfg, data_obj, lSet, pseudo_train_data, train_data, uSet=uSet)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)

    if valSet_loader is None:
        valSet_loader = lSet_loader

    # Initialize the model.  
    print("model: {}\n".format(cfg.MODEL.TYPE))
    logger.info("model: {}\n".format(cfg.MODEL.TYPE))

    # Construct the optimizer
    # model = model_builder.build_model(cfg)
    model = mh.get_model(cfg)
    optimizer = optim.construct_optimizer(cfg, model)
    opt_init_state = deepcopy(optimizer.state_dict()) if not (cfg.MODEL.USE_1NN or cfg.EVAL_MODEL_TYPE == 'from_mlp_sklearn') else None
    model_init_state = deepcopy(model.state_dict().copy()) if not (cfg.MODEL.USE_1NN or cfg.EVAL_MODEL_TYPE == 'from_mlp_sklearn') else None

    print("optimizer: {}\n".format(optimizer))
    logger.info("optimizer: {}\n".format(optimizer))

    print("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    logger.info("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))


    eval_steps = np.arange(cfg.ACTIVE_LEARNING.START_ITER, cfg.ACTIVE_LEARNING.MAX_ITER + 1, cfg.ACTIVE_LEARNING.EVAL_FREQUENCY)
    checkpoint_file = None
    num_classes = len(np.unique(train_labels))

    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER+1):

        print("======== EPISODE {} BEGINS ========\n".format(cur_episode))
        logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))
        
        # Check if we need to switch AL methods at this round
        if (hasattr(cfg, 'SWITCH_METHOD_AT_ROUND') and cfg.SWITCH_METHOD_AT_ROUND is not None and 
            hasattr(cfg, 'SWITCH_TO_METHOD') and cfg.SWITCH_TO_METHOD is not None and
            cur_episode == cfg.SWITCH_METHOD_AT_ROUND):
            print(f"======== SWITCHING AL METHOD FROM {cfg.ACTIVE_LEARNING.SAMPLING_FN} TO {cfg.SWITCH_TO_METHOD} ========\n")
            logger.info(f"======== SWITCHING AL METHOD FROM {cfg.ACTIVE_LEARNING.SAMPLING_FN} TO {cfg.SWITCH_TO_METHOD} ========\n")
            
            # Update the sampling function name in config
            old_method = cfg.ACTIVE_LEARNING.SAMPLING_FN
            cfg.ACTIVE_LEARNING.SAMPLING_FN = cfg.SWITCH_TO_METHOD
            
            # Store CK manager if it exists (to reattach after switching)
            old_ck_manager = al_obj.ck_manager if hasattr(al_obj, 'ck_manager') else None
            
            # Reinitialize the sampling function
            al_obj.update_sampling_function(train_labels, lSet)
            
            # Reattach CK manager if it existed
            if old_ck_manager is not None:
                al_obj.attach_ck_manager(old_ck_manager)
            
            print(f"AL method switched successfully. Now using: {cfg.ACTIVE_LEARNING.SAMPLING_FN}\n")
            logger.info(f"AL method switched from {old_method} to {cfg.ACTIVE_LEARNING.SAMPLING_FN}\n")

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir
        
        if cfg.EVAL_MODEL_TYPE in ['from_mlp_sklearn', 'mlp_dropout']:
            model = mh.get_model(cfg)
            optimizer = optim.construct_optimizer(cfg, model)


        train_episode = cur_episode in eval_steps
        checkpoint_file, per_class_accuracy, model = train_and_eval_model(cfg, cur_episode, lSet_loader, model, optimizer, test_loader,
                                               valSet_loader, train_episode, num_classes=num_classes)
            # DCoM's delta-s updating
        Dcom_delta_update(cfg, data_obj, checkpoint_file, lSet, train_data, uSet, model)

        # No need to perform active sampling in the last episode iteration
        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            # Save current lSet, uSet in the final episode directory
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            break

        # Active Sample 
        lSet, lSet_loader, uSet, valSet_loader = active_sampling_part(cfg, checkpoint_file, model, cur_episode, data_obj, lSet,
                                                                      lSet_loader, train_data, uSet, valSet,
                                                                      valSet_loader, al_obj, pseudo_train_data,
                                                                      per_class_accuracy=per_class_accuracy)



        # add avg delta to cfg.ACTIVE_LEARNING.DELTA_LST towards the next active sampling
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ['dcom']:
            delta_lst_float = [float(delta) for delta in cfg.ACTIVE_LEARNING.DELTA_LST]
            next_initial_deltas = [str(round(np.average(delta_lst_float), 2))] * cfg.ACTIVE_LEARNING.BUDGET_SIZE
            cfg.ACTIVE_LEARNING.DELTA_LST.extend(next_initial_deltas)
            print("Current delta list: ", cfg.ACTIVE_LEARNING.DELTA_LST)
            print("Current delta avg list: ", delta_avg_lst)
            print("Current delta std list: ", delta_std_lst)
        print('Current accuracy values: ', plot_episode_yvalues)

        if cfg.EVAL_MODEL_TYPE in ['from_mlp_sklearn', 'mlp_dropout']:
            model = mh.get_model(cfg)
            optimizer = optim.construct_optimizer(cfg, model)
        elif not (cfg.ACTIVE_LEARNING.FINE_TUNE):
            # start model from scratch
            print('Starting model from scratch - ignoring existing weights.')
            model = mh.get_model(cfg, model_init_state=model_init_state)
            # Construct the optimizer
            optimizer = optim.construct_optimizer(cfg, model, opt_init_state=opt_init_state)

        if checkpoint_file is not None:
            os.remove(checkpoint_file)


def get_lset_loader(al_obj, cfg, data_obj, lSet, pseudo_train_data, train_data, uSet=None, decay_rate=None):
    # Check if distillation training is enabled
    distillation_enabled = getattr(cfg, 'DISTILLATION_TRAINING', False)
    
    if distillation_enabled and hasattr(al_obj, 'sampling_fn') and al_obj.sampling_fn is not None:
        # Distillation training mode: use soft targets from C matrix
        # Returns GPU tensors for efficiency
        soft_targets, distill_mask, distill_weights = prepare_distillation_targets(
            al_obj, cfg.DISTILLATION_THRESHOLD, cfg, lSet=lSet, uSet=uSet, decay_rate=decay_rate
        )
        
        # Find samples above threshold that are not in lSet (from uSet)
        if uSet is not None:
            # Move to CPU for numpy indexing, then back to GPU
            distill_mask_cpu = distill_mask.cpu()
            distill_uset_mask = distill_mask_cpu[uSet.astype(int)]
            distill_uset_indices = uSet[distill_uset_mask.numpy()]
            # Combine lSet with distillation samples from uSet
            training_indices = np.concatenate([lSet, distill_uset_indices])
            print(f"[Training] Distillation Training: {len(lSet)} labeled + {len(distill_uset_indices)} distillation samples = {len(training_indices)} total")
            logger.info(f"[Training] Distillation Training: {len(lSet)} labeled + {len(distill_uset_indices)} distillation samples = {len(training_indices)} total")
            logger.info(f"Number of distill Labels = {len(distill_uset_indices)}")
        else:
            training_indices = lSet
            print(f"[Training] Distillation Training: {len(lSet)} labeled samples (no uSet provided)")
            logger.info(f"[Training] Distillation Training: {len(lSet)} labeled samples (no uSet provided)")
        
        # Create labeled mask (True for lSet samples, False for distillation-only samples)
        # Keep on CPU to avoid CUDA kernel compatibility issues during tensor creation
        labeled_mask = torch.zeros(len(train_data.targets), dtype=torch.bool)
        labeled_mask[torch.as_tensor(lSet.astype(int))] = True
        
        # Attach distillation data to train_data (GPU tensors stay on GPU, CPU masks stay on CPU)
        train_data.return_index = True
        train_data.soft_targets = soft_targets  # Already on GPU
        train_data.distill_mask = distill_mask  # Already on GPU
        train_data.distill_weights = distill_weights  # Already on GPU
        train_data.labeled_mask = labeled_mask  # CPU tensor, moved to device in training loop
        train_data.pseudo_mask = None
        train_data.pseudo_weights = None
        
        lSet_loader = data_obj.getIndexesDataLoader(
            indexes=training_indices, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data
        )
    
    elif cfg.TRAIN_PSEUDO_LABELS:
        # Use CK manager's C_general if it exists (for separate matrices), otherwise use sampling_fn's
        if hasattr(al_obj, 'ck_manager') and al_obj.ck_manager is not None:
            C_matrix = al_obj.ck_manager.C_general.cpu().numpy()
            use_ck_manager = True
        else:
            C_matrix = al_obj.sampling_fn.C_general.cpu().numpy()
            use_ck_manager = False
        thresh = cfg.PSEUDO_LABELS_THRESHOLD

        if C_matrix.ndim == 1:
            # Vector coverage (CoverageVectorMethod): per-point max similarity + teacher index
            C_vec = C_matrix.astype(np.float64)
            label_cov = al_obj.sampling_fn.label_coverage_general.cpu().numpy().astype(np.int64)
            targets_all = np.array(train_data.targets)
            # Global max normalization (no per-class columns; PSEUDO_LABEL_CLASS_THRESHOLD uses same scalar thresh)
            max_values = C_vec / (np.max(C_vec) + 1e-8)
            point_thresholds = thresh
            valid_points_indx = np.where((max_values > point_thresholds) & (label_cov >= 0))[0]
            valid_pseudo_labels = targets_all[label_cov[valid_points_indx]]
            new_mask = ~np.isin(valid_points_indx, lSet)
            valid_points_indx = valid_points_indx[new_mask]
            valid_pseudo_labels = valid_pseudo_labels[new_mask]
        else:
            # Normalize C matrix based on configuration
            ck_c_normalization = getattr(cfg, 'CK_C_NORMALIZATION', 'sum')
            if use_ck_manager and ck_c_normalization == 'softmax':
                # Softmax normalization with temperature scaling
                import scipy.special
                C_norm = scipy.special.softmax(C_matrix / al_obj.ck_manager.alpha, axis=1)
            else:
                # Sum normalization (default)
                row_sums = np.sum(C_matrix, axis=1)[:, None]
                C_norm = np.divide(
                    C_matrix,
                    row_sums,
                    out=np.zeros_like(C_matrix, dtype=float),  # Default output is 0
                    where=row_sums != 0  # Only divide where the sum is not 0
                )

            if cfg.PSEUDO_LABEL_CLASS_THRESHOLD:
                if cfg.PSEUDO_LABEL_CLASS_THRESHOLD_TOPK:
                    # Use top-k mean per class (k = num_points / num_classes)
                    num_points = C_norm.shape[0]
                    num_classes = C_norm.shape[1]
                    k = num_points // num_classes
                    
                    # For each class, select top-k points and compute mean
                    C_norm_per_class = np.zeros(num_classes)
                    for c in range(num_classes):
                        class_probs = C_norm[:, c]
                        top_k_indices = np.argpartition(class_probs, -min(k, num_points))[-min(k, num_points):]
                        C_norm_per_class[c] = class_probs[top_k_indices].mean()
                else:
                    # Use mean over all points
                    C_norm_per_class = np.mean(C_norm, axis=0)
                
                C_norm_per_class_norm_per_max = C_norm_per_class / np.max(C_norm_per_class)
                class_threshold_vec = thresh * C_norm_per_class_norm_per_max
                thresh = class_threshold_vec

            # Find the class with maximum coverage for each point
            max_values_label_indx = np.argmax(C_norm, axis=1)
            max_values = C_norm[np.arange(C_norm.shape[0]), max_values_label_indx]

            # Get appropriate threshold for each point (handle both scalar and vector)
            if isinstance(thresh, np.ndarray):
                point_thresholds = thresh[max_values_label_indx]
            else:
                point_thresholds = thresh

            valid_points_indx = np.where(max_values > point_thresholds)[0]
            valid_pseudo_labels = max_values_label_indx[valid_points_indx]
            # keep only points not already in lSet
            new_mask = ~np.isin(valid_points_indx, lSet)
            valid_points_indx = valid_points_indx[new_mask]
            valid_pseudo_labels = valid_pseudo_labels[new_mask]

        print(f"Number of Pseudo labels {valid_points_indx.size}\n")
        if valid_points_indx.size == 0:
            # fall back to original loader without adding pseudo-labeled samples
            train_data.return_index = True
            train_data.pseudo_mask = None
            train_data.pseudo_weights = None
            train_data.soft_targets = None
            train_data.distill_mask = None
            train_data.distill_weights = None
            train_data.labeled_mask = None
            return data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        logger.info(f"Number of Pseudo Labels = {valid_pseudo_labels.size}")
        lset_with_pseudo_labels = np.concatenate([lSet, valid_points_indx])
        valid_labels = np.array(train_data.targets)
        valid_labels[valid_points_indx] = valid_pseudo_labels
        pseudo_train_data.targets = list(valid_labels)

        # track which samples are pseudo-labeled
        pseudo_mask = np.zeros(len(train_data.targets), dtype=bool)
        pseudo_mask[valid_points_indx] = True
        pseudo_train_data.pseudo_mask = pseudo_mask
        pseudo_weights = None
        if cfg.PSEUDO_LABEL_WEIGHTING_FUNC == 'dynamic':
            pseudo_weights = np.ones(len(train_data.targets), dtype=np.float32)
            if C_matrix.ndim == 1:
                C_vec_w = C_matrix.astype(np.float64)
                max_vals_w = C_vec_w / (np.max(C_vec_w) + 1e-8)
                pseudo_weights[valid_points_indx] = max_vals_w[valid_points_indx].astype(np.float32)
            else:
                pseudo_weights[valid_points_indx] = C_norm[valid_points_indx, valid_pseudo_labels]
        pseudo_train_data.pseudo_weights = pseudo_weights
        pseudo_train_data.return_index = True

        lSet_loader = data_obj.getIndexesDataLoader(indexes=lset_with_pseudo_labels, batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    data=pseudo_train_data)

    else:
        train_data.return_index = True
        train_data.pseudo_mask = None
        train_data.pseudo_weights = None
        train_data.soft_targets = None
        train_data.distill_mask = None
        train_data.distill_weights = None
        train_data.labeled_mask = None
        lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    return lSet_loader


def active_sampling_part(cfg, checkpoint_file, model, cur_episode, data_obj, lSet, lSet_loader, train_data, uSet, valSet,
                         valSet_loader, al_obj, pseudo_train_data, per_class_accuracy=None, decay_rate=None):
    print("======== ACTIVE SAMPLING ========\n")
    logger.info("======== ACTIVE SAMPLING ========\n")
    # al_obj = ActiveLearning(data_obj, cfg) if al_obj is None else al_obj
    # clf_model = model_builder.build_model(cfg)
    # clf_model = cu.load_checkpoint(checkpoint_file, clf_model)
    clf_model = mh.get_model(cfg, checkpoint_file) if checkpoint_file is not None else None
    clf_model = model if model is not None else clf_model
    activeSet, new_uSet = al_obj.sample_from_uSet(clf_model, lSet, uSet, train_data, 
                                                   per_class_accuracy=per_class_accuracy, 
                                                   data_obj=data_obj)
    
    # Always store the most recent AL picks for al_constant propagation sender/receiver modes
    al_obj.propagation_sender_al_indices = activeSet.copy()
    al_obj.propagation_receiver_al_indices = activeSet.copy()
    
    # Update C matrix if using CK manager
    if hasattr(al_obj, 'ck_manager') and al_obj.ck_manager is not None:
        newly_labeled_indices = activeSet
        labels = np.array(train_data.targets)[newly_labeled_indices.astype(int)]
        al_obj.ck_manager.update_C_after_selection(newly_labeled_indices, labels)
        
        # Only update reference in sampling_fn if method doesn't have native C_general
        # (i.e., the sampling_fn.C_general is actually pointing to ck_manager.C_general)
        if (hasattr(al_obj.sampling_fn, 'C_general') and 
            al_obj.sampling_fn.C_general is al_obj.ck_manager.C_general):
            # Already pointing to the same object, no need to update
            pass
        elif not hasattr(al_obj.sampling_fn, 'C_general'):
            # Method doesn't have C_general at all, expose it
            al_obj.sampling_fn.C_general = al_obj.ck_manager.C_general
    
    # Save current lSet, new_uSet and activeSet in the episode directory
    data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)
    # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
    lSet = np.append(lSet, activeSet)
    uSet = new_uSet

    lSet_loader = get_lset_loader(al_obj, cfg, data_obj, lSet, pseudo_train_data, train_data, uSet=uSet, decay_rate=decay_rate)

    # lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    if valSet_loader is None:
        valSet_loader = lSet_loader
    # uSet_loader = data_obj.getSequentialDataLoader(indexes=uSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    print(
        "Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(
            cur_episode, len(lSet), len(uSet), len(activeSet)))
    logger.info(
        "Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(
            cur_episode, len(lSet), len(uSet), len(activeSet)))
    print("================================\n\n")
    logger.info("================================\n\n")
    return lSet, lSet_loader, uSet, valSet_loader


def test_1nn(test_loader, model, cur_epoch, train_loader):
    # assert test_loader.dataset.no_aug

    if torch.cuda.is_available():
        model.cuda()

    test_meter = TestMeter(len(test_loader))
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.
    totalSamples = 0.
    if train_loader is not None and cfg.MODEL.USE_1NN:
        total_train_inputs = []
        total_train_labels = []

        len_loader_info = len(list(train_loader)[0])
        if len_loader_info == 3:
            for train_inputs, train_labels, _ in train_loader:
                total_train_inputs.append(train_inputs)
                total_train_labels.append(train_labels)
        elif len_loader_info == 2:
            for train_inputs, train_labels in train_loader:
                total_train_inputs.append(train_inputs)
                total_train_labels.append(train_labels)

        total_train_inputs = torch.cat(total_train_inputs).cuda()
        total_train_labels = torch.cat(total_train_labels).cuda()
        print(total_train_labels.size)

    test_logits = []
    test_labels = []
    test_entropies = []
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)

            # Compute the predictions
            if cfg.MODEL.USE_1NN:
                output_dict = model(total_train_inputs, total_train_labels, inputs)
            else:
                output_dict = model(inputs, labels)

            preds = output_dict['preds'].to(dtype=torch.float16)
            if 'labels' in output_dict:
                labels = output_dict['labels']

            probs = torch.nn.functional.softmax(preds, dim=1)
            entropy = probs * torch.log(probs + 1e-8)
            entropy = -1 * torch.sum(entropy, dim=1)
            test_entropies.append(entropy.detach().cpu().numpy())

            # Compute the errors
            top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])

            # Copy the errors from GPU to CPU (sync point)
            top1_err = top1_err.item()
            # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
            misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
            totalSamples += inputs.size(0) * cfg.NUM_GPUS
            test_meter.iter_toc()
            # Update and log stats
            test_meter.update_stats(
                top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
            )
            test_meter.log_iter_stats(cur_epoch, cur_iter)
            test_meter.iter_tic()

            test_logits.append(preds)
            test_labels.append(labels)

    if test_entropies:
        test_mean_entropy = np.mean(np.concatenate(test_entropies))
    else:
        test_mean_entropy = 0.0
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()


    return 100 - (misclassifications / totalSamples)


def train_and_eval_model(cfg, cur_episode, lSet_loader, model, optimizer, test_loader, valSet_loader, train_episode=True, num_classes=None):
    # Train model
    # if cfg.MODEL.USE_1NN:
    #     test_acc = test_1nn(test_loader, model, cur_episode, lSet_loader,
    #            )
    #     checkpoint_file = None


    # else:
    if train_episode:
        print("======== TRAINING ========")
        logger.info("======== TRAINING ========")
        best_val_acc, best_val_epoch, checkpoint_file, model = train_model(lSet_loader, valSet_loader, model, optimizer, cfg)
        print("Best Validation Accuracy: {}\nBest Epoch: {}\n".format(round(best_val_acc, 4), best_val_epoch))
        logger.info("EPISODE {} Best Validation Accuracy: {}\tBest Epoch: {}\n".format(cur_episode, round(best_val_acc, 4),
                                                                                       best_val_epoch))
    else:
        checkpoint_file = None
    # Test best model checkpoint
    print("======== TESTING ========\n")
    logger.info("======== TESTING ========\n")



    test_acc = test_model(test_loader, checkpoint_file, cfg, cur_episode, model, lSet_loader)
    print("Test Accuracy: {}.\n".format(round(test_acc, 4)))
    logger.info("EPISODE {} Test Accuracy {}.\n".format(cur_episode, test_acc))
    
    # Compute per-class accuracy for NN fusion mode
    per_class_accuracy = None
    nn_fusion_enabled = getattr(cfg, 'NN_FUSION', False)
    nn_fusion_start = getattr(cfg, 'NN_FUSION_START_ROUND', 10)
    nn_fusion_interval = getattr(cfg, 'NN_FUSION_INTERVAL', 10)
    
    # Check if this is a fusion round: at or after start, and at interval steps
    is_fusion_round = (
        nn_fusion_enabled and 
        cur_episode >= nn_fusion_start and 
        (cur_episode - nn_fusion_start) % nn_fusion_interval == 0
    )
    
    if is_fusion_round and checkpoint_file is not None and num_classes is not None:
        print(f"======== COMPUTING PER-CLASS ACCURACY FOR NN FUSION (round {cur_episode}) ========\n")
        eval_model = mh.get_model(cfg, checkpoint_file)
        per_class_accuracy = compute_per_class_accuracy(test_loader, eval_model, num_classes, cfg)
        print(f"Per-class accuracy: {per_class_accuracy}")
        logger.info(f"EPISODE {cur_episode} Per-class accuracy: {per_class_accuracy}")
    
    return checkpoint_file, per_class_accuracy, model


def Dcom_delta_update(cfg, data_obj, checkpoint_file, lSet, train_data, uSet, model):

    if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["dcom"]:
        print("======== Update the deltas dynamically ========\n")
        from pycls.al.DCoM import DCoM
        al_algo = DCoM(cfg, lSet, uSet, budgetSize=cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                       max_delta=cfg.ACTIVE_LEARNING.MAX_DELTA,
                       lSet_deltas=cfg.ACTIVE_LEARNING.DELTA_LST)

        lSet_labels = np.take(train_data.targets, np.asarray(lSet, dtype=np.int64))
        all_images_idx = np.array(list(lSet) + list(uSet))
        images_loader = data_obj.getSequentialDataLoader(indexes=all_images_idx,
                                                         batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        all_labels = np.take(train_data.targets, np.asarray(all_images_idx, dtype=np.int64))

        images_pseudo_labels = get_label_from_model(images_loader, checkpoint_file, cfg, model)
        cfg.ACTIVE_LEARNING.DELTA_LST[
        -1 * cfg.ACTIVE_LEARNING.BUDGET_SIZE:] = al_algo.new_centroids_deltas(lSet_labels,
                                                                              all_labels=all_labels,
                                                                              pseudo_labels=images_pseudo_labels,
                                                                              budget=cfg.ACTIVE_LEARNING.BUDGET_SIZE)

        delta_lst_float = [float(delta) for delta in cfg.ACTIVE_LEARNING.DELTA_LST]
        delta_avg_lst.append(np.average(delta_lst_float))
        delta_std_lst.append(np.std(delta_lst_float))


def train_mlp_with_distillation(train_loader, val_loader, cfg):
    """Train a PyTorch MLP matching sklearn's behavior with distillation support."""
    from pycls.core.builders import MLPClassifierTorch
    
    # Determine input dimension from data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[1] if len(sample_batch[0].shape) > 1 else sample_batch[0].shape[0]
    
    # Create MLP with SAME architecture as sklearn
    num_features = 384 if cfg.DATASET.NAME in ['IMAGENET50', 'IMAGENET100', 'IMAGENET200'] else 256
    num_features = 2 if cfg.DATASET.NAME in ['SCENARIO_A', 'HALF_MOON'] else num_features
    model = MLPClassifierTorch(input_dim, num_features, cfg.MODEL.NUM_CLASSES).cuda()
    
    # Match sklearn's Adam optimizer settings EXACTLY
    # sklearn uses: learning_rate_init=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
    # sklearn's alpha parameter (L2 penalty) = weight_decay in PyTorch
    l2_alpha = 3e-2  # sklearn's alpha parameter
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.001,           # sklearn's learning_rate_init (default)
        betas=(0.9, 0.999), # sklearn's beta_1, beta_2 (defaults)
        eps=1e-8,           # sklearn's epsilon (default)
        weight_decay=l2_alpha  # Use PyTorch's efficient weight_decay instead of manual L2
    )
    
    # Loss function
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Training parameters matching sklearn
    max_iter = 300 if not cfg.DEBUG else 1  # sklearn's max_iter
    n_iter_no_change = 10  # sklearn's early stopping patience
    tol = 1e-4  # sklearn's convergence tolerance
    
    best_val_acc = 0.0
    best_epoch = 0
    best_loss = float('inf')
    no_improvement_count = 0
    
    distillation_enabled = getattr(cfg, 'DISTILLATION_TRAINING', False)
    has_distill_data = (hasattr(train_loader.dataset, 'soft_targets') and 
                        train_loader.dataset.soft_targets is not None)
    
    print(f"Training PyTorch MLP (sklearn-style) with distillation for max {max_iter} iterations")
    logger.info(f"Training PyTorch MLP (sklearn-style) with distillation for max {max_iter} iterations")
    
    # Training loop - sklearn iterates over the dataset max_iter times
    for iteration in range(max_iter):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            if len(batch) == 3:
                inputs, labels, idx = batch
            else:
                inputs, labels = batch
                idx = None
            
            inputs = inputs.cuda().type(torch.cuda.FloatTensor)
            labels = labels.cuda()
            
            # Forward pass
            logits = model(inputs)
            
            # Compute loss based on distillation or standard training
            if distillation_enabled and has_distill_data and idx is not None:
                # Get soft targets and masks - already GPU tensors, just index them
                # Ensure idx is on the correct device and dtype matches logits
                idx_device = train_loader.dataset.soft_targets.device
                if idx.device != idx_device:
                    idx = idx.to(idx_device)
                
                soft_targets_batch = train_loader.dataset.soft_targets[idx].to(dtype=logits.dtype)
                distill_mask_batch = train_loader.dataset.distill_mask[idx]
                labeled_mask_batch = train_loader.dataset.labeled_mask.to(idx_device)[idx]
                distill_weights_batch = train_loader.dataset.distill_weights[idx]
                
                # Distill-only mask: points above threshold that are NOT labeled
                distill_only_mask = distill_mask_batch & ~labeled_mask_batch
                
                # CE loss on labeled points
                if labeled_mask_batch.any():
                    ce_loss = ce_loss_fn(logits[labeled_mask_batch], labels[labeled_mask_batch]).mean()
                else:
                    ce_loss = torch.tensor(0.0, device=logits.device)
                
                # KL loss on distill-only points
                distill_temp = getattr(cfg, 'DISTILLATION_TEMPERATURE', 1.0)
                distill_soft_target_norm = getattr(cfg, 'DISTILL_SOFT_TARGET_NORMALIZATION', 'power')
                distill_loss_raw = losses.distillation_loss(logits, soft_targets_batch, distill_only_mask, 
                                                           temperature=distill_temp, 
                                                           soft_target_normalization=distill_soft_target_norm)
                
                # Weight distillation loss
                distill_factor = getattr(cfg, 'DISTILL_FACTOR', 8.0)
                if distill_only_mask.any():
                    avg_weight = distill_weights_batch[distill_only_mask].mean()
                    weighted_distill_factor = distill_factor * avg_weight
                    distill_loss = distill_loss_raw * weighted_distill_factor
                else:
                    distill_loss = distill_loss_raw * distill_factor
                
                loss = ce_loss + distill_loss
            else:
                # Standard CE loss
                loss = ce_loss_fn(logits, labels).mean()
            
            # Note: L2 regularization is now handled by optimizer weight_decay (more efficient)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Validation (sklearn checks after each iteration)
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    inputs, labels, _ = batch
                else:
                    inputs, labels = batch
                
                inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                labels = labels.cuda()
                
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = 100.0 * correct / total
        
        # Track best model (like sklearn does)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = iteration + 1
        
        # Early stopping check (like sklearn's n_iter_no_change)
        if avg_loss < best_loss - tol:
            best_loss = avg_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Log progress (print every 10 iterations like before)
        if (iteration + 1) % 10 == 0 or iteration == max_iter - 1:
            print(f'Iter {iteration+1}/{max_iter} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}% - No improve: {no_improvement_count}')
            logger.info(f'Iter {iteration+1}/{max_iter} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}% - No improve: {no_improvement_count}')
        
        # Early stopping (sklearn stops if no improvement for n_iter_no_change iterations)
        if no_improvement_count >= n_iter_no_change:
            print(f"Early stopping at iteration {iteration+1} (no improvement for {n_iter_no_change} iterations)")
            logger.info(f"Early stopping at iteration {iteration+1} (no improvement for {n_iter_no_change} iterations)")
            break
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}% at iteration {best_epoch}")
    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.2f}% at iteration {best_epoch}")
    
    return best_val_acc, best_epoch, None, model


def train_model(train_loader, val_loader, model, optimizer, cfg):
    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    if cfg.MODEL.USE_1NN:
        return 0, 0, None, model
        train_data_and_labels = list(iter(train_loader))
        train_data = torch.concat([inner_list[0] for inner_list in train_data_and_labels])
        train_labels = torch.concat([inner_list[1] for inner_list in train_data_and_labels])
        model.fit(train_data, train_labels)

        val_data_and_labels = list(iter(val_loader))
        val_data = torch.concat([inner_list[0] for inner_list in val_data_and_labels])
        val_labels = torch.concat([inner_list[1] for inner_list in val_data_and_labels])
        preds = model.predict(val_data)
        test_acc = 100. * (preds == np.array(val_labels)).mean()

        return test_acc, 0, None, model
    # elif cfg.EVAL_MODEL_TYPE == 'from_mlp':
    #     # PyTorch MLP - uses the same training logic as from_mlp_sklearn with distillation
    #     print("Using PyTorch MLP (from_mlp) with sklearn-matching training")
    #     logger.info("Using PyTorch MLP (from_mlp) with sklearn-matching training")
    #     return train_mlp_with_distillation(train_loader, val_loader, cfg)
    
    elif cfg.EVAL_MODEL_TYPE == 'mlp_dropout':
        # MLP with Dropout - always use PyTorch training
        print("Using PyTorch MLP with Dropout for MC uncertainty estimation")
        logger.info("Using PyTorch MLP with Dropout for MC uncertainty estimation")
        return train_mlp_with_distillation(train_loader, val_loader, cfg)
    
    elif cfg.EVAL_MODEL_TYPE == 'from_mlp_sklearn':
        # Check if distillation training is enabled
        distillation_enabled = getattr(cfg, 'DISTILLATION_TRAINING', False)
        has_soft_targets = (hasattr(train_loader.dataset, 'soft_targets') and 
                           train_loader.dataset.soft_targets is not None)
        
        # If distillation is enabled, use PyTorch MLP with soft targets
        # if distillation_enabled and has_soft_targets:
        #     print("Using PyTorch MLP with distillation on soft targets (from_mlp_sklearn + distillation)")
        #     logger.info("Using PyTorch MLP with distillation on soft targets (from_mlp_sklearn + distillation)")
        #     return train_mlp_with_distillation(train_loader, val_loader, cfg)
        
        # Otherwise use standard sklearn MLP
        print("Using sklearn MLP (no distillation)")
        logger.info("Using sklearn MLP (no distillation)")
        
        # Extract data and labels from train_loader
        train_data_and_labels = list(iter(train_loader))
        
        # Handle batch format (with or without index)
        if len(train_data_and_labels[0]) == 3:
            train_data = torch.cat([batch[0] for batch in train_data_and_labels])
            train_labels = torch.cat([batch[1] for batch in train_data_and_labels])
            train_indices = torch.cat([batch[2] for batch in train_data_and_labels])
        else:
            train_data = torch.cat([batch[0] for batch in train_data_and_labels])
            train_labels = torch.cat([batch[1] for batch in train_data_and_labels])
            train_indices = None
        
        # Log pseudo-label usage if applicable
        if train_indices is not None and hasattr(train_loader.dataset, 'pseudo_mask') and train_loader.dataset.pseudo_mask is not None:
            pseudo_mask = train_loader.dataset.pseudo_mask
            train_indices_np = train_indices.cpu().numpy()
            pseudo_count = pseudo_mask[train_indices_np].sum()
            print(f"sklearn MLP: Training on {len(train_labels)} samples ({len(train_labels) - pseudo_count} labeled + {pseudo_count} pseudo-labeled)")
            logger.info(f"sklearn MLP: Training on {len(train_labels)} samples ({len(train_labels) - pseudo_count} labeled + {pseudo_count} pseudo-labeled)")
        
        train_data_np = train_data.cpu().numpy()
        train_labels_np = train_labels.cpu().numpy()
        
        model.fit(train_data_np, train_labels_np)
        
        val_data_and_labels = list(iter(val_loader))
        val_data = torch.cat([inner_list[0] for inner_list in val_data_and_labels])
        val_labels = torch.cat([inner_list[1] for inner_list in val_data_and_labels])
        
        val_data_np = val_data.cpu().numpy()
        val_labels_np = val_labels.cpu().numpy()
        
        preds = model.predict(val_data_np)
        test_acc = 100. * (preds == val_labels_np).mean()
        
        return test_acc, 0, None, model
    start_epoch = 0
    loss_fun = losses.get_loss_fun(reduction="none")

    # Create meters
    train_meter = TrainMeter(len(train_loader))
    val_meter = ValMeter(len(val_loader))

    # Perform the training loop
    # print("Len(train_loader):{}".format(len(train_loader)))
    logger.info('Start epoch: {}'.format(start_epoch + 1))
    val_set_acc = 0.

    temp_best_val_acc = 0.
    temp_best_val_epoch = 0

    # Best checkpoint model and optimizer states
    best_model_state = None
    best_opt_state = None

    val_acc_epochs_x = []
    val_acc_epochs_y = []

    clf_train_iterations = cfg.OPTIM.MAX_EPOCH * int(len(train_loader)/cfg.TRAIN.BATCH_SIZE)
    clf_change_lr_iter = clf_train_iterations // 25
    clf_iter_count = 0

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):

        # Train for one epoch
        train_loss, clf_iter_count = train_epoch(train_loader, model, loss_fun, optimizer, train_meter, \
                                        cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_train_iterations)

        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)


        # Model evaluation
        if is_eval_epoch(cur_epoch):
            # Original code[PYCLS] passes on testLoader but we want to compute on val Set
            val_loader.dataset.no_aug = True
            val_set_err = test_epoch(val_loader, model, val_meter, cur_epoch)
            val_set_acc = 100. - val_set_err
            val_loader.dataset.no_aug = False
            if temp_best_val_acc < val_set_acc:
                temp_best_val_acc = val_set_acc
                temp_best_val_epoch = cur_epoch + 1

                # Save best model and optimizer state for checkpointing
                model.eval()

                best_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
                best_opt_state = optimizer.state_dict()

                model.train()

            # Since we start from 0 epoch
            val_acc_epochs_x.append(cur_epoch+1)
            val_acc_epochs_y.append(val_set_acc)

        plot_epoch_xvalues.append(cur_epoch+1)
        plot_epoch_yvalues.append(train_loss)

        # save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y],\
        #     ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR, isDebug=False)
        logger.info("Successfully logged numpy arrays!!")

        # Plot arrays
        # plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
        # x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
        #
        # plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
        # x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

        # save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y], \
        #         ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR)

        print('Training Epoch: {}/{}\tTrain Loss: {}\tVal Accuracy: {}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4), round(val_set_acc, 4)))

    # Save the best model checkpoint (Episode level)
    checkpoint_file = cu.save_checkpoint(info="vlBest_acc_"+str(int(temp_best_val_acc)), \
        model_state=best_model_state, optimizer_state=best_opt_state, epoch=temp_best_val_epoch, cfg=cfg)

    print('\nWrote Best Model Checkpoint to: {}\n'.format(checkpoint_file.split('/')[-1]))
    logger.info('Wrote Best Model Checkpoint to: {}\n'.format(checkpoint_file))

    # plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
    #     x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
    #
    # plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, \
    #     x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
    #
    # plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
    #     x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

    plot_epoch_xvalues = []
    plot_epoch_yvalues = []
    plot_it_x_values = []
    plot_it_y_values = []

    best_val_acc = temp_best_val_acc
    best_val_epoch = temp_best_val_epoch

    return best_val_acc, best_val_epoch, checkpoint_file, None


def test_model(test_loader, checkpoint_file, cfg, cur_episode, model, lset_loader):

    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    test_meter = TestMeter(len(test_loader))

    # model = model_builder.build_model(cfg)
    # model = cu.load_checkpoint(checkpoint_file, model, weights_only=False)


    if cfg.MODEL.USE_1NN:
        # # 1NN model does not have a predict method, so we use the fit method to train it
        # data = test_loader.dataset.features
        # labels = test_loader.dataset.targets
        # preds = model.predict(data)
        # test_acc = 100. * (preds == np.array(labels)).mean()
        test_acc = test_1nn(test_loader, model, cur_episode, lset_loader)
        #            )



        # return test_acc
    elif cfg.EVAL_MODEL_TYPE == 'mlp_dropout' and model is not None:
        # MLP with Dropout - test with MC uncertainty estimation
        print("Testing MLP with Dropout (MC dropout)")
        model.eval()  # Note: dropout will be activated during mc_uncertainty calls
        correct = 0
        total = 0
        uncertainties = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    inputs, labels, _ = batch
                else:
                    inputs, labels = batch
                
                inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                labels = labels.cuda()
                
                # Standard forward pass for accuracy
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Compute MC uncertainty (optional, for monitoring)
                if hasattr(model, 'mc_uncertainty'):
                    uncertainty = model.mc_uncertainty(inputs, T=20)
                    uncertainties.append(uncertainty.cpu().numpy())
        
        test_acc = 100.0 * correct / total
        
        if uncertainties:
            avg_uncertainty = np.mean(np.concatenate(uncertainties))
            print(f"Average MC Uncertainty: {avg_uncertainty:.4f}")
            logger.info(f"Average MC Uncertainty: {avg_uncertainty:.4f}")
    
    elif cfg.EVAL_MODEL_TYPE == 'from_mlp' and model is not None:
        # PyTorch MLP (always PyTorch for from_mlp)
        print("Testing PyTorch MLP model (from_mlp)")
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    inputs, labels, _ = batch
                else:
                    inputs, labels = batch
                
                inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                labels = labels.cuda()
                
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        test_acc = 100.0 * correct / total
    elif cfg.EVAL_MODEL_TYPE == 'from_mlp_sklearn' and model is not None:
        # Check if model is PyTorch or sklearn
        if isinstance(model, torch.nn.Module):
            # PyTorch MLP (from distillation training)
            print("Testing PyTorch MLP model")
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    if len(batch) == 3:
                        inputs, labels, _ = batch
                    else:
                        inputs, labels = batch
                    
                    inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                    labels = labels.cuda()
                    
                    logits = model(inputs)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            
            test_acc = 100.0 * correct / total
        else:
            # sklearn MLP
            from sklearn.exceptions import NotFittedError
            try:
                test_data = test_loader.dataset.features
                test_labels = np.array(test_loader.dataset.targets)
                
                preds = model.predict(test_data)
                test_acc = 100. * (preds == test_labels).mean()
            except NotFittedError:
                print("WARNING: sklearn model not fitted yet, skipping test for this episode")
                logger.info("sklearn model not fitted yet, skipping test for this episode")
                test_acc = np.nan
    elif checkpoint_file is not None:
        model = mh.get_model(cfg, checkpoint_file)
        test_err = test_epoch(test_loader, model, test_meter, cur_episode)
        test_acc = 100. - test_err
    else:
        test_acc = np.nan

    plot_episode_xvalues.append(cur_episode)
    plot_episode_yvalues.append(test_acc)
    print(f"OUTPUT FOLDER IS {cfg.EXP_DIR}")
    plot_arrays(x_vals=plot_episode_xvalues, y_vals=plot_episode_yvalues, \
                x_name="Episodes", y_name="Test Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EXP_DIR)

    save_plot_values([plot_episode_xvalues, plot_episode_yvalues], \
                     ["plot_episode_xvalues", "plot_episode_yvalues"], out_dir=cfg.EXP_DIR)

    return test_acc


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_max_iter):
    """Performs one epoch of training."""
    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    # Shuffle the data
    #loader.shuffle(train_loader, cur_epoch)
    if cfg.NUM_GPUS>1:  train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    # Currently we only support LR schedules for only 'SGD' optimizer
    lr = optim.get_epoch_lr(cfg, cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    if torch.cuda.is_available():
        model.cuda()

    # Enable training mode
    model.train()
    train_meter.iter_tic() #This basically notes the start time in timer class defined in utils/timer.py

    # Check if distillation training is enabled
    distillation_enabled = getattr(cfg, 'DISTILLATION_TRAINING', False)
    has_distill_data = (hasattr(train_loader.dataset, 'soft_targets') and 
                        train_loader.dataset.soft_targets is not None and
                        hasattr(train_loader.dataset, 'distill_mask') and 
                        train_loader.dataset.distill_mask is not None and
                        hasattr(train_loader.dataset, 'distill_weights') and
                        train_loader.dataset.distill_weights is not None)
    
    len_train_loader = len(train_loader)
    epoch_ce_grad_norms = []
    epoch_kl_grad_norms = []

    for cur_iter, batch in enumerate(train_loader):
        if len(batch) == 3:
            if type(batch) == dict:
                inputs, labels, idx = batch['image'], batch['target'], batch['meta']['index']
            else:
                inputs, labels, idx = batch
        else:
            inputs, labels = batch
            idx = None
        #ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        
        # Perform the forward pass
        preds = model(inputs)
        
        # Handle distillation training
        if distillation_enabled and has_distill_data and idx is not None:
            # Get soft targets and masks for this batch - already GPU tensors, just index them
            # Ensure idx is on the correct device and dtype matches preds
            idx_device = train_loader.dataset.soft_targets.device
            if idx.device != idx_device:
                idx = idx.to(idx_device)
            
            soft_targets_batch = train_loader.dataset.soft_targets[idx].to(dtype=preds.dtype)
            distill_mask_batch = train_loader.dataset.distill_mask[idx]
            labeled_mask_batch = train_loader.dataset.labeled_mask.to(idx_device)[idx]
            distill_weights_batch = train_loader.dataset.distill_weights[idx]
            
            # Distill-only mask: points above threshold that are NOT labeled
            distill_only_mask = distill_mask_batch & ~labeled_mask_batch
            
            # CE loss on labeled points only
            if labeled_mask_batch.any():
                ce_loss = loss_fun(preds[labeled_mask_batch], labels[labeled_mask_batch]).mean()
            else:
                ce_loss = torch.tensor(0.0, device=preds.device)
            
            # KL loss on distill-only points (unlabeled above threshold)
            distill_temp = getattr(cfg, 'DISTILLATION_TEMPERATURE', 1.0)
            distill_soft_target_norm = getattr(cfg, 'DISTILL_SOFT_TARGET_NORMALIZATION', 'power')
            distill_loss_raw = losses.distillation_loss(preds, soft_targets_batch, distill_only_mask, 
                                                       temperature=distill_temp, 
                                                       soft_target_normalization=distill_soft_target_norm)
            
            # Weight distillation loss by per-sample max probabilities * global factor
            # Each sample's weight = max_prob * distill_factor
            distill_factor = getattr(cfg, 'DISTILL_FACTOR', 0.0)
            if distill_only_mask.any():
                # Get average weight for distill-only samples
                avg_weight = distill_weights_batch[distill_only_mask].mean()
                weighted_distill_factor = distill_factor * avg_weight
                distill_loss = distill_loss_raw * weighted_distill_factor
            else:
                distill_loss = distill_loss_raw * distill_factor
            
            loss = ce_loss + distill_loss
            
            # Compute per-loss gradient norms for monitoring (only if explicitly enabled via --monitor_grad_norms)
            # This adds significant overhead (~30-40%) so it's disabled by default
            monitor_grad_norms = getattr(cfg, 'MONITOR_GRAD_NORMS', False)
            grad_norm_log_freq = getattr(cfg, 'GRAD_NORM_LOG_FREQ', 500)
            if monitor_grad_norms and cur_iter % grad_norm_log_freq == 0:
                ce_grad_norm = 0.0
                kl_grad_norm = 0.0
                optimizer.zero_grad()
                if ce_loss.grad_fn is not None:
                    ce_loss.backward(retain_graph=True)
                    ce_grad_norm = _compute_grad_norm(model)
                    optimizer.zero_grad()
                if distill_loss.grad_fn is not None:
                    distill_loss.backward(retain_graph=True)
                    kl_grad_norm = _compute_grad_norm(model)
                    optimizer.zero_grad()
                epoch_ce_grad_norms.append(ce_grad_norm)
                epoch_kl_grad_norms.append(kl_grad_norm)
        else:
            # Original loss computation path (non-distillation)
            use_dynamic = getattr(cfg, "PSEUDO_LABEL_WEIGHTING_FUNC", "constant") == "dynamic"
            if use_dynamic and idx is not None and hasattr(train_loader.dataset, "pseudo_weights") and train_loader.dataset.pseudo_weights is not None:
                sample_weights = torch.as_tensor(train_loader.dataset.pseudo_weights[idx], device=labels.device, dtype=torch.float32)
            else:
                if idx is not None and hasattr(train_loader.dataset, "pseudo_mask") and train_loader.dataset.pseudo_mask is not None:
                    is_pseudo = torch.as_tensor(train_loader.dataset.pseudo_mask[idx], device=labels.device, dtype=torch.bool)
                else:
                    is_pseudo = torch.zeros_like(labels, dtype=torch.bool)
                sample_weights = torch.where(is_pseudo, torch.tensor(cfg.PSEUDO_LABEL_WEIGHT, device=labels.device, dtype=torch.float32), torch.tensor(1.0, device=labels.device, dtype=torch.float32))
            # Compute the loss
            raw_loss = loss_fun(preds, labels)
            loss = (raw_loss * sample_weights).mean()
        
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parametersSWA
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs
        # if cfg.NUM_GPUS > 1:
        #     #Average error and losses across GPUs
        #     #Also this this calls wait method on reductions so we are ensured
        #     #to obtain synchronized results
        #     loss, top1_err = du.scaled_all_reduce(
        #         [loss, top1_err]
        #     )
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()
        # #Only master process writes the logs which are used for plotting
        # if du.is_master_proc():
        if cur_iter != 0 and cur_iter%19 == 0:
            #because cur_epoch starts with 0
            plot_it_x_values.append((cur_epoch)*len_train_loader + cur_iter)
            plot_it_y_values.append(loss)
            # save_plot_values([plot_it_x_values, plot_it_y_values],["plot_it_x_values", "plot_it_y_values"], out_dir=cfg.EPISODE_DIR, isDebug=False)
            # print(plot_it_x_values)
            # print(plot_it_y_values)
            #Plot loss graphs
            # plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR,)
            log_msg = 'Training Epoch: {}/{}\tIter: {}/{}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, cur_iter, len(train_loader))
            if epoch_ce_grad_norms:
                log_msg += '\tCE GradNorm: {:.4f}\tKL GradNorm: {:.4f}'.format(
                    epoch_ce_grad_norms[-1], epoch_kl_grad_norms[-1])
            print(log_msg)

        #Compute the difference in time now from start time initialized just before this for loop.
        train_meter.iter_toc()
        train_meter.update_stats(top1_err=top1_err, loss=loss, \
            lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    if epoch_ce_grad_norms:
        avg_ce_gn = np.mean(epoch_ce_grad_norms)
        avg_kl_gn = np.mean(epoch_kl_grad_norms)
        print('Epoch {} - Avg CE GradNorm: {:.4f}, Avg KL GradNorm: {:.4f}'.format(cur_epoch + 1, avg_ce_gn, avg_kl_gn))
        logger.info('Epoch {} - Avg CE GradNorm: {:.4f}, Avg KL GradNorm: {:.4f}'.format(cur_epoch + 1, avg_ce_gn, avg_kl_gn))
    train_meter.reset()
    return loss, clf_iter_count


def get_label_from_model(images_loader, checkpoint_file, cfg, model=None):
    """
    returns the labels of the images according to the checkpoint file model
    """
    get_label_meter = TestMeter(len(images_loader))
    if model is None:
        # model = model_builder.build_model(cfg)
        # model = cu.load_checkpoint(checkpoint_file, model)
        model = mh.get_model(cfg, checkpoint_file)


    pred = get_label_epoch(images_loader, model, get_label_meter, cfg)
    return pred

@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.
    totalSamples = 0.

    for cur_iter, batch in enumerate(test_loader):
        if len(batch) == 3:
            inputs, labels, _ = batch
        else:
            inputs, labels = batch
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            # Compute the predictions
            preds = model(inputs)
            # Compute the errors
            top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
            # Combine the errors across the GPUs
            # if cfg.NUM_GPUS > 1:
            #     top1_err = du.scaled_all_reduce([top1_err])
            #     #as above returns a list
            #     top1_err = top1_err[0]
            # Copy the errors from GPU to CPU (sync point)
            top1_err = top1_err.item()
            # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
            misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
            totalSamples += inputs.size(0)*cfg.NUM_GPUS
            test_meter.iter_toc()
            # Update and log stats
            test_meter.update_stats(
                top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
            )
            test_meter.log_iter_stats(cur_epoch, cur_iter)
            test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()

    return misclassifications/totalSamples


@torch.no_grad()
def compute_per_class_accuracy(test_loader, model, num_classes, cfg):
    """
    Compute per-class accuracy from test set evaluation.
    
    Args:
        test_loader: DataLoader for test set
        model: Trained model
        num_classes: Number of classes
        cfg: Configuration object
        
    Returns:
        np.ndarray: Per-class accuracy vector of shape (num_classes,), values in [0, 1]
    """
    if torch.cuda.is_available():
        model.cuda()
    
    model.eval()
    
    # Track correct predictions and total samples per class
    class_correct = np.zeros(num_classes, dtype=np.float32)
    class_total = np.zeros(num_classes, dtype=np.float32)
    
    for batch in test_loader:
        if len(batch) == 3:
            inputs, labels, _ = batch
        else:
            inputs, labels = batch
        
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        inputs = inputs.type(torch.cuda.FloatTensor)
        
        preds = model(inputs)
        pred_classes = torch.argmax(preds, dim=1)
        
        # Update per-class counts
        for label_val in range(num_classes):
            mask = (labels == label_val)
            class_total[label_val] += mask.sum().item()
            class_correct[label_val] += ((pred_classes == labels) & mask).sum().item()
    
    # Compute per-class accuracy, handle classes with no samples
    per_class_accuracy = np.divide(
        class_correct, 
        class_total, 
        out=np.zeros_like(class_correct), 
        where=class_total > 0
    )
    
    return per_class_accuracy


@torch.no_grad()
def get_label_epoch(images_loader, model, get_label_meter, cfg):
    """get labels according to the model (PyTorch or sklearn)."""
    # Check if model is PyTorch or sklearn based on instance type
    if isinstance(model, torch.nn.Module):
        # PyTorch model path (including PyTorch MLP from distillation)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        get_label_meter.iter_tic()

        all_preds = []
        for cur_iter, batch in enumerate(images_loader):
            if len(batch) == 3:
                inputs, _, _ = batch
            else:
                inputs, _ = batch
            with torch.no_grad():
                # Transfer the data to the current GPU device
                inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                # Compute the predictions
                preds = model(inputs)
                all_preds += preds

        final_preds = [torch.argmax(p).item() for p in all_preds]
        model.train()
    else:
        # sklearn model path
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(model)
        except (NotFittedError, AttributeError):
            print(f'Model is not fitted yet. Returning uniform margin scores.')
            # Return uniform scores (all zeros) so selection is based purely on density
            return np.zeros(len(images_loader) * images_loader.batch_size)

        get_label_meter.iter_tic()
        all_features = []
        for cur_iter, batch in enumerate(images_loader):
            if len(batch) == 3:
                inputs, _, _ = batch
            else:
                inputs, _ = batch
            # sklearn models expect numpy arrays
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.cpu().numpy()
            all_features.append(inputs)
        
        # Concatenate all features
        all_features = np.concatenate(all_features, axis=0)
        # Get predictions from sklearn model
        final_preds = model.predict(all_features).tolist()

    return final_preds


def define_eval_model_type(cfg, debug=False):
    if cfg.EVAL_MODEL_TYPE == 'NN1':
        cfg.MODEL.USE_1NN = True
        cfg.MODEL.LINEAR_FROM_FEATURES = True
        cfg.MODEL.LINEAR_FROM_IMAGES = False
        if debug:
            cfg.OPTIM.MAX_EPOCH = 1
        else:
            cfg.OPTIM.MAX_EPOCH = 200
    elif cfg.EVAL_MODEL_TYPE == 'from_features':
        cfg.MODEL.USE_1NN = False
        cfg.MODEL.LINEAR_FROM_FEATURES = True
        cfg.MODEL.LINEAR_FROM_IMAGES = False
        if debug:
            cfg.OPTIM.MAX_EPOCH = 1
        else:
            cfg.OPTIM.MAX_EPOCH = 200

    elif cfg.EVAL_MODEL_TYPE == 'from_mlp':
        cfg.MODEL.USE_1NN = False
        cfg.MODEL.LINEAR_FROM_FEATURES = True
        cfg.MODEL.LINEAR_FROM_IMAGES = False
        cfg.OPTIM.TYPE = 'adam'
        cfg.OPTIM.BASE_LR = 0.001  # Match sklearn's learning_rate_init
        cfg.OPTIM.WEIGHT_DECAY = 3e-2
        if debug:
            cfg.OPTIM.MAX_EPOCH = 1
        else:
            cfg.OPTIM.MAX_EPOCH = 300  # Match sklearn's max_iter

    elif cfg.EVAL_MODEL_TYPE == 'from_mlp_sklearn':
        cfg.MODEL.USE_1NN = False
        cfg.MODEL.LINEAR_FROM_FEATURES = True
        cfg.MODEL.LINEAR_FROM_IMAGES = False
        cfg.OPTIM.TYPE = 'adam'
        cfg.OPTIM.WEIGHT_DECAY = 3e-2
        if debug:
            cfg.OPTIM.MAX_EPOCH = 1
        else:
            cfg.OPTIM.MAX_EPOCH = 300

    elif cfg.EVAL_MODEL_TYPE == 'mlp_dropout':
        cfg.MODEL.USE_1NN = False
        cfg.MODEL.LINEAR_FROM_FEATURES = True
        cfg.MODEL.LINEAR_FROM_IMAGES = False
        cfg.OPTIM.TYPE = 'adam'
        cfg.OPTIM.BASE_LR = 0.001
        cfg.OPTIM.WEIGHT_DECAY = 3e-2
        if debug:
            cfg.OPTIM.MAX_EPOCH = 1
        else:
            cfg.OPTIM.MAX_EPOCH = 300

    elif cfg.EVAL_MODEL_TYPE == 'from_images':
        cfg.MODEL.USE_1NN = False
        cfg.MODEL.LINEAR_FROM_FEATURES = False
        cfg.MODEL.LINEAR_FROM_IMAGES = True
        if debug:
            cfg.OPTIM.MAX_EPOCH = 1
        else:
            cfg.OPTIM.MAX_EPOCH = 200
            # cfg.OPTIM.MAX_EPOCH = 400

if __name__ == "__main__":
    # print(1)
    args = argparser().parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.EXP_NAME = args.exp_name
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.al
    cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
    cfg.ACTIVE_LEARNING.INITIAL_DELTA = args.initial_delta
    cfg.ACTIVE_LEARNING.INITIAL_SIGMA = args.initial_sigma
    # Set CK_SIGMA separately if provided, otherwise use the same as INITIAL_SIGMA
    cfg.CK_SIGMA = args.ck_sigma if args.ck_sigma is not None else args.initial_sigma
    cfg.KERNEL_TYPE = args.kernel_type
    cfg.DIFF_METHOD = args.diff_method
    cfg.CONT_METHOD = args.cont_method
    cfg.DISTRIBUTION_CONT_WEIGHT_METHOD = args.distribution_cont_weight_method
    cfg.C_NORMALIZATION = args.c_normalization
    cfg.CLASS_WEIGHTING_METHOD = args.class_weighting_method

    cfg.SPARSE_K = args.sparse_K
    cfg.K_SPARSITY_THRESHOLD = args.K_sparsity_threshold if  cfg.KERNEL_TYPE != 'tophat' else cfg.ACTIVE_LEARNING.INITIAL_DELTA
    cfg.PSEUDO_LABELS_THRESHOLD = args.pseudo_labels_threshold
    cfg.TRAIN_PSEUDO_LABELS = args.train_pseudo_labels
    cfg.ALPHA_LOWER_BOUND = args.alpha_lower_bound
    cfg.ALPHA_UPPER_BOUND = args.alpha_upper_bound
    cfg.LOCAL_ALPHA = args.local_alpha
    cfg.LOCAL_ALPHA_ORACLE_METHOD = args.local_alpha_oracle_method
    cfg.USE_K_TOP50_MASK = args.use_k_top50_mask
    cfg.UPDATE_K_MATRIX = args.update_k_matrix
    cfg.UPDATE_K_MATRIX_FACTOR = args.update_k_matrix_factor
    cfg.UPDATE_C_MATRIX = args.update_c_matrix
    cfg.DECREASING_ALPHA = args.decrease_alpha
    debug = cfg.DEBUG = args.debug
    cfg.HIGH_BUDGET = args.high_budget
    cfg.CONFIDENCE_METHOD = args.confidence_method
    if debug:
        cfg.RNG_SEED = 0
    else:
        cfg.RNG_SEED = args.seed
    cfg.ALPHA = args.alpha
    # Set CK_ALPHA separately if provided, otherwise use the same as ALPHA
    cfg.CK_ALPHA = args.ck_alpha if args.ck_alpha is not None else args.alpha
    cfg.CK_C_NORMALIZATION = args.ck_c_normalization
    # Set CK_K_SPARSITY_THRESHOLD separately if provided, otherwise use the same as K_SPARSITY_THRESHOLD
    cfg.CK_K_SPARSITY_THRESHOLD = args.ck_K_sparsity_threshold
    # Set CK_SPARSE_K separately if provided, otherwise use the same as SPARSE_K
    cfg.CK_SPARSE_K = args.ck_sparse_K
    cfg.CK_K = args.ck_k
    cfg.LOCAL_RBF_ALPHA = args.local_rbf_alpha
    cfg.DEGREE_NORMALIZATION_METHOD = args.degree_normalization_method
    cfg.NORM_IMPORTANCE = args.norm_importance
    cfg.SPARSE_DS = args.sparse_ds
    cfg.PSEUDO_LABEL_WEIGHT = args.pseudo_label_weight
    cfg.PSEUDO_LABEL_WEIGHTING_FUNC = args.pseudo_label_weighting_func
    cfg.PSEUDO_LABEL_CLASS_THRESHOLD = args.pseudo_label_class_threshold
    cfg.PSEUDO_LABEL_CLASS_THRESHOLD_TOPK = args.pseudo_label_class_threshold_topk
    cfg.SWITCH_ALPHA_LOW_TO_HIGH = args.switch_alpha_low_to_high
    cfg.SWITCH_ALPHA_HIGH_TO_LOW = args.switch_alpha_high_to_low
    cfg.SWITCH_ALPHA_ALLTIME = args.switch_alpha_alltime
    cfg.ALPHA_INIT_MODE = args.alpha_init_mode
    cfg.ALPHA_VECTOR_PATH = args.alpha_vector_path
    cfg.NN_FUSION = args.nn_fusion
    cfg.NN_FUSION_START_ROUND = args.nn_fusion_start_round
    cfg.NN_FUSION_INTERVAL = args.nn_fusion_interval
    cfg.DISTILLATION_TRAINING = args.distillation_training
    cfg.BAYESIAN_DISTILLATION = args.bayesian_distillation
    cfg.BAYESIAN_DELTA = args.bayesian_delta
    cfg.BAYESIAN_REF_PRIOR = args.bayesian_ref_prior
    cfg.DISTILLATION_THRESHOLD = args.distillation_threshold
    cfg.RAW_DISTILLATION_THRESHOLD = args.raw_distillation_threshold
    cfg.DISTILLATION_TEMPERATURE = args.distillation_temperature
    cfg.DISTILL_FACTOR = args.distill_factor
    cfg.DISTILL_SOFT_TARGET_NORMALIZATION = args.distill_soft_target_normalization
    cfg.ALPHA_DECAY_GAMMA = args.alpha_decay_gamma
    cfg.CALC_METHOD = args.calc_method
    cfg.SWITCH_METHOD_AT_ROUND = args.switch_method_at_round
    cfg.SWITCH_TO_METHOD = args.switch_to_method
    cfg.PROPAGATE_C_LABELS = args.propagate_c_labels
    cfg.PROPAGATION_ITERATIONS = args.propagation_iterations
    cfg.PROPAGATION_CONFIDENCE_THRESHOLD = args.propagation_confidence_threshold
    cfg.PROPAGATION_SENDER_MODE = args.propagation_sender_mode
    cfg.PROPAGATION_SENDER_THRESHOLD = args.propagation_sender_threshold
    cfg.PROPAGATION_SENDER_BUDGET = args.propagation_sender_budget
    cfg.PROPAGATION_RECEIVER_MODE = args.propagation_receiver_mode
    cfg.PROPAGATION_RECEIVER_THRESHOLD = args.propagation_receiver_threshold
    cfg.PROPAGATION_RECEIVER_BUDGET = args.propagation_receiver_budget
    cfg.PROPAGATION_DECAY_RATE = args.propagation_decay_rate
    cfg.MONITOR_GRAD_NORMS = args.monitor_grad_norms
    cfg.GRAD_NORM_LOG_FREQ = args.grad_norm_log_freq

    print("RNG_SEED is set to {}".format(cfg.RNG_SEED))
    # cfg.MODEL.LINEAR_FROM_FEATURES = args.linear_from_features
    cfg.ACTIVE_LEARNING.A_LOGISTIC = args.a_logistic
    cfg.ACTIVE_LEARNING.K_LOGISTIC = args.k_logistic
    cfg.EVAL_MODEL_TYPE = args.eval_model_type
    cfg.ACTIVE_LEARNING.MAX_ITER = args.max_iter
    cfg.ACTIVE_LEARNING.START_ITER = args.start_iter
    cfg.ACTIVE_LEARNING.EVAL_FREQUENCY = args.eval_frequency

    cfg.FIXED_DISTILL_MLP = args.fixed_distill_mlp
    cfg.FIXED_PROPAGATION_V1 = args.fixed_propagation_v1
    cfg.FIXED_INITIAL_SET_DIR = args.fixed_initial_set_dir

    define_eval_model_type(cfg, debug)
    if args.max_epoch is not None:
        cfg.OPTIM.MAX_EPOCH = args.max_epoch

    main(cfg)


# --cfg ../configs/cifar10/al/RESNET18.yaml --al probcover --exp-name auto --initial_size 0 --budget 10 --initial_delta 0.75
# """
# #!/bin/bash
# #SBATCH --mem=20g ## amount of memory
# #SBATCH -c 20 ## number of cpu
# #SBATCH --time=40:00:00 ## time limit for the process
# #SBATCH --gres=gpu:a10:1,vmem:20g ## gpu::<type of gpu>:<number of gpus>,vmem:<amount of memory on gpu (max 10gb/24gb/40gb)>
# #SBATCH --array=0-2 ## array of jobs to run (job 0, job 1, job 2)
# #SBATCH --output /cs/labs/daphna/maorni_cse/SupContrast/run_outputs/run_exp_%A_%a.txt
# ## those are the type of the potential gpus : gpu:rtx2080:1, gpu:a10:1, gpu:g4:1 (can check in the university for more)
# dir=/cs/labs/daphna/maorni_cse/SupContrast
#
# cd $dir
#
# source /cs/labs/daphna/maorni_cse/virs/reembedding/bin/activate
#
# python run_it.py --job_id ${SLURM_ARRAY_TASK_ID}
# """

###some utils functions should be copied to train_al.py instead of train.py

### need to change it to save the metrics in the test function of the eval model, it is not saving by default

## change in the sbatch to my paths instead of Maor

## scancel <job_id> to cancel the job

## sbatch <script_name> to run the job

## sbatch <script_name> --killable to run the job with killable resources


## change alpha values (on from features)