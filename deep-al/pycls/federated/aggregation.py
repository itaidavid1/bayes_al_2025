from collections import OrderedDict
from typing import Dict, Iterable, List

import torch


def fedavg(client_state_dicts: List[Dict[str, torch.Tensor]], client_weights: Iterable[float]) -> Dict[str, torch.Tensor]:
    client_weights = list(client_weights)
    total = float(sum(client_weights))
    if total <= 0:
        raise ValueError("Sum of client weights must be > 0 for FedAvg.")

    avg_state = OrderedDict()
    for key in client_state_dicts[0].keys():
        weighted = None
        for state, w in zip(client_state_dicts, client_weights):
            tensor = state[key].detach().clone().float()
            weighted = tensor * (w / total) if weighted is None else weighted + tensor * (w / total)
        avg_state[key] = weighted
    return avg_state


def fedprox_step_loss(model: torch.nn.Module, global_params: Dict[str, torch.Tensor], mu: float) -> torch.Tensor:
    if mu <= 0:
        # keep graph-friendly zero on the right device
        return next(model.parameters()).new_tensor(0.0)
    prox = next(model.parameters()).new_tensor(0.0)
    for name, param in model.named_parameters():
        if name in global_params:
            ref = global_params[name].to(param.device)
            prox = prox + torch.sum((param - ref) ** 2)
    return 0.5 * mu * prox
