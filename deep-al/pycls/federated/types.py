from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class ClientPartition:
    client_id: int
    indices: np.ndarray


@dataclass
class ClientUpdate:
    client_id: int
    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    metrics: Dict[str, float]
