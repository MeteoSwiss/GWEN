# Standard library
from typing import List
from typing import Tuple

# Third-party
import mlflow  # type: ignore
import numpy as np
import torch
import xarray as xr
from _typeshed import Incomplete
from matplotlib import animation
from torch import nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data  # type: ignore
from torch_geometric.data import Dataset as Dataset_GNN  # type: ignore

# First-party
from gwen.create_gif import get_member_name as get_member_name
from gwen.loggers_configs import load_config as load_config
from gwen.loggers_configs import setup_logger as setup_logger

logger: Incomplete
config: Incomplete

class ConvDataset(Dataset):
    data: Incomplete
    split: Incomplete
    input_indices: Incomplete
    target_indices: Incomplete
    def __init__(self, data: xr.Dataset, split: int) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def __iter__(self): ...
    def get_target_indices(self) -> np.ndarray: ...

class GraphDataset(Dataset_GNN):
    data: Incomplete
    split: Incomplete
    transform: Incomplete
    pre_transform: Incomplete
    nodes: Incomplete
    edge_index: Incomplete
    input_indices: Incomplete
    target_indices: Incomplete
    channels: Incomplete
    def __init__(
        self,
        xr_data,
        split,
        transform: Incomplete | None = ...,
        pre_transform: Incomplete | None = ...,
    ) -> None: ...
    def len(self): ...
    def get(self, idx) -> Data: ...

def animate(data: xr.Dataset, member: str, preds: str) -> animation.FuncAnimation: ...
def create_animation(
    data: dict, member_pred: int, member_target: int, preds: str
) -> str: ...
def downscale_data(data: xr.Dataset, factor: int) -> xr.Dataset: ...
def get_runs(experiment_name: str) -> List[mlflow.entities.Run]: ...
def load_best_model(experiment_name: str) -> nn.Module: ...
def load_config_and_data() -> Tuple[dict, xr.Dataset, xr.Dataset]: ...
def load_data(configs: dict) -> Tuple[xr.Dataset, xr.Dataset]: ...
