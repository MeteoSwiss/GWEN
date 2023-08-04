# Standard library
from typing import Any
from typing import List
from typing import Tuple

# Third-party
import mlflow  # type: ignore
import numpy as np
import xarray as xr
from _typeshed import Incomplete
from matplotlib import animation
from torch import nn
from torch.utils.data import Dataset

# First-party
from weathergraphnet.logger import setup_logger as setup_logger

logger: Incomplete

class CRPSLoss(nn.Module):
    def __init__(self) -> None: ...
    def forward(self, outputs: Any, target: Any, dim: int = ...) -> Any: ...

class EnsembleVarRegLoss(nn.Module):
    alpha: Incomplete
    def __init__(self, alpha: float = ...) -> None: ...
    def forward(self, outputs: Any, target: Any) -> Any: ...

class MaskedLoss(nn.Module):
    loss_fn: Incomplete
    def __init__(self, loss_fn: nn.Module) -> None: ...
    def forward(self, outputs: Any, target: Any, mask: Any) -> Any: ...

class MyDataset(Dataset):
    data: Incomplete
    split: Incomplete
    train_indices: Incomplete
    test_indices: Incomplete
    def __init__(self, data: xr.Dataset, split: int) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]: ...
    def __iter__(self): ...

def animate(
    data: xr.Dataset, member: int = ..., preds: str = ...
) -> animation.FuncAnimation: ...
def create_animation(data: dict, member: int, preds: str) -> str: ...
def downscale_data(data: xr.Dataset, factor: int) -> xr.Dataset: ...
def get_runs(experiment_name: str) -> List[mlflow.entities.Run]: ...
def load_best_model(experiment_name: str) -> nn.Module: ...
def load_config(): ...
def load_config_and_data() -> Tuple[dict, xr.Dataset, xr.Dataset]: ...
def load_data(config: dict) -> Tuple[xr.Dataset, xr.Dataset]: ...
def setup_mlflow() -> Tuple[str, str]: ...
def suppress_warnings() -> None: ...
