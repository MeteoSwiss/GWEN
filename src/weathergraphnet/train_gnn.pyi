# Standard library
from typing import Tuple

# Third-party
import torch
import xarray as xr
from _typeshed import Incomplete
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader  # type: ignore

# First-party
from weathergraphnet.loggers_configs import setup_logger as setup_logger
from weathergraphnet.models_gnn import EvaluationConfigGNN as EvaluationConfigGNN
from weathergraphnet.models_gnn import GNNConfig as GNNConfig
from weathergraphnet.models_gnn import GNNModel as GNNModel
from weathergraphnet.models_gnn import TrainingConfigGNN as TrainingConfigGNN
from weathergraphnet.utils import MaskedLoss as MaskedLoss
from weathergraphnet.utils import MyDataset as MyDataset
from weathergraphnet.utils import create_animation as create_animation
from weathergraphnet.utils import load_config_and_data as load_config_and_data
from weathergraphnet.utils import setup_mlflow as setup_mlflow

logger: Incomplete


def create_data_loader(
    data: xr.Dataset, edge_index: torch.Tensor, nodes: int, batch: int
) -> DataLoader: ...


class CustomSampler(Sampler):
    data: Incomplete
    edge_index: Incomplete
    batch: Incomplete

    def __init__(
        self, data: xr.Dataset, edge_index: torch.Tensor, batch: int
    ) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, index: int) -> Tuple[xr.Dataset, xr.Dataset]: ...


def create_data_sampler(
    data: xr.Dataset, edge_index: torch.Tensor, nodes: int, batch: int, workers: int
) -> DataLoader: ...
def main() -> None: ...
