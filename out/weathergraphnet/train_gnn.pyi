import torch
import xarray as xr
from _typeshed import Incomplete
from torch import nn, optim
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader
from typing import Optional, Tuple, Union
from weathergraphnet.models import GNNModel as GNNModel
from weathergraphnet.utils import EnsembleVarianceRegularizationLoss as EnsembleVarianceRegularizationLoss, MaskedLoss as MaskedLoss, MyDataset as MyDataset, create_animation as create_animation, load_best_model as load_best_model, load_config_and_data as load_config_and_data, setup_logger as setup_logger, setup_mlflow as setup_mlflow

logger: Incomplete

class GNNConfig(dict):
    nodes_in: int
    nodes_out: int
    in_channels: int
    out_channels: int
    hidden_feats: int
    def __init__(self, nodes_in, nodes_out, in_channels, out_channels, hidden_feats) -> None: ...

def create_data_loader(data: xr.Dataset, edge_index: torch.Tensor, nodes: int, batch: int) -> DataLoader: ...

class CustomSampler(Sampler):
    data: Incomplete
    edge_index: Incomplete
    batch: Incomplete
    def __init__(self, data: xr.Dataset, edge_index: torch.Tensor, batch: int) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, index: int) -> Tuple[xr.Dataset, xr.Dataset]: ...

class TrainingConfig(dict):
    loader_train_in: DataLoader
    loader_train_out: DataLoader
    optimizer: Union[optim.Optimizer, torch.optim.Optimizer, Adam]
    scheduler: Union[CyclicLR, torch.optim.lr_scheduler.CyclicLR]
    loss_fn: Union[nn.Module, EnsembleVarianceRegularizationLoss, nn.MSELoss, nn.DataParallel]
    mask: Optional[torch.Tensor]
    num_epochs: int
    device: str
    seed: int
    def __init__(self, loader_train_in, loader_train_out, optimizer, scheduler, loss_fn, mask, num_epochs, device, seed) -> None: ...

class EvaluationConfig(dict):
    loader_in: DataLoader
    loader_out: DataLoader
    loss_fn: Union[nn.Module, EnsembleVarianceRegularizationLoss, nn.MSELoss, nn.DataParallel]
    mask: Optional[torch.Tensor]
    device: str
    seed: int
    def __init__(self, loader_in, loader_out, loss_fn, mask, device, seed) -> None: ...

def create_data_sampler(data: xr.Dataset, edge_index: torch.Tensor, nodes: int, batch: int, workers: int) -> DataLoader: ...
