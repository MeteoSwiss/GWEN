import torch
from _typeshed import Incomplete
from gwen.loggers_configs import setup_logger as setup_logger, setup_mlflow as setup_mlflow, suppress_warnings as suppress_warnings
from gwen.utils import GraphDataset as GraphDataset
from torch import nn, optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from typing import List, Optional, Union

logger: Incomplete

class TrainingConfigGNN(dict):
    dataset: GraphDataset
    optimizer: Union[optim.Optimizer, torch.optim.Optimizer, Adam]
    scheduler: Union[CyclicLR, torch.optim.lr_scheduler.CyclicLR]
    loss_fn: Union[nn.Module, nn.MSELoss]
    batch_size: int
    mask: Optional[torch.Tensor]
    epochs: int
    device: str
    seed: int
    def __init__(self, dataset, optimizer, scheduler, loss_fn, batch_size, mask, epochs, device, seed) -> None: ...

class EvaluationConfigGNN(dict):
    dataset: GraphDataset
    loss_fn: Union[nn.Module, nn.MSELoss]
    mask: Optional[torch.Tensor]
    device: str
    batch_size: int
    seed: int
    def __init__(self, dataset, loss_fn, mask, device, batch_size, seed) -> None: ...

class GNNConfig(dict):
    nodes_in: int
    nodes_out: int
    channels_in: int
    channels_out: int
    hidden_feats: int
    def __init__(self, nodes_in, nodes_out, channels_in, channels_out, hidden_feats) -> None: ...

class DownConvLayers(torch.nn.Module):
    conv1: Incomplete
    conv2: Incomplete
    conv3: Incomplete
    conv4: Incomplete
    conv5: Incomplete
    def __init__(self, gnn_configs: GNNConfig) -> None: ...
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor: ...

class UpConvLayers(torch.nn.Module):
    upconv1: Incomplete
    upconv2: Incomplete
    upconv3: Incomplete
    upconv4: Incomplete
    upconv5: Incomplete
    def __init__(self, gnn_configs: GNNConfig) -> None: ...
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor: ...

class GCNConvLayers(torch.nn.Module):
    down_conv_layers: Incomplete
    up_conv_layers: Incomplete
    def __init__(self, gnn_configs: GNNConfig) -> None: ...
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor: ...

def loss_func(output, target, target_mask): ...

class GNNModel(torch.nn.Module):
    conv_layers: Incomplete
    activation: Incomplete
    def __init__(self, gnn_configs: GNNConfig) -> None: ...
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor: ...
    def train_with_configs(self, rank, configs_train_gnn: TrainingConfigGNN, world_size) -> None: ...
    def eval_gnn_with_configs(self, rank, configs_eval_gnn: EvaluationConfigGNN, world_size, queue) -> tuple[float, List[torch.Tensor]]: ...
