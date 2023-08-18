import torch
from _typeshed import Incomplete as Incomplete
from torch import nn as nn, optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch_geometric.data import Data as Data
from torch_geometric.loader import DataLoader as DataLoader
from typing import List, Optional, Tuple, Union

logger: Incomplete

class TrainingConfigGNN(dict):
    loader_train_in: DataLoader
    loader_train_out: DataLoader
    optimizer: Union[optim.Optimizer, torch.optim.Optimizer, Adam]
    scheduler: Union[CyclicLR, torch.optim.lr_scheduler.CyclicLR]
    loss_fn: Union[nn.Module, nn.MSELoss]
    mask: Optional[torch.Tensor]
    epochs: int
    device: str
    seed: int
    def __init__(self, loader_train_in, loader_train_out, optimizer, scheduler, loss_fn, mask, epochs, device, seed) -> None: ...

class EvaluationConfigGNN(dict):
    loader_in: DataLoader
    loader_out: DataLoader
    loss_fn: Union[nn.Module, nn.MSELoss]
    mask: Optional[torch.Tensor]
    device: str
    seed: int
    def __init__(self, loader_in, loader_out, loss_fn, mask, device, seed) -> None: ...

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

class TopKPoolingLayer(torch.nn.Module):
    pool: Incomplete
    def __init__(self, gnn_configs: GNNConfig) -> None: ...
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

class GNNModel(torch.nn.Module):
    conv_layers: Incomplete
    pool_layer: Incomplete
    activation: Incomplete
    def __init__(self, gnn_configs: GNNConfig) -> None: ...
    def forward(self, data: Data) -> torch.Tensor: ...
    def train_with_configs(self, configs_train_gnn: TrainingConfigGNN) -> None: ...
    def eval_with_configs(self, configs_eval_gnn: EvaluationConfigGNN) -> tuple[float, List[torch.Tensor]]: ...
