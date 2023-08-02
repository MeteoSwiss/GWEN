import torch
from _typeshed import Incomplete
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, StepLR
from torch.utils.data import DataLoader
from typing import Optional, Union
from weathergraphnet.models import UNet as UNet
from weathergraphnet.utils import EnsembleVarianceRegularizationLoss as EnsembleVarianceRegularizationLoss, MaskedLoss as MaskedLoss, MyDataset as MyDataset, create_animation as create_animation, load_best_model as load_best_model, load_config_and_data as load_config_and_data, setup_logger as setup_logger, setup_mlflow as setup_mlflow

logger: Incomplete

class TrainingConfig(dict):
    dataloader: DataLoader
    optimizer: Union[torch.optim.Optimizer, Adam]
    scheduler: Union[CyclicLR, StepLR]
    loss_fn: Union[nn.Module, MaskedLoss, EnsembleVarianceRegularizationLoss, nn.MSELoss, nn.DataParallel]
    mask: Optional[torch.Tensor]
    num_epochs: int
    device: str
    seed: int
    def __init__(self, dataloader, optimizer, scheduler, loss_fn, mask, num_epochs, device, seed) -> None: ...

class EvaluateConfig(dict):
    dataloader: DataLoader
    loss_fn: Union[nn.Module, MaskedLoss, EnsembleVarianceRegularizationLoss, nn.MSELoss, nn.DataParallel]
    mask: Optional[torch.Tensor]
    device: str
    seed: int
    def __init__(self, dataloader, loss_fn, mask, device, seed) -> None: ...
