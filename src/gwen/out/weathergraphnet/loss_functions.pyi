from _typeshed import Incomplete
from torch import nn
from typing import Any
from gwen.loggers_configs import setup_logger as setup_logger

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
