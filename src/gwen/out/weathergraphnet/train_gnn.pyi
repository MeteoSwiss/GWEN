# Standard library
from typing import Tuple

# Third-party
import torch
import xarray as xr
from _typeshed import Incomplete as Incomplete
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader as DataLoader

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
