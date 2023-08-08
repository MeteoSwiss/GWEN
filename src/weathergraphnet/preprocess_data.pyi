# Standard library
from typing import Tuple

# Third-party
import xarray as xr
from _typeshed import Incomplete

# First-party
from weathergraphnet.loggers_configs import setup_logger as setup_logger

logger: Incomplete


def split_data(
    data: xr.Dataset, test_size: float = ..., random_state: int = ...
) -> Tuple[xr.Dataset, xr.Dataset]: ...


def normalize_data(
    data_train_raw: xr.Dataset, data_test_raw: xr.Dataset, method: str = ...
) -> Tuple[xr.Dataset, xr.Dataset]: ...


compressor: Incomplete
data_zarr: Incomplete
data_theta: Incomplete
data_train: Incomplete
data_test: Incomplete
data_train_norm: Incomplete
data_test_norm: Incomplete
