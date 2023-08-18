import xarray as xr
from _typeshed import Incomplete as Incomplete

config_dict: Incomplete
logger: Incomplete

def append_or_create_zarr(data_out: xr.Dataset, config: dict) -> None: ...
def load_data(config: dict) -> None: ...
