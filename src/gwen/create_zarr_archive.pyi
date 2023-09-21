# Third-party
import xarray as xr
from _typeshed import Incomplete

# First-party
from gwen.loggers_configs import load_config as load_config
from gwen.loggers_configs import setup_logger as setup_logger

config_dict: Incomplete
logger: Incomplete

def append_or_create_zarr(data_out: xr.Dataset, config: dict) -> None: ...
def load_data(config: dict) -> None: ...
