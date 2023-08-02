import xarray as xr
from _typeshed import Incomplete
from typing import List, Pattern
from weathergraphnet.utils import setup_logger as setup_logger

hostname: str
SCRATCH: str
data_path: str
filename_regex: str
filename_pattern: Pattern[str]
zarr_path: str
folders: List[str]
compressor: Incomplete
logger: Incomplete

def load_data() -> None: ...
def append_or_create_zarr(data: xr.Dataset) -> None: ...
def handle_file_not_found_error(file: str, e: Exception) -> None: ...
def handle_invalid_file_format_error(file: str, e: Exception) -> None: ...
