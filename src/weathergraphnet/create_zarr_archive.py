"""Load weather data from NetCDF files and store it in a Zarr archive."""
# Standard library
import os
import re
import socket
from typing import List
from typing import Pattern

# Third-party
import numcodecs  # type: ignore
import xarray as xr

# First-party
from weathergraphnet.utils import setup_logger

hostname: str = socket.gethostname()
if "nid" in hostname:
    SCRATCH: str = "/scratch/e1000/meteoswiss/scratch/sadamov/"
else:
    SCRATCH = "/scratch/sadamov/"

data_path: str = f"{SCRATCH}/icon/icon-nwp/cpu/experiments/"
filename_regex: str = r"atmcirc-straka_93_(.*)DOM01_ML_20080801T000000Z.nc"
filename_pattern: Pattern[str] = re.compile(filename_regex)
zarr_path: str = f"{SCRATCH}/icon/icon-nwp/cpu/experiments/data_combined_2.zarr"
folders: List[str] = os.listdir(data_path)
compressor = numcodecs.Zlib(level=1)

logger = setup_logger()


def load_data() -> None:
    """Load weather data from NetCDF files and store it in a Zarr archive.

    The data is assumed to be in a specific directory structure and file naming
    convention, which is checked using regular expressions. The loaded data is chunked
    along the "member" and "time" dimensions for efficient storage in the Zarr archive.
    If the Zarr archive already exists, new data is appended to it. Otherwise, a new
    Zarr archive is created.

    Args:
        None

    Returns:
        None

    """
    for folder in folders:
        if folder.startswith("atmcirc-straka_93_"):
            file_path: str = os.path.join(data_path, folder)
            files: List[str] = os.listdir(file_path)
            for file in files:
                try:
                    match = filename_pattern.match(file)
                    if match:
                        data: xr.Dataset = xr.open_dataset(
                            os.path.join(file_path, file), engine="netcdf4"
                        )
                        data = data.assign_coords(member=match.group(1))
                        data = data.expand_dims({"member": 1})
                        data = data.chunk(
                            chunks={
                                "time": 32,
                                "member": -1,
                                "height": -1,
                                "height_2": -1,
                                "height_3": -1,
                            },
                            compressor=compressor,
                        )
                        append_or_create_zarr(data)
                        print(f"Loaded {file}", flush=True)
                except FileNotFoundError as e:
                    handle_file_not_found_error(file, e)
                except ValueError as e:
                    handle_invalid_file_format_error(file, e)


def append_or_create_zarr(data: xr.Dataset) -> None:
    if os.path.exists(zarr_path):
        data.to_zarr(
            store=zarr_path,
            mode="a",
            encoding={"theta_v": {"compressor": compressor}},
            consolidated=True,
            append_dim="member",
        )
    else:
        data.to_zarr(
            zarr_path,
            mode="w",
            encoding={"theta_v": {"compressor": compressor}},
            consolidated=True,
        )


def handle_file_not_found_error(file: str, e: Exception) -> None:
    """Handle the case when a file is not found."""
    logger.error("File not found: %s. %s", file, str(e))
    raise e


def handle_invalid_file_format_error(file: str, e: Exception) -> None:
    """Handle the case when a file has an invalid format."""
    logger.error("Invalid file format: %s. %s", file, str(e))
    raise e


load_data()
