"""Load weather data from NetCDF files and store it in a Zarr archive."""
# Standard library
import os
import re
from typing import List

# Third-party
import numcodecs  # type: ignore
import xarray as xr

# First-party
from gwen.loggers_configs import setup_logger
from gwen.loggers_configss_configs import load_config

config_dict = load_config()

config_dict.update(
    {
        "folders": os.listdir(config_dict["data_path"]),
        "filename_pattern": re.compile(config_dict["filename_regex"]),
        "compressor": numcodecs.Zlib(level=config_dict["zlib_compression_level"]),
    }
)

logger = setup_logger()


def append_or_create_zarr(data_out: xr.Dataset, config: dict) -> None:
    """Append data to an existing Zarr archive or create a new one."""
    if os.path.exists(config["zarr_path"]):
        data_out.to_zarr(
            store=config["zarr_path"],
            mode="a",
            consolidated=True,
            append_dim="member",
        )
    else:
        data_out.to_zarr(
            config["zarr_path"],
            mode="w",
            consolidated=True,
        )


def load_data(config: dict) -> None:
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
    for folder in config["folders"]:
        if folder.startswith("atmcirc-straka_93_"):
            file_path: str = os.path.join(config["data_path"], folder)
            files: List[str] = os.listdir(file_path)
            for file in files:
                try:
                    match = config["filename_pattern"].match(file)
                    if not match:
                        continue

                    data: xr.Dataset = xr.open_dataset(
                        os.path.join(file_path, file), engine="netcdf4"
                    )

                    # Specify the encoding for theta_v
                    if "theta_v" in data:
                        data["theta_v"].encoding = {"compressor": config["compressor"]}

                    data = data.assign_coords(member=match.group(1))
                    data = data.expand_dims({"member": 1})
                    data = data.chunk(
                        chunks={
                            "time": 32,
                            "member": -1,
                            "height": -1,
                            "height_2": -1,
                            "height_3": -1,
                        }
                    )
                    append_or_create_zarr(data, config)
                    logger.info("Loaded %s", file)
                except (FileNotFoundError, OSError) as e:
                    logger.error("Error loading %s: %s", file, e)


if __name__ == "__main__":
    load_data(config_dict)
