"""Functions to import and preprocess weather data.

Functions:
- split_data(data: xr.Dataset, test_size: float = 0.3, random_state: int = 42) ->
    Tuple[xr.Dataset, xr.Dataset]: Splits the data into training and testing sets.
- normalize_data(data_train_raw: xr.Dataset, data_test_raw: xr.Dataset,
    method: str = "mean") -> Tuple[xr.Dataset, xr.Dataset]:
    Normalizes the training and testing data.
"""

# Standard library
from typing import Tuple

# Third-party
import numcodecs  # type: ignore
import numpy as np
import xarray as xr
from pyprojroot import here

# First-party
from weathergraphnet.utils import setup_logger

logger = setup_logger()


def split_data(
    data: xr.Dataset, test_size: float = 0.3, random_state: int = 42
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Split the data into training and testing sets.

    Args:
        data (xr.Dataset): The data to split.
        test_size (float, optional): The proportion of the data to use for testing.
            Defaults to 0.3.
        random_state (int, optional): The random seed used for splitting. Default is 42.

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: Two xarray.Datasets containing the training and
        testing data.

    """
    try:
        # Get the number of samples in the data
        n_samples = len(data.time)

        # Shuffle the indices of the samples
        indices = np.random.RandomState(seed=random_state).permutation(n_samples)

        # Calculate the number of samples in the test set
        n_test_samples = int(test_size * n_samples)

        # Select the test samples
        test_indices = indices[:n_test_samples]
        test_data = data.isel(time=test_indices)

        # Select the training samples
        train_indices = indices[n_test_samples:]
        train_data = data.isel(time=train_indices)

        return train_data, test_data

    except Exception as error:
        logger.exception("Error occurred while splitting data: %s", str(error))
        raise ValueError(
            "Failed to split data. Please check the input data and parameters."
        ) from error


def normalize_data(
    data_train_raw: xr.Dataset, data_test_raw: xr.Dataset, method: str = "mean"
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Normalize the training and testing data.

    Args:
        data_train_raw (xr.Dataset): The training data to normalize. data_test_raw
        (xr.Dataset): The testing data to normalize.
        method (str, optional): The normalization method to use.
            Must be 'mean' or 'median'. Defaults to "mean".

    Raises:
        ValueError: If an invalid normalization method is provided.

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: The normalized training and testing data.

    """
    try:
        if method == "mean":
            center: np.floating = np.array(data_train_raw.mean().values)[0]
            scale: np.floating = np.array(data_train_raw.std().values)[0]

        elif method == "median":
            center = np.nanmedian(data_train_raw)
            centered_ds = data_train_raw - center
            scale = np.nanmedian(np.abs(centered_ds))

        else:
            raise ValueError("Invalid method. Must be 'mean' or 'median'.")

        data_train_scaled = (data_train_raw - center) / scale
        data_test_scaled = (data_test_raw - center) / scale

        with open(str(here()) + "/data/scaling.txt", "w", encoding="utf-8") as f:
            f.write("center: " + str(center) + "\n" + "scale: " + str(scale))
        return data_train_scaled, data_test_scaled

    except Exception as error:
        logger.exception("Error occurred while normalizing data: %s", str(error))
        raise ValueError(
            "Failed to normalize data. Please check the input data and parameters."
        ) from error


# Create a compressor using the zlib codec
compressor = numcodecs.Zlib(level=1)

# Data Import
try:
    data_zarr = xr.open_zarr(
        str(here()) + "/data/data_combined.zarr", consolidated=True
    )
    data_theta = (
        data_zarr["theta_v"]
        .sel(ncells=slice(2632, None))
        .transpose("time", "member", "height", "ncells")
    )

    # Check for missing data
    if np.isnan(data_theta.to_numpy()).any():
        logger.warning("Data contains missing values.")

    # Interpolate missing data
    data_theta = data_theta.interpolate_na(
        dim="x", method="linear", fill_value="extrapolate"
    )

except Exception as e:
    logger.exception("Error occurred while importing data: %s", str(e))
    raise ValueError(
        "Failed to import data. Please check the input data and parameters."
    ) from e

# Split the data into training and testing sets
try:
    data_train, data_test = split_data(data_theta)

except ValueError as e:
    logger.exception("Error occurred while splitting data: %s", str(e))
    raise

# Normalize the training and testing data
try:
    data_train_norm, data_test_norm = normalize_data(data_train, data_test, "mean")

except ValueError as e:
    logger.exception("Error occurred while normalizing data: %s", str(e))
    raise

# Chunk and compress the normalized data and save to zarr files
try:
    data_train_norm.chunk(
        chunks={
            "time": 32,
            "member": len(data_train_norm.member),
            "height": -1,
            "ncells": -1,
        }
    ).to_zarr(
        str(here()) + "/data/data_train.zarr",
        encoding={"theta_v": {"compressor": compressor}},
        mode="w",
    )

    data_test_norm.chunk(
        chunks={
            "time": 32,
            "member": len(data_test_norm.member),
            "height": -1,
            "ncells": -1,
        }
    ).to_zarr(
        str(here()) + "/data/data_test.zarr",
        encoding={"theta_v": {"compressor": compressor}},
        mode="w",
    )

except Exception as e:
    logger.exception("Error occurred while chunking and compressing data: %s", str(e))
    raise ValueError(
        "Failed to chunk and compress data."
        " Please check the input data and parameters."
    ) from e
