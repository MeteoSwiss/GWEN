"""Import and preprocess the data."""

# pylint: disable=R0801

# Third-party
import numcodecs  # type: ignore
import numpy as np

# Third-party import numpy as np
import xarray as xr
from pyprojroot import here  # type: ignore


def split_data(data, test_size=0.3, random_state=42):
    """Split the data into training and testing sets.

    Args:
        data (xarray.Dataset): The data to split. test_size (float): The proportion of
        the data to use for testing. random_state (int): The random seed to use for
        splitting.

    Returns:
        Two xarray.Datasets containing the training and testing data.

    """
    # Get the number of samples in the data
    n_samples = len(data.time)

    # Shuffle the indices of the samples # pylint: disable = no-member
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


# Data Import
data_zarr = xr.open_zarr(str(here()) + "/data/data_combined.zarr")
data_theta = (
    data_zarr["theta_v"]
    .sel(ncells=slice(2632, None))
    .transpose("time", "member", "height", "ncells")
)

# Check for missing data print(np.argwhere(np.isnan(data_theta.to_array().to_numpy())))
# data_theta = data_theta.interpolate_na(dim="x", method="linear",
# fill_value="extrapolate")

# Split the data into training and testing sets
data_train, data_test = split_data(data_theta)

center = data_train.mean()
scale = data_train.std()

with open(str(here()) + "/data/scaling.txt", "w", encoding="utf-8") as f:
    f.write("center: " + str(center.values) + "\n" + "scale: " + str(scale.values))

data_train_norm = (data_train - center) / scale
data_test_norm = (data_test - center) / scale

# Create a compressor using the zlib codec
compressor = numcodecs.Zlib(level=1)

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
)
