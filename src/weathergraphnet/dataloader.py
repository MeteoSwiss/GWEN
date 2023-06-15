"""Functionality for loading weather data from NetCDF files.

The data is assumed to be in a specific directory structure and file naming convention,
which is checked using regular expressions. The loaded data is chunked along the
"member" and "time" dimensions for efficient storage in the Zarr archive. If the Zarr
archive already exists, new data is appended to it. Otherwise, a new Zarr archive is
created.
"""
# Import the necessary libraries
# Standard library
import os
import re

# Third-party
import xarray as xr  # type: ignore

# Define the path to the directory containing the data files
data_path = "/scratch/sadamov/icon/icon-nwp/cpu/experiments/"

# Define a regular expression to match the filename of the data files
filename_regex = r"atmcirc-straka_93_(.*)DOM01_ML_20080801T000000Z.nc"

# Compile the regular expression into a pattern object
filename_pattern = re.compile(filename_regex)

# Define the path to the Zarr archive where the data will be stored
zarr_path = "/scratch/sadamov/icon/icon-nwp/cpu/experiments/data_combined.zarr"

# Get a list of all the folders in the data directory
folders = os.listdir(data_path)

# Loop over each folder and load the data files
for folder in folders:
    # Check if the folder matches the naming convention for the data files
    if folder.startswith("atmcirc-straka_93_"):
        # Define the path to the folder containing the data files
        file_path = os.path.join(data_path, folder)
        # Get a list of all the files in the folder
        files = os.listdir(file_path)
        # Loop over each file and load the data if it matches the filename pattern
        for file in files:
            match = filename_pattern.match(file)
            if match:
                # Load the data using xarray
                data = xr.open_dataset(os.path.join(file_path, file), engine="netcdf4")
                # Add a new dimension called "member" and set its values based on the
                # matched string
                data.coords["member"] = xr.DataArray([match.group(1)], dims=["member"])
                # Make "member" a dimension as well as a coordinate
                data = data.set_index(member="member")
                # Chunk the data along the "member" and "time" dimensions for efficient
                # storage
                data = data.chunk(
                    chunks={
                        "member": 1,
                        "time": -1,
                        "height": -1,
                        "height_2": -1,
                        "height_3": -1,
                    }
                )
                # Check if the Zarr archive already exists
                if os.path.exists(zarr_path):
                    # If the Zarr archive exists, open it in "a" (append) mode
                    data.to_zarr(
                        zarr_path,
                        mode="a",
                        compute=True,
                        consolidated=True,
                        append_dim="member",
                    )
                else:
                    # If the Zarr archive does not exist, create it in "w" (write) mode
                    data.to_zarr(zarr_path, mode="w", compute=True, consolidated=True)
