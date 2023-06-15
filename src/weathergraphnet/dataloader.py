import os
import re

import xarray as xr
import zarr # type: ignore

# Define the path to the data
data_path = "/scratch/sadamov/icon/icon-nwp/cpu/experiments/"

# Define the regular expression to match the filename
filename_regex = r"atmcirc-straka_93_(.*)DOM01_ML_20080801T000000Z.nc"

# Compile the regular expression
filename_pattern = re.compile(filename_regex)

# Define the path to the Zarr archive
zarr_path = "/scratch/sadamov/icon/icon-nwp/cpu/experiments/data.zarr"

# Define the encoding for the Zarr archive
encoding = {"member": {"chunks": 1}, "time": {"chunks": "auto"}}

# Get a list of all the folders in the data path
folders = os.listdir(data_path)

# Loop over each folder and load the data
for folder in folders:
    # Check if the folder matches the naming convention
    if folder.startswith("atmcirc-straka_93_"):
        # Define the path to the data file
        file_path = os.path.join(data_path, folder)
        # Get a list of all the files in the folder
        files = os.listdir(file_path)
        # Loop over each file and load the data if it matches the filename pattern
        for file in files:
            match = filename_pattern.match(file)
            if match:
                # Load the data using xarray
                data = xr.open_dataset(os.path.join(file_path, file))
                # Add a new dimension called "member" and set its values based on the
                # matched string
                data.coords["member"] = match.group(1)
                # Store the data in the Zarr archive with chunking along the "member"
                # and "time" dimensions
                data.to_zarr(
                    zarr_path,
                    mode="a",
                    group=folder,
                    compute=False,
                    encoding=encoding
                )

# Combine the data from the Zarr archive into a single xarray dataset
# Open the Zarr archive
zarr_file = zarr.open(zarr_path, mode="r")

# Initialize an empty list to hold the arrays
arrays = []

# Iterate over all the groups in the archive
for group in zarr.hierarchy.walk_groups(zarr_file):
    # Access the data in the group
    data = group["my_array"]
    # Append the array to the list
    arrays.append(data)

# Concatenate the arrays along the first axis
combined_data = zarr.concatenate(arrays, axis=0)

# Compute the combined dataset and store it in the Zarr archive with chunking along the
# "member" and "time" dimensions
combined_data.to_zarr(
    zarr_path.replace("data.zarr", "data_combined.zarr"),
    mode="a",
    compute=True,
    encoding=encoding
)