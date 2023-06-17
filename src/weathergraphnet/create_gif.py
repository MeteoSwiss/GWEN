"""Create an animated gif of a variable from a NetCDF file using matplotlib.

The script opens a NetCDF file, extracts a variable, and creates an animated gif of the
variable over time. The output gif is saved to the specified filename.

Example usage:
    $ python create_gif.py

Attributes:
    nc_file (str): The name of the NetCDF file to open.
    nc_path (str): The path to the NetCDF file.
    var_name (str): The name of the variable to plot.
    var_min (float): The minimum value of the variable.
    var_max (float): The maximum value of the variable.
    fig (matplotlib.figure.Figure): The figure object for the plot.
    ax (matplotlib.axes.Axes): The axes object for the plot.
    im (matplotlib.image.AxesImage): The image object for the plot.
    ani (matplotlib.animation.FuncAnimation): The animation object for the plot.
    output_filename (str): The name of the output gif file.

"""
# Standard library
import os

# Third-party
import matplotlib.pyplot as plt  # type: ignore
import xarray as xr
from matplotlib import animation  # type: ignore
from pyprojroot import here

# Prompt the user for the variable name
var_name = input("Enter the name of the variable to plot (default: tkvh): ") or "tkhv"

# Prompt the user for the file path
nc_path = (
    input(
        "Enter the path to the NetCDF file (default: /scratch/sadamov/icon/"
        "icon-nwp/cpu/experiments/atmcirc-straka_93_-30.0_4000.0_2000.0/"
        "atmcirc-straka_93_-30.0_4000.0_2000.0_DOM01_ML_20080801T000000Z.nc): "
    )
    or "/scratch/sadamov/icon/icon-nwp/cpu/experiments/atmcirc-straka_"
    "93_-30.0_4000.0_2000.0/atmcirc-straka_93_-30.0_4000.0_2000.0_DOM01_"
    "ML_20080801T000000Z.nc"
)

# Split the file path into directory and filename
nc_dir, nc_file = os.path.split(nc_path)

# Extract the relevant parts of the filename
part = nc_file.split("_")[2:5]

# Open the NetCDF file
ds = xr.open_dataset(nc_path)

var = ds[var_name]
# Calculate the minimum and maximum values of the variable
var_min = var.min()
var_max = var.max()

# Create a new figure object
fig, ax = plt.subplots()

# Replace ".0" with " K", " m", " m"
member_parts = [
    f"Temp: {p.replace('.0', '')} K;"
    if i == 0
    else f"Height: {p.replace('.0', '')} m;"
    if i == 1
    else f"Width: {p.replace('.0', '')} m"
    for i, p in enumerate(part)
]
member = " ".join(member_parts)

# Plot the first time step of the variable
im = var.isel(time=0).plot(vmin=var_min, vmax=var_max, ax=ax)
plt.gca().invert_yaxis()  # type: ignore # invert the y-axis
plt.title(f"Var: {var_name}; Time: 0 s\n{member}")


def update(frame):
    # Update the data of the current plot
    time_in_seconds = round((var.time[frame] - var.time[0]).item() * 24 * 3600)
    im.set_array(var.isel(time=frame))
    plt.title(f"Var: {var_name}; Time: {time_in_seconds:.0f} s\n{member}")
    return im


ani = animation.FuncAnimation(
    fig, update, frames=var.shape[0], interval=100, blit=False
)

# Define the filename for the output gif
output_filename = f"{here()}/output/{var_name}/animation_member_{'_'.join(part)}.gif"

# Save the animation as a gif
ani.save(output_filename, writer="imagemagick", dpi=100)
