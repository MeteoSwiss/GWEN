"""Create an animated gif of a variable from a NetCDF file using matplotlib.

The module imports the necessary libraries, defines the path to the NetCDF file, and
opens the file using xarray. It then defines the variable to plot, calculates the
minimum and maximum values of the variable, and creates a new figure object. The first
time step of the variable is plotted, and an update function is defined to update the
plot for each frame of the animation. Finally, the animation is saved as a gif with the
specified filename.

Example usage:
    $ python create_gif.py

Output:
    A gif animation of the specified variable from the NetCDF file.
"""
# Third-party
import matplotlib.pyplot as plt  # type: ignore
import xarray as xr  # type: ignore
from matplotlib import animation  # type: ignore

# Define the variable to plot
var_name = "temp"

# Define the path to the zarr archive
zarr_path = "/scratch/sadamov/icon/icon-nwp/cpu/experiments/data_combined.zarr"

# Open the NetCDF file
ds = xr.open_zarr(zarr_path)

var = ds[var_name]
# Calculate the minimum and maximum values of the variable
var_min = float(var.min())
var_max = float(var.max())

for member in ds.member.values:
    # Create a new figure object
    fig, ax = plt.subplots()
    print(member)
    var_mem = var.sel(member=member)
    # Plot the first time step of the variable
    im = var_mem.isel(time=0).plot(vmin=var_min, vmax=var_max, ax=ax)
    plt.gca().invert_yaxis()  # type: ignore # invert the y-axis

    # string split member by "_"
    member_parts = member.split("_")[:-1]
    # Replace ".0" with " K", " m", " m"
    member_parts = [
        f"Temp: {part} K;"
        if i == 0
        else f"Height: {part} m;"
        if i == 1
        else f"Width: {part} m"
        for i, part in enumerate([part.replace(".0", "") for part in member_parts])
    ]
    member_name = " ".join(member_parts)
    plt.title(f"Var: {var_name}; Time: 0 s\n{member_name}")

    # pylint: disable=W0640
    def update(frame):
        # Update the data of the current plot
        time_in_seconds = round(
            (var_mem.time[frame] - var_mem.time[0]).item() * 24 * 3600
        )
        im.set_array(var_mem.isel(time=frame))
        plt.title(f"Var: {var_name}; Time: {time_in_seconds:.0f} s\n{member_name}")
        return im

    ani = animation.FuncAnimation(
        fig, update, frames=var_mem.shape[0], interval=100, blit=False
    )

    # Define the filename for the output gif
    output_filename = f"output/{var_name}/animation_member_{member}.gif"

    # Save the animation as a gif
    ani.save(output_filename, writer="imagemagick", dpi=100)
