"""Creates an animated gif of from a NetCDF or Zarr file using matplotlib.

The module contains the following functions:
    - get_member_parts: Extracts the relevant parts of the filename.
    - get_var_min_max: Calculate the minimum and maximum values of the variable.
    - create_animation: Create a gif from a NetCDF or Zarr file using matplotlib.
    - open_input_file: Open a NetCDF or Zarr file.
    - select_variable: Select a variable from a dataset.
    - plot_first_time_step: Plot the first time step of a variable.
    - get_member_name: Get the name of the member for use in the plot title.
    - create_update_function: Create the update function for the animation.
"""

# Standard library
import os
import sys
from typing import Callable
from typing import Tuple

# Third-party
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from pyprojroot import here


def get_member_parts(nc_file: str) -> Tuple[str, ...]:
    """Extract the relevant parts of the filename.

    Args:
        nc_file (str): The path to the NetCDF file.

    Returns:
        Tuple[str, ...]: A tuple containing the relevant parts of the filename.

    Raises:
        IndexError: If the filename does not contain the expected number of parts.

    """
    try:
        filename_parts = os.path.splitext(os.path.basename(nc_file))[0].split("_")
        return tuple(filename_parts[2:5])
    except IndexError:
        print("Error in getting member parts, check your file!")
        sys.exit(1)


def get_var_min_max(var: xr.DataArray) -> Tuple[float, float]:
    """Calculate the minimum and maximum values of the variable.

    Args:
        var (xr.DataArray): The variable to calculate the minimum and maximum values
        for.

    Returns:
        Tuple[float, float]: A tuple containing the minimum and maximum values of
        the variable.

    """
    try:
        return float(var.min()), float(var.max())
    except IndexError:
        print("Error in getting min and max of variable!")
        sys.exit(1)


def create_animation(input_file: str, var_name: str) -> None:
    """Create an animated gif from a NetCDF or Zarr file using matplotlib.

    Args:
        input_file (str): The path to the input file.
        var_name (str): The name of the variable to plot.

    """
    # Open the input file
    ds = open_input_file(input_file)

    for member in ds.member.values:
        # Create a new figure object
        fig, ax = plt.subplots()

        # Select the variable for the current member
        var = select_variable(ds, var_name, member)

        # Plot the first time step of the variable
        im = plot_first_time_step(var, ax)

        # Create a title for the plot
        member_name = get_member_name(input_file)
        plt.title(f"Var: {var_name}; Time: 0 s\n{member_name}")

        # Define the update function for the animation
        update_func = create_update_function(im, var, member_name, var_name)

        # Create the animation object
        ani = create_animation_object(fig, update_func, var.shape[0])

        # Save the animation as a gif
        save_animation(ani, var_name, member)

        # Close the figure object
        plt.close()


def open_input_file(input_file: str) -> xr.Dataset:
    """Open a NetCDF or Zarr file.

    Args:
        input_file (str): The path to the input file.

    Returns:
        xr.Dataset: The opened dataset.

    """
    try:
        if input_file.endswith(".nc"):
            ds = xr.open_dataset(input_file)
        elif input_file.endswith(".zarr"):
            ds = xr.open_zarr(input_file)
        else:
            raise ValueError(
                "Invalid file format. Please input either .nc or .zarr file"
            )
    except (FileNotFoundError, ValueError) as error:
        print(f"Error in opening the file: {error}")
        sys.exit(1)
    return ds


def select_variable(ds: xr.Dataset, var_name: str, member: str) -> xr.DataArray:
    """Select a variable from a dataset.

    Args:
        ds (xr.Dataset): The dataset to select the variable from.
        var_name (str): The name of the variable to select.
        member (str): The member to select the variable for.

    Returns:
        xr.DataArray: The selected variable.

    """
    try:
        var = ds[var_name].sel(member=member)
    except KeyError:
        print(f"Error: Variable {var_name} not found in the dataset")

    return var


def plot_first_time_step(var: xr.DataArray, ax: plt.Axes) -> AxesImage:
    """Plot the first time step of a variable.

    Args:
        var (xr.DataArray): The variable to plot.
        ax (plt.Axes): The axes to plot the variable on.

    Returns:
        AxesImage: The plotted image.

    """
    var_min, var_max = get_var_min_max(var)
    im = var.isel(time=0).plot(vmin=var_min, vmax=var_max, ax=ax)
    plt.gca().invert_yaxis()
    return im


def get_member_name(input_file: str) -> str:
    """Get the name of the member for use in the plot title.

    Args:
        input_file (str): The path to the input file.

    Returns:
        str: The name of the member.

    """
    member_parts = get_member_parts(input_file)
    member_name = " ".join(
        [
            f"Temp: {part.replace('.0', '')} K;"
            if i == 0
            else f"Height: {part.replace('.0', '')} m;"
            if i == 1
            else f"Width: {part.replace('.0', '')} m"
            for i, part in enumerate(member_parts)
        ]
    )
    return member_name


def create_update_function(
    im: AxesImage, var: xr.DataArray, member_name: str, var_name: str
) -> Callable[[int], AxesImage]:
    """Create the update function for the animation.

    Args:
        im (AxesImage): The image to update.
        var (xr.DataArray): The variable to update the image with.
        member_name (str): The name of the member to use in the plot title.
        var_name (str): The name of the variable to use in the plot title.

    Returns:
        animation.FuncAnimation: The update function for the animation.

    """

    def update(frame: int) -> AxesImage:
        time_in_seconds = round((var.time[frame] - var.time[0]).item() * 24 * 3600)
        im.set_array(var.isel(time=frame))
        plt.title(f"Var: {var_name}; Time: {time_in_seconds:.0f} s\n{member_name}")
        return im

    return update


def create_animation_object(
    fig: Figure, update_func: Callable[[int], AxesImage], num_frames: int
) -> animation.FuncAnimation:
    """Create the animation object.

    Args:
        fig (plt.Figure): The figure object to use for the animation.
        update_func (animation.FuncAnimation): The update function for animation.
        num_frames (int): The number of frames in the animation.

    Returns:
        animation.FuncAnimation: The animation object.

    """
    ani = animation.FuncAnimation(
        fig, update_func, frames=num_frames, interval=100, blit=False
    )
    return ani


def save_animation(ani: animation.FuncAnimation, var_name: str, member: str) -> None:
    """Save the animation as a gif.

    Args:
        ani (animation.FuncAnimation): The animation object to save.
        var_name (str): The name of the variable being plotted.
        member (str): The member being plotted.

    """
    output_filename = f"{here()}/output/{var_name}/animation_member_{member}.gif"
    try:
        ani.save(output_filename, writer="imagemagick", dpi=100)
    except RuntimeError as error:
        print(f"Error in saving output file: {error}")


def main(input_file: str, var_name: str) -> None:
    """Create the animation.

    Args:
        input_file (str): The path to the input file.
        var_name (str): The name of the variable to plot.

    """
    # Create the output directory if it doesn't exist
    output_dir = f"{here()}/output/{var_name}"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except FileExistsError as error:
        print(f"Error in creating output directory: {error}")
        sys.exit(1)

    # Create the animation
    create_animation(input_file, var_name)


if __name__ == "__main__":
    if sys.stdin.isatty():
        # Third-party
        import click

        @click.command()
        @click.option(
            "--input-file",
            type=click.Path(exists=True),
            default=str(here()) + "/data/data_combined.zarr",
            help="The path to the input file.",
        )
        @click.option(
            "--var-name",
            type=str,
            default="theta_v",
            help="The name of the variable to plot.",
        )
        def cli(input_file: str, var_name: str) -> None:
            try:
                main(input_file, var_name)
            except FileNotFoundError as e:
                print(f"Error: {e}. Please check the input file path.")

    else:
        in_file = (
            input("Enter the path to the input file: ")
            or str(here()) + "/data/data_combined.zarr"
        )
        in_var = input("Enter the name of the variable to plot: ") or "theta_v"

        main(in_file, in_var)
