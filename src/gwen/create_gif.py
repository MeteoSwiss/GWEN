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
import random
import sys
from typing import Callable
from typing import Tuple

# Third-party
import click
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from pyprojroot import here

# First-party
from gwen.loggers_configs import setup_logger

logger = setup_logger()


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
        if nc_file in ".nc":
            filename_parts = os.path.splitext(os.path.basename(nc_file))[0].split("_")[
                2:5
            ]
        else:
            filename_parts = nc_file.split("_")[:-1]
    except IndexError:
        logger.error("Error in getting member parts, check your file!")
    return tuple(filename_parts)


def get_var_min_max(var: xr.DataArray) -> Tuple[float, float]:
    """Calculate the minimum and maximum values of the variable.

    Args:
        var (xr.DataArray): The variable to calculate the minimum and maximum values
        for.

    Returns:
        Tuple[float, float]: A tuple containing the minimum and maximum values of
        the variable.

    """
    return float(var.min()), float(var.max())


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
        logger.error("Error in opening the file: %s", error)

    return ds


def select_variable(ds: xr.Dataset, var_name: str, member: int) -> xr.DataArray:
    """Select a variable from a dataset.

    Args:
        ds (xr.Dataset): The dataset to select the variable from.
        var_name (str): The name of the variable to select.
        member (str): The member to select the variable for.

    Returns:
        xr.DataArray: The selected variable.

    Raises:
        KeyError: If the variable is not found in the dataset.

    """
    try:
        var = ds[var_name].isel(member=member)
    except KeyError:
        logger.error("Variable %s not found in the dataset", var_name)
        raise
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
            f"Temp: {part.replace('.0', '')} °C;"
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


def save_animation(ani: animation.FuncAnimation, output_filename: str) -> None:
    """Save the animation as a gif.

    Args:
        ani (animation.FuncAnimation): The animation object to save.
        output_filename (str): The path to save the animation to.

    """
    try:
        ani.save(output_filename, writer="imagemagick", dpi=100)
    except RuntimeError as error:
        logger.error("Error in saving output file: %s", error)


# pylint: disable=too-many-locals
def main(input_file: str, var_name: str, out_dir: str) -> None:
    """Create the animation.

    Args:
        input_file (str): The path to the input file.
        var_name (str): The name of the variable to plot.
        out_dir (str): The path to the output directory.

    """
    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Open the dataset
    try:
        ds = xr.open_dataset(input_file).sortby("time", ascending=True)
    except ValueError as error:
        logger.error("Error in opening the file: %s", error)

    # Select the variable and member
    random_mode = False
    for i in range(len(ds.member)):
        try:
            if random_mode:
                member = random.randint(0, len(ds.member) - 1)
            else:
                member = i
            var = select_variable(ds, var_name, member=member)
            member_str = get_member_name(ds.member[member].item())
        except KeyError as error:
            logger.error("Error in selecting the variable: %s", error)

        # Create the figure and plot the first time step
        fig, ax = plt.subplots()
        im = plot_first_time_step(var, ax)

        # Create the update function and animation object
        update_func = create_update_function(im, var, member_str, var_name)
        num_frames = len(var.time)
        ani = create_animation_object(fig, update_func, num_frames)

        member_filename = member_str.replace(" ", "_").lower()
        # Save the animation
        try:
            logger.info("Saving animation for member %s", member_filename)
            save_animation(ani, f"{out_dir}/animation_{str(member_filename)}_ICON.gif")
        except RuntimeError as error:
            logger.error("Error in saving the animation: %s", error)


if __name__ == "__main__":
    if len(sys.argv) > 1:

        @click.command()
        @click.option(
            "--input_file_cli",
            type=click.Path(exists=True),
            default=str(here()) + "/data/data_test.zarr",
            help="The path to the input file.",
        )
        @click.option(
            "--var_name_cli",
            type=str,
            default="theta_v",
            help="The name of the variable to plot.",
        )
        @click.option(
            "--output_dir_cli",
            type=click.Path(exists=True),
            default=str(here()) + "/output",
            help="The path to the output directory.",
        )
        def cli(input_file_cli: str, var_name_cli: str, output_dir_cli: str) -> None:
            try:
                main(input_file_cli, var_name_cli, output_dir_cli)
            except FileNotFoundError as e:
                logger.error("Error: %s. Please check the input file path.", e)

        # pylint: disable=no-value-for-parameter
        cli()

    else:
        in_file = (
            input("Enter the path to the input file: ")
            or str(here()) + "/data/data_test.zarr"
        )
        in_var = input("Enter the name of the variable to plot: ") or "theta_v"
        out_directory = (
            input("Enter the path to the output directory: ") or str(here()) + "/output"
        )

        main(in_file, in_var, out_directory)
