import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import animation
from pyprojroot import here


def get_member_parts(nc_file: str) -> Tuple[str, str, str]:
    """Extract the relevant parts of the filename."""
    try:
        filename_parts = os.path.splitext(
            os.path.basename(nc_file))[0].split("_")
        return tuple(filename_parts[2:5])
    except IndexError:
        print("Error in getting member parts, check your file!")
        sys.exit(1)


def get_var_min_max(var: xr.DataArray) -> Tuple[float, float]:
    """Calculate the minimum and maximum values of the variable."""
    try:
        return float(var.min()), float(var.max())
    except IndexError:
        print("Error in getting min and max of variable!")
        sys.exit(1)


def create_animation(input_file: str, var_name: str) -> None:
    """Create an animated gif of a variable from a NetCDF or Zarr file using
    matplotlib."""
    # Open the input file
    try:
        if input_file.endswith(".nc"):
            ds = xr.open_dataset(input_file)
        elif input_file.endswith(".zarr"):
            ds = xr.open_zarr(input_file)
        else:
            raise ValueError(
                "Invalid file format. Please input either .nc or .zarr file")
    except Exception as error:
        print(f"Error in opening the file: {error}")
        sys.exit(1)

    # Extract the relevant parts of the filename
    member_parts = get_member_parts(input_file)

    for member in ds.member.values:
        # Create a new figure object
        fig, ax = plt.subplots()

        try:
            # Select the variable for the current member
            var = ds[var_name].sel(member=member)
        except KeyError:
            print(f"Error: Variable {var_name} not found in the dataset")
            continue

        # Calculate the minimum and maximum values of the variable
        var_min, var_max = get_var_min_max(var)

        # Plot the first time step of the variable
        im = var.isel(time=0).plot(vmin=var_min, vmax=var_max, ax=ax)
        plt.gca().invert_yaxis()

        # Create a title for the plot
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
        plt.title(f"Var: {var_name}; Time: 0 s\n{member_name}")

        # Define the update function for the animation
        def update(frame):
            time_in_seconds = round(
                (var.time[frame] - var.time[0]).item() * 24 * 3600)
            im.set_array(var.isel(time=frame))
            plt.title(
                f"Var: {var_name}; Time: {time_in_seconds:.0f} s\n{member_name}")
            return im

        # Create the animation object
        try:
            ani = animation.FuncAnimation(
                fig, update, frames=var.shape[0], interval=100, blit=False
            )
        except Exception as error:
            print(f"Error in creating animation: {error}")
            continue

        # Define the filename for the output gif
        output_filename = f"{here()}/output/{var_name}/animation_member_{member}.gif"

        # Save the animation as a gif
        try:
            ani.save(output_filename, writer="imagemagick", dpi=100)
        except Exception as error:
            print(f"Error in saving output file: {error}")
            continue

        # Close the figure object
        plt.close()


def main(input_file: str = None, var_name: str = None) -> None:
    """Create an animated gif of a variable from a NetCDF or Zarr file using
    matplotlib."""

    if input_file is None:
        input_file = input("Enter the path to the input file: ") or str(
            here()) + "/data/data_combined.zarr"
    if var_name is None:
        var_name = input(
            "Enter the name of the variable to plot: ") or "theta_v"

    # Create the output directory if it doesn't exist
    output_dir = f"{here()}/output/{var_name}"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as error:
        print(f"Error in creating output directory: {error}")
        sys.exit(1)

    # Create the animation
    create_animation(input_file, var_name)


if __name__ == "__main__":
    if sys.stdin.isatty():
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
            except Exception as e:
                print("Error: " + str(e))

        cli()
    else:
        main()
