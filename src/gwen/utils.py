"""Utility functions and classes for the gwen project.

Classes:
    ConvDataset: Custom dataset class.

Functions:
    animate: Animate the prediction evolution.
    create_animation: Create an animation of the prediction evolution.
    downscale_data: Downscale the data by the given factor.
    get_runs: Get all runs from the specified experiment.
    load_best_model: Load the best checkpoint of the model recent MLflow run.
    load_config_and_data: Load the configuration and data.
    load_data: Load the data.

"""
# Standard library
import os
from typing import Any
from typing import List
from typing import Tuple

# Third-party
import dask
import matplotlib.pyplot as plt
import mlflow  # type: ignore
import numpy as np
import torch
import xarray as xr
from matplotlib import animation
from matplotlib import cm
from matplotlib.image import AxesImage
from numpy import dtype
from numpy import signedinteger
from pyprojroot import here
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as Dataset_GNN
from torch_geometric.utils import erdos_renyi_graph

# First-party
from gwen.create_gif import get_member_name
from gwen.loggers_configs import load_config
from gwen.loggers_configs import setup_logger

logger = setup_logger()
config = load_config()


class ConvDataset(Dataset):
    """Custom dataset class.

    This class implements a custom dataset class, which is used to load and preprocess
    data for training and testing machine learning models.

    Args:
        data: The data to use.
        split: The split between train and test sets.

    Returns:
        x, y: Data for the train and test sets.

    """

    def __init__(self, data: xr.Dataset, split: int):
        """Initialize the custom dataset class.

        Args:
            data: The data to use.
            split: The split between input and target sets.

        """
        super().__init__()
        try:
            self.data: xr.Dataset = data
            self.split = split

            # Get the number of members in the dataset
            num_members: int = self.data.sizes["member"]

            # Get the indices of the members
            member_indices: np.ndarray[Any, dtype[signedinteger[Any]]] = np.arange(
                num_members
            )

            # Shuffle the member indices
            np.random.shuffle(member_indices)

            # Split the member indices into input and target sets
            self.input_indices: np.ndarray = member_indices[: self.split]
            self.target_indices: np.ndarray = member_indices[self.split:]
            if config["simplify"]:
                simple_input_index = np.random.choice(self.input_indices)
                self.input_indices = np.array(
                    [simple_input_index, simple_input_index+1])
                simple_target_index = simple_input_index + 1
                self.target_indices = np.array(
                    [simple_target_index, simple_target_index+1])

        except Exception as e:
            logger.exception("Error initializing custom dataset: %s", e)
            raise

    def __len__(self) -> int:
        """Get the length of the dataset."""
        try:
            return len(self.data.time)
        except Exception as e:
            logger.exception("Error getting length of dataset: %s", e)
            raise

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the data for the input and target sets.

        This method gets the data for the input and target sets.

        Args:
            idx: Index of the data.

        Returns:
            x, y: Data for the input and target sets.

        """
        try:
            # Get the data for the input and target sets
            if config["simplify"]:
                x = self.data.isel(member=slice(
                    self.input_indices[0], self.input_indices[1]), time=idx)
                y = self.data.isel(member=slice(
                    self.target_indices[0], self.target_indices[1]), time=idx)
            else:
                x = self.data.isel(
                    member=self.input_indices, time=idx)
                y = self.data.isel(
                    member=self.target_indices, time=idx)

        except Exception as e:
            logger.exception(
                "Error getting data for input and target sets: %s", e)
            raise

        return x, y

    def __iter__(self):
        """Get an iterator for the dataset."""
        try:
            return iter(self.data.time)
        except Exception as e:
            logger.exception("Error getting iterator for dataset: %s", e)
            raise

    def get_target_indices(self) -> np.ndarray:
        """Get the test indices.

        Returns:
            test_indices: Indices for the test set.

        """
        return self.target_indices


class GraphDataset(Dataset_GNN):
    def __init__(self, xr_data, split, transform=None, pre_transform=None):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.data = xr_data
        self.split = split
        self._indices = None
        self.transform = transform
        self.pre_transform = pre_transform

        # Calculate nodes and edges
        self.nodes = self.data.sizes["member"]
        self.edge_index = erdos_renyi_graph(self.nodes, edge_prob=1)
        # self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)

        num_members = self.data.sizes["member"]
        member_indices = np.arange(num_members)
        np.random.shuffle(member_indices)

        # Determine the split index
        self.input_indices = member_indices[:self.split]
        self.target_indices = member_indices[self.split:]

        # Calculate channels
        self.channels = self.data.sizes["height"] * self.data.sizes["ncells"]

    def len(self):
        return len(self.data.time)

    def get(self, idx) -> Data:
        # Load only the necessary data
        x = self.data.isel(member=np.arange(self.nodes), time=idx).stack(
            features=["height", "ncells"]).load()

        # Convert to tensor
        x = torch.tensor(x.values, dtype=torch.float)

        # Create masks for input and target nodes
        target_mask = torch.zeros(self.nodes, dtype=torch.bool)
        target_mask[self.target_indices] = 1

        # Create a Data object for the graph
        data = Data(x=x, edge_index=self.edge_index, target_mask=target_mask)

        return data


def animate(data: xr.Dataset, member: str, preds: str) -> animation.FuncAnimation:
    """Animate the prediction evolution.

    Args:
        data: The data to animate.
        member: The member to use.
        preds: The predictions to use.

    Returns:
        ani: The animation.

    """
    try:
        # Create a new figure object
        fig, ax = plt.subplots()

        # Calculate the 5% and 95% percentile of the y_mem data
        vmin, vmax = np.percentile(np.array(data.values), [1, 99])
        # Create a colormap with grey for values outside of the range
        cmap = cm.get_cmap("RdBu_r")
        cmap.set_bad(color="grey")

        im: AxesImage = data.isel(time=0).plot(
            ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

        plt.gca().invert_yaxis()

        text = ax.text(
            0.5,
            1.05,
            f"Theta_v - Time: 0 s\n Member: {member} - None",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=12,
        )
        plt.tight_layout()
        ax.set_title("")  # Remove the plt.title

        def update(frame: int) -> Tuple:
            """Update the data of the current plot.

            Args:
                frame: The frame to update.

            Returns:
                im, text: The updated plot.

            """
            try:
                time_in_seconds = round(
                    (data.time[frame] - data.time[0]).item() * 24 * 3600
                )
                im.set_array(data.isel(time=frame))
                title = (
                    f"Var: Theta_v - Time: {time_in_seconds:.0f} s\n"
                    f"Member: {member} - {preds}"
                )
                text.set_text(title)
                return im, text
            except Exception as e:
                logger.error("Error updating plot: %s", e)
                raise e

        ani = animation.FuncAnimation(
            fig, update, frames=range(len(data.time)), interval=50, blit=True
        )
        return ani
    except Exception as e:
        logger.error("Error creating animation: %s", e)
        raise e


def create_animation(
    data: dict, member_pred: int, member_target: int, preds: str
) -> str:
    """Create an animation of weather data for a given member and prediction type.

    Args:
        data (dict): A dictionary containing the weather data.
        member (int): The member index to plot.
        preds (str): The type of prediction to plot.

    Returns:
        str: The filepath of the output gif.

    Raises:
        ValueError: If the member index is out of range.
        ValueError: If the prediction type is not recognized.

    """
    y_pred_reshaped = data["y_pred_reshaped"]
    data_test = data["data_test"]
    y_pred_reshaped["time"] = data_test["time"]
    y_pred_reshaped["height"] = data_test["height"]

    # Plot the first time step of the variable
    if preds == "ICON":
        y_mem = data_test.isel(member=member_target)
    elif preds in ["CNN", "GNN"]:
        y_mem = y_pred_reshaped.isel(member=member_pred)
    else:
        raise ValueError(f"Unrecognized prediction type: {preds}")

    y_mem = y_mem.sortby(y_mem.time, ascending=True)
    member_name = get_member_name(data_test.member[member_target].item())

    try:
        ani = animate(y_mem, member=member_name, preds=preds)
    except Exception as e:
        logger.exception(
            f"Error creating animation for member {member_target}"
            f" and prediction type {preds}: %s",
            e,
        )
        raise

    # Define the filename for the output gif
    output_filename = (
        (f"{here()}/output/animations_{member_name}_{preds}.gif")
        .lower()
        .replace(" ", "_")
    )

    try:
        # Save the animation as a gif
        logger.info("Saving animation to %s", output_filename)
        ani.save(output_filename, writer="imagemagick", dpi=100)

    except Exception as e:
        logger.exception(
            f"Error saving animation for member {member_target}"
            f"and prediction type {preds}: %s",
            e,
        )
        raise
    finally:
        plt.close("all")

    return output_filename


def downscale_data(data: xr.Dataset, factor: int) -> xr.Dataset:
    """Downscale the data by the given factor.

    Args:
        data: The data to downscale.
        factor: The factor by which to downscale the data.

    Returns:
        The downscaled data.

    Raises:
        ValueError: If the factor is not a positive integer.

    """
    if not isinstance(factor, int) or factor <= 0:
        raise ValueError(
            f"Factor must be a positive integer, but got {factor}")

    with dask.config.set(
        dict[str, bool](**{"array.slicing.split_large_chunks": False})
    ):
        # Coarsen the height and ncells dimensions by the given factor
        data_coarse = data.coarsen(
            height=factor, ncells=factor).reduce(np.mean).compute()
        return data_coarse


def get_runs(experiment_name: str) -> List[mlflow.entities.Run]:
    """Retrieve a list of runs for a given experiment name.

    Args:
        experiment_name (str): The name of the experiment to retrieve runs for.

    Returns:
        List[mlflow.entities.Run]: A list of runs for the given experiment name.

    Raises:
        ValueError: If no runs are found for the given experiment name.

    """
    # Define a function to filter runs by artifact_uri

    def filter_runs(row):
        artifact_uri = row["artifact_uri"]
        if os.path.isdir(artifact_uri) and os.listdir(artifact_uri):
            return True
        else:
            return False

    runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=[
                              "attributes.start_time DESC"])
    filtered_runs = runs[runs.apply(filter_runs, axis=1)]

    if len(runs) == 0:
        raise ValueError(f"No runs found in experiment: {experiment_name}")

    return filtered_runs


def load_best_model(experiment_name: str) -> nn.Module:
    """Load the best model from a given MLflow experiment.

    Args:
        experiment_name (str): The name of the MLflow experiment.

    Returns:
        nn.Module: The PyTorch model object.

    Raises:
        ValueError: If no runs are found for the given experiment name.
        FileNotFoundError: If the best model path does not exist.

    """
    try:
        runs = get_runs(experiment_name)
        # [ ]: actually get the best model

        run_id: str = runs.iloc[0].run_id  # type: ignore [attr-defined]
        best_model_path = mlflow.get_artifact_uri()
        best_model_path = os.path.abspath(
            os.path.join(best_model_path, "../../"))
        best_model_path = os.path.join(
            best_model_path, run_id, "artifacts", "models")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(
                f"Best model path does not exist: {best_model_path}"
            )
        model = mlflow.pytorch.load_model(best_model_path)
    except (ValueError, FileNotFoundError) as e:
        logger.exception(str(e))
        raise e
    return model


def load_config_and_data() -> Tuple[dict, xr.Dataset, xr.Dataset]:
    """Load configuration and data for the gwen project.

    Returns:
        Tuple[dict, xr.Dataset, xr.Dataset]: A tuple containing the configuration
        dictionary, and two xarray DataArrays containing the training and testing data.

    Raises:
        FileNotFoundError: If the configuration file or data files do not exist.
        ValueError: If the coarsen factor is not a positive integer.

    """
    try:
        config = load_config()

        data_train, data_test = load_data(config)

        if config["coarsen"] > 1:
            # Coarsen the data
            if not isinstance(config["coarsen"], int) or config["coarsen"] <= 0:
                raise ValueError(
                    f"Coarsen factor must be a positive integer, "
                    f"but got {config['coarsen']}"
                )
            data_test = downscale_data(data_test, config["coarsen"])
            data_train = downscale_data(data_train, config["coarsen"])

    except (FileNotFoundError, ValueError) as e:
        logger.exception(str(e))
        raise e
    return config, data_train, data_test


def load_data(config: dict) -> Tuple[xr.Dataset, xr.Dataset]:
    """Load training and test data from zarr files and return them as xarray DataArrays.

    Args:
        config (dict): A dictionary containing the paths to the training and test data.

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: A tuple containing the training and test data
        as xarray Dataset.

    Raises:
        FileNotFoundError: If the data files do not exist.

    """
    try:
        # Load the training data
        data_train = (
            xr.open_zarr(
                str(here()) + config["data_train"]).to_array().squeeze()
        )
        data_train = data_train.transpose(
            "time",
            "member",
            "height",
            "ncells",
        )

        # Load the test data
        data_test = (
            xr.open_zarr(str(here()) + config["data_test"])
            .to_array()
            .squeeze(drop=False)
        )
        data_test = data_test.transpose(
            "time",
            "member",
            "height",
            "ncells",
        )

        return data_train, data_test
    except FileNotFoundError as e:
        logger.exception(str(e))
        raise e
