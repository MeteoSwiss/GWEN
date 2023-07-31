"""Utils for the command line tool."""

# Standard library
import json
import logging
import os
import socket
import warnings
from typing import Any
from typing import List

# Third-party
import dask

# Third-party libraries
import matplotlib.pyplot as plt
import mlflow  # type: ignore
import numpy as np
import torch
import xarray as xr
from matplotlib import animation
from pyprojroot import here
from torch import nn
from torch.distributions import Normal
from torch.utils.data import Dataset


class CRPSLoss(nn.Module):
    """Continuous Ranked Probability Score (CRPS) loss function.

    Args:
        nn.Module: PyTorch module.

    Returns:
        crps_loss: CRPS loss for each sample in the batch.

    """

    def __init__(self):
        """Initialize the CRPS loss function."""
        super().__init__()

    def forward(self, outputs, target, dim=0):
        """Calculate the CRPS loss for each sample in the batch.

        Args:
            outputs: Predicted values. target: Target values. dim: Dimension over which
            to calculate the mean and standard deviation.

        Returns:
            crps_loss: CRPS loss for each sample in the batch.

        """
        # Calculate the mean and standard deviation of the predicted distribution
        mu = torch.mean(outputs, dim=dim)  # Mean over ensemble members
        # Stddev over ensemble members
        sigma = torch.std(outputs, dim=dim) + 1e-6

        # Create a normal distribution with the predicted mean and standard deviation
        dist = Normal(mu, sigma)

        # Calculate the CRPS loss for each sample in the batch Mean over ensemble
        # members and spatial locations
        crps_loss = torch.mean((dist.cdf(target) - 0.5) ** 2, dim=[1, 2, 3])

        return crps_loss


class EnsembleVarianceRegularizationLoss(nn.Module):
    """Ensemble variance regularization loss function.

    Args:
        nn.Module: PyTorch module. alpha: Regularization strength.

    Returns:
        l1_loss + regularization_loss: Loss for each sample in the batch.

    """

    def __init__(self, alpha=0.1):
        """Initialize the ensemble variance regularization loss function."""
        super().__init__()
        self.alpha = alpha  # Regularization strength

    def forward(self, outputs, target):
        """Calculate the loss for each sample using the specified loss function.

        Args:
            outputs: Predicted values. target: Target values.

        Returns:
            l1_loss + regularization_loss: Loss for each sample in the batch.

        """
        l1_loss = torch.mean(torch.abs(outputs - target))
        ensemble_variance = torch.var(outputs, dim=1)
        regularization_loss = -self.alpha * torch.mean(ensemble_variance)
        return l1_loss + regularization_loss


class MaskedLoss(nn.Module):
    """Masked loss function.

    Args:
        nn.Module: PyTorch module. loss_fn: Loss function to use.

    Returns:
        mean_loss: Mean loss over all unmasked cells.

    """

    def __init__(self, loss_fn):
        """Initialize the masked loss function."""
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, outputs, target, mask):
        """Calculate the loss for each sample using the specified loss function.

        Args:
            outputs: Predicted values. target: Target values. mask: Mask for cells where
            the values stay constant over all observed times.

        Returns:
            mean_loss: Mean loss over all unmasked cells.

        """
        # Calculate the loss for each sample in the batch using the specified loss
        # function
        loss = self.loss_fn(outputs, target)

        # Mask the loss for cells where the values stay constant over all observed times
        masked_loss = loss * mask

        # Calculate the mean loss over all unmasked cells
        mean_loss = torch.sum(masked_loss) / torch.sum(mask)

        return mean_loss


class MyDataset(Dataset):
    """Custom dataset class.

    Args:
        Dataset: PyTorch dataset. data: The data to use. split: The split between train
        and test sets.

    Returns:
        x, y: Data for the train and test sets.

    """

    def __init__(self, data, split):
        """Initialize the custom dataset class."""
        self.data = data
        self.split = split

        # Get the number of members in the dataset
        num_members = self.data.sizes["member"]

        # Get the indices of the members
        member_indices = np.arange(num_members)

        # Shuffle the member indices
        np.random.shuffle(member_indices)

        # Split the member indices into train and test sets
        self.train_indices = member_indices[: self.split]
        self.test_indices = member_indices[self.split :]

    def __len__(self):
        return len(self.data.time)

    def __getitem__(self, idx):
        """Get the data for the train and test sets.

        Args:
            idx: Index of the data.

        Returns:
            x, y: Data for the train and test sets.

        """
        # Get the data for the train and test sets
        x = self.data.isel(member=self.train_indices, time=idx).values
        y = self.data.isel(member=self.test_indices, time=idx).values

        # If x and y are 2D arrays, add a new dimension
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=0)

        return torch.from_numpy(x), torch.from_numpy(y)


def animate(data, member=0, preds="CNN"):
    """Animate the prediction evolution.

    Args:
        data: The data to animate. member: The member to use. preds: The predictions to
        use.

    Returns:
        ani: The animation.

    """
    # Create a new figure object
    fig, ax = plt.subplots()

    # Calculate the 5% and 95% percentile of the y_mem data
    vmin, vmax = np.percentile(data.values, [1, 99])
    # Create a colormap with grey for values outside of the range
    cmap = plt.cm.RdBu_r
    cmap.set_bad(color="grey")

    im = data.isel(time=0).plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.gca().invert_yaxis()

    text = ax.text(
        0.5,
        1.05,
        "Theta_v - Time: 0 s\n Member: 0 - None",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=12,
    )
    plt.tight_layout()
    ax.set_title("")  # Remove the plt.title

    def update(frame):
        """Update the data of the current plot.

        Args:
            frame: The frame to update.

        Returns:
            im, text: The updated plot.

        """
        time_in_seconds = round((data.time[frame] - data.time[0]).item() * 24 * 3600)
        im.set_array(data.isel(time=frame))
        title = (
            f"Var: Theta_v - Time: {time_in_seconds:.0f} s\n Member: {member} - {preds}"
        )
        text.set_text(title)
        return im, text

    ani = animation.FuncAnimation(
        fig, update, frames=range(len(data.time)), interval=50, blit=True
    )
    return ani


def count_to_log_level(count: int) -> int:
    """Map occurrence of the command line option verbose to the log level.

    Args:
        count: The count of the command line option verbose.

    Returns:
        logging.ERROR: If count is 0. logging.WARNING: If count is 1. logging.INFO: If
        count is 2. logging.DEBUG: If count is greater than 2.

    """
    if count == 0:
        return logging.ERROR
    elif count == 1:
        return logging.WARNING
    elif count == 2:
        return logging.INFO
    else:
        return logging.DEBUG


def create_animation(data, member, preds):
    y_pred_reshaped = data["y_pred_reshaped"]
    data_test = data["data_test"]
    y_pred_reshaped["time"] = data_test["time"]
    y_pred_reshaped["height"] = data_test["height"]

    # Plot the first time step of the variable
    if preds == "ICON":
        y_mem = data_test.isel(member=member)
    else:
        y_mem = y_pred_reshaped.isel(member=member)

    y_mem = y_mem.sortby(y_mem.time, ascending=True)

    ani = animate(y_mem, member=member, preds=preds)

    # Define the filename for the output gif
    output_filename = f"{here()}/output/animation_member_{member}_{preds}.gif"

    # Save the animation as a gif
    ani.save(output_filename, writer="imagemagick", dpi=100)
    return output_filename


def downscale_data(data, factor):
    """Downscale the data by the given factor.

    Args:
        data (xarray.DataArray): The data to downscale. factor (int): The factor by
        which to downscale the data.

    Returns:
        xarray.DataArray: The downscaled data.

    """
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        # Coarsen the height and ncells dimensions by the given factor
        data_coarse = data.coarsen(height=factor, ncells=factor).mean()
        return data_coarse


def get_runs(experiment_name: str) -> List[mlflow.entities.Run]:
    """Get all runs from the specified experiment."""
    runs = mlflow.search_runs(experiment_names=experiment_name)
    if not runs:
        raise ValueError(f"No runs found in experiment: {experiment_name}")
    return runs


def load_best_model(experiment_name: str) -> nn.Module:
    """Load the best checkpoint of the model from the most recent MLflow run."""
    runs = get_runs(experiment_name)
    run_id = runs.iloc[0].run_id
    best_model_path = mlflow.get_artifact_uri()
    best_model_path = os.path.abspath(os.path.join(best_model_path, "../../"))
    best_model_path = os.path.join(best_model_path, run_id, "artifacts", "models")
    model: Any = mlflow.pytorch.load_model(best_model_path)

    return model


def load_config_and_data():
    with open(
        str(here()) + "/src/weathergraphnet/config.json", "r", encoding="UTF-8"
    ) as f:
        config = json.load(f)

    # Suppress all warnings
    suppress_warnings()

    data_train, data_test = load_data(config)

    if config.coarsen > 1:
        # Coarsen the data
        data_test = downscale_data(data_test, config.coarsen)
        data_train = downscale_data(data_train, config.coarsen)

    return config, data_train, data_test


def load_data(config):
    # Load the training data
    data_train = xr.open_zarr(str(here()) + config.data_train).to_array().squeeze()
    data_train = data_train.transpose(
        "time",
        "member",
        "height",
        "ncells",
    )

    # Load the test data
    data_test = (
        xr.open_zarr(str(here()) + config.data_test).to_array().squeeze(drop=False)
    )
    data_test = data_test.transpose(
        "time",
        "member",
        "height",
        "ncells",
    )

    return data_train, data_test


def move_to_device(model, loss_fn):
    """Move the loss function and model to the cuda device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Move the loss function and model to the cuda device
    loss_fn = loss_fn.to(device)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    return model, loss_fn


def setup_mlflow() -> tuple:
    hostname = socket.gethostname()
    # Set the artifact path based on the hostname
    if "nid" in hostname:
        artifact_path = (
            "/scratch/e1000/meteoswiss/scratch/sadamov/"
            "pyprojects_data/weathergraphnet/mlruns"
        )
        experiment_name = "WGN_balfrin"
    else:
        artifact_path = "/scratch/sadamov/pyprojects_data/weathergraphnet/mlruns"
        experiment_name = "WGN"

    mlflow.set_tracking_uri(str(here()) + "/mlruns")
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_path)
    mlflow.set_experiment(experiment_name=experiment_name)
    return artifact_path, experiment_name


def suppress_warnings():
    warnings.simplefilter("always")
    warnings.filterwarnings("ignore", message="Setuptools is replacing dist")
    warnings.filterwarnings(
        "ignore",
        message="Encountered an unexpected error while inferring pip requirements",
    )
