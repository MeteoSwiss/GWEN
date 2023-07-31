"""Utils for the command line tool."""
# Standard library
import logging

import dask

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import animation
from torch.distributions import Normal
from torch.utils.data import Dataset


class CRPSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, dim=0):
        # Calculate the mean and standard deviation of the predicted distribution
        mu = torch.mean(outputs, dim=dim)  # Mean over ensemble members
        sigma = torch.std(outputs, dim=dim) + 1e-6  # Stddev over ensemble members

        # Create a normal distribution with the predicted mean and standard deviation
        dist = Normal(mu, sigma)

        # Calculate the CRPS loss for each sample in the batch
        # Mean over ensemble members and spatial locations
        crps_loss = torch.mean((dist.cdf(target) - 0.5) ** 2, dim=[1, 2, 3]) #TODO check this!

        return crps_loss


class EnsembleVarianceRegularizationLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha  # Regularization strength

    def forward(self, outputs, target):
        l1_loss = torch.mean(torch.abs(outputs - target))
        ensemble_variance = torch.var(outputs, dim=1)
        regularization_loss = -self.alpha * torch.mean(ensemble_variance)
        return l1_loss + regularization_loss


class MaskedLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, outputs, target, mask):

        # Calculate the loss for each sample in the batch using the specified loss
        # function
        loss = self.loss_fn(outputs, target)

        # Mask the loss for cells where the values stay constant over all observed times
        masked_loss = loss * mask

        # Calculate the mean loss over all unmasked cells
        mean_loss = torch.sum(masked_loss) / torch.sum(mask)

        return mean_loss

class MyDataset(Dataset):
    def __init__(self, data, split):
        self.data = data
        self.split = split

        # Get the number of members in the dataset
        num_members = self.data.sizes["member"]

        # Get the indices of the members
        member_indices = np.arange(num_members)

        # Shuffle the member indices
        np.random.shuffle(member_indices)

        # Split the member indices into train and test sets
        self.train_indices = member_indices[:self.split]
        self.test_indices = member_indices[self.split:]

    def __len__(self):
        return len(self.data.time)

    def __getitem__(self, idx):
        # Get the data for the train and test sets
        x = self.data.isel(member=self.train_indices, time=idx).values
        y = self.data.isel(member=self.test_indices, time=idx).values

        # If x and y are 2D arrays, add a new dimension
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=0)

        return torch.from_numpy(x), torch.from_numpy(y)


def animate(data, member=0, preds="CNN"):
    """Animate the prediction evolution."""
    # Create a new figure object
    fig, ax = plt.subplots()

    # Calculate the 5% and 95% percentile of the y_mem data
    vmin, vmax = np.percentile(data.values, [1, 99])
    # Create a colormap with grey for values outside of the range
    cmap = plt.cm.RdBu_r
    cmap.set_bad(color='grey')

    im = data.isel(time=0).plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.gca().invert_yaxis()

    text = ax.text(
        0.5,
        1.05,
        "Theta_v - Time: 0 s\n Member: 0 - None",
        ha='center',
        va='bottom',
        transform=ax.transAxes,
        fontsize=12)
    plt.tight_layout()
    ax.set_title("")  # Remove the plt.title

    def update(frame):
        """Update the data of the current plot."""
        time_in_seconds = round(
            (data.time[frame] - data.time[0]).item() * 24 * 3600
        )
        im.set_array(data.isel(time=frame))
        title = f"Var: Theta_v - Time: {time_in_seconds:.0f} s\n Member: {member} - {preds}"
        text.set_text(title)
        return im, text

    ani = animation.FuncAnimation(
        fig, update, frames=range(len(data.time)), interval=50, blit=True
    )
    return ani


def count_to_log_level(count: int) -> int:
    """Map occurrence of the command line option verbose to the log level."""
    if count == 0:
        return logging.ERROR
    elif count == 1:
        return logging.WARNING
    elif count == 2:
        return logging.INFO
    else:
        return logging.DEBUG
    

def downscale_data(data, factor):
    """Downscale the data by the given factor.

    Args:
        data (xarray.Dataset): The data to downscale.
        factor (int): The factor by which to downscale the data.

        Returns:
            The downscaled data.

    """
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # Coarsen the height and ncells dimensions by the given factor
        data_coarse = data.coarsen(height=factor, ncells=factor).mean()
        return data_coarse
