"""Training a Graph Neural Network (GNN) with PyTorch and the Torch Geometric library.

The script defines a GNN model using linear layers and ReLU activation functions, and
trains the model on input and output data using the mean squared error loss function and
the Adam optimizer. The input and output data are loaded using PyTorch's DataLoader
class, and are processed using the Torch Geometric library. The trained model is then
saved to a file for later use.

Usage: To use this module, simply run the script from the command line. The script
assumes that the input and output data are stored in separate NumPy arrays, and that the
arrays have the same number of samples. The script also assumes that the input and
output data have the same number of features, and that the input data is stored in
row-major order.

Dependencies: - torch - torch_geometric

Example usage:

    python train_gnn.py --input data_in.npy --output data_out.npy --channels-in 10
    --channels-out 5

This will load the input and output data from the specified NumPy arrays, train a GNN
model on the data, and save the trained model to a file. The script will also print the
training loss for each epoch.

Arguments:
    --input (str): The path to the NumPy array containing the input data. --output
    (str): The path to the NumPy array containing the output data. --channels-in (int):
    The number of input channels. --channels-out (int): The number of output channels.
    --epochs (int): The number of epochs to train for. Default is 10. --lr (float): The
    learning rate for the optimizer. Default is 0.001. --batch-size (int): The batch
    size for the DataLoader. Default is 1. --shuffle (bool): Whether to shuffle the
    DataLoader. Default is True.

"""

# Standard library
import socket
import warnings
from typing import List

# Third-party
import matplotlib.pyplot as plt  # type: ignore
import mlflow  # type: ignore
import numpy as np
import torch
import xarray as xr  # type: ignore
from matplotlib import animation  # type: ignore
from pyprojroot import here
from pytorch_lightning.loggers import MLFlowLogger  # type: ignore
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CyclicLR
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torch_geometric.nn import GCNConv  # type: ignore
from torch_geometric.utils import erdos_renyi_graph  # type: ignore


class GNNModel(torch.nn.Module):
    # pylint: disable=too-many-instance-attributes
    """A Graph Neural Network (GNN) model for weather prediction.

    Args:
        in_channels (int): The number of input channels. out_channels (int): The number
        of output channels.

    Methods:
        forward(data): Performs a forward pass through the GNN model.

    """

    def __init__(self, in_channels, out_channels, start_channels=1024):
        """Initialize the GNN model."""
        super().__init__()
        self.conv1 = GCNConv(in_channels, start_channels)
        self.conv2 = GCNConv(start_channels, start_channels // 2)
        self.conv3 = GCNConv(start_channels // 2, start_channels // 4)
        self.conv4 = GCNConv(start_channels // 4, start_channels // 8)
        self.conv5 = GCNConv(start_channels // 8, start_channels // 16)
        self.upconv1 = GCNConv(start_channels // 16, start_channels // 8)
        self.upconv2 = GCNConv(start_channels // 8, start_channels // 4)
        self.upconv3 = GCNConv(start_channels // 4, start_channels // 2)
        self.upconv4 = GCNConv(start_channels // 2, start_channels)
        self.upconv5 = GCNConv(start_channels, out_channels)

    def forward(self, data):
        """Perform a forward pass through the GNN model.

        Args:
            data (torch_geometric.data.Data): The input data.

        Returns:
            The output of the GNN model.

        """
        x, edge_index = (
            data.x,
            data.edge_index,
        )
        x1 = nn.functional.relu(self.conv1(x, edge_index))
        x2 = nn.functional.relu(self.conv2(x1, edge_index))
        x3 = nn.functional.relu(self.conv3(x2, edge_index))
        x4 = nn.functional.relu(self.conv4(x3, edge_index))
        x5 = nn.functional.relu(self.conv5(x4, edge_index))
        x6 = nn.functional.relu(self.upconv1(x5, edge_index) + x4)
        x7 = nn.functional.relu(self.upconv2(x6, edge_index) + x3)
        x8 = nn.functional.relu(self.upconv3(x7, edge_index) + x2)
        x9 = nn.functional.relu(self.upconv4(x8, edge_index) + x1)
        x10 = nn.functional.relu(self.upconv5(x9, edge_index))
        return x10


# pylint: disable=no-member
def create_data_loader(data: List, edge_index: torch.Tensor, nodes: int) -> DataLoader:
    """Create a PyTorch DataLoader object from a list of data samples and an edge index.

    Args:
        data (List): A list of data samples. edge_index (torch.Tensor): The edge index
        for the graph.

    Returns:
        A PyTorch DataLoader object.

    """
    dataset = [
        Data(
            x=torch.tensor(sample, dtype=torch.float32).view(nodes, -1),
            edge_index=edge_index,
        )
        for sample in data.values  # type: ignore[attr-defined]
    ]
    return DataLoader(dataset, batch_size=1, shuffle=True)


# pylint: disable=too-many-arguments
def train_model(
    model: nn.Module,
    train_loader_in: DataLoader,
    train_loader_out: DataLoader,
    criterion_sel: nn.Module,
    optimizer_sel: optim.Optimizer,
    num_epochs: int = 10,
) -> None:
    """Train a GNN model and output data using the specified loss function.

    Args:
        model (torch.nn.Module): The GNN model to train. loader_in
        (torch.utils.data.DataLoader): The data loader for the input data. loader_out
        (torch.utils.data.DataLoader): The data loader for the output data. channels_in
        (int): The number of input channels. criterion (torch.nn.Module): The loss
        function to use for training. optimizer (torch.optim.Optimizer): The optimizer
        to use for training. num_epochs (int): The number of epochs to train for.

    Returns:
        None

    """
    best_loss = float("inf")
    # Train the GNN model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data_in, data_out in zip(train_loader_in, train_loader_out):
            data_in = data_in.to(device)
            data_out = data_out.to(device)
            optimizer_sel.zero_grad()
            outputs = model(data_in)
            loss = criterion_sel(outputs, data_out.x)
            loss.backward()
            optimizer_sel.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader_in)
        print(f"Epoch {epoch + 1}: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            mlflow.pytorch.log_model(model, "models")
            mlflow.log_metric("best_loss", best_loss)


# type: ignore[return-value]
def evaluate(
    model: nn.Module,
    loader_in: DataLoader,
    loader_out: DataLoader,
    criterion_sel: nn.Module,
    return_predictions: bool = False,
) -> tuple[float, list[torch.Tensor]]:
    """Evaluate the performance of a GNN model on a given dataset.

    Args:
        model (torch.nn.Module): The GNN model to evaluate. loader_in
        (torch_geometric.loader.DataLoader): The input data loader. loader_out
        (torch_geometric.loader.DataLoader): The output data loader. criterion
        (torch.nn.Module): The loss function to use for evaluation. return_predictions
        (bool, optional): Whether to return the model predictions. Defaults to False.

    Returns:
        If `return_predictions` is False, returns the evaluation loss. If
        `return_predictions` is True, returns a tuple containing the evaluation loss and
        the model predictions.

    """
    model.eval()
    with torch.no_grad():
        loss = 0.0
        if return_predictions:
            y_preds: List[torch.Tensor] = []
        for data_in, data_out in zip(loader_in, loader_out):
            data_in = data_in.to(device)
            data_out = data_out.to(device)
            outputs = model(data_in)
            loss += criterion_sel(outputs, data_out.x)
            if return_predictions:
                y_pred.append(outputs.cpu())
        loss /= len(loader_in)
        if return_predictions:
            return loss, y_preds  # type: ignore[return-value]
        else:
            return loss  # type: ignore[return-value]


def update(frame):
    """Update the data of the current plot."""
    if preds == "truth":
        time_in_seconds = round((y_mem.time[frame] - y_mem.time[0]).item() * 24 * 3600)
    else:
        time_in_seconds = y_mem.time[frame] * 10
    im.set_array(y_mem.isel(time=frame))
    plt.title(f"Var: Theta_v - Time: {time_in_seconds:.0f} s\n Member: {member}")
    return im


if __name__ == "__main__":
    # Suppress the warning message
    warnings.filterwarnings("ignore", message="Setuptools is replacing distutils.")
    warnings.filterwarnings(
        "ignore",
        message="Encountered an unexpected error while inferring pip requirements",
    )

    # Load the data
    data_train = (
        xr.open_zarr(str(here()) + "/data/data_train.zarr").to_array().squeeze()
    )
    data_train = data_train.transpose("time", "member", "height", "ncells")
    data_train_in = data_train.isel(member=slice(0, 62))
    data_train_out = data_train.isel(member=slice(62, 124))

    data_test = xr.open_zarr(str(here()) + "/data/data_test.zarr").to_array().squeeze()
    data_test = data_test.transpose("time", "member", "height", "ncells")
    data_test_in = data_test.isel(member=slice(0, 62))
    data_test_out = data_test.isel(member=slice(62, 124))

    # Define the Graph Neural Network architecture
    nodes_in = data_train_in.shape[1]
    nodes_out = data_train_out.shape[1]
    channels_in = data_train_in.shape[2] * data_train_in.shape[3]
    channels_out = data_train_out.shape[2] * data_train_out.shape[3]
    # Define the edge indices for the graph
    edge_index_in = erdos_renyi_graph(nodes_in, edge_prob=1)
    edge_index_out = erdos_renyi_graph(nodes_out, edge_prob=1)

    model_gnn = GNNModel(channels_in, channels_out)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model_gnn.parameters(), lr=0.0001)
    scheduler = CyclicLR(
        optimizer,
        base_lr=0.0001,
        max_lr=0.001,
        mode="triangular2",
        cycle_momentum=False,
    )
    # Create data loaders
    loader_train_in = create_data_loader(data_train_in, edge_index_in, nodes_in)
    loader_train_out = create_data_loader(data_train_out, edge_index_out, nodes_out)
    loader_test_in = create_data_loader(data_test_in, edge_index_in, nodes_in)
    loader_test_out = create_data_loader(data_test_out, edge_index_out, nodes_out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_path)

    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(str(here()) + "/mlruns")
    mlflow_logger = MLFlowLogger(experiment_name=experiment_name)
    # Get the hostname of the machine

    retrain = True
    if retrain:
        # Train the model with MLflow logging
        with mlflow.start_run():
            # Train the model
            train_model(
                model_gnn.to(device),
                loader_train_in,
                loader_train_out,
                criterion.to(device),
                optimizer,
                num_epochs=10,
            )
    # Load the best checkpoint of the model from MLflow
    y_pred: List[torch.Tensor] = []
    with mlflow.start_run():
        test_loss, y_pred = evaluate(
            model_gnn.to(device),
            loader_test_in,
            loader_test_out,
            criterion.to(device),
            return_predictions=True,
        )
        print(f"Best model test loss: {test_loss:.4f}")
        mlflow.log_metric("test_loss", test_loss)

# Plot the predictions
# pylint: disable=no-member
# type: ignore
y_pred_reshaped = xr.DataArray(
    torch.cat(y_pred).numpy().reshape((72, 62, 128, 2632)),
    dims=["time", "member", "height", "ncells"],
)
member = 61
preds = "preds"
sort_indices = np.argsort(data_test_out.time.values)

# Create a new figure object
fig, ax = plt.subplots()
# Plot the first time step of the variable
if preds == "truth":
    y_mem = data_test_out.isel(member=member)
else:
    y_mem = y_pred_reshaped.isel(member=member)

# Reorder y_mem by the time dimension
y_mem = y_mem.isel(time=sort_indices)
# Plot the first time step of the variable
im = y_mem.isel(time=0).plot(ax=ax)
plt.gca().invert_yaxis()  # type: ignore # invert the y-axis
plt.title(f"Theta_v - Time: 0 s\n Member: {member}")

ani = animation.FuncAnimation(
    fig, update, frames=y_mem.shape[0], interval=100, blit=False
)

# Define the filename for the output gif
output_filename = f"{here()}/output/animation_member_{member}_{preds}.gif"

# Save the animation as a gif
ani.save(output_filename, writer="imagemagick", dpi=100)
