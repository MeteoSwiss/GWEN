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
from typing import Optional

# Third-party
import mlflow  # type: ignore
import torch
import xarray as xr  # type: ignore
from pyprojroot import here
from pytorch_lightning.loggers import MLFlowLogger  # type: ignore
from torch import nn
from torch import optim
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torch_geometric.nn import GCNConv  # type: ignore
from torch_geometric.utils import erdos_renyi_graph  # type: ignore


# pylint: disable=too-many-arguments
def train_model(
    model: nn.Module,
    train_loader_in: DataLoader,
    train_loader_out: DataLoader,
    criterion_sel: nn.Module,
    optimizer_sel: optim.Optimizer,
    num_epochs: int = 10,
    logger: Optional[MLFlowLogger] = None,
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
        print(f"Epoch {epoch + 1}: {running_loss / len(train_loader_in)}")
        # Log metrics to the logger
        if logger is not None:
            logger.log_metrics(
                {"train_loss": running_loss / len(train_loader_in)}, step=epoch
            )


class GNNModel(torch.nn.Module):
    # pylint: disable=too-many-instance-attributes
    """A Graph Neural Network (GNN) model for weather prediction.

    Args:
        in_channels (int): The number of input channels. out_channels (int): The number
        of output channels.

    Methods:
        forward(data): Performs a forward pass through the GNN model.

    """

    def __init__(self, in_channels, out_channels):
        """Initialize the GNN model."""
        super().__init__()
        self.conv1 = GCNConv(in_channels, 163840)
        self.conv2 = GCNConv(163840, 81920)
        self.conv3 = GCNConv(81920, 40960)
        self.conv4 = GCNConv(40960, 20480)
        self.conv5 = GCNConv(20480, 10240)
        self.upconv1 = GCNConv(10240, 20480)
        self.upconv2 = GCNConv(20480, 40960)
        self.upconv3 = GCNConv(40960, 81920)
        self.upconv4 = GCNConv(81920, 163840)
        self.upconv5 = GCNConv(163840, out_channels)

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


if __name__ == "__main__":
    # Load the data
    data_train = (
        xr.open_zarr(str(here()) + "/data/data_train.zarr").to_array().squeeze()
    )
    data_train = data_train.transpose("time", "member", "height", "ncells")
    data_train_in = data_train.isel(member=slice(0, 62))
    data_train_out = data_train.isel(member=slice(62, 124))

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
    optimizer = optim.Adam(model_gnn.parameters(), lr=0.001)

    # Define the data loader for the training set
    # pylint: disable=no-member
    train_in_dataset = [
        Data(
            x=torch.tensor(sample, dtype=torch.float32).view(nodes_in, -1),
            edge_index=edge_index_in,
        )
        for sample in data_train_in.values
    ]
    loader_train_in = DataLoader(train_in_dataset, batch_size=1, shuffle=True)

    # Define the data loader for the training set
    # pylint: disable=no-member
    train_out_dataset = [
        Data(
            x=torch.tensor(sample, dtype=torch.float32).view(nodes_out, -1),
            edge_index=edge_index_out,
        )
        for sample in data_train_out.values
    ]
    loader_train_out = DataLoader(train_out_dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mlflow.set_tracking_uri(str(here()) + "/mlruns")
    mlflow.set_experiment(experiment_name="WGN")
    # Create an MLflow logger
    mlflow_logger = MLFlowLogger(experiment_name="WGN")
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
            logger=mlflow_logger,
        )
