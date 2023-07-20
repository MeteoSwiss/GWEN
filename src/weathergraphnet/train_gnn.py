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
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt

# Third-party
import mlflow
import torch
import xarray as xr  # type: ignore
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


def create_data_loader(data: List, edge_index: torch.Tensor, nodes: int) -> DataLoader:
    """
    Create a PyTorch DataLoader object from a list of data samples and an edge index.

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
        for sample in data.values
    ]
    return DataLoader(dataset, batch_size=1, shuffle=True)


# pylint: disable=too-many-arguments
def train_model(
        model: nn.Module,
        train_loader_in: DataLoader,
        train_loader_out: DataLoader,
        criterion_sel: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int = 10,
        logger: Optional[MLFlowLogger] = None) -> None:
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

    best_loss = float('inf')
    # Train the GNN model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data_in, data_out in zip(train_loader_in, train_loader_out):
            data_in = data_in.to(device)
            data_out = data_out.to(device)
            optimizer.zero_grad()
            outputs = model(data_in)
            loss = criterion_sel(outputs, data_out.x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader_in)
        print(f"Epoch {epoch + 1}: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            mlflow.pytorch.log_model(model, "models")
            mlflow.log_metric("best_loss", best_loss)


def evaluate(
    model: nn.Module,
    loader_in: DataLoader,
    loader_out: DataLoader,
    criterion: nn.Module,
    return_predictions: bool = False,
) -> Union[float, Tuple[float, torch.Tensor]]:
    """
    Evaluate the performance of a GNN model on a given dataset.

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
            if return_predictions:
                y_pred = []
            for data_in, data_out in zip(loader_in, loader_out):
                data_in = data_in.to(device)
                data_out = data_out.to(device)
                output = model(data_in)
                loss += criterion(output, data_out).item() * data_in.size(0)
                if return_predictions:
                    y_pred.append(output.cpu())
            loss /= len(loader_in.dataset)
            if return_predictions:
                y_pred = torch.cat(y_pred, dim=0)
                return loss, y_pred
            else:
                return loss


if __name__ == "__main__":
    # Load the data
    data_train = (
        xr.open_zarr(str(here()) + "/data/data_train.zarr").to_array().squeeze()
    )
    data_train = data_train.transpose("time", "member", "height", "ncells")
    data_train_in = data_train.isel(member=slice(0, 62))
    data_train_out = data_train.isel(member=slice(62, 124))

    data_test = (
        xr.open_zarr(str(here()) + "/data/data_test.zarr").to_array().squeeze()
    )
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
        mode='triangular2',
        cycle_momentum=False)
    # Create data loaders
    loader_train_in = create_data_loader(data_train_in, edge_index_in, nodes_in)
    loader_train_out = create_data_loader(data_train_out, edge_index_out, nodes_out)
    loader_test_in = create_data_loader(data_test_in, edge_index_in, nodes_in)
    loader_test_out = create_data_loader(data_test_out, edge_index_out, nodes_out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mlflow.set_tracking_uri(str(here()) + "/mlruns")
    mlflow.set_experiment(experiment_name="WGN")
    mlflow_logger = MLFlowLogger(experiment_name="WGN")

    mode = "evaluate"
    if mode == "train":
        # Train the model with MLflow logging
        with mlflow.start_run():
            # Train the model
            train_model(
                model_gnn.to(device),
                loader_train_in,
                loader_train_out,
                criterion.to(device),
                optimizer,
                num_epochs=4,
                logger=mlflow_logger,
            )
    elif mode == "evaluate":
        # Load the best checkpoint of the model from MLflow
        y_pred = None
        with mlflow.start_run():
            test_loss, y_pred = evaluate(
                model_gnn, loader_test_in, loader_test_out, criterion,
                return_predictions=True)
            print(f"Best model test loss: {test_loss:.4f}")
            mlflow.log_metric("test_loss", test_loss)

# # Plot the predictions
# fig, ax = plt.subplots()
# ax.scatter(loader_test_out.dataset.y.numpy(), y_pred.numpy())
# ax.plot(loader_test_out.dataset.y.numpy(), loader_test_out.dataset.y.numpy(), color="r")
# ax.set_xlabel("True values")
# ax.set_ylabel("Predicted values")
# ax.set_title("Predictions vs. true values")
# plt.show()
