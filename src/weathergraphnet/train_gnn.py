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
import os
import socket
import warnings
from typing import List

# Third-party
import mlflow  # type: ignore
import torch
import torch_geometric  # type: ignore
import utils
import xarray as xr  # type: ignore
from pyprojroot import here
from pytorch_lightning.loggers import MLFlowLogger  # type: ignore
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import BatchSampler
from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torch_geometric.nn import GCNConv  # type: ignore
from torch_geometric.nn import TopKPooling  # type: ignore
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

    def __init__(self, nodes_in, nodes_out, in_channels,
                 out_channels, start_channels=1024):
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
        self.pool = TopKPooling(out_channels, ratio=nodes_out / nodes_in)

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

    # pylint: disable=too-many-arguments
    def train_model(
        self,
        train_loader_in: DataLoader,
        train_loader_out: DataLoader,
        optimizer_sel: optim.Optimizer,
        scheduler: None,
        loss_fn: nn.Module,
        mask=None,
        num_epochs: int = 10,
        device: str = "cuda",
    ) -> None:
        """Train a GNN model and output data using the specified loss function.

        Args:
            model (torch.nn.Module): The GNN model to train. loader_in
            (torch.utils.data.DataLoader): The data loader for the input data.
            loader_out (torch.utils.data.DataLoader): The data loader for the output
            data. channels_in (int): The number of input channels. criterion
            (torch.nn.Module): The loss function to use for training. optimizer
            (torch.optim.Optimizer): The optimizer to use for training. num_epochs
            (int): The number of epochs to train for.

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
                output = self(data_in)
                if mask is not None:
                    loss = loss_fn(output, data_out.x, mask.to(device))
                else:
                    loss = loss_fn(output, data_out.x)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()   # update the learning rate
                running_loss += loss.item()

            if len(train_loader_in) > 0:
                avg_loss = running_loss / len(train_loader_in)
                print(f"Epoch {epoch + 1}: {avg_loss}")
                mlflow.log_metric("loss", loss)
            else:
                avg_loss = None
                print(f"Skipping epoch {epoch + 1} due to empty data loader")
            if avg_loss is not None and avg_loss < best_loss:
                best_loss = avg_loss
                mlflow.pytorch.log_model(self, "models")
                mlflow.log_metric("best_loss", best_loss)

    def evaluate(
        self,
        loader_in: DataLoader,
        loader_out: DataLoader,
        loss_fn: nn.Module,
        mask: torch.Tensor,
        return_predictions: bool = False,
        device: str = "cuda",
    ) -> tuple[float, list[torch.Tensor]]:
        """Evaluate the performance of the GNN model on a given dataset.

        Args:
            loader_in (torch.utils.data.DataLoader): The input data loader. loader_out
            (torch.utils.data.DataLoader): The output data loader. loss_fn
            (torch.nn.Module): The loss function to use for evaluation.
            return_predictions (bool, optional): Whether to return the model
            predictions. Defaults to False.

        Returns:
            If `return_predictions` is False, returns the evaluation loss. If
            `return_predictions` is True, returns a tuple containing the evaluation loss
            and the model predictions.

        """
        self.eval()
        with torch.no_grad():
            loss = 0.0
            if return_predictions:
                y_preds: List[torch.Tensor] = []
            for data_in, data_out in zip(loader_in, loader_out):
                data_in = data_in.to(device)
                data_out = data_out.to(device)
                output = self(data_in)
                if mask is not None:
                    loss = loss_fn(output, data_out.x, mask.to(device))
                else:
                    loss = loss_fn(output, data_out.x)
                if return_predictions:
                    y_preds.append(output.cpu())
            loss /= len(loader_in)
            if return_predictions:
                return loss, y_preds
            else:
                return loss


# pylint: disable=no-member
def create_data_loader(data: List, edge_index: torch.Tensor,
                       nodes: int, batch_size: int) -> DataLoader:
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# pylint: disable=no-member


class CustomSampler(Sampler):
    def __init__(self, data, edge_index, batch_size):
        self.data = data
        self.edge_index = edge_index
        self.batch_size = batch_size

    def __len__(self):
        return len(self.edge_index)

    def __iter__(self):
        # Generate the indices for each mini-batch based on the edge index
        batch_sampler = BatchSampler(
            SequentialSampler(self.edge_index),
            batch_size=self.batch_size,
            drop_last=False,
        )
        indices = [index for batch in batch_sampler for index in batch]
        return iter(indices)

    def __getitem__(self, index):
        src, dst = self.edge_index[index]
        x = self.data.isel(ncells=src)
        y = self.data.isel(ncells=dst)
        return x, y





def create_data_sampler(data, edge_index, nodes, batch_size, num_workers):
    # Convert the data and labels to PyTorch tensors
    dataset = [
        Data(
            x=torch.tensor(sample, dtype=torch.float32).view(nodes, -1),
            edge_index=edge_index)
        for sample in data.values]
    # Create a dataset from the data and labels
    tensor_dataset = torch.utils.data.TensorDataset(*dataset)
    # Create a random sampler for the data
    sampler = CustomSampler(data, edge_index, batch_size)
    # Create a collate function to convert the data into mini-batches

    def collate_fn(batch):
        # Extract the subgraphs and labels from the batch
        subgraphs = [item[0] for item in batch]
        labels_batch = [item[1:] for item in batch]

        # Pad the subgraphs to the same number of nodes
        max_nodes = max([subgraph.num_nodes for subgraph in subgraphs])
        for subgraph in subgraphs:
            num_nodes = subgraph.num_nodes
            if num_nodes < max_nodes:
                subgraph.x = torch.cat([subgraph.x, torch.zeros(
                    (max_nodes - num_nodes, subgraph.num_node_features))], dim=0)
                subgraph.edge_index = torch.cat(
                    [subgraph.edge_index, torch.zeros((2, max_nodes - num_nodes))], dim=1)

        # Convert the subgraphs and labels to tensors
        subgraphs = torch_geometric.data.Batch.from_data_list(subgraphs)
        labels_batch = [torch.cat(labels, dim=0) for labels in zip(*labels_batch)]

        return subgraphs, *labels_batch
    # Create a data loader with the specified batch size and collate function
    loader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader










if __name__ == "__main__":
    # Suppress the warning message
    warnings.simplefilter("always")
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
    # Load the data
    data_test = (
        xr.open_zarr(str(here()) + "/data/data_test.zarr").to_array().squeeze()
    )
    data_train = data_train.transpose("time", "member", "height", "ncells")

    member_split=100
    data_train_in, data_train_out = utils.MyDataset(data_train, member_split)
    data_test_in, data_test_out = utils.MyDataset(data_test, member_split)

    data_test = xr.open_zarr(str(here()) + "/data/data_test.zarr").to_array().squeeze()
    data_test = data_test.transpose("time", "member", "height", "ncells")

    # Define the Graph Neural Network architecture
    nodes_in = data_train_in.shape[1]
    nodes_out = data_train_out.shape[1]
    channels_in = data_train_in.shape[2] * data_train_in.shape[3]
    channels_out = data_train_out.shape[2] * data_train_out.shape[3]
    # Define the edge indices for the graph
    edge_index_in = erdos_renyi_graph(nodes_in, edge_prob=1)
    edge_index_out = erdos_renyi_graph(nodes_out, edge_prob=1)

    # Create data loaders
    sample = False
    if sample:
        loader_train_in = create_data_sampler(
            data_train_in, edge_index_in, nodes_in, 13, 4)
        loader_train_out = create_data_sampler(
            data_train_out, edge_index_out, nodes_out, 13, 4)
        loader_test_in = create_data_sampler(
            data_test_in, edge_index_in, nodes_in, 13, 4)
        loader_test_out = create_data_sampler(
            data_test_out, edge_index_out, nodes_out, 13, 4)
    else:
        loader_train_in = create_data_loader(
            data_train_in, edge_index_in, nodes_in, batch_size=8)
        loader_train_out = create_data_loader(
            data_train_out, edge_index_out, nodes_out, batch_size=8)
        loader_test_in = create_data_loader(
            data_test_in, edge_index_in, nodes_in, batch_size=8)
        loader_test_out = create_data_loader(
            data_test_out, edge_index_out, nodes_out, batch_size=8)

    loss_fn = utils.EnsembleVarianceRegularizationLoss(alpha=0.1)
    model = GNNModel(nodes_in, nodes_out, channels_in, channels_out, 1024)
    optimizer = optim.Adam(model.parameters())
    scheduler = CyclicLR(
        optimizer,
        base_lr=0.0001,
        max_lr=0.001,
        mode="triangular2",
        cycle_momentum=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Move the loss function and dataloader to the cuda device
    loss_fn = loss_fn.to(device)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    train_model = model.module.train_model if isinstance(
        model, nn.DataParallel) else model.train_model
    
    if loss_fn == utils.MaskedLoss:
        # Create a mask that masks all cells that stay constant over all time steps
        variance = data_train.var(dim='time')
        # Create a mask that hides all data with zero variance
        mask = variance <= 1e-6
        torch.from_numpy(mask.values.astype(float))
        print(f"Number of masked cells: {(mask[0].values == 1).sum()}", flush=True)

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
    
    retrain = True
    if retrain:
        # Train the model with MLflow logging
        MLFlowLogger(experiment_name=experiment_name)
        with mlflow.start_run():
            # Train the model
            train_model(
                loader_train_in,
                loader_train_out,
                optimizer,
                scheduler,
                loss_fn,
                mask=None,
                num_epochs=20,
                device=device
            )
    else:

        # Load the best checkpoint of the model from the most recent MLflow run
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["start_time desc"],
            max_results=1)

        if len(runs) == 0:
            print("No runs found in experiment:", experiment_name)
        run_id = runs.iloc[0].run_id
        best_model_path = mlflow.get_artifact_uri()
        best_model_path = os.path.abspath(os.path.join(best_model_path, "../../"))
        best_model_path = os.path.join(
            best_model_path,
            run_id,
            "artifacts",
            "models")
        model = mlflow.pytorch.load_model(best_model_path)
        model.to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        evaluate = model.module.evaluate if isinstance(
            model, nn.DataParallel) else model.evaluate
    y_pred: List[torch.Tensor] = []
    test_loss, y_pred = evaluate(
        loader_test_in,
        loader_test_out,
        loss_fn,
        None,
        return_predictions=True,
        device=device,
    )
    print(f"Best model test loss: {test_loss:.4f}")


# Plot the predictions
# pylint: disable=no-member
# type: ignore
y_pred_reshaped = xr.DataArray(
    torch.cat(y_pred).numpy().reshape((data_test_out.values.shape)),
    dims=["time", "member", "height", "ncells"],
)
member = 0
preds = "GNN"

# Reorder y_mem by the time dimension
y_pred_reshaped["time"] = data_test["time"]
y_pred_reshaped["height"] = data_test["height"]

# Plot the first time step of the variable
if preds == "ICON":
    y_mem = data_test.isel(member=member)
else:
    y_mem = y_pred_reshaped.isel(member=member)

y_mem = y_mem.sortby(y_mem.time, ascending=True)

ani = utils.animate(y_mem, member=member, preds=preds)

# Define the filename for the output gif
output_filename = f"{here()}/output/animation_member_{member}_{preds}.gif"

# Save the animation as a gif
ani.save(output_filename, writer="imagemagick", dpi=100)
