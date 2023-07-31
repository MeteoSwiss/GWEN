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
from dataclasses import dataclass
from typing import List
from typing import Optional

# Third-party
import mlflow
import torch
import torch_geometric
import xarray as xr
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import BatchSampler
from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import erdos_renyi_graph

# First-party
from weathergraphnet.utils import create_animation
from weathergraphnet.utils import EnsembleVarianceRegularizationLoss
from weathergraphnet.utils import load_best_model
from weathergraphnet.utils import load_config_and_data
from weathergraphnet.utils import MaskedLoss
from weathergraphnet.utils import move_to_device
from weathergraphnet.utils import MyDataset
from weathergraphnet.utils import setup_mlflow


@dataclass
class GNNConfig:
    """Configuration parameters for the GNN model."""

    nodes_in: int
    nodes_out: int
    in_channels: int
    out_channels: int
    hidden_feats: int


class DownConvLayers(torch.nn.Module):
    def __init__(self, configs: GNNConfig):
        """Initialize the GNN model."""
        super().__init__()
        self.conv1 = GCNConv(configs.in_channels, configs.hidden_feats)
        self.conv2 = GCNConv(configs.hidden_feats, configs.hidden_feats // 2)
        self.conv3 = GCNConv(configs.hidden_feats // 2, configs.hidden_feats // 4)
        self.conv4 = GCNConv(configs.hidden_feats // 4, configs.hidden_feats // 8)
        self.conv5 = GCNConv(configs.hidden_feats // 8, configs.hidden_feats // 16)

    def forward(self, x, edge_index):
        """Perform a forward pass through the GNN model."""
        x = self.activation(self.conv1(x, edge_index))
        x = self.activation(self.conv2(x, edge_index))
        x = self.activation(self.conv3(x, edge_index))
        x = self.activation(self.conv4(x, edge_index))
        x = self.activation(self.conv5(x, edge_index))
        return x


class UpConvLayers(torch.nn.Module):
    """A Graph Neural Network (GNN) model for weather prediction."""

    def __init__(self, configs: GNNConfig):
        """Initialize the GNN model."""
        super().__init__()
        self.upconv1 = GCNConv(configs.hidden_feats // 16, configs.hidden_feats // 8)
        self.upconv2 = GCNConv(configs.hidden_feats // 8, configs.hidden_feats // 4)
        self.upconv3 = GCNConv(configs.hidden_feats // 4, configs.hidden_feats // 2)
        self.upconv4 = GCNConv(configs.hidden_feats // 2, configs.hidden_feats)
        self.upconv5 = GCNConv(configs.hidden_feats, configs.out_channels)

    def forward(self, x, edge_index):
        """Perform a forward pass through the GNN model."""
        x = self.activation(self.upconv1(x, edge_index))
        x = self.activation(self.upconv2(x, edge_index))
        x = self.activation(self.upconv3(x, edge_index))
        x = self.activation(self.upconv4(x, edge_index))
        x = self.upconv5(x, edge_index)
        return x


class GCNConvLayers(torch.nn.Module):
    """A Graph Neural Network (GNN) model for weather prediction."""

    def __init__(self, configs: GNNConfig):
        """Initialize the GNN model."""
        super().__init__()
        self.down_conv_layers = DownConvLayers(configs)
        self.up_conv_layers = UpConvLayers(configs)

    def forward(self, x, edge_index):
        """Perform a forward pass through the GNN model."""
        x = self.down_conv_layers(x, edge_index)
        x = self.up_conv_layers(x, edge_index)
        return x


class TopKPoolingLayer(torch.nn.Module):
    """A Graph Neural Network (GNN) model for weather prediction."""

    def __init__(self, configs: GNNConfig):
        """Initialize the GNN model."""
        super().__init__()
        self.pool = TopKPooling(
            configs.out_channels, ratio=configs.nodes_out / configs.nodes_in
        )

    def forward(self, x, edge_index):
        """Perform a forward pass through the GNN model."""
        x, edge_index, _, _, _, _ = self.pool(x, edge_index)
        return x, edge_index


class GNNModel(torch.nn.Module):
    """A Graph Neural Network (GNN) model for weather prediction.

    Args:
        config (GNNConfig): Configuration parameters for the GNN model.

    Methods:
        forward(data): Performs a forward pass through the GNN model.

    """

    def __init__(self, configs: GNNConfig):
        """Initialize the GNN model."""
        super().__init__()
        self.conv_layers = GCNConvLayers(configs)
        self.pool_layer = TopKPoolingLayer(configs)
        self.activation = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv_layers(x, edge_index)
        x, edge_index = self.pool_layer(x, edge_index)
        return x

    def train_model(self, configs) -> None:
        """Train a GNN model and output data using the specified loss function.

        Args:
            model (torch.nn.Module): The GNN model to train. loader_in
            (torch.data.DataLoader): The data loader for the input data.
            loader_out (torch.data.DataLoader): The data loader for the output
            data. channels_in (int): The number of input channels. criterion
            (torch.nn.Module): The loss function to use for training. optimizer
            (torch.optim.Optimizer): The optimizer to use for training. num_epochs
            (int): The number of epochs to train for.

        Returns:
            None

        """
        # Set the seed for reproducibility
        torch.manual_seed(configs.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(configs.seed)

        best_loss = float("inf")
        # Train the GNN model
        for epoch in range(configs.num_epochs):
            running_loss = 0.0
            for data_in, data_out in zip(
                configs.train_loader_in, configs.train_loader_out
            ):
                data_in = data_in.to(configs.device)
                data_out = data_out.to(configs.device)
                configs.optimizer_sel.zero_grad()
                output = self(data_in)
                if configs.mask is not None:
                    loss = configs.loss_fn(
                        output, data_out.x, configs.mask.to(configs.device)
                    )
                else:
                    loss = configs.loss_fn(output, data_out.x)
                loss.backward()
                configs.optimizer_sel.step()
                if configs.scheduler is not None:
                    configs.scheduler.step()  # update the learning rate
                running_loss += loss.item()

            if len(configs.train_loader_in) > 0:
                avg_loss = running_loss / len(configs.train_loader_in)
                print(f"Epoch {epoch + 1}: {avg_loss}")
                mlflow.log_metric("loss", avg_loss)
            else:
                avg_loss = None
                print(f"Skipping epoch {epoch + 1} due to empty data loader")
            if avg_loss is not None and avg_loss < best_loss:
                best_loss = avg_loss
                mlflow.pytorch.log_model(self, "models")
                mlflow.log_metric("best_loss", best_loss)

    def evaluate(
        self,
        configs,
    ) -> tuple[float, list[torch.Tensor]]:
        """Evaluate the performance of the GNN model on a given dataset.

        Args:
            loader_in (torch.data.DataLoader): The input data loader. loader_out
            (torch.data.DataLoader): The output data loader. loss_fn
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
            if configs.return_predictions:
                y_preds: List[torch.Tensor] = []
            for data_in, data_out in zip(configs.loader_in, configs.loader_out):
                data_in = data_in.to(configs.device)
                data_out = data_out.to(configs.device)
                output = self(data_in)
                if mask is not None:
                    loss = configs.loss_fn(output, data_out.x, mask.to(configs.device))
                else:
                    loss = configs.loss_fn(output, data_out.x)
                if configs.return_predictions:
                    y_preds.append(output.cpu())
            loss /= len(configs.loader_in)
            if configs.return_predictions:
                return loss, y_preds
            else:
                return loss


def create_data_loader(
    data: List, edge_index: torch.Tensor, nodes: int, batch: int
) -> DataLoader:
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
        for sample in data.values
    ]
    return DataLoader(dataset, batch=batch, shuffle=True)


class CustomSampler(Sampler):
    """A custom sampler for the PyTorch DataLoader class."""

    def __init__(self, data, edge_index, batch):
        """Initialize the custom sampler."""
        self.data = data
        self.edge_index = edge_index
        self.batch = batch

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.edge_index)

    def __iter__(self):
        """Return an iterator over the indices of the samples in the dataset."""
        batch_sampler = BatchSampler(
            SequentialSampler(self.edge_index),
            batch_size=self.batch,
            drop_last=False,
        )
        indices = [index for batch in batch_sampler for index in batch]
        return iter(indices)

    def __getitem__(self, index):
        """Return the data sample at the specified index."""
        src, dst = self.edge_index[index]
        x = self.data.isel(ncells=src)
        y = self.data.isel(ncells=dst)
        return x, y


class TrainingConfig:
    """Training configuration parameters."""

    loader_train_in: DataLoader
    loader_train_out: DataLoader
    optimizer: nn.Module
    scheduler: nn.Module
    loss_fn: nn.Module
    mask: Optional[torch.Tensor] = None
    num_epochs: int = 10
    device: str = "cuda"
    seed: int = 42


class EvaluationConfig:
    """Evaluation configuration parameters."""

    loader_in: DataLoader
    loader_out: DataLoader
    loss_fn: nn.Module
    mask: Optional[torch.Tensor] = None
    return_predictions: bool = True
    device: str = "cuda"
    seed: int = 42


def create_data_sampler(data, edge_index, nodes, batch, workers):
    """Create a PyTorch DataLoader object from a list of data samples and edges."""
    dataset = [
        Data(
            x=torch.tensor(sample, dtype=torch.float32).view(nodes, -1),
            edge_index=edge_index,
        )
        for sample in data.values
    ]
    # Create a dataset from the data and labels
    tensor_dataset = torch.data.TensorDataset(*dataset)
    # Create a random sampler for the data
    sampler = CustomSampler(data, edge_index, batch)
    # Create a collate function to convert the data into mini-batches

    def collate_fn(batch):
        """Collate function for the data loader."""
        # Extract the subgraphs and labels from the batch
        subgraphs = [item[0] for item in batch]
        labels_batch = [item[1:] for item in batch]

        # Pad the subgraphs to the same number of nodes
        max_nodes = max(subgraph.num_nodes for subgraph in subgraphs)
        for subgraph in subgraphs:
            num_nodes = subgraph.num_nodes
            if num_nodes < max_nodes:
                subgraph.x = torch.cat(
                    [
                        subgraph.x,
                        torch.zeros(
                            (max_nodes - num_nodes, subgraph.num_node_features)
                        ),
                    ],
                    dim=0,
                )
                subgraph.edge_index = torch.cat(
                    [subgraph.edge_index, torch.zeros((2, max_nodes - num_nodes))],
                    dim=1,
                )

        # Convert the subgraphs and labels to tensors
        subgraphs = torch_geometric.data.Batch.from_data_list(subgraphs)
        labels_batch = [torch.cat(labels, dim=0) for labels in zip(*labels_batch)]

        return subgraphs, *labels_batch

    # Create a data loader with the specified batch size and collate function
    loader = DataLoader(
        tensor_dataset,
        batch_size=batch,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


if __name__ == "__main__":
    # Load the configuration parameters and the input and output data
    config, data_train, data_test = load_config_and_data()

    data_train_in, data_train_out = MyDataset(data_train, config.member_split)
    data_test_in, data_test_out = MyDataset(data_test, config.member_split)

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
            data_train_in,
            edge_index_in,
            nodes_in,
            config.batch_size,
            config.num_workers,
        )
        loader_train_out = create_data_sampler(
            data_train_out,
            edge_index_out,
            nodes_out,
            config.batch_size,
            config.num_workers,
        )
        loader_test_in = create_data_sampler(
            data_test_in, edge_index_in, nodes_in, config.batch_size, config.num_workers
        )
        loader_test_out = create_data_sampler(
            data_test_out,
            edge_index_out,
            nodes_out,
            config.batch_size,
            config.num_workers,
        )
    else:
        loader_train_in = create_data_loader(
            data_train_in, edge_index_in, nodes_in, config.batch_size
        )
        loader_train_out = create_data_loader(
            data_train_out, edge_index_out, nodes_out, config.batch_size
        )
        loader_test_in = create_data_loader(
            data_test_in, edge_index_in, nodes_in, config.batch_size
        )
        loader_test_out = create_data_loader(
            data_test_out, edge_index_out, nodes_out, config.batch_size
        )

    loss_fn = EnsembleVarianceRegularizationLoss(alpha=0.1)
    model_config = GNNConfig(
        nodes_in, nodes_out, channels_in, channels_out, config.hidden_feats
    )
    model = GNNModel(model_config)
    optimizer = optim.Adam(model.parameters())
    scheduler = CyclicLR(
        optimizer,
        base_lr=config.lr,
        max_lr=10 * config.lr,
        mode="triangular2",
        cycle_momentum=False,
    )

    model, loss_fn = move_to_device(model, loss_fn)

    train_model = (
        model.module.train_model
        if isinstance(model, nn.DataParallel)
        else model.train_model
    )

    if loss_fn == MaskedLoss:
        # Create a mask that masks all cells that stay constant over all time steps
        variance = data_train.var(dim="time")
        # Create a mask that hides all data with zero variance
        mask = variance <= config.mask_threshold
        torch.from_numpy(mask.values.astype(float))
        print(f"Number of masked cells: {(mask[0].values == 1).sum()}", flush=True)

    artifact_path, experiment_name = setup_mlflow()

    retrain = True
    if retrain:
        # Train the model with MLflow logging
        MLFlowLogger(experiment_name=experiment_name)
        with mlflow.start_run():
            # Train the model Create a TrainingConfig object that contains both the
            # local variables and the JSON parameters
            config = TrainingConfig(
                loader_train_in=loader_train_in,
                loader_train_out=loader_train_out,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                mask=mask,
                num_epochs=config.epochs,
                device=config.device,
                seed=config.seed,
            )
            # Pass the TrainingConfig object to the train_model method
            train_model(config)
    else:
        # Load the best model from the most recent MLflow run
        model = load_best_model(experiment_name)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        evaluate = (
            model.module.evaluate
            if isinstance(model, nn.DataParallel)
            else model.evaluate
        )
    y_pred: List[torch.Tensor] = []
    # Evaluate the model on the test data
    # pylint: disable=R0801
    config = EvaluationConfig(
        loader_in=loader_test_in,
        loader_out=loader_test_out,
        loss_fn=loss_fn,
        mask=mask,
        return_predictions=True,
        device=config.device,
        seed=config.seed,
    )
    test_loss, y_pred = evaluate(config)
    print(f"Best model test loss: {test_loss:.4f}")

    # Plot the predictions

    y_pred_reshaped = xr.DataArray(
        torch.cat(y_pred).numpy().reshape((data_test_out.values.shape)),
        dims=["time", "member", "height", "ncells"],
    )

    data_gif = {
        "y_pred_reshaped": y_pred_reshaped,
        "data_test": data_test,
    }

    output_filename = create_animation(data_gif, member=0, preds="GNN")
