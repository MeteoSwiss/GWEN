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
    --input (str): The path to the NumPy array containing the input data.
    --output (str): The path to the NumPy array containing the output data.
    --channels-in (int): The number of input channels.
    --channels-out (int): The number of output channels.
    --epochs (int): The number of epochs to train for. Default is 10.
    --lr (float): The learning rate for the optimizer. Default is 0.001.
    --batch-size (int): The batch size for the DataLoader. Default is 1.
    --shuffle (bool): Whether to shuffle the DataLoader. Default is True.
    --workers (int): The number of workers to use for the DataLoader. Default is 0.
    --seed (int): The random seed to use for training. Default is 42.
    --device (str): The device to use for training. Default is "cuda".
    --save-model (str): The path to save the trained model to. Default is "model.pt".

"""

# Standard library
import random
from typing import cast
from typing import List
from typing import Tuple

# Third-party
import mlflow  # type: ignore
import numpy as np
import torch
import torch_geometric  # type: ignore
import xarray as xr
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import BatchSampler
from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torch_geometric.utils import erdos_renyi_graph  # type: ignore

# First-party
from weathergraphnet.logger import setup_logger
from weathergraphnet.models import EvaluationConfigGNN
from weathergraphnet.models import GNNConfig
from weathergraphnet.models import GNNModel
from weathergraphnet.models import TrainingConfigGNN
from weathergraphnet.utils import create_animation
from weathergraphnet.utils import load_best_model
from weathergraphnet.utils import load_config_and_data
from weathergraphnet.utils import MaskedLoss
from weathergraphnet.utils import MyDataset
from weathergraphnet.utils import setup_mlflow

logger = setup_logger()

# TODO: add dropout layers to all my models!


def create_data_loader(
    data: xr.Dataset, edge_index: torch.Tensor, nodes: int, batch: int
) -> DataLoader:
    """Create a PyTorch DataLoader object from a list of data samples and an edge index.

    Args:
        data (List): A list of data samples.
        edge_index (torch.Tensor): The edge index for the graph.

    Returns:
        A PyTorch DataLoader object.

    """
    try:
        dataset = [
            Data(
                x=torch.tensor(sample, dtype=torch.float32).view(nodes, -1),
                edge_index=edge_index,
            )
            for sample in np.array(data.values)
        ]
        return DataLoader(dataset, batch_size=batch, shuffle=True)
    except Exception as error:
        logger.error("Error creating data loader: %s", error)
        raise


class CustomSampler(Sampler):
    """A custom sampler for the PyTorch DataLoader class.

    Args:
        data (torch.Tensor): The input data.
        edge_index (torch.Tensor): The edge index for the graph.
        batch (int): The batch size.

    """

    def __init__(self, data: xr.Dataset, edge_index: torch.Tensor, batch: int) -> None:
        """Initialize the custom sampler.

        Args:
            data (torch.Tensor): The input data.
            edge_index (torch.Tensor): The edge index for the graph.
            batch (int): The batch size.

        """
        try:
            self.data = data
            self.edge_index = edge_index
            self.batch = batch
        except Exception as error:
            logger.error("Error initializing custom sampler: %s", error)
            raise

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            The number of samples in the dataset.

        """
        try:
            return len(self.edge_index)
        except Exception as error:
            logger.error("Error getting length of custom sampler: %s", error)
            raise

    def __iter__(self):
        """Return an iterator over the indices of the samples in the dataset.

        Returns:
            An iterator over the indices of the samples in the dataset.

        """
        try:
            batch_sampler = BatchSampler(
                SequentialSampler(self.edge_index),
                batch_size=self.batch,
                drop_last=False,
            )
            indices = [index for batch in batch_sampler for index in batch]
            return iter(indices)
        except Exception as error:
            logger.error("Error iterating over custom sampler: %s", error)
            raise

    def __getitem__(self, index: int) -> Tuple[xr.Dataset, xr.Dataset]:
        """Return the data sample at the specified index.

        Args:
            index (int): The index of the data sample to return.

        Returns:
            The data sample at the specified index.

        """
        try:
            src, dst = self.edge_index[index]
            x = self.data.isel(ncells=src)
            y = self.data.isel(ncells=dst)
            return x, y
        except Exception as error:
            logger.error("Error getting item from custom sampler: %s", error)
            raise


def create_data_sampler(
    data: xr.Dataset,
    edge_index: torch.Tensor,
    nodes: int,
    batch: int,
    workers: int,
) -> DataLoader:
    """Create a data loader for the specified data.

    Args:
        data (List): The data to load.
        edge_index (torch.Tensor): The edge index tensor.
        nodes (int): The number of nodes in the data.
        batch (int): The batch size.
        workers (int): The number of workers to use.

    Returns:
        DataLoader: The data loader.

    """
    try:
        dataset = [
            torch_geometric.data.Data(
                x=torch.tensor(sample, dtype=torch.float32).view(nodes, -1),
                edge_index=edge_index,
            )
            for sample in np.array(data.values)
        ]
        # Create a dataset from the data and labels
        tensor_dataset = torch.utils.data.TensorDataset(*dataset)
        # Create a random sampler for the data
        sampler = CustomSampler(data, edge_index, batch)
        # Create a collate function to convert the data into mini-batches

        def collate_fn(batch):
            """Collate function for the data loader.

            Args:
                batch (List): The batch of data to collate.

            Returns:
                torch_geometric.data.Batch: The collated data.

            """
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
            labels_batch = [torch.cat(labels, dim=0) for labels in zip(*labels_batch)]
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
            verbose=True,
        )
        return loader
    except Exception as error:
        logger.error("Error creating data sampler: %s", error)
        raise


if __name__ == "__main__":
    # Load the configuration parameters and the input and output data
    config, data_train, data_test = load_config_and_data()

    try:
        # TODO fix this ugly naming and difference from CNN and test/train meaning
        data_train_set = MyDataset(data_train, config["member_split"])
        data_train_in = data_train_set.data.isel(member=data_train_set.train_indices)
        data_train_out = data_train_set.data.isel(member=data_train_set.test_indices)
        data_test_set = MyDataset(data_test, config["member_split"])
        data_test_in = data_test_set.data.isel(member=data_test_set.train_indices)
        data_test_out = data_test_set.data.isel(member=data_test_set.test_indices)
    except IndexError as e:
        logger.exception("Error occurred while creating datasets: %s", e)
    try:
        # Define the Graph Neural Network architecture
        nodes_in = data_train_in.shape[1]
        nodes_out = data_train_out.shape[1]
        channels_in = data_train_in.shape[2] * data_train_in.shape[3]
        channels_out = data_train_out.shape[2] * data_train_out.shape[3]
        # Define the edge indices for the graph
        edge_index_in = erdos_renyi_graph(nodes_in, edge_prob=1)
        edge_index_out = erdos_renyi_graph(nodes_out, edge_prob=1)
    except IndexError as e:
        logger.exception("Error occurred while defining GNN architecture: %s", e)

    try:
        # Create data loaders
        sample = False
        if sample:
            loader_train_in = create_data_sampler(
                data_train_in,
                edge_index_in,
                nodes_in,
                config["batch_size"],
                config["num_workers"],
            )
            loader_train_out = create_data_sampler(
                data_train_out,
                edge_index_out,
                nodes_out,
                config["batch_size"],
                config["num_workers"],
            )
            loader_test_in = create_data_sampler(
                data_test_in,
                edge_index_in,
                nodes_in,
                config["batch_size"],
                config["num_workers"],
            )
            loader_test_out = create_data_sampler(
                data_test_out,
                edge_index_out,
                nodes_out,
                config["batch_size"],
                config["num_workers"],
            )
        else:
            loader_train_in = create_data_loader(
                data_train_in, edge_index_in, nodes_in, config["batch_size"]
            )
            loader_train_out = create_data_loader(
                data_train_out, edge_index_out, nodes_out, config["batch_size"]
            )
            loader_test_in = create_data_loader(
                data_test_in, edge_index_in, nodes_in, config["batch_size"]
            )
            loader_test_out = create_data_loader(
                data_test_out, edge_index_out, nodes_out, config["batch_size"]
            )
    except (IndexError, TypeError) as e:
        logger.exception("Error occurred while creating data loaders: %s", e)

    try:
        # loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        loss_fn = nn.L1Loss()

        if isinstance(loss_fn, MaskedLoss):
            # Create a mask that masks all cells that stay constant over all time steps
            variance = data_train.var(dim="time")
            # Create a mask that hides all data with zero variance
            mask = variance <= config["mask_threshold"]
            logger.info("Number of masked cells: %d", (mask[0].values == 1).sum())
            logger.info("Number of masked cells: %d", (mask[0].values == 1).sum())
        else:
            mask = None
    except (ValueError, TypeError) as e:
        logger.exception("Error occurred while creating loss function: %s", e)
    artifact_path, experiment_name = setup_mlflow()
    try:
        if config["retrain"]:
            gnn_config = GNNConfig(
                nodes_in=nodes_in,
                nodes_out=nodes_out,
                channels_in=channels_in,
                channels_out=channels_out,
                hidden_feats=64,
            )
            model = GNNModel(gnn_config)
            optimizer = optim.Adam(model.parameters())
            scheduler = CyclicLR(
                optimizer,
                base_lr=config["lr"],
                max_lr=10 * config["lr"],
                mode="triangular2",
                cycle_momentum=False,
            )

            # Train the model with MLflow logging
            MLFlowLogger(experiment_name=experiment_name)
            with mlflow.start_run():
                # Train the model Create a TrainingConfig object that contains both the
                # local variables and the JSON parameters
                config_train = (
                    TrainingConfigGNN(  # pylint: disable=too-many-function-args
                        loader_train_in=loader_train_in,
                        loader_train_out=loader_train_out,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        mask=mask,
                        epochs=config["epochs"],
                        device=config["device"],
                        seed=config["seed"],
                    )
                )
                # Pass the TrainingConfig object to the train method
                if torch.cuda.device_count() > 1:
                    logger.info(
                        "Using %d GPUs for Evaluation", torch.cuda.device_count()
                    )
                    model = cast(GNNModel, nn.DataParallel(model).module)
                    loss_fn = loss_fn.to(config["device"])
                model.train_with_configs(config_train)

        else:
            # Load the best model from the most recent MLflow run
            model_best = load_best_model(experiment_name)
            if isinstance(model_best, GNNModel):
                model = model_best
            else:
                model = cast(GNNModel, model_best)
    except mlflow.exceptions.MlflowException as e:
        logger.exception("Error occurred while setting up MLflow: %s", e)

    try:
        y_pred: List[torch.Tensor] = []
        # Evaluate the model on the test data
        # pylint: disable=R0801
        config_eval = EvaluationConfigGNN(
            loader_in=loader_test_in,
            loader_out=loader_test_out,
            loss_fn=loss_fn,
            mask=mask,
            device=config["device"],
            seed=config["seed"],
        )
        if torch.cuda.device_count() > 1:
            logger.info("Using %d GPUs for Training", torch.cuda.device_count())
            model = cast(GNNModel, nn.DataParallel(model).module)
            loss_fn = loss_fn.to(config["device"])
        test_loss, y_pred = model.eval_with_configs(config_eval)
        # test_loss = test_loss.mean().item()
        logger.info("Best model test loss: %f", test_loss)
    except (RuntimeError, ValueError) as e:
        logger.exception("Error occurred while evaluating model: %s", e)
    try:
        # Plot the predictions

        y_pred_reshaped = xr.DataArray(
            torch.cat(y_pred).numpy().reshape((np.array(data_test_out.values).shape)),
            dims=["time", "member", "height", "ncells"],
        )
        logger.info(
            "The shape of the raw model prediction: %s", torch.cat(y_pred).numpy().shape
        )
        logger.info("Reshaped into form: %s", y_pred_reshaped.shape)
        data_gif = {
            "y_pred_reshaped": y_pred_reshaped,
            "data_test": data_test,
        }

        for i in range(10):
            member = random.randint(
                0, data_test.sizes["member"] - config["member_split"] - 1
            )
            output_filename = create_animation(data_gif, member=member, preds="GNN")
    except (ValueError, TypeError) as e:
        logger.exception("Error occurred while creating animation: %s", e)
