"""Contains the models used for weather prediction."""
# Standard library
import os
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Union

# Third-party
import mlflow
import torch
from torch import distributed as dist
from torch import nn
from torch import optim
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborSampler  # type: ignore
from torch_geometric.nn import GCNConv  # type: ignore

from weathergraphnet.loggers_configs import setup_logger
from weathergraphnet.utils import GraphDataset

# First-party

logger = setup_logger()


@dataclass
class TrainingConfigGNN(dict):  # pylint: disable=too-many-instance-attributes
    """Training configuration parameters.

    Args:
        loader_train_in (DataLoader): The data loader for the input training data.
        loader_train_out (DataLoader): The data loader for the output training data.
        optimizer (nn.Module): The optimizer to use for training.
        scheduler (nn.Module): The learning rate scheduler to use for training.
        loss_fn (nn.Module): The loss function to use for training.
        mask (Optional[torch.Tensor]): The mask to use for training.
        epochs (int): The number of epochs to train for. Default is 10.
        device (str): The device to use for training. Default is "cuda".
        seed (int): The random seed to use for training. Default is 42.

    """

    dataset: GraphDataset
    optimizer: Union[optim.Optimizer, torch.optim.Optimizer, Adam]
    scheduler: Union[CyclicLR, torch.optim.lr_scheduler.CyclicLR]
    loss_fn: Union[
        nn.Module,
        nn.MSELoss,
    ]
    batch_size: int = 21
    mask: Optional[torch.Tensor] = None
    epochs: int = 10
    device: str = "cuda"
    seed: int = 42


@dataclass
class EvaluationConfigGNN(dict):
    """Configuration class for evaluation of a GNN model.

    Attributes:
        loader_in (DataLoader): Input data loader.
        loader_out (DataLoader): Output data loader.
        loss_fn (nn.Module): Loss function for the evaluation.
        mask (Optional[torch.Tensor], optional): Mask tensor for the evaluation.
        device (str, optional): Device to use for evaluation. Defaults to "cuda".
        seed (int, optional): Random seed for evaluation. Defaults to 42.

    """

    dataloader: GraphDataset
    loss_fn: Union[
        nn.Module,
        nn.MSELoss,
    ]
    mask: Optional[torch.Tensor] = None
    device: str = "cuda"
    batch_size: int = 21
    seed: int = 42


@dataclass
class GNNConfig(dict):
    """Configuration parameters for the GNN model.

    Attributes:
        nodes_in (int): The number of input nodes.
        nodes_out (int): The number of output nodes.
        channels_in (int): The number of input channels.
        channels_out (int): The number of output channels.
        hidden_feats (int): The number of hidden features.

    """

    nodes_in: int
    nodes_out: int
    channels_in: int
    channels_out: int
    hidden_feats: int


class DownConvLayers(torch.nn.Module):
    """The down-convolutional layers of the GNN model."""

    def __init__(self, gnn_configs: GNNConfig):
        """Initialize the down-convolutional layers of the GNN model.

        Args:
            gnn_configs (GNNConfig): The configuration parameters for the GNN model.

        """
        super().__init__()
        try:
            self.conv1 = GCNConv(gnn_configs.channels_in,
                                 gnn_configs.hidden_feats)
            self.conv2 = GCNConv(
                gnn_configs.hidden_feats, gnn_configs.hidden_feats // 2
            )
            self.conv3 = GCNConv(
                gnn_configs.hidden_feats // 2, gnn_configs.hidden_feats // 4
            )
            self.conv4 = GCNConv(
                gnn_configs.hidden_feats // 4, gnn_configs.hidden_feats // 8
            )
            self.conv5 = GCNConv(
                gnn_configs.hidden_feats // 8, gnn_configs.hidden_feats // 16
            )
        except KeyError as e:
            logger.error(
                "Error occurred while initializing DownConvLayers: %s", e)
            raise

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the down-convolutional layers of the GNN.

        Args:
            x (torch.Tensor): The input tensor.
            edge_index (torch.Tensor): The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        try:
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            x = torch.relu(self.conv3(x, edge_index))
            # x = torch.relu(self.conv4(x, edge_index))
            # x = torch.relu(self.conv5(x, edge_index))
        except Exception as e:
            logger.error(
                "Error occurred while performing forward pass in DownConvLayers: %s", e
            )
            raise
        return x


class UpConvLayers(torch.nn.Module):
    """The up-convolutional layers of the GNN model."""

    def __init__(self, gnn_configs: GNNConfig):
        """Initialize the up-convolutional layers of the GNN model.

        Args:
            gnn_configs (GNNConfig): The configuration parameters for the GNN model.

        """
        super().__init__()
        try:
            self.upconv1 = GCNConv(
                gnn_configs.hidden_feats // 16, gnn_configs.hidden_feats // 8
            )
            self.upconv2 = GCNConv(
                gnn_configs.hidden_feats // 8, gnn_configs.hidden_feats // 4
            )
            self.upconv3 = GCNConv(
                gnn_configs.hidden_feats // 4, gnn_configs.hidden_feats // 2
            )
            self.upconv4 = GCNConv(
                gnn_configs.hidden_feats // 2, gnn_configs.hidden_feats
            )
            self.upconv5 = GCNConv(
                gnn_configs.hidden_feats, gnn_configs.channels_out)
        except KeyError as e:
            logger.error(
                "Error occurred while initializing UpConvLayers: %s", e)
            raise

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the up-convolutional layers of the GNN model.

        Args:
            x (torch.Tensor): The input tensor.
            edge_index (torch.Tensor): The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        # TODO: do i need skip connections or batch normalization?
        try:
            # x = torch.relu(self.upconv1(x, edge_index))
            # x = torch.relu(self.upconv2(x, edge_index))
            x = torch.relu(self.upconv3(x, edge_index))
            x = torch.relu(self.upconv4(x, edge_index))
            x = self.upconv5(x, edge_index)
        except Exception as e:
            logger.error(
                "Error occurred while performing forward pass in UpConvLayers: %s", e
            )
            raise
        return x


class GCNConvLayers(torch.nn.Module):
    """A Graph Neural Network (GNN) model for weather prediction.

    Args:
        config (GNNConfig): Configuration parameters for the GNN model.

    Methods:
        forward(data): Performs a forward pass through the GNN model.

    """

    def __init__(self, gnn_configs: GNNConfig):
        """Initialize the GNN model.

        Args:
            gnn_configs (GNNConfig): The configuration parameters for the GNN model.

        """
        super().__init__()
        try:
            self.down_conv_layers = DownConvLayers(gnn_configs)
            self.up_conv_layers = UpConvLayers(gnn_configs)
        except (TypeError, ValueError) as e:
            logger.error("Error initializing GCNConvLayers: %s", e)
            raise

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the GNN model.

        Args:
            x (torch.Tensor): The input tensor.
            edge_index (torch.Tensor): The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        try:
            x = self.down_conv_layers(x, edge_index)
            x = self.up_conv_layers(x, edge_index)
        except (TypeError, ValueError) as e:
            logger.error("Error in GCNConvLayers forward method: %s", e)
            raise
        return x


def loss_func(output, target, target_mask):
    output_target = output[target_mask]
    target_target = target[target_mask]
    loss = nn.L1Loss()
    return loss(output_target, target_target)


class GNNModel(torch.nn.Module):
    """A Graph Neural Network (GNN) model for weather prediction.

    Args:
        config (GNNConfig): Configuration parameters for the GNN model.

    Methods:
        forward(data): Performs a forward pass through the GNN model.
        train_with_configs(gnn_configs): Trains the GNN model.
        eval_with_configs(gnn_configs): Evaluates the performance of the GNN model.

    """

    def __init__(self, gnn_configs: GNNConfig) -> None:
        """Initialize the GNN model.

        Args:
            gnn_configs (GNNConfig): The configuration parameters for the GNN model.

        """
        super().__init__()
        self.conv_layers = GCNConvLayers(gnn_configs)
        self.activation = torch.nn.ReLU()

    def forward(self, data) -> torch.Tensor:
        """Perform a forward pass through the GNN model.

        Args:
            data (Data): The input data.

        Returns:
            torch.Tensor: The output tensor.

        """
        x, edge_index = data.x, data.edge_index
        x = self.conv_layers(x, edge_index)
        return x

    def train_with_configs(self, rank, configs_train_gnn: TrainingConfigGNN, world_size) -> None:
        """Train a GNN model and output data using the specified loss function.

        Args:
            configs_train_gnn (TrainingConfigGNN): The configuration parameters for the
            training

        Returns:
            None

        """
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # TODO: remove this after debugging
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"  # choose an available port
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl", rank=rank, world_size=world_size)
        if dist.get_rank() == 0:
            print("Training UNet network with configurations:", flush=True)
            print(configs_train_gnn, flush=True)
        # Set the seed for reproducibility
        torch.manual_seed(configs_train_gnn.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(configs_train_gnn.seed)

        sampler = NeighborSampler(
            configs_train_gnn.dataset.edge_index, sizes=[-1]*2,
            batch_size=configs_train_gnn.batch_size, shuffle=True)
        data_train_loader = DataLoader(
            configs_train_gnn.dataset, batch_sampler=sampler)

        device = configs_train_gnn.device
        model = self.to(device)
        # model = nn.parallel.DistributedDataParallel(model)
        if configs_train_gnn.mask is not None:
            configs_train_gnn.mask = configs_train_gnn.mask.to(device)

        try:

            # Train the GNN model
            # Train the GNN model
            for epoch in range(configs_train_gnn.epochs):
                running_loss = 0.0
                for batch_size, n_id, adjs in data_train_loader:
                    # Get the data for the nodes in the subgraph
                    subgraph_data = configs_train_gnn.dataset[n_id].to(
                        device)

                    # Crop the target_mask to match the size of the subgraph
                    subgraph_target_mask = configs_train_gnn.dataset.target_mask[n_id].to(
                        device)

                    configs_train_gnn.optimizer.zero_grad()

                    # Use the input data from the subgraph
                    output = model(subgraph_data)

                    # Use the target data from the subgraph and the cropped mask
                    loss = loss_func(output, subgraph_data.x,
                                     subgraph_target_mask)

                    loss.backward()
                    configs_train_gnn.optimizer.step()
                    if configs_train_gnn.scheduler is not None:
                        configs_train_gnn.scheduler.step()  # update the learning rate
                    running_loss += loss.item()

            avg_loss = running_loss / len(data_train_loader)
            # gathered_losses = [torch.zeros_like(avg_loss) for _ in range(
            #     dist.get_world_size())] if dist.get_rank() == 0 else []
            # dist.gather(
            #     avg_loss, gather_list=gathered_losses, dst=0)

            # if dist.get_rank() == 0:
            # avg_loss = torch.stack(gathered_losses).mean().item()
            logger.info("Epoch: %d, Loss: %f4", epoch, avg_loss)
            mlflow.log_metric("loss", avg_loss)
            mlflow.pytorch.log_model(model, f"model_epoch_{epoch}")
            best_loss = torch.tensor(float("inf")).to(device)
            if avg_loss < best_loss:
                best_loss = avg_loss
                mlflow.pytorch.log_model(model, "models")

        except RuntimeError as e:
            logger.error(
                "Error occurred while training GNN: %s", str(e))

    def eval_with_configs(
        self,
        configs_eval_gnn: EvaluationConfigGNN,
    ) -> tuple[float, List[torch.Tensor]]:
        """Evaluate the performance of the GNN model on a given dataset.

        Args:
            configs_eval_gnn(TrainingConfigGNN): The configuration parameters for the
            evaluation

        Returns:
            float: The loss achieved during evaluation.

        """
        data_eval_loader = DataLoader(
            configs_eval_gnn.dataloader,
        )

        device = configs_eval_gnn.device
        model = self.to(device)
        self.eval()
        with torch.no_grad():
            loss: float = 0.0
            running_loss = 0.0
            y_preds: List[torch.Tensor] = []
            for data in data_eval_loader:
                data = data.to(configs_eval_gnn.device)
                # pylint: disable=not-callable
                output = model(data)
                if configs_eval_gnn.mask is not None:
                    try:
                        configs_eval_gnn.mask = configs_eval_gnn.mask.to(
                            configs_eval_gnn.device)
                        loss += configs_eval_gnn.loss_fn(
                            output, data.x, configs_eval_gnn.mask
                        )
                    except Exception as e:
                        logger.error(
                            "Error occurred while calculating loss: %s", e)
                        raise e
                else:
                    try:
                        loss = loss_func(
                            output, data.x, data.target_mask)
                    except Exception as e:
                        logger.error(
                            "Error occurred while calculating loss: %s", e)
                        raise e
                y_preds.append(output.cpu())
                running_loss += loss.item()
            avg_loss = running_loss / len(data_eval_loader)

            return avg_loss, y_preds
