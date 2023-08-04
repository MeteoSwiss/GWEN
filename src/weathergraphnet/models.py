"""Contains the models used for weather prediction."""
# Standard library
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# Third-party
import mlflow  # type: ignore  # type: ignore  # type: ignore
import torch
from torch import nn
from torch import optim
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data  # type: ignore  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torch_geometric.nn import GCNConv  # type: ignore
from torch_geometric.nn import TopKPooling  # type: ignore

# First-party
from weathergraphnet.logger import setup_logger

logger = setup_logger()


class InitPrintMeta(type):
    def __new__(mcs, name, bases, attrs):
        new_mcs = super().__new__(mcs, name, bases, attrs)
        if "__init__" in attrs:
            old_init = new_mcs.__init__

            def new_init(self, *args, **kwargs):
                old_init(self, *args, **kwargs)
                print(f"Initialized {name} instance")

            new_mcs.__init__ = new_init
        return new_mcs


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

    loader_train_in: DataLoader
    loader_train_out: DataLoader
    optimizer: Union[optim.Optimizer, torch.optim.Optimizer, Adam]
    scheduler: Union[CyclicLR, torch.optim.lr_scheduler.CyclicLR]
    loss_fn: Union[
        nn.Module,
        nn.MSELoss,
        nn.DataParallel,
    ]
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

    loader_in: DataLoader
    loader_out: DataLoader
    loss_fn: Union[
        nn.Module,
        nn.MSELoss,
        nn.DataParallel,
    ]
    mask: Optional[torch.Tensor] = None
    device: str = "cuda"
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


@dataclass
class TrainingConfigCNN(dict):  # pylint: disable=too-many-instance-attributes
    """Configuration class for training a CNN model.

    Attributes:
        dataloader (DataLoader): The data loader for the training dataset.
        optimizer (nn.Module): The optimizer used for training.
        scheduler (nn.Module): The learning rate scheduler used for training.
        loss_fn (nn.Module): The loss function used for training.
        mask (Optional[torch.Tensor]): The mask to apply to the input data.
        epochs (int): The number of epochs to train for.
        device (str): The device to use for training (default is "cuda").
        seed (int): The random seed to use for reproducibility (default is 42).

    """

    dataloader: DataLoader
    optimizer: Union[torch.optim.Optimizer, Adam]
    scheduler: Union[CyclicLR, StepLR]
    loss_fn: Union[
        nn.Module,
        nn.MSELoss,
        nn.DataParallel,
    ]
    mask: Optional[torch.Tensor] = None
    epochs: int = 10
    device: str = "cuda"
    seed: int = 42


# pylint: disable=R0902,R0801
@dataclass
class EvaluationConfigCNN(dict):
    """Configuration class for evaluating a CNN model.

    Attributes:
        dataloader (DataLoader): The data loader for the evaluation dataset.
        loss_fn (nn.Module): The loss function to use for evaluation.
        mask (Optional[torch.Tensor], optional): A mask for the evaluation data.
            Defaults to None.
        device (str, optional): The device to use for evaluation. Defaults to "cuda".
        seed (int, optional): The random seed to use for evaluation. Defaults to 42.

    """

    dataloader: DataLoader
    loss_fn: Union[
        nn.Module,
        nn.MSELoss,
        nn.DataParallel,
    ]
    mask: Optional[torch.Tensor] = None
    device: str = "cuda"
    seed: int = 42


class BaseNet(nn.Module):
    """Base class for the encoder and decoder networks."""

    def __init__(self, channels_in: int, channels_out: int, hidden_size: int) -> None:
        """Initialize the BaseNet class.

        Args:
            channels_in (int): Number of input channels.
            channels_out (int): Number of output channels.
            hidden_size (int): Size of the hidden layer.

        """
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.hidden_size = hidden_size
        try:
            self.conv_layers = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.channels_in,
                        self.hidden_size // 8,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.Conv2d(
                        self.hidden_size // 8,
                        self.hidden_size // 4,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.Conv2d(
                        self.hidden_size // 4,
                        self.hidden_size // 2,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.Conv2d(
                        self.hidden_size // 2,
                        self.hidden_size,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.Conv2d(
                        self.channels_out,
                        self.channels_out,
                        kernel_size=1,
                        stride=1,
                    ),
                ]
            )
            self.conv_transposed_layers = nn.ModuleList(
                [
                    nn.ConvTranspose2d(
                        self.hidden_size,
                        self.hidden_size // 2,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ConvTranspose2d(
                        self.hidden_size,
                        self.hidden_size // 4,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ConvTranspose2d(
                        self.hidden_size // 2,
                        self.hidden_size // 8,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ConvTranspose2d(
                        self.hidden_size // 4,
                        self.channels_out,
                        kernel_size=3,
                        padding=1,
                    ),
                ]
            )
            self.batch_norm_layers = nn.ModuleList(
                [
                    nn.BatchNorm2d(self.hidden_size // 8),
                    nn.BatchNorm2d(self.hidden_size // 4),
                    nn.BatchNorm2d(self.hidden_size // 2),
                    nn.BatchNorm2d(self.hidden_size),
                ],
            )
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        except ValueError as e:
            logger.error("Error occurred while initializing the Decoder class: %s", e)

    def forward(self, x):
        """Forward pass through the network."""
        raise NotImplementedError


class Encoder(BaseNet):
    """Encoder network."""

    def __init__(self, channels_in: int, channels_out: int, hidden_size: int) -> None:
        """Initialize the Encoder class.

        Args:
            channels_in (int): Number of input channels.
            channels_out (int): Number of output channels.
            hidden_size (int): Size of the hidden layer.

        """
        # pylint: disable=useless-parent-delegation
        super().__init__(channels_in, channels_out, hidden_size)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of
            output tensors.

        """
        try:
            x1 = self.conv_layers[0](x)

            x1 = self.maxpool(x1)

            x1 = self.batch_norm_layers[0](x1)

            x1 = self.activation(x1)

            x2 = self.conv_layers[1](x1)

            x2 = self.maxpool(x2)

            x2 = self.batch_norm_layers[1](x2)

            x2 = self.activation(x2)

            x3 = self.conv_layers[2](x2)

            x3 = self.maxpool(x3)

            x3 = self.batch_norm_layers[2](x3)

            x3 = self.activation(x3)

            x4 = self.conv_layers[3](x3)

            x4 = self.maxpool(x4)

            x4 = self.batch_norm_layers[3](x4)

            x4 = self.activation(x4)

        except IndexError as e:
            logger.error(
                "Error occurred while performing forward"
                " pass through the encoder network: %s",
                e,
            )
        return (x1, x2, x3, x4)


class Decoder(BaseNet):
    """Decoder network."""

    def __init__(self, channels_in: int, channels_out: int, hidden_size: int) -> None:
        """Initialize the Decoder class.

        Args:
            channels_in (int): Number of input channels.
            channels_out (int): Number of output channels.
            hidden_size (int): Size of the lowest hidden layer
                (highest number of convolutions lowest spatial resolution)

        """
        # pylint: disable=useless-parent-delegation
        super().__init__(channels_in, channels_out, hidden_size)

    def crop(
        self, encoder_layer: torch.Tensor, decoder_layer: torch.Tensor
    ) -> torch.Tensor:
        """Crop the encoder layer to the size of the decoder layer.

        Args:
            encoder_layer (torch.Tensor): Encoder tensor.
            decoder_layer (torch.Tensor): Decoder tensor.

        Returns:
            torch.Tensor: Cropped tensor.

        """
        try:
            diff_y = encoder_layer.size()[2] - decoder_layer.size()[2]
            diff_x = encoder_layer.size()[3] - decoder_layer.size()[3]
            encoder_layer = encoder_layer[
                :,
                :,
                diff_y // 2 : encoder_layer.size()[2] - diff_y // 2,
                diff_x // 2 : encoder_layer.size()[3] - diff_x // 2,
            ]
            if diff_y % 2 == 1:
                encoder_layer = encoder_layer[:, :, 1 : encoder_layer.size()[2], :]
            if diff_x % 2 == 1:
                encoder_layer = encoder_layer[:, :, :, 1 : encoder_layer.size()[3]]
        except IndexError as e:
            logger.error("Error occurred while cropping the encoder layer: %s", e)
        return encoder_layer

    def forward(  # pylint: disable=too-many-locals, too-many-statements
        self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass of the CNN model.

        Args:
            x (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): Tuple of
            four tensors representing the input data.

        Returns:
            torch.Tensor: Output tensor after passing through the CNN model.

        """
        x1, x2, x3, x4 = x
        cropped = 0

        try:
            y1 = self.conv_transposed_layers[0](x4)

            y1 = self.upsample(y1)

            y1 = self.batch_norm_layers[2](y1)

            y1 = self.activation(y1)

            if y1.shape != x3.shape:
                x3 = self.crop(x3, y1)
                cropped += 1
            y1 = torch.cat([x3, y1], dim=1)

            y1 = self.activation(y1)

            y2 = self.conv_transposed_layers[1](y1)

            y2 = self.upsample(y2)

            y2 = self.batch_norm_layers[1](y2)

            y2 = self.activation(y2)

            if y2.shape != x2.shape:
                x2 = self.crop(x2, y2)
                cropped += 1
            y2 = torch.cat([x2, y2], dim=1)

            y2 = self.activation(y2)

            y3 = self.conv_transposed_layers[2](y2)

            y3 = self.upsample(y3)

            y3 = self.batch_norm_layers[0](y3)

            y3 = self.activation(y3)

            if y3.shape != x1.shape:
                x1 = self.crop(x1, y3)
                cropped += 1
            y3 = torch.cat([x1, y3], dim=1)

            y3 = self.activation(y3)

            y4 = self.conv_transposed_layers[3](y3)

            y4 = self.upsample(y4)

            y4 = self.activation(y4)

            out = self.conv_layers[4](y4)

            out = nn.functional.pad(out, (cropped * 2, 0, 0, 0), mode="replicate")

            logged_messages = set()
            # Log the messages after each epoch, but only if they haven't been logged
            # before
            if "Shapes of the UNET" not in logged_messages:
                logger.info("Shapes of the UNET")
                logged_messages.add("Shapes of the UNET")
            variables = [
                ("X1", x1),
                ("X2", x2),
                ("X3", x3),
                ("X4", x4),
                ("Y1", y1),
                ("Y2", y2),
                ("Y3", y3),
                ("Y4", y4),
                ("OUT", out),
            ]
            for name, var in variables:
                message = f"{name}: {var.shape}"
                if message not in logged_messages:
                    logger.info(message)
                    logged_messages.add(message)
            cropped_message = f"The spatial dimensions were cropped {cropped} times."
            if cropped_message not in logged_messages:
                logger.info(cropped_message)
                logged_messages.add(cropped_message)

        except ValueError as e:
            logger.error(
                "Error occurred during the forward pass of the CNN model: %s", e
            )
        return out


class UNet(BaseNet):
    """A class representing the UNet network.

    Args:
    - channels_in (int): The number of input channels.
    - channels_out (int): The number of output channels.
    - hidden_size (int): The number of hidden units.

    Attributes:
    - encoder (Encoder): The encoder module.
    - decoder (Decoder): The decoder module.

    """

    def __init__(self, channels_in: int, channels_out: int, hidden_size: int) -> None:
        """Initialize the UNet network."""
        super().__init__(channels_in, channels_out, hidden_size)

        try:
            self.encoder = Encoder(channels_in, channels_out, hidden_size)
            self.decoder = Decoder(channels_in, channels_out, hidden_size)
        except RuntimeError as e:
            logger.error("Error occurred while initializing UNet network: %s", str(e))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet network.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - out (torch.Tensor): The output tensor.

        """
        try:
            x1, x2, x3, x4 = self.encoder(x)

            out: torch.Tensor = self.decoder((x1, x2, x3, x4))

        except RuntimeError as e:
            logger.error(
                "Error occurred during forward pass through UNet network: %s", str(e)
            )
        return out

    def train_with_configs(self, configs_train_cnn: TrainingConfigCNN) -> None:
        """Train the model.

        Args:
        configs_train_cnn (Any): The configuration object.

        Returns:
        None

        """
        print("Training UNet network with configurations:", flush=True)
        print(configs_train_cnn, flush=True)

        try:
            for epoch in range(configs_train_cnn.epochs):
                print(f"Epoch: {epoch}", flush=True)
                running_loss = 0.0
                for input_data, target_data in configs_train_cnn.dataloader:
                    print(input_data.shape, flush=True)
                    input_data = input_data.to(configs_train_cnn.device)
                    target_data = target_data.to(configs_train_cnn.device)
                    configs_train_cnn.optimizer.zero_grad()
                    print(configs_train_cnn.device)
                    print(type(self))
                    model = self.to(configs_train_cnn.device)
                    print(type(model))
                    output = model(input_data)
                    if configs_train_cnn.mask is not None:
                        loss = configs_train_cnn.loss_fn(
                            output,
                            target_data,
                            configs_train_cnn.mask.to(configs_train_cnn.device),
                        )
                    else:
                        loss = configs_train_cnn.loss_fn(output, target_data)
                    loss.backward()
                    configs_train_cnn.optimizer.step()
                    if configs_train_cnn.scheduler is not None:
                        configs_train_cnn.scheduler.step()  # update the learning rate
                    running_loss += loss.item()
                    print(f"Running Loss: {running_loss}", flush=True)
                avg_loss = running_loss / float(len(configs_train_cnn.dataloader))
                logger.info("Epoch: %d, Loss: %f", epoch, avg_loss)
                mlflow.log_metric("loss", avg_loss)

        except RuntimeError as e:
            logger.error("Error occurred while training UNet network: %s", str(e))

    def eval_with_configs(
        self, configs_eval_cnn: EvaluationConfigCNN
    ) -> Tuple[float, List[torch.Tensor]]:
        """Evaluate the model.

        Args:
        - configs_eval_cnn (Any): The configuration object.

        Returns:
        - loss (float): The loss achieved during evaluation.

        """
        try:
            self.eval()
            with torch.no_grad():
                loss: float = 0.0
                y_preds: List[torch.Tensor] = []
                for input_data, target_data in configs_eval_cnn.dataloader:
                    input_data = input_data.to(configs_eval_cnn.device)
                    target_data = target_data.to(configs_eval_cnn.device)
                    model = self.to(configs_eval_cnn.device)
                    output = model(input_data)
                    if configs_eval_cnn.mask is not None:
                        loss += configs_eval_cnn.loss_fn(
                            output,
                            target_data,
                            configs_eval_cnn.mask.to(configs_eval_cnn.device),
                        )
                    else:
                        loss += configs_eval_cnn.loss_fn(output, target_data)
                    y_preds.append(output.cpu())
                loss /= len(configs_eval_cnn.dataloader)
        except RuntimeError as e:
            logger.error("Error occurred while evaluating UNet network: %s", str(e))
        return loss, y_preds


class DownConvLayers(torch.nn.Module):
    """The down-convolutional layers of the GNN model."""

    def __init__(self, gnn_configs: GNNConfig):
        """Initialize the down-convolutional layers of the GNN model.

        Args:
            gnn_configs (GNNConfig): The configuration parameters for the GNN model.

        """
        super().__init__()
        try:
            self.conv1 = GCNConv(gnn_configs.channels_in, gnn_configs.hidden_feats)
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
            logger.error("Error occurred while initializing DownConvLayers: %s", e)
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
            x = torch.relu(self.conv4(x, edge_index))
            x = torch.relu(self.conv5(x, edge_index))
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
            self.upconv5 = GCNConv(gnn_configs.hidden_feats, gnn_configs.channels_out)
        except KeyError as e:
            logger.error("Error occurred while initializing UpConvLayers: %s", e)
            raise

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the up-convolutional layers of the GNN model.

        Args:
            x (torch.Tensor): The input tensor.
            edge_index (torch.Tensor): The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        # TODO: do i need skip connections?
        # TODO: check if batch normalization is used?
        try:
            x = torch.relu(self.upconv1(x, edge_index))
            x = torch.relu(self.upconv2(x, edge_index))
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


class TopKPoolingLayer(torch.nn.Module):
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
            self.pool = TopKPooling(
                gnn_configs.channels_out,
                ratio=gnn_configs.nodes_out / gnn_configs.nodes_in,
            )
        except (TypeError, ValueError) as e:
            logger.error("Error initializing TopKPoolingLayer: %s", e)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the GNN model.

        Args:
            x (torch.Tensor): The input tensor.
            edge_index (torch.Tensor): The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        try:
            x, edge_index, _, _, _, _ = self.pool(x, edge_index)
        except RuntimeError as e:
            logger.error("Error in TopKPoolingLayer forward method: %s", e)
            raise
        return x, edge_index


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
        self.pool_layer = TopKPoolingLayer(gnn_configs)
        self.activation = torch.nn.ReLU()

    def forward(self, data: Data) -> torch.Tensor:
        """Perform a forward pass through the GNN model.

        Args:
            data (Data): The input data.

        Returns:
            torch.Tensor: The output tensor.

        """
        x, edge_index = data.x, data.edge_index
        x = self.conv_layers(x, edge_index)
        x, edge_index = self.pool_layer(x, edge_index)
        return x

    def train_with_configs(self, configs_train_gnn: TrainingConfigGNN) -> None:
        """Train a GNN model and output data using the specified loss function.

        Args:
            configs_train_gnn (TrainingConfigGNN): The configuration parameters for the
            training

        Returns:
            None

        """
        # Set the seed for reproducibility
        torch.manual_seed(configs_train_gnn.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(configs_train_gnn.seed)

        # Train the GNN model
        for epoch in range(configs_train_gnn.epochs):
            running_loss = 0.0
            for data_in, data_out in zip(
                configs_train_gnn.loader_train_in, configs_train_gnn.loader_train_out
            ):
                data_in = data_in.to(configs_train_gnn.device)
                data_out = data_out.to(configs_train_gnn.device)
                configs_train_gnn.optimizer.zero_grad()
                model = self.to(configs_train_gnn.device)
                output = model(data_in)
                if configs_train_gnn.mask is not None:
                    try:
                        loss = configs_train_gnn.loss_fn(
                            output,
                            data_out.x,
                            configs_train_gnn.mask.to(configs_train_gnn.device),
                        )
                    except Exception as e:
                        logger.error("Error occurred while calculating loss: %s", e)
                        raise e
                else:
                    try:
                        loss = configs_train_gnn.loss_fn(output, data_out.x)
                    except Exception as e:
                        logger.error("Error occurred while calculating loss: %s", e)
                        raise e
                loss.backward()
                configs_train_gnn.optimizer.step()
                if configs_train_gnn.scheduler is not None:
                    configs_train_gnn.scheduler.step()  # update the learning rate
                running_loss += loss.item()

            avg_loss = running_loss / float(len(configs_train_gnn.loader_train_in))
            logger.info("Epoch: %d, Loss: %f4", epoch, avg_loss)
            mlflow.log_metric("loss", avg_loss)

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
        self.eval()
        with torch.no_grad():
            loss: float = 0.0
            y_preds: List[torch.Tensor] = []
            for data_in, data_out in zip(
                configs_eval_gnn.loader_in, configs_eval_gnn.loader_out
            ):
                data_in = data_in.to(configs_eval_gnn.device)
                data_out = data_out.to(configs_eval_gnn.device)
                model = self.to(configs_eval_gnn.device)
                output = model(data_in)
                if configs_eval_gnn.mask is not None:
                    try:
                        loss += configs_eval_gnn.loss_fn(
                            output,
                            data_out.x,
                            configs_eval_gnn.mask.to(configs_eval_gnn.device),
                        )
                    except Exception as e:
                        logger.error("Error occurred while calculating loss: %s", e)
                        raise e
                else:
                    try:
                        loss += configs_eval_gnn.loss_fn(output, data_out.x)
                    except Exception as e:
                        logger.error("Error occurred while calculating loss: %s", e)
                        raise e
                y_preds.append(output.cpu())
            loss /= float(len(configs_eval_gnn.loader_in))

            return loss, y_preds
