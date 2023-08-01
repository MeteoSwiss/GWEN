from typing import List
from typing import Tuple

import mlflow  # type: ignore
import torch
from torch import nn
from torch_geometric.data import Data  # type: ignore
from torch_geometric.nn import GCNConv  # type: ignore
from torch_geometric.nn import TopKPooling  # type: ignore


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
        self.activation = nn.ReLU()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.hidden_size = hidden_size


class Encoder(BaseNet):
    """Encoder network."""

    def __init__(self, channels_in: int, channels_out: int, hidden_size: int) -> None:
        """Initialize the Encoder class.

        Args:
            channels_in (int): Number of input channels.
            channels_out (int): Number of output channels.
            hidden_size (int): Size of the hidden layer.

        """
        super().__init__(channels_in, channels_out, hidden_size)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels_in, self.hidden_size //
                      8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.hidden_size // 8, self.hidden_size // 4, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                self.hidden_size // 4, self.hidden_size // 2, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.hidden_size // 2, self.hidden_size, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.hidden_size, self.hidden_size,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_size, self.hidden_size,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

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
        x1 = self.conv_layers[0](x)
        x1 = self.activation(x1)
        x2 = self.conv_layers[2](self.conv_layers[1](x1))
        x2 = self.activation(x2)
        x3 = self.conv_layers[4](self.conv_layers[3](x2))
        x3 = self.activation(x3)
        x4 = self.conv_layers[7](self.conv_layers[6](self.conv_layers[5](x3)))
        x4 = self.activation(x4)
        return x1, x2, x3, x4


class Decoder(BaseNet):
    """Decoder network."""

    def __init__(self, channels_in: int, channels_out: int, hidden_size: int) -> None:
        """Initialize the Decoder class.

        Args:
            channels_in (int): Number of input channels.
            channels_out (int): Number of output channels.
            hidden_size (int): Size of the hidden layer.

        """
        super().__init__(channels_in, channels_out, hidden_size)
        self.conv_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    self.hidden_size, self.hidden_size // 2, kernel_size=2, stride=2
                ),
                nn.Conv2d(
                    self.hidden_size, self.hidden_size // 2, kernel_size=3, padding=1
                ),
                nn.ConvTranspose2d(
                    self.hidden_size // 2,
                    self.hidden_size // 4,
                    kernel_size=2,
                    stride=2,
                ),
                nn.Conv2d(
                    self.hidden_size // 2,
                    self.hidden_size // 4,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ConvTranspose2d(
                    self.hidden_size // 4,
                    self.hidden_size // 8,
                    kernel_size=2,
                    stride=2,
                ),
                nn.Conv2d(
                    self.hidden_size // 4,
                    self.hidden_size // 8,
                    kernel_size=3,
                    padding=1,
                ),
                nn.Conv2d(self.hidden_size // 8,
                          self.channels_out, kernel_size=1),
            ]
        )

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
        diff_y = encoder_layer.size()[2] - decoder_layer.size()[2]
        diff_x = encoder_layer.size()[3] - decoder_layer.size()[3]
        encoder_layer = encoder_layer[
            :,
            :,
            diff_y // 2: encoder_layer.size()[2] - diff_y // 2,
            diff_x // 2: encoder_layer.size()[3] - diff_x // 2,
        ]
        if diff_x % 2 == 1:
            encoder_layer = encoder_layer[:, :, :, 1: encoder_layer.size()[3]]
        if diff_y % 2 == 1:
            encoder_layer = encoder_layer[:, :, 1: encoder_layer.size()[2], :]
        return encoder_layer

    def forward(
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
        y1 = self.conv_layers[0](x4)
        y1 = self.activation(y1)
        if y1.shape != x3.shape:
            x3 = self.crop(x3, y1)
            cropped += 1
        y1 = self.conv_layers[1](torch.cat([x3, y1], dim=1))
        y1 = self.activation(y1)
        y2 = self.conv_layers[2](y1)
        y2 = self.activation(y2)
        if y2.shape != x2.shape:
            x2 = self.crop(x2, y2)
            cropped += 1
        y2 = self.conv_layers[3](torch.cat([x2, y2], dim=1))
        y2 = self.activation(y2)
        y3 = self.conv_layers[4](y2)
        y3 = self.activation(y3)
        if y3.shape != x1.shape:
            x1 = self.crop(x1, y3)
            cropped += 1
        y3 = self.conv_layers[5](torch.cat([x1, y3], dim=1))
        y3 = self.activation(y3)
        out = self.conv_layers[6](y3)
        out = nn.functional.pad(out, (cropped, 0, 0, 0), mode="replicate")
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
        self.encoder = Encoder(channels_in, channels_out, hidden_size)
        self.decoder = Decoder(channels_in, channels_out, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet network.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - out (torch.Tensor): The output tensor.

        """
        x1, x2, x3, x4 = self.encoder(x)
        out = self.decoder((x1, x2, x3, x4))
        return out

    def train_with_configs(self, configs: dict) -> None:
        """Train the model.

        Args:
        - configs (Any): The configuration object.

        """
        for epoch in range(configs["num_epochs"]):
            running_loss = 0.0
            for input_data, target_data in configs["loader_train"]:
                input_data = input_data.to(configs["device"])
                target_data = target_data.to(configs["device"])
                configs["optimizer"].zero_grad()
                output = self(input_data)
                if configs["mask"] is not None:
                    loss = configs["loss_fn"](
                        output, target_data, configs["mask"].to(
                            configs["device"])
                    )
                else:
                    loss = configs["loss_fn"](output, target_data)
                loss.backward()
                configs["optimizer"].step()
                if configs["scheduler"] is not None:
                    configs["scheduler"].step()  # update the learning rate
                running_loss += loss.item()
            avg_loss = running_loss / len(configs["dataloader"])
            print(f"Epoch {epoch + 1}: {avg_loss}")
            mlflow.log_metric("loss", avg_loss)

    def eval_with_configs(
        self, configs: dict
    ) -> Tuple[float, List[torch.Tensor]]:
        """Evaluate the model.

        Args:
        - configs (Any): The configuration object.

        Returns:
        - loss (float): The loss achieved during evaluation.

        """
        self.eval()
        with torch.no_grad():
            loss: float = 0.0
            y_preds: List[torch.Tensor] = []
            for input_data, target_data in configs["dataloader"]:
                input_data = input_data.to(configs["device"])
                target_data = target_data.to(configs["device"])
                output = self(input_data)
                if configs["mask"] is not None:
                    loss += configs["loss_fn"](
                        output, target_data, configs["mask"].to(
                            configs["device"])
                    )
                else:
                    loss += configs["loss_fn"](output, target_data)
                y_preds.append(output.cpu())
            loss /= len(configs["dataloader"])
            return loss, y_preds


class DownConvLayers(torch.nn.Module):
    """The down-convolutional layers of the GNN model."""

    def __init__(self, gnn_configs: dict):
        """Initialize the down-convolutional layers of the GNN model.

        Args:
            gnn_configs (dict): The configuration parameters for the GNN model.

        """
        super().__init__()
        self.conv1 = GCNConv(
            gnn_configs["in_channels"], gnn_configs["hidden_feats"])
        self.conv2 = GCNConv(
            gnn_configs["hidden_feats"], gnn_configs["hidden_feats"] // 2
        )
        self.conv3 = GCNConv(
            gnn_configs["hidden_feats"] // 2, gnn_configs["hidden_feats"] // 4
        )
        self.conv4 = GCNConv(
            gnn_configs["hidden_feats"] // 4, gnn_configs["hidden_feats"] // 8
        )
        self.conv5 = GCNConv(
            gnn_configs["hidden_feats"] // 8, gnn_configs["hidden_feats"] // 16
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the down-convolutional layers of the GNN.

        Args:
            x (torch.Tensor): The input tensor.
            edge_index (torch.Tensor): The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        x = torch.relu(self.conv5(x, edge_index))
        return x


class UpConvLayers(torch.nn.Module):
    """The up-convolutional layers of the GNN model."""

    def __init__(self, gnn_configs: dict):
        """Initialize the up-convolutional layers of the GNN model.

        Args:
            gnn_configs (dict): The configuration parameters for the GNN model.

        """
        super().__init__()
        self.upconv1 = GCNConv(
            gnn_configs["hidden_feats"] // 16, gnn_configs["hidden_feats"] // 8
        )
        self.upconv2 = GCNConv(
            gnn_configs["hidden_feats"] // 8, gnn_configs["hidden_feats"] // 4
        )
        self.upconv3 = GCNConv(
            gnn_configs["hidden_feats"] // 4, gnn_configs["hidden_feats"] // 2
        )
        self.upconv4 = GCNConv(
            gnn_configs["hidden_feats"] // 2, gnn_configs["hidden_feats"]
        )
        self.upconv5 = GCNConv(
            gnn_configs["hidden_feats"], gnn_configs["out_channels"])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the up-convolutional layers of the GNN model.

        Args:
            x (torch.Tensor): The input tensor.
            edge_index (torch.Tensor): The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = torch.relu(self.upconv1(x, edge_index))
        x = torch.relu(self.upconv2(x, edge_index))
        x = torch.relu(self.upconv3(x, edge_index))
        x = torch.relu(self.upconv4(x, edge_index))
        x = self.upconv5(x, edge_index)
        return x


class GCNConvLayers(torch.nn.Module):
    """A Graph Neural Network (GNN) model for weather prediction.

    Args:
        config (dict): Configuration parameters for the GNN model.

    Methods:
        forward(data): Performs a forward pass through the GNN model.

    """

    def __init__(self, gnn_configs: dict):
        """Initialize the GNN model.

        Args:
            gnn_configs (dict): The configuration parameters for the GNN model.

        """
        super().__init__()
        self.down_conv_layers = DownConvLayers(gnn_configs)
        self.up_conv_layers = UpConvLayers(gnn_configs)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the GNN model.

        Args:
            x (torch.Tensor): The input tensor.
            edge_index (torch.Tensor): The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.down_conv_layers(x, edge_index)
        x = self.up_conv_layers(x, edge_index)
        return x


class TopKPoolingLayer(torch.nn.Module):
    """A Graph Neural Network (GNN) model for weather prediction.

    Args:
        config (dict): Configuration parameters for the GNN model.

    Methods:
        forward(data): Performs a forward pass through the GNN model.

    """

    def __init__(self, gnn_configs: dict):
        """Initialize the GNN model.

        Args:
            gnn_configs (dict): The configuration parameters for the GNN model.

        """
        super().__init__()
        self.pool = TopKPooling(
            gnn_configs["out_channels"],
            ratio=gnn_configs["nodes_out"] / gnn_configs["nodes_in"],
        )

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
        x, edge_index, _, _, _, _ = self.pool(x, edge_index)
        return x, edge_index


class GNNModel(torch.nn.Module):
    """A Graph Neural Network (GNN) model for weather prediction.

    Args:
        config (dict): Configuration parameters for the GNN model.

    Methods:
        forward(data): Performs a forward pass through the GNN model.
        train_with_configs(gnn_configs): Trains the GNN model.
        eval_with_configs(gnn_configs): Evaluates the performance of the GNN model.

    """

    def __init__(self, gnn_configs: dict) -> None:
        """Initialize the GNN model.

        Args:
            gnn_configs (dict): The configuration parameters for the GNN model.

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

    def train_with_configs(self, configs: dict) -> None:
        """Train a GNN model and output data using the specified loss function.

        Args:
            configs (dict): The configuration parameters for the training

        Returns:
            None

        """
        # Set the seed for reproducibility
        torch.manual_seed(configs["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(configs["seed"])

        # Train the GNN model
        for epoch in range(configs["num_epochs"]):
            running_loss = 0.0
            for data_in, data_out in zip(
                configs["train_loader_in"], configs["train_loader_out"]
            ):
                data_in = data_in.to(configs["device"])
                data_out = data_out.to(configs["device"])
                configs["optimizer"].zero_grad()
                output = self(data_in)
                if configs["mask"] is not None:
                    loss = configs["loss_fn"](
                        output, data_out.x, configs["mask"].to(
                            configs["device"])
                    )
                else:
                    loss = configs["loss_fn"](output, data_out.x)
                loss.backward()
                configs["optimizer"].step()
                if configs["scheduler"] is not None:
                    configs["scheduler"].step()  # update the learning rate
                running_loss += loss.item()

            avg_loss = running_loss / len(configs["dataloader"])
            print(f"Epoch {epoch + 1}: {avg_loss}")
            mlflow.log_metric("loss", avg_loss)

    def eval_with_configs(
        self,
        configs: dict,
    ) -> tuple[float, List[torch.Tensor]]:
        """Evaluate the performance of the GNN model on a given dataset.

        Args:
            configs(dict): The configuration parameters for the evaluation

        Returns:
            float: The loss achieved during evaluation.
        """
        self.eval()
        with torch.no_grad():
            loss: float = 0.0
            y_preds: List[torch.Tensor] = []
            for data_in, data_out in zip(configs["loader_in"], configs["loader_out"]):
                data_in = data_in.to(configs["device"])
                data_out = data_out.to(configs["device"])
                output = self(data_in)
                if configs["mask"] is not None:
                    loss += configs["loss_fn"](
                        output, data_out.x, configs["mask"].to(
                            configs["device"])
                    )
                else:
                    loss += configs["loss_fn"](output, data_out.x)
                y_preds.append(output.cpu())
            loss /= len(configs["loader_in"])
            return loss, y_preds

    def test_me(self) -> None:
        print(123)
