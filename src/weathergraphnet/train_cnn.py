"""Train a CNN to predict the future state of the atmosphere."""

# Standard library
from dataclasses import dataclass
from typing import List
from typing import Optional

# Third-party
import mlflow
import torch
import torch.nn.functional as F
import xarray as xr
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# First-party
from weathergraphnet.utils import create_animation
from weathergraphnet.utils import EnsembleVarianceRegularizationLoss
from weathergraphnet.utils import load_best_model
from weathergraphnet.utils import load_config_and_data
from weathergraphnet.utils import MaskedLoss
from weathergraphnet.utils import move_to_device
from weathergraphnet.utils import MyDataset
from weathergraphnet.utils import setup_mlflow


class BaseNet(nn.Module):
    """Base class for the encoder and decoder networks."""

    def __init__(self, channels_in, channels_out, hidden_size):
        """Initialize the base class."""
        super().__init__()
        self.activation = nn.ReLU()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.hidden_size = hidden_size

    def forward(self, x):
        """Forward pass through the network."""
        raise NotImplementedError("Subclasses must implement the forward method.")


class Encoder(BaseNet):
    """Encoder network."""

    def __init__(self, channels_in, channels_out, hidden_size):
        """Initialize the encoder network."""
        super().__init__(channels_in, channels_out, hidden_size)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels_in, self.hidden_size // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                self.hidden_size // 8, self.hidden_size // 4, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                self.hidden_size // 4, self.hidden_size // 2, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                self.hidden_size // 2, self.hidden_size, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass through the encoder network."""
        x = self.conv_layers(x)
        return x


class Decoder(BaseNet):
    """Decoder network."""

    def __init__(self, channels_in, channels_out, hidden_size):
        """Initialize the decoder network."""
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
                nn.Conv2d(self.hidden_size // 8, self.channels_out, kernel_size=1),
            ]
        )

    def crop(self, encoder_layer, decoder_layer):
        """Crop the encoder layer to the size of the decoder layer."""
        diff_y = encoder_layer.size()[2] - decoder_layer.size()[2]
        diff_x = encoder_layer.size()[3] - decoder_layer.size()[3]
        encoder_layer = encoder_layer[
            :,
            :,
            diff_y // 2 : encoder_layer.size()[2] - diff_y // 2,
            diff_x // 2 : encoder_layer.size()[3] - diff_x // 2,
        ]
        if diff_x % 2 == 1:
            encoder_layer = encoder_layer[:, :, :, 1 : encoder_layer.size()[3]]
        if diff_y % 2 == 1:
            encoder_layer = encoder_layer[:, :, 1 : encoder_layer.size()[2], :]
        return encoder_layer

    def forward(self, x):
        """Perform forward pass through the decoder network."""
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
        out = F.pad(out, (cropped, 0, 0, 0), mode="replicate")
        return out


class UNet(BaseNet):
    """UNet network."""

    def __init__(self, channels_in, channels_out, hidden_size):
        """Initialize the UNet network."""
        super().__init__(channels_in, channels_out, hidden_size)
        self.encoder = Encoder(channels_in, channels_out, hidden_size)
        self.decoder = Decoder(channels_in, channels_out, hidden_size)

    def forward(self, x):
        """Forward pass through the UNet network."""
        x1, x2, x3, x4 = self.encoder(x)
        out = self.decoder(x1, x2, x3, x4)
        return out


def train_model(self, configs):
    """Train the model."""
    best_loss = float("inf")
    for epoch in range(configs.num_epochs):
        running_loss = 0.0
        for input_data, target_data in configs.loader_train:
            input_data = input_data.to(configs.device)
            target_data = target_data.to(configs.device)
            configs.optimizer.zero_grad()
            output = self(input_data)
            if configs.mask is not None:
                loss = configs.loss_fn(
                    output, target_data, configs.mask.to(configs.device)
                )
            else:
                loss = configs.loss_fn(output, target_data)
            loss.backward()
            configs.optimizer.step()
            if configs.scheduler is not None:
                configs.scheduler.step()  # update the learning rate
            running_loss += loss.item()
            avg_loss = running_loss / len(output)
            print(f"Epoch {epoch + 1}: {avg_loss}")
            mlflow.log_metric("loss", avg_loss)
        if avg_loss is not None and avg_loss < best_loss:
            best_loss = avg_loss
    return best_loss


def evaluate(self, configs):
    """Evaluate the model."""
    self.eval()
    with torch.no_grad():
        loss = 0.0
        if configs.return_predictions:
            y_preds = []
        for input_data, target_data in configs.dataloader:
            input_data = input_data.to(configs.device)
            target_data = target_data.to(configs.device)
            output = self(input_data)
            if configs.mask is not None:
                loss += configs.loss_fn(
                    output, target_data, configs.mask.to(configs.device)
                )
            else:
                loss += configs.loss_fn(output, target_data)
            if configs.return_predictions:
                y_preds.append(output.cpu())
        loss /= len(configs.dataloader)
        if configs.return_predictions:
            return loss, y_preds
        else:
            return loss


# pylint: disable=R0902,R0801
@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    dataloader: DataLoader
    optimizer: nn.Module
    scheduler: nn.Module
    loss_fn: nn.Module
    mask: Optional[torch.Tensor] = None
    num_epochs: int = 10
    device: str = "cuda"
    seed: int = 42


# pylint: disable=R0902,R0801
@dataclass
class EvaluateConfig:
    """Evaluation configuration parameters."""

    dataloader: DataLoader
    loss_fn: nn.Module
    mask: Optional[torch.Tensor] = None
    return_predictions: bool = False
    device: str = "cuda"
    seed: int = 42


if __name__ == "__main__":
    # Load the configuration parameters and the input and output data
    config, data_train, data_test = load_config_and_data()
    # Create the dataset and dataloader
    dataset = MyDataset(data_train, config.member_split)
    dataloader = DataLoader(dataset, config.batch_size, shuffle=True)
    dataset_test = MyDataset(data_test, config.member_split)
    dataloader_test = DataLoader(dataset_test, config.batch_size, shuffle=False)

    model = UNet(
        channels_in=config.member_split,
        channels_out=data_train.shape[1] - config.member_split,
        hidden_size=config.hidden_feats,
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # loss_fn = MaskedLoss(loss_fn=nn.L1Loss())
    # loss_fn = nn.L1Loss()
    loss_fn = EnsembleVarianceRegularizationLoss(alpha=0.1)

    if loss_fn == MaskedLoss:
        # Create a mask that masks all cells that stay constant over all time steps
        variance = data_train.var(dim="time")
        # Create a mask that hides all data with zero variance
        mask = variance <= config.mask_threshold
        torch.from_numpy(mask.values.astype(float))
        print(f"Number of masked cells: {(mask[0].values == 1).sum()}", flush=True)

    model, loss_fn = move_to_device(model, loss_fn)

    train_model_fn = (
        model.module.train_model
        if isinstance(model, nn.DataParallel)
        else model.train_model
    )

    artifact_path, experiment_name = setup_mlflow()

    retrain = True
    if retrain:
        # Train the model with MLflow logging
        MLFlowLogger(experiment_name=experiment_name)
        with mlflow.start_run():
            # Train the model Create a TrainingConfig object that contains both the
            # local variables and the JSON parameters
            config_train = TrainingConfig(
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                mask=mask,
                num_epochs=config.epochs,
                device=config.device,
                seed=config.seed,
            )
            train_model_fn(config_train)
    else:
        # Load the best model from the most recent MLflow run
        model = load_best_model(experiment_name)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        evaluate_fn = (
            model.module.evaluate
            if isinstance(model, nn.DataParallel)
            else model.evaluate
        )
    y_pred: List[torch.Tensor] = []
    config_eval = EvaluateConfig(
        dataloader=dataloader_test,
        loss_fn=loss_fn,
        mask=mask,
        return_predictions=True,
        device=config.device,
        seed=config.seed,
    )
    test_loss, y_pred = evaluate_fn(config_eval)
    print(f"Best model test loss: {test_loss:.4f}")

    # Plot the predictions

    y_pred_reshaped = xr.DataArray(
        torch.cat(y_pred)
        .numpy()
        .reshape(
            (
                data_test.isel(
                    member=slice(config.member_split, data_test.sizes["member"])
                ).values.shape
            )
        ),
        dims=["time", "member", "height", "ncells"],
    )

    data_gif = {
        "y_pred_reshaped": y_pred_reshaped,
        "data_test": data_test,
    }

    output_filename = create_animation(data_gif, member=0, preds="CNN")
