"""Contains the models used for weather prediction."""
# Standard library
import os
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# Third-party
import mlflow  # type: ignore  # type: ignore  # type: ignore
import torch
from pytorch_lightning.loggers import MLFlowLogger
from torch import distributed as dist
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# First-party
from weathergraphnet.loggers_configs import setup_logger
from weathergraphnet.loggers_configs import setup_mlflow
from weathergraphnet.utils import ConvDataset

logger = setup_logger()


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

    dataset: ConvDataset
    optimizer: Union[torch.optim.Optimizer, Adam]
    scheduler: Union[CyclicLR, StepLR]
    loss_fn: Union[
        nn.Module,
        nn.MSELoss,
    ]
    batch_size: int
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

    dataset: ConvDataset
    loss_fn: Union[
        nn.Module,
        nn.MSELoss,
    ]
    batch_size: int
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
            logger.error(
                "Error occurred while initializing the Decoder class: %s", e)

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
                diff_y // 2: encoder_layer.size()[2] - diff_y // 2,
                diff_x // 2: encoder_layer.size()[3] - diff_x // 2,
            ]
            if diff_y % 2 == 1:
                encoder_layer = encoder_layer[:, :, 1: encoder_layer.size()[
                    2], :]
            if diff_x % 2 == 1:
                encoder_layer = encoder_layer[:, :, :, 1: encoder_layer.size()[
                    3]]
        except IndexError as e:
            logger.error(
                "Error occurred while cropping the encoder layer: %s", e)
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
                cropped += x3.shape[3] - y1.shape[3]
            y1 = torch.cat([x3, y1], dim=1)

            y1 = self.activation(y1)

            y2 = self.conv_transposed_layers[1](y1)

            y2 = self.upsample(y2)

            y2 = self.batch_norm_layers[1](y2)

            y2 = self.activation(y2)

            if y2.shape != x2.shape:
                x2 = self.crop(x2, y2)
                cropped += x2.shape[3] - y2.shape[3]
            y2 = torch.cat([x2, y2], dim=1)

            y2 = self.activation(y2)

            y3 = self.conv_transposed_layers[2](y2)

            y3 = self.upsample(y3)

            y3 = self.batch_norm_layers[0](y3)

            y3 = self.activation(y3)

            if y3.shape != x1.shape:
                x1 = self.crop(x1, y3)
                cropped += x1.shape[3] - y3.shape[3]
            y3 = torch.cat([x1, y3], dim=1)

            y3 = self.activation(y3)

            y4 = self.conv_transposed_layers[3](y3)

            y4 = self.upsample(y4)

            y4 = self.activation(y4)

            out = self.conv_layers[4](y4)

            out = nn.functional.pad(
                out, (cropped * 2, 0, 0, 0), mode="replicate")

            if dist.get_rank() == 0:
                logged_messages = set()
                # Log the messages after each epoch, but only if they haven't been
                # logged before
                if "Shapes of the UNET" not in logged_messages:
                    logger.debug("Shapes of the UNET")
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
                ]
                for name, var in variables:
                    message = f"{name}: {var.shape}"
                    if message not in logged_messages:
                        logger.debug(message)
                        logged_messages.add(message)

        except ValueError as e:
            logger.error(
                "Error occurred during the forward pass of the CNN model: %s", e
            )
        return out


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack([torch.from_numpy(x.values) for x in inputs])
    targets = torch.stack([torch.from_numpy(y.values) for y in targets])
    return inputs, targets


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
            logger.error(
                "Error occurred while initializing UNet network: %s", str(e))

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
            if x.shape[3] != out.shape[3]:
                out = nn.functional.pad(
                    out, (x.shape[3] - out.shape[3], 0, 0, 0), mode="replicate"
                )  # pad the output tensor
            if dist.get_rank() == 0:
                logger.info(f"Output UNet shape: {out.shape}")

        except RuntimeError as e:
            logger.error(
                "Error occurred during forward pass through UNet network: %s", str(
                    e)
            )
        return out

    def train_with_configs(
        self, rank, configs_train_cnn: TrainingConfigCNN, world_size
    ) -> None:
        """Train the model.

        Args:
        configs_train_cnn (Any): The configuration object.

        Returns:
        None

        """

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"  # choose an available port
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl", rank=rank, world_size=world_size)
        if dist.get_rank() == 0:
            print("Training UNet network with configurations:", flush=True)
            print(configs_train_cnn, flush=True)
        torch.manual_seed(configs_train_cnn.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(configs_train_cnn.seed)

        if dist.get_rank() == 0:
            # Train the model with MLflow logging
            artifact_path, experiment_name = setup_mlflow()
            MLFlowLogger(experiment_name=experiment_name)
            mlflow.start_run()

        sampler = DistributedSampler(
            configs_train_cnn.dataset, seed=configs_train_cnn.seed)
        dataloader = DataLoader(
            configs_train_cnn.dataset,
            batch_size=configs_train_cnn.batch_size,
            sampler=sampler,
            num_workers=16,
            pin_memory=True,
            collate_fn=collate_fn  # Add the custom collate function here
        )

        device = configs_train_cnn.device
        model = self.to(device)
        model = nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
        if configs_train_cnn.mask is not None:
            configs_train_cnn.mask = configs_train_cnn.mask.to(device)
        best_loss = torch.tensor(float("inf")).to(device)
        try:
            for epoch in range(configs_train_cnn.epochs):
                sampler.set_epoch(epoch)
                running_loss = 0.0
                for input_data, target_data in dataloader:
                    input_data = input_data.to(device)
                    target_data = target_data.to(device)
                    configs_train_cnn.optimizer.zero_grad()
                    # pylint: disable=not-callable
                    output = model(input_data)
                    if configs_train_cnn.mask is not None:
                        loss = configs_train_cnn.loss_fn(
                            output,
                            target_data,
                            configs_train_cnn.mask,
                        )
                    else:
                        loss = configs_train_cnn.loss_fn(output, target_data)
                    loss.backward()
                    configs_train_cnn.optimizer.step()
                    if configs_train_cnn.scheduler is not None:
                        configs_train_cnn.scheduler.step()  # update the learning rate
                    running_loss += loss.item()
                avg_loss = torch.tensor(running_loss /
                                        float(len(dataloader))).to(device)
                gathered_losses = [torch.zeros_like(avg_loss) for _ in range(
                    dist.get_world_size())] if dist.get_rank() == 0 else []
                dist.gather(
                    avg_loss, gather_list=gathered_losses, dst=0)
                torch.distributed.barrier()  # Wait for all workers to finish
                if dist.get_rank() == 0:
                    avg_loss_gathered = torch.stack(
                        gathered_losses).mean().item()
                    logger.info("Epoch: %d, Loss: %f",
                                epoch, avg_loss_gathered)
                    mlflow.log_metric("loss", float(
                        avg_loss_gathered), step=epoch)
                    # mlflow.pytorch.log_model(model, f"model_epoch_{epoch}")

                    if avg_loss_gathered < best_loss:
                        best_loss = avg_loss_gathered
                        mlflow.pytorch.log_model(model.module,
                                                 "models", pip_requirements=[
                                                     f"torch=={torch.__version__}"])

        except RuntimeError as e:
            logger.error(
                "Error occurred while training UNet: %s", str(e))
        if dist.get_rank() == 0:
            mlflow.end_run()
            dist.destroy_process_group()

    def eval_with_configs(
        self, rank, configs_eval_cnn, world_size: EvaluationConfigCNN, queue, event
    ):
        """Evaluate the model.

        Args:
        - configs_eval_cnn (Any): The configuration object.

        Returns:
        - loss (float): The loss achieved during evaluation.

        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"  # choose an available port
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl", rank=rank, world_size=1)
        # dist.init_process_group(
        #     "nccl", rank=rank, world_size=world_size) #TODO: eval on >1 GPU
        if dist.get_rank() == 0:
            print("Evaluating UNet network with configurations:", flush=True)
            print(configs_eval_cnn, flush=True)
        torch.manual_seed(configs_eval_cnn.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(configs_eval_cnn.seed)

        sampler = DistributedSampler(
            configs_eval_cnn.dataset, shuffle=False, drop_last=False)
        dataloader_test = DataLoader(
            configs_eval_cnn.dataset, configs_eval_cnn.batch_size, sampler=sampler,
            num_workers=16, pin_memory=True, collate_fn=collate_fn)
        device = configs_eval_cnn.device
        model = self.to(device)
        model = nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
        ranks = []
        try:
            model.eval()
            with torch.no_grad():
                loss: float = 0.0
                y_preds: List[torch.Tensor] = []
                for input_data, target_data in dataloader_test:
                    input_data = input_data.to(device)
                    target_data = target_data.to(device)
                    # pylint: disable=not-callable
                    output = model(input_data)
                    if configs_eval_cnn.mask is not None:
                        configs_eval_cnn.mask = configs_eval_cnn.mask.to(
                            device)
                        loss += configs_eval_cnn.loss_fn(
                            output,
                            target_data,
                            configs_eval_cnn.mask,
                        )
                    else:
                        loss += configs_eval_cnn.loss_fn(output, target_data)
                    ranks.append(rank)
                    y_preds.append(output)

                avg_loss = torch.tensor(loss.item() /
                                        float(len(dataloader_test))).to(device)
                gathered_losses = [torch.zeros_like(avg_loss) for _ in range(
                    dist.get_world_size())] if dist.get_rank() == 0 else []

                ranks_tensor = torch.tensor(ranks, device=device)
                ranks_list = [torch.zeros_like(ranks_tensor)
                              for _ in range(dist.get_world_size())]
                y_preds_tensor = torch.cat(y_preds)
                y_preds_list = [torch.zeros_like(y_preds_tensor)
                                for _ in range(dist.get_world_size())]

                dist.barrier()
                dist.all_gather(ranks_list, ranks_tensor)
                dist.all_gather(y_preds_list, y_preds_tensor)
                dist.gather(avg_loss, gather_list=gathered_losses, dst=0)

                y_preds_ordered = [y_pred for _, y_pred in sorted(
                    zip(ranks_list, y_preds_list), key=lambda x: x[0])]

                dist.barrier()
                if dist.get_rank() == 0:
                    avg_loss_gathered = torch.stack(
                        gathered_losses).mean().item()
                    y_preds_ordered_tensor = torch.cat(y_preds_ordered)
                    logger.info("Loss: %f", avg_loss_gathered)

        except RuntimeError as e:
            logger.error(
                "Error occurred while evaluating UNet network: %s", str(e))
        finally:
            dist.barrier()
            torch.cuda.synchronize()
            if dist.get_rank() == 0:
                queue.put((float(avg_loss_gathered),
                          y_preds_ordered_tensor.cpu()))
            dist.barrier()
            dist.destroy_process_group()
