"""Train a CNN to predict the future state of the atmosphere."""

# Standard library
from dataclasses import dataclass
from typing import cast
from typing import List
from typing import Optional
from typing import Union

# Third-party
import mlflow  # type: ignore
import numpy as np
import torch
import xarray as xr
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn
from torch import optim
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# First-party
from weathergraphnet.models import UNet
from weathergraphnet.utils import create_animation
from weathergraphnet.utils import EnsembleVarianceRegularizationLoss
from weathergraphnet.utils import load_best_model
from weathergraphnet.utils import load_config_and_data
from weathergraphnet.utils import MaskedLoss
from weathergraphnet.utils import MyDataset
from weathergraphnet.utils import setup_mlflow


# pylint: disable=R0902,R0801
@dataclass
class TrainingConfig(dict):
    """Configuration class for training a CNN model.

    Attributes:
        dataloader (DataLoader): The data loader for the training dataset.
        optimizer (nn.Module): The optimizer used for training.
        scheduler (nn.Module): The learning rate scheduler used for training.
        loss_fn (nn.Module): The loss function used for training.
        mask (Optional[torch.Tensor]): The mask to apply to the input data.
        num_epochs (int): The number of epochs to train for.
        device (str): The device to use for training (default is "cuda").
        seed (int): The random seed to use for reproducibility (default is 42).

    """

    dataloader: DataLoader
    optimizer: Union[torch.optim.Optimizer, Adam]
    scheduler: Union[CyclicLR, StepLR]
    loss_fn: Union[
        nn.Module,
        MaskedLoss,
        EnsembleVarianceRegularizationLoss,
        nn.MSELoss,
        nn.DataParallel,
    ]
    mask: Optional[torch.Tensor] = None
    num_epochs: int = 10
    device: str = "cuda"
    seed: int = 42


# pylint: disable=R0902,R0801
@dataclass
class EvaluateConfig(dict):
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
        MaskedLoss,
        EnsembleVarianceRegularizationLoss,
        nn.MSELoss,
        nn.DataParallel,
    ]
    mask: Optional[torch.Tensor] = None
    device: str = "cuda"
    seed: int = 42


if __name__ == "__main__":
    # Load the configuration parameters and the input and output data
    config, data_train, data_test = load_config_and_data()
    # Create the dataset and dataloader
    dataset = MyDataset(data_train, config["member_split"])
    dataloader = DataLoader(dataset, config["batch_size"], shuffle=True)
    dataset_test = MyDataset(data_test, config["member_split"])
    dataloader_test = DataLoader(dataset_test, config["batch_size"], shuffle=False)

    loss_fn: Union[
        EnsembleVarianceRegularizationLoss, MaskedLoss, nn.MSELoss, nn.Module
    ] = EnsembleVarianceRegularizationLoss(alpha=0.1)

    if loss_fn == MaskedLoss:
        # Create a mask that masks all cells that stay constant over all time steps
        variance = data_train.var(dim="time")
        # Create a mask that hides all data with zero variance
        mask = variance <= config["mask_threshold"]
        torch.from_numpy(mask.values.astype(float))
        print(f"Number of masked cells: {(mask[0].values == 1).sum()}", flush=True)

    artifact_path, experiment_name = setup_mlflow()

    if config["retrain"]:
        model = UNet(
            channels_in=config["member_split"],
            channels_out=data_train.shape[1] - config["member_split"],
            hidden_size=config["hidden_feats"],
        )

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

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
                num_epochs=config["epochs"],
                device=config["device"],
                seed=config["seed"],
            )
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs")
                model = cast(UNet, nn.DataParallel(model).module)
            model.train_with_configs(config_train)
    else:
        # Load the best model from the most recent MLflow run
        model_best = load_best_model(experiment_name)
        if isinstance(model_best, UNet):
            model = model_best
        else:
            model = cast(UNet, model_best)

    y_pred: List[torch.Tensor] = []
    config_eval = EvaluateConfig(
        dataloader=dataloader_test,
        loss_fn=loss_fn,
        mask=mask,
        device=config["device"],
        seed=config["seed"],
    )
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = cast(UNet, nn.DataParallel(model).module)
    test_loss, y_pred = model.eval_with_configs(config_eval)
    print(f"Best model test loss: {test_loss:.4f}")

    # Plot the predictions

    y_pred_reshaped = xr.DataArray(
        torch.cat(y_pred)
        .numpy()
        .reshape(
            (
                np.array(
                    data_test.isel(
                        member=slice(config["member_split"], data_test.sizes["member"])
                    ).values
                ).shape
            )
        ),
        dims=["time", "member", "height", "ncells"],
    )

    data_gif = {
        "y_pred_reshaped": y_pred_reshaped,
        "data_test": data_test,
    }

    output_filename = create_animation(data_gif, member=0, preds="CNN")
