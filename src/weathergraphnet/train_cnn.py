"""Train a CNN to predict the future state of the atmosphere."""

# Standard library
import random
from typing import cast
from typing import List
from typing import Union

# Third-party
import mlflow  # type: ignore
import numpy as np
import torch
import xarray as xr
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# First-party
from weathergraphnet.logger import setup_logger
from weathergraphnet.models import EvaluationConfigCNN
from weathergraphnet.models import TrainingConfigCNN
from weathergraphnet.models import UNet
from weathergraphnet.utils import create_animation
from weathergraphnet.utils import EnsembleVarRegLoss
from weathergraphnet.utils import load_best_model
from weathergraphnet.utils import load_config_and_data
from weathergraphnet.utils import MaskedLoss
from weathergraphnet.utils import MyDataset
from weathergraphnet.utils import setup_mlflow

logger = setup_logger()

# pylint: disable=R0902,R0801


if __name__ == "__main__":
    try:
        # Load the configuration parameters and the input and output data
        config, data_train, data_test = load_config_and_data()
        logger.info(data_train.shape)
        logger.info(data_test.shape)
        # Create the dataset and dataloader
        dataset = MyDataset(data_train, config["member_split"])
        dataloader = DataLoader(dataset, config["batch_size"], shuffle=True)
        dataset_test = MyDataset(data_test, config["member_split"])
        dataloader_test = DataLoader(dataset_test, config["batch_size"], shuffle=False)

        loss_fn: Union[
            EnsembleVarRegLoss, MaskedLoss, nn.MSELoss, nn.Module
        ] = EnsembleVarRegLoss(alpha=0.1)

        if isinstance(loss_fn, MaskedLoss):
            # Create a mask that masks all cells that stay constant over all time steps
            variance = data_train.var(dim="time")
            # Create a mask that hides all data with zero variance
            mask = variance <= config["mask_threshold"]
            torch.from_numpy(mask.values.astype(float))
            logger.info("Number of masked cells: %d", sum((mask[0].values == 1)))

        else:
            mask = None

        artifact_path, experiment_name = setup_mlflow()

        if config["retrain"]:
            model: Union[UNet, nn.Module] = UNet(
                channels_in=config["member_split"],
                channels_out=data_train.sizes["member"] - config["member_split"],
                hidden_size=config["hidden_feats"],
            )

            optimizer = optim.Adam(model.parameters(), lr=config["lr"] * 100)
            scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

            # Train the model with MLflow logging
            MLFlowLogger(experiment_name=experiment_name)
            with mlflow.start_run():
                # Train the model Create a TrainingConfig object that contains both the
                # local variables and the JSON parameters
                config_train = TrainingConfigCNN(
                    dataloader=dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    mask=mask,
                    epochs=config["epochs"],
                    device=config["device"],
                    seed=config["seed"],
                )
                if torch.cuda.device_count() > 1:
                    logger.info("Using %d GPUs for Training", torch.cuda.device_count())
                    model = cast(UNet, nn.DataParallel(model).module)

                model.train_with_configs(config_train)  # type: ignore
        else:
            # Load the best model from the most recent MLflow run
            model = load_best_model(experiment_name)
    except FileNotFoundError as e:
        logger.error("Could not find file: %s", e)
        raise e

    y_pred: List[torch.Tensor] = []
    config_eval = EvaluationConfigCNN(
        dataloader=dataloader_test,
        loss_fn=loss_fn,
        mask=mask,
        device=config["device"],
        seed=config["seed"],
    )
    if torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs for Evaluation", torch.cuda.device_count())

        model = cast(UNet, nn.DataParallel(model).module)
    test_loss, y_pred = model.eval_with_configs(config_eval)  # type: ignore [operator]
    # if isinstance(loss_fn, nn.CrossEntropyLoss):
    # test_loss = test_loss.mean().item()
    logger.info("Best model test loss: %4f", test_loss)

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

    try:
        member = random.randint(
            0, data_test.sizes["member"] - config["member_split"] - 1
        )
        output_filename = create_animation(data_gif, member=member, preds="CNN")
    except Exception as e:
        logger.error("An error occurred while creating the animation: %s", e)

        raise
