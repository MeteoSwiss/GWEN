"""Train a CNN to predict the future state of the atmosphere."""

# Standard library
from typing import List
from typing import Union

# Third-party
import numpy as np
import torch
import torch.multiprocessing as mp
import xarray as xr
from torch import nn
from torch import optim

# First-party
from weathergraphnet.logger import setup_logger
from weathergraphnet.models_cnn import EvaluationConfigCNN
from weathergraphnet.models_cnn import TrainingConfigCNN
from weathergraphnet.models_cnn import UNet
from weathergraphnet.utils import create_animation
from weathergraphnet.utils import load_best_model
from weathergraphnet.utils import load_config_and_data
from weathergraphnet.utils import MaskedLoss
from weathergraphnet.utils import MyDataset
from weathergraphnet.utils import setup_mlflow

logger = setup_logger()

# pylint: disable=R0902,R0801


def main():
    try:
        # This line is used to set multiprocessing start method
        mp.set_start_method("spawn")

        # Load the configuration parameters and the input and output data
        config, data_train, data_test = load_config_and_data()
        logger.info(data_train.shape)
        logger.info(data_test.shape)
        # Create the dataset and dataloader
        dataset = MyDataset(data_train, config["member_split"])
        dataset_test = MyDataset(data_test, config["member_split"])

        # loss_fn: Union[
        #     EnsembleVarRegLoss, MaskedLoss, nn.MSELoss, nn.Module
        # ] = EnsembleVarRegLoss(alpha=0.1)

        loss_fn = nn.L1Loss()

        if isinstance(loss_fn, MaskedLoss):
            # Create a mask that masks all cells that stay constant over all time steps
            variance = data_train.var(dim="time")
            # Create a mask that hides all data with zero variance
            mask = variance <= config["mask_threshold"]
            torch.from_numpy(mask.values.astype(float))
            logger.info("Number of masked cells: %d", sum((mask[0].values == 1)))

        else:
            mask = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config["retrain"]:
            model: Union[UNet, nn.Module] = UNet(
                channels_in=config["member_split"],
                channels_out=data_train.sizes["member"] - config["member_split"],
                hidden_size=config["hidden_feats"],
            )

            optimizer = optim.Adam(model.parameters(), lr=config["lr"] * 100)
            # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
            # TODO: currently breaks the setup with DDP optimizer.step and
            # scheduler.step
            scheduler = None

            # Train the model Create a TrainingConfig object that contains both the
            # local variables and the JSON parameters
            config_train = TrainingConfigCNN(
                dataset=dataset,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                batch_size=config["batch_size"],
                mask=mask,
                epochs=config["epochs"],
                device=device,
                seed=config["seed"],
            )
            logger.info("Using %d GPUs for Training", torch.cuda.device_count())

            world_size = torch.cuda.device_count()
            mp.spawn(
                model.train_with_configs,
                args=(
                    config_train,
                    world_size,
                ),
                nprocs=world_size,
                join=True,
            )
        else:
            _, experiment_name = setup_mlflow()
            load_best_model(experiment_name)

    except FileNotFoundError as e:
        logger.error("Could not find file: %s", e)
        raise e
    try:
        y_pred: List[torch.Tensor] = []
        config_eval = EvaluationConfigCNN(
            dataset=dataset_test,
            loss_fn=loss_fn,
            batch_size=config["batch_size"],
            mask=mask,
            device=device,
            seed=config["seed"],
        )

        test_loss, y_pred = model.eval_with_configs(
            config_eval
        )  # type: ignore [operator]
        # if isinstance(loss_fn, nn.CrossEntropyLoss):
        # test_loss = test_loss.mean().item()
        logger.info("Best model test loss: %4f", test_loss)

    except (RuntimeError, ValueError) as e:
        logger.exception("Error occurred while evaluating model: %s", e)

    # Plot the predictions

    try:
        y_pred_reshaped = xr.DataArray(
            torch.cat(y_pred)
            .numpy()
            .reshape(
                (
                    np.array(
                        data_test.isel(
                            member=slice(
                                config["member_split"], data_test.sizes["member"]
                            )
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

        test_out_members = dataset_test.get_test_indices()
        for member_pred, member_target in enumerate(test_out_members):
            create_animation(
                data_gif,
                member_pred=member_pred,
                member_target=member_target,
                preds="CNN",
            )
            create_animation(
                data_gif,
                member_pred=member_pred,
                member_target=member_target,
                preds="ICON",
            )
    except Exception as e:
        logger.error("An error occurred while creating the animation: %s", e)

        raise


if __name__ == "__main__":
    main()
