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

# Third-party
import mlflow  # type: ignore
import numpy as np
import torch
import torch.multiprocessing as mp

# import torch.multiprocessing as mp
import xarray as xr
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn
from torch import optim

# First-party
from gwen.loggers_configs import setup_logger
from gwen.loggers_configs import setup_mlflow
from gwen.loggers_configs import suppress_warnings
from gwen.loss_functions import MaskedLoss
from gwen.models_gnn import EvaluationConfigGNN
from gwen.models_gnn import GNNConfig
from gwen.models_gnn import GNNModel
from gwen.models_gnn import TrainingConfigGNN
from gwen.utils import create_animation
from gwen.utils import GraphDataset
from gwen.utils import load_best_model
from gwen.utils import load_config_and_data

logger = setup_logger()
suppress_warnings()

# TODO: add dropout layers to all my models!


def main():
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue(10000)
    event = mp.Event()

    # Load the configuration parameters and the input and output data
    config, data_train, data_test = load_config_and_data()

    dataset_train = GraphDataset(data_train, config["member_split"])
    dataset_test = GraphDataset(data_test, config["member_split"])

    try:
        # loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        loss_fn = nn.L1Loss()
        # loss_fn = EnsembleVarRegLoss()
        if isinstance(loss_fn, MaskedLoss):
            # Create a mask that masks all cells that stay constant over all time steps
            variance = data_train.var(dim="time")
            # Create a mask that hides all data with zero variance
            mask = variance <= config["mask_threshold"]
            torch.from_numpy(mask.values.astype(float))
            logger.info("Number of masked cells: %d", (mask[0].values == 1).sum())
        else:
            mask = None
    except (ValueError, TypeError) as e:
        logger.exception("Error occurred while creating loss function: %s", e)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if config["retrain"]:
            gnn_config = GNNConfig(
                nodes_in=len(dataset_train.input_indices),
                nodes_out=len(dataset_train.target_indices),
                channels_in=dataset_train.channels,
                channels_out=dataset_train.channels,
                hidden_feats=config["hidden_feats"],
            )
            model = GNNModel(gnn_config)
            optimizer = optim.Adam(model.parameters(), lr=config["lr"] * 10)
            # scheduler = CyclicLR(
            #     optimizer,
            #     base_lr=config["lr"],
            #     max_lr=10 * config["lr"],
            #     mode="triangular2",
            #     cycle_momentum=False,
            # ) #BUG: this scheduler is not working with DDP
            scheduler = None

            # Train the model with MLflow logging
            _, experiment_name = setup_mlflow()
            MLFlowLogger(experiment_name=experiment_name)
            with mlflow.start_run():
                # Train the model Create a TrainingConfig object that contains both the
                # local variables and the JSON parameters
                config_train = (
                    TrainingConfigGNN(  # pylint: disable=too-many-function-args
                        dataset=dataset_train,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        batch_size=config["batch_size"],
                        mask=mask,
                        epochs=config["epochs"],
                        device=device,
                        seed=config["seed"],
                    )
                )
                # Pass the TrainingConfig object to the train method
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
                # model.train_with_configs(config_train)
        else:
            artifact_path, experiment_name = setup_mlflow()
            model = load_best_model(experiment_name)

    except mlflow.exceptions.MlflowException as e:
        logger.exception("Error occurred while setting up MLflow: %s", e)

    try:
        config_eval = EvaluationConfigGNN(
            dataset=dataset_test,
            loss_fn=loss_fn,
            mask=mask,
            device=device,
            batch_size=config["batch_size"],
            seed=config["seed"],
        )

        world_size = torch.cuda.device_count()  # TODO: eval on >1 GPU
        # world_size = 1

        mp.spawn(
            model.eval_gnn_with_configs,
            args=(config_eval, world_size, queue, event),
            nprocs=world_size,
            join=True,
        )
        print("mp spawn is done.")

        test_loss, b = queue.get()
        y_pred = b.clone()
        del b

        logger.info("Best model test loss: %f", test_loss)
    except (RuntimeError, ValueError) as e:
        logger.exception("Error occurred while evaluating model: %s", e)
    try:
        # Plot the predictions
        # TODO: This might have changed check data_test_out dims
        y_pred_reshaped = xr.DataArray(
            y_pred.numpy().reshape(np.array(data_test.values).shape),
            dims=["time", "member", "height", "ncells"],
        )

        logger.info("The shape of the raw model prediction: %s", y_pred.numpy().shape)
        logger.info("Reshaped into form: %s", y_pred_reshaped.shape)
        data_gif = {
            "y_pred_reshaped": y_pred_reshaped,
            "data_test": data_test,
        }

        test_out_members = dataset_test.target_indices

        for member_pred, member_target in enumerate(test_out_members):
            create_animation(
                data_gif,
                member_pred=member_pred,
                member_target=member_target,
                preds="GNN",
            )
        for member_pred, member_target in enumerate(test_out_members):
            create_animation(
                data_gif,
                member_pred=member_pred,
                member_target=member_target,
                preds="ICON",
            )
    except (ValueError, TypeError) as e:
        logger.exception("Error occurred while creating animation: %s", e)


if __name__ == "__main__":
    main()
