"""Provide logging for the gwen package."""
# Standard library
import json
import logging
import re
import socket
import warnings
from typing import Tuple

# Third-party
import matplotlib
import mlflow
from pyprojroot import here
from torch import distributed as dist

# Create a logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logger() -> logging.Logger:
    """Set up a logger for the gwen package.

    The logger will log to the console and to a file called "logfile.log" in the
    current working directory.

    The logger will log messages with level DEBUG and above to the console and
    messages with level INFO and above to the file.

    """
    # If handlers have already been added to the logger, return it
    if logger.hasHandlers():
        return logger

    if dist.is_available() and dist.is_initialized():
        # If the current process is not the master process, do not add handlers
        if dist.get_rank() != 0:
            return logger

    # Create a console handler and set its level to DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a file handler and set its level to INFO
    file_handler = logging.FileHandler("logfile.log")
    file_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def load_config():
    """Load the configuration for the gwen project."""
    with open(str(here()) + "/src/gwen/config.json", "r", encoding="UTF-8") as f:
        config = json.load(f)
    return config


def setup_mlflow() -> Tuple[str, str]:
    """Set up the MLflow experiment and artifact path based on the hostname.

    Returns the artifact path and experiment name as a tuple.

    """
    try:
        hostname = socket.gethostname()
        # Set the artifact path based on the hostname
        if "nid" in hostname:
            artifact_path = (
                "/scratch/e1000/meteoswiss/scratch/sadamov/"
                "pyprojects_data/gwen/mlruns"
            )
            experiment_name = "GWEN_balfrin"
        else:
            artifact_path = "/scratch/sadamov/pyprojects_data/gwen/mlruns"
            experiment_name = "GWEN"

        mlflow.set_tracking_uri(str(here()) + "/mlruns")
        existing_experiment = mlflow.get_experiment_by_name(experiment_name)
        if existing_experiment is None:
            mlflow.create_experiment(
                name=experiment_name, artifact_location=artifact_path
            )
        mlflow.set_experiment(experiment_name=experiment_name)
    except Exception as e:
        logger.exception(str(e))
        raise e
    logger.info("MLflow experiment name: %s", experiment_name)
    return artifact_path, experiment_name


def suppress_warnings():
    """Suppresses certain warnings that are not relevant to the user."""
    warnings.simplefilter("always")
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
    warnings.filterwarnings("ignore", message="Setuptools is replacing dist")
    warnings.filterwarnings(
        "ignore",
        message="Encountered an unexpected error while inferring pip requirements",
    )
    warnings.filterwarnings(
        "ignore",
        message=re.escape("Using '")
        + ".*"
        + re.escape(
            "' without a 'pyg-lib' installation is deprecated and "
            "will be removed soon. Please install 'pyg-lib' for "
            "accelerated neighborhood sampling"
        ),
        category=UserWarning,
    )
