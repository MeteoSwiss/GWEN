"""Provide logging for the weathergraphnet package."""
# Standard library
import json
import logging

from pyprojroot import here

# Third-party
from torch import distributed as dist

# Create a logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logger() -> logging.Logger:
    """Set up a logger for the weathergraphnet package.

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
    """Load the configuration for the weathergraphnet project."""
    with open(
        str(here()) + "/src/weathergraphnet/config.json", "r", encoding="UTF-8"
    ) as f:
        config = json.load(f)
    return config
