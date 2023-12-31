# Standard library
import logging
from typing import Tuple

# Third-party
from _typeshed import Incomplete

logger: Incomplete

def setup_logger() -> logging.Logger: ...
def load_config(): ...
def setup_mlflow() -> Tuple[str, str]: ...
def suppress_warnings() -> None: ...
