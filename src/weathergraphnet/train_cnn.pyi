# Third-party
from _typeshed import Incomplete

# First-party
from weathergraphnet.loggers_configs import setup_logger as setup_logger
from weathergraphnet.models_cnn import EvaluationConfigCNN as EvaluationConfigCNN
from weathergraphnet.models_cnn import TrainingConfigCNN as TrainingConfigCNN
from weathergraphnet.models_cnn import UNet as UNet
from weathergraphnet.utils import MaskedLoss as MaskedLoss
from weathergraphnet.utils import MyDataset as MyDataset
from weathergraphnet.utils import create_animation as create_animation
from weathergraphnet.utils import load_config_and_data as load_config_and_data
from weathergraphnet.utils import setup_mlflow as setup_mlflow

logger: Incomplete


def main() -> None: ...
