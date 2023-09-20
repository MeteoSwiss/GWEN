from _typeshed import Incomplete
from gwen.loggers_configs import setup_logger as setup_logger, setup_mlflow as setup_mlflow
from gwen.loss_functions import MaskedLoss as MaskedLoss
from gwen.models_cnn import EvaluationConfigCNN as EvaluationConfigCNN, TrainingConfigCNN as TrainingConfigCNN, UNet as UNet
from gwen.utils import ConvDataset as ConvDataset, create_animation as create_animation, load_best_model as load_best_model, load_config_and_data as load_config_and_data

logger: Incomplete

def main() -> None: ...
