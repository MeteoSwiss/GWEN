import json
import os

import pytest
import xarray as xr
from weathergraphnet.models import GNNModel
from weathergraphnet.models import UNet
from weathergraphnet.utils import load_best_model
from weathergraphnet.utils import load_config_and_data
from weathergraphnet.utils import load_data
from weathergraphnet.utils import setup_mlflow


@pytest.fixture(scope="module")
def config():
    with open(
        os.path.join(os.path.dirname(__file__),
                     "../../src/weathergraphnet/config.json"),
        "r",
        encoding="UTF-8",
    ) as f:
        config = json.load(f)
    return config


@pytest.fixture(scope="module")
def data(config):
    data_train, data_test = load_data(config)
    return data_train, data_test


@pytest.fixture(scope="module")
def experiment_name():
    return "WGN"


def test_load_best_model(experiment_name):
    with pytest.raises(ValueError):
        load_best_model("nonexistent_experiment")
    with pytest.raises(FileNotFoundError):
        load_best_model(experiment_name)
    model = load_best_model(experiment_name)
    assert isinstance(model, (GNNModel, UNet))


def test_load_config_and_data(config):
    with pytest.raises(FileNotFoundError):
        load_config_and_data()
    with pytest.raises(ValueError):
        config["coarsen"] = -1
        load_config_and_data()
    config, data_train, data_test = load_config_and_data()
    assert isinstance(config, dict)
    assert isinstance(data_train, xr.Dataset)
    assert isinstance(data_test, xr.Dataset)


def test_load_data(config, data):
    with pytest.raises(FileNotFoundError):
        config["data_train"] = "nonexistent_file.zarr"
        load_data(config)
    data_train, data_test = data
    assert isinstance(data_train, xr.Dataset)
    assert isinstance(data_test, xr.Dataset)


def test_setup_mlflow():
    artifact_path, experiment_name = setup_mlflow()
    assert isinstance(artifact_path, str)
    assert isinstance(experiment_name, str)
