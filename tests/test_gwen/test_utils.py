"""Tests for the gwen.utils module."""
# Standard library
import json
import os

# Third-party
import pytest
import xarray as xr

# First-party
from gwen.loggers_configs import setup_mlflow
from gwen.models_cnn import UNet
from gwen.models_gnn import GNNModel
from gwen.utils import load_best_model
from gwen.utils import load_config_and_data
from gwen.utils import load_data


@pytest.fixture(scope="module")
def config():
    with open(
        os.path.join(os.path.dirname(__file__), "../../src/gwen/config.json"),
        "r",
        encoding="UTF-8",
    ) as f:
        config_json = json.load(f)
    return config_json


@pytest.fixture(scope="module")
def data(config_json):
    data_train, data_test = load_data(config_json)
    return data_train, data_test


@pytest.fixture(scope="module")
def experiment_name():
    return "GWEN"


def test_load_best_model(exp_name):
    with pytest.raises(ValueError):
        load_best_model("nonexistent_experiment")
    with pytest.raises(FileNotFoundError):
        load_best_model(exp_name)
    model = load_best_model(exp_name)
    assert isinstance(model, (GNNModel, UNet))


def test_load_config_and_data(config_json):
    with pytest.raises(FileNotFoundError):
        load_config_and_data()
    with pytest.raises(ValueError):
        config_json["coarsen"] = -1
        load_config_and_data()
    config_json, data_train, data_test = load_config_and_data()
    assert isinstance(config_json, dict)
    assert isinstance(data_train, xr.Dataset)
    assert isinstance(data_test, xr.Dataset)


def test_load_data(config_json, data_load):
    with pytest.raises(FileNotFoundError):
        config_json["data_train"] = "nonexistent_file.zarr"
        load_data(config_json)
    data_train, data_test = data_load
    assert isinstance(data_train, xr.Dataset)
    assert isinstance(data_test, xr.Dataset)


def test_setup_mlflow():
    artifact_path, exp_name = setup_mlflow()
    assert isinstance(artifact_path, str)
    assert isinstance(exp_name, str)
