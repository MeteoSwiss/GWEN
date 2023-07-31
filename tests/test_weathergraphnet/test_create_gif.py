import os

import pytest
import xarray as xr
from weathergraphnet.create_gif import create_animation


@pytest.fixture
def input_file():
    return "data/data_combined.zarr"


@pytest.fixture
def var_name():
    return "theta_v"


def test_create_animation(input_file, var_name):
    """Test the create_animation function."""
    create_animation(input_file, var_name)
    output_dir = f"output/{var_name}"
    assert os.path.exists(output_dir)
    ds = xr.open_zarr(input_file)
    for member in ds.member.values:
        output_filename = f"{output_dir}/animation_member_{member}.gif"
        assert os.path.exists(output_filename)
