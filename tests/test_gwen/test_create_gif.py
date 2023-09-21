"""Tests for the create_gif module."""

# pylint: disable=redefined-outer-name

# Standard library
import glob
import os
from typing import Any

# Third-party
import matplotlib.pyplot as plt
import pytest
import xarray as xr
from matplotlib import animation
from pyprojroot import here

# First-party
from gwen.create_gif import create_animation_object
from gwen.create_gif import create_update_function
from gwen.create_gif import get_member_name
from gwen.create_gif import main
from gwen.create_gif import save_animation
from gwen.loggers_configs import load_config

config = load_config()


@pytest.fixture
def input_file() -> str:
    return str(here()) + (
        "/tests/test_data/atmcirc-straka_93_-10."
        "0_3000.0_2000.0_DOM01_ML_20080801T000000Z.nc"
    )


@pytest.fixture
def var_name():
    return "theta_v"


@pytest.fixture
def member_name():
    return "Temp: -10 °C; Height: 3000 m; Width: 2000 m"


@pytest.fixture
def var(input_file: Any, var_name: str = "theta_v"):
    ds = xr.open_dataset(input_file)
    return ds[var_name].isel(member=0)


@pytest.fixture
def fig():
    return plt.figure()


@pytest.fixture
def ax(fig):
    return fig.add_subplot(111)


@pytest.fixture
def im(ax: Any, var: Any):
    return ax.imshow(var.isel(time=0))


def test_get_member_name(
    input_file: Any, member_name: str = "Temp: -10 °C; Height: 3000 m; Width: 2000 m"
):
    assert get_member_name(input_file) == member_name


def test_create_update_function(
    im: Any,
    var: Any,
    member_name: str = "Temp: -10 °C; Height: 3000 m; Width: 20000 m",
    var_name: str = "theta_v",
):
    update_func = create_update_function(im, var, member_name, var_name)
    assert callable(update_func)


def test_create_animation_object(fig, var, im, member_name, var_name):
    num_frames = len(var.time)
    update_func = create_update_function(im, var, member_name, var_name)
    ani = create_animation_object(fig, update_func, num_frames)
    assert isinstance(ani, animation.FuncAnimation)


# pylint: disable=too-many-arguments
def test_save_animation(tmp_path, im, var, member_name, var_name, fig):
    output_filename = os.path.join(tmp_path, "animation_member_test.gif")
    update_func = create_update_function(im, var, member_name, var_name)
    ani = create_animation_object(fig, update_func, num_frames=len(var.time))
    save_animation(ani, output_filename)
    assert os.path.exists(output_filename)


def test_main(tmp_path, input_file, var_name):
    output_dir = os.path.join(tmp_path, var_name)
    main(
        input_file,
        var_name,
        output_dir,
    )
    assert os.path.exists(output_dir)
    output_files = glob.glob(os.path.join(output_dir, "*.gif"))
    assert len(output_files) > 0
