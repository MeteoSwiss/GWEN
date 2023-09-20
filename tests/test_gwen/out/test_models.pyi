# Standard library
import unittest

# Third-party
from _typeshed import Incomplete as Incomplete

class TestGNNModel(unittest.TestCase):
    configs_train: Incomplete
    configs_test: Incomplete
    configs_gnn: Incomplete
    def setUp(self) -> None: ...
    def test_train_with_configs(self) -> None: ...
    def test_eval_with_configs(self) -> None: ...

class TestUNet(unittest.TestCase):
    channels_in: int
    channels_out: int
    hidden_size: int
    batch_size: int
    input_shape: Incomplete
    output_shape: Incomplete
    x: Incomplete
    def setUp(self) -> None: ...
    def test_forward_pass(self) -> None: ...
    def test_train_with_configs(self) -> None: ...
    def test_eval_with_configs(self) -> None: ...
