"""Tests for the models module."""
# Standard library
import unittest
from unittest import mock

# Third-party
import torch
from torch import Tensor

# First-party
from gwen.models_cnn import EvaluationConfigCNN
from gwen.models_cnn import TrainingConfigCNN
from gwen.models_cnn import UNet
from gwen.models_gnn import EvaluationConfigGNN
from gwen.models_gnn import GNNConfig
from gwen.models_gnn import GNNModel
from gwen.models_gnn import TrainingConfigGNN


class TestGNNModel(unittest.TestCase):
    def setUp(self):
        configs_train = {
            "loader_train_in": [torch.randn(10, 5) for _ in range(3)],
            "loader_train_out": [torch.randn(10, 1) for _ in range(3)],
            "optimizer": mock.MagicMock(),
            "scheduler": None,
            "loss_fn": torch.nn.MSELoss(),
            "mask": Tensor | None,
            "epochs": 10,
            "device": "cpu",
            "seed": 42,
        }

        configs_gnn = {
            "nodes_in": 10,
            "nodes_out": 10,
            "channels_in": 10,
            "channels_out": 10,
            "hidden_feats": 10,
        }

        configs_test = {
            "loader_in": [torch.randn(10, 5) for _ in range(3)],
            "loader_out": [torch.randn(10, 1) for _ in range(3)],
            "loss_fn": torch.nn.MSELoss(),
            "mask": Tensor | None,
            "device": "cpu",
            "seed": 42,
        }

        self.configs_train = TrainingConfigGNN(**configs_train)
        self.configs_test = EvaluationConfigGNN(**configs_test)
        self.configs_gnn = GNNConfig(**configs_gnn)

    def test_train_with_configs(self):
        model = GNNModel(self.configs_gnn)
        model.conv_layers = mock.MagicMock()
        model.pool_layer = mock.MagicMock()
        model.pool_layer.return_value = (
            torch.randn(10, 5),
            torch.tensor([0, 1, 1, 2, 3, 0, 3, 2, 1, 0]),
        )
        model.activation = torch.nn.ReLU()

        model.train_with_configs(self.configs_train)

        self.assertEqual(model.conv_layers.call_count, 6)
        self.assertEqual(model.pool_layer.call_count, 6)
        self.assertEqual(self.configs_train.optimizer.zero_grad.call_count, 18)
        self.assertEqual(self.configs_train.optimizer.step.call_count, 18)

    def test_eval_with_configs(self):
        model = GNNModel(self.configs_gnn)
        model.conv_layers = mock.MagicMock()
        model.pool_layer = mock.MagicMock()
        model.pool_layer.return_value = (
            torch.randn(10, 5),
            torch.tensor([0, 1, 1, 2, 3, 0, 3, 2, 1, 0]),
        )
        model.activation = torch.nn.ReLU()

        self.configs_test.loader_in = [torch.randn(10, 5) for _ in range(3)]
        self.configs_test.loader_out = [torch.randn(10, 1) for _ in range(3)]

        loss, y_preds = model.eval_with_configs(self.configs_test)

        self.assertIsInstance(loss, float)
        self.assertEqual(len(y_preds), 3)
        self.assertIsInstance(y_preds[0], torch.Tensor)


class TestUNet(unittest.TestCase):
    # HACK test with different input shapes
    def setUp(self):
        self.channels_in = 100
        self.channels_out = 25
        self.hidden_size = 1024
        self.batch_size = 13
        self.input_shape = (self.batch_size, self.channels_in, 256, 256)
        self.output_shape = (self.batch_size, self.channels_out, 256, 256)
        self.x = torch.randn(self.input_shape)

    def test_forward_pass(self):
        model = UNet(self.channels_in, self.channels_out, self.hidden_size)
        with mock.patch.object(model, "encoder") as mock_encoder, mock.patch.object(
            model, "decoder"
        ) as mock_decoder:
            mock_encoder.return_value = (
                torch.randn(self.batch_size, self.hidden_size, 128, 128),
                torch.randn(self.batch_size, self.hidden_size * 2, 64, 64),
                torch.randn(self.batch_size, self.hidden_size * 4, 32, 32),
                torch.randn(self.batch_size, self.hidden_size * 8, 16, 16),
            )
            mock_decoder.return_value = torch.randn(self.output_shape)
            output = model(self.x)
            self.assertEqual(output.shape, self.output_shape)

    def test_train_with_configs(self):
        model = UNet(self.channels_in, self.channels_out, self.hidden_size)
        configs = {
            "epochs": 2,
            "dataset": [
                (torch.randn(self.input_shape), torch.randn(self.output_shape))
            ],
            "device": "cpu",
            "optimizer": torch.optim.Adam(model.parameters()),
            "scheduler": None,
            "loss_fn": torch.nn.MSELoss(),
            "mask": None,
            "seed": 42,
        }
        configs_train = TrainingConfigCNN(**configs)
        model.train_with_configs(0, configs_train, 1)

    def test_eval_with_configs(self):
        model = UNet(self.channels_in, self.channels_out, self.hidden_size)
        configs = {
            "dataset": [
                (torch.randn(self.input_shape), torch.randn(self.output_shape))
            ],
            "device": "cpu",
            "loss_fn": torch.nn.MSELoss(),
            "mask": None,
            "seed": 42,
        }
        configs_test = EvaluationConfigCNN(**configs)
        loss, y_preds = model.eval_with_configs(configs_test)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(y_preds, list)
        self.assertEqual(len(y_preds), 1)
        self.assertEqual(y_preds[0].shape, self.output_shape)
