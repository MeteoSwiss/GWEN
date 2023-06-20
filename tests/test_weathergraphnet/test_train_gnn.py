"""Test the GNNModel class in the weathergraphnet.train_gnn module."""

# Third-party
import torch
from torch_geometric.data import Data  # type: ignore

# First-party
from weathergraphnet.train_gnn import GNNModel  # type: ignore


def test_gnn_model():
    """
    Test the GNN model.

    This function tests the GNNModel class from the weathergraphnet.train_gnn module. It
    creates a GNNModel instance with the specified input and output channels, and a
    random input data tensor. Then it passes the input data through the model and checks
    if the output tensor has the expected shape.

    Returns:
        None

    """
    in_channels = 2
    out_channels = 1
    model = GNNModel(in_channels, out_channels)
    data = Data(
        x=torch.randn((in_channels, 1)),  # pylint: disable=no-member
        edge_index=torch.tensor(  # pylint: disable=no-member
            [[0, 1]], dtype=torch.long  # pylint: disable=no-member
        ),
    )
    output = model(data)
    assert output.shape == (out_channels, 1)
