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

    # Define the input and output channel sizes
    in_channels = 10
    out_channels = 10

    # Create an instance of the GNNModel class
    model = GNNModel(in_channels, out_channels)

    # Create some dummy input data
    data = Data(
        x=torch.randn((in_channels, 1)),  # pylint: disable=no-member
        edge_index=torch.tensor(  # pylint: disable=no-member
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]],
            dtype=torch.long,  # pylint: disable=no-member
        ),
    )

    # Pass the input data through the model
    output = model(data)

    # Check that the output has the expected shape
    assert output.shape == (out_channels, 1)
