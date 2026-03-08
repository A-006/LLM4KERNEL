import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs torch.select operation.
    """

    def __init__(self, dim, index):
        """
        Initialize the Model.

        Args:
            dim (int): The dimension to select from.
            index (int): The index to select.
        """
        super(Model, self).__init__()
        self.dim = dim
        self.index = index

    def forward(self, x):
        """
        Forward pass, selecting a slice.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Selected slice (dimension reduced).
        """
        return torch.select(x, dim=self.dim, index=self.index)

def get_inputs():
    # x: (2, 3, 4)
    return [torch.rand(2, 3, 4)]

def get_init_inputs():
    # Select index 0 along dim 1
    return [1, 0]