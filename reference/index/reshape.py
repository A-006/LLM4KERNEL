import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs torch.reshape operation.
    """

    def __init__(self, shape):
        """
        Initialize the Model.

        Args:
            shape (tuple): The new shape.
        """
        super(Model, self).__init__()
        self.shape = shape

    def forward(self, x):
        """
        Forward pass, reshaping the tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Reshaped tensor.
        """
        return torch.reshape(x, self.shape)

def get_inputs():
    # Total elements 2*3*4 = 24. Reshape to (6, 4)
    return [torch.rand(2, 3, 4)]

def get_init_inputs():
    return [(6, 4)]