import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs the polygamma function.
    """

    def __init__(self, n):
        """
        Initialize the PolygammaModel.

        Args:
            n (int): Order of the derivative.
        """
        super(Model, self).__init__()
        self.n = n

    def forward(self, x):
        """
        Forward pass, computing the polygamma function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying polygamma.
        """
        return torch.polygamma(self.n, x)

# Define input dimensions and parameters
batch_size = 1024
input_shape = (1024,)
n = 1

def get_inputs():
    # polygamma requires positive input for stability
    return [torch.rand(batch_size, *input_shape) + 0.1]

def get_init_inputs():
    return [n]