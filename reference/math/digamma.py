import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs the digamma function.
    """

    def __init__(self):
        """
        Initialize the DigammaModel.
        """
        super(Model, self).__init__()

    def forward(self, x):
        """
        Forward pass, computing the digamma function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying digamma.
        """
        return torch.digamma(x)

# Define input dimensions and parameters
batch_size = 1024
input_shape = (1024,)

def get_inputs():
    # digamma requires positive input to avoid NaN/Inf at 0
    return [torch.rand(batch_size, *input_shape) + 0.1]

def get_init_inputs():
    return []