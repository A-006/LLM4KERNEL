import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs the log-gamma function.
    """

    def __init__(self):
        """
        Initialize the LgammaModel.
        """
        super(Model, self).__init__()

    def forward(self, x):
        """
        Forward pass, computing the lgamma function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying lgamma.
        """
        return torch.lgamma(x)

# Define input dimensions and parameters
batch_size = 1024
input_shape = (1024,)

def get_inputs():
    # lgamma requires positive input for stability
    return [torch.rand(batch_size, *input_shape) + 0.1]

def get_init_inputs():
    return []