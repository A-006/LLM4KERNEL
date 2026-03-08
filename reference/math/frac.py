import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes the fractional part of the input.
    """

    def __init__(self):
        """
        Initialize the FracModel.
        """
        super(Model, self).__init__()

    def forward(self, x):
        """
        Forward pass, computing fractional part.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing fractional parts.
        """
        return torch.frac(x)

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)

def get_inputs():
    # Add integer component to make frac meaningful
    return [torch.rand(batch_size, *input_shape) * 10.0]

def get_init_inputs():
    return []