import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that returns the imaginary part of a complex tensor.
    """

    def __init__(self):
        """
        Initialize the ImagModel.
        """
        super(Model, self).__init__()

    def forward(self, x):
        """
        Forward pass, extracting imaginary part.

        Args:
            x (torch.Tensor): Complex input tensor.

        Returns:
            torch.Tensor: Real tensor containing imaginary parts.
        """
        return torch.imag(x)

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)

def get_inputs():
    # imag requires complex input
    return [torch.rand(batch_size, *input_shape, dtype=torch.cfloat)]

def get_init_inputs():
    return []