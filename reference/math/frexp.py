import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that decomposes elements into mantissa and exponent.
    """

    def __init__(self):
        """
        Initialize the FrexpModel.
        """
        super(Model, self).__init__()

    def forward(self, x):
        """
        Forward pass, computing frexp.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Mantissa (returning first element of tuple to match Tensor type hint).
        """
        mantissa, exponent = torch.frexp(x)
        return mantissa

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)

def get_inputs():
    return [torch.rand(batch_size, *input_shape) + 0.1]

def get_init_inputs():
    return []