import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs the modified Bessel function of order 0.
    """

    def __init__(self):
        """
        Initialize the I0Model.
        """
        super(Model, self).__init__()

    def forward(self, x):
        """
        Forward pass, computing the i0 function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying i0.
        """
        return torch.i0(x)

# Define input dimensions and parameters
batch_size = 1024
input_shape = (1024,)

def get_inputs():
    # i0 is defined for all real numbers, rand is safe
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []