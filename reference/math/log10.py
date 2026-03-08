import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs the log10 function.
    """

    def __init__(self):
        """
        Initialize the Log10Model.
        """
        super(Model, self).__init__()

    def forward(self, x):
        """
        Forward pass, computing the log10 function.

        Args:
            x (torch.Tensor): Input tensor with values > 0.

        Returns:
            torch.Tensor: Tensor after applying log10.
        """
        return torch.log10(x)

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)

def get_inputs():
    # log10 requires input > 0. torch.rand can be 0, so add epsilon.
    return [torch.rand(batch_size, *input_shape) + 1e-6]

def get_init_inputs():
    return []