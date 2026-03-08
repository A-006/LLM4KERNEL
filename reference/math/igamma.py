import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs the incomplete gamma function.
    """

    def __init__(self):
        """
        Initialize the IgammaModel.
        """
        super(Model, self).__init__()

    def forward(self, a, x):
        """
        Forward pass, computing the igamma function.

        Args:
            a (torch.Tensor): Input tensor a.
            x (torch.Tensor): Input tensor x.

        Returns:
            torch.Tensor: Tensor after applying igamma.
        """
        return torch.igamma(a, x)

# Define input dimensions and parameters
batch_size = 1024
input_shape = (1024,)

def get_inputs():
    # igamma requires positive inputs
    return [
        torch.rand(batch_size, *input_shape) + 0.1, 
        torch.rand(batch_size, *input_shape) + 0.1
    ]

def get_init_inputs():
    return []