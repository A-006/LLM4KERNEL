import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs the complementary incomplete gamma function.
    """

    def __init__(self):
        """
        Initialize the IgammacModel.
        """
        super(Model, self).__init__()

    def forward(self, a, x):
        """
        Forward pass, computing the igammac function.

        Args:
            a (torch.Tensor): Input tensor a.
            x (torch.Tensor): Input tensor x.

        Returns:
            torch.Tensor: Tensor after applying igammac.
        """
        return torch.igammac(a, x)

# Define input dimensions and parameters
batch_size = 1024
input_shape = (1024,)

def get_inputs():
    # igammac requires positive inputs
    return [
        torch.rand(batch_size, *input_shape) + 0.1, 
        torch.rand(batch_size, *input_shape) + 0.1
    ]

def get_init_inputs():
    return []