import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs the multivariate log-gamma function.
    """

    def __init__(self, p):
        """
        Initialize the MvlgammaModel.

        Args:
            p (int): The number of dimensions.
        """
        super(Model, self).__init__()
        self.p = p

    def forward(self, x):
        """
        Forward pass, computing the mvlgamma function.

        Args:
            x (torch.Tensor): Input tensor where last dim >= p.

        Returns:
            torch.Tensor: Tensor after applying mvlgamma.
        """
        return torch.mvlgamma(x, p=self.p)

# Define input dimensions and parameters
batch_size = 1024
# Last dimension must be >= p (set to 3)
input_shape = (1024, 5) 
p = 3

def get_inputs():
    # mvlgamma requires positive input
    return [torch.rand(batch_size, *input_shape) + 0.1]

def get_init_inputs():
    return [p]