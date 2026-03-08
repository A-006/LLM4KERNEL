import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs the pow function.
    """

    def __init__(self):
        """
        Initialize the PowModel.
        """
        super(Model, self).__init__()

    def forward(self, input, exponent):
        """
        Forward pass, computing the pow function.

        Args:
            input (torch.Tensor): Base tensor.
            exponent (torch.Tensor): Exponent tensor.

        Returns:
            torch.Tensor: Tensor after applying pow.
        """
        return torch.pow(input, exponent)

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)

def get_inputs():
    # pow takes two tensors. Add epsilon to base to avoid 0^0 issues if exponent is also near 0.
    return [
        torch.rand(batch_size, *input_shape) + 1e-6, 
        torch.rand(batch_size, *input_shape)
    ]

def get_init_inputs():
    return []