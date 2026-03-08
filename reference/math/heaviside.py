import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs the heaviside function.
    """

    def __init__(self):
        """
        Initialize the HeavisideModel.
        """
        super(Model, self).__init__()

    def forward(self, input, values):
        """
        Forward pass, computing the heaviside function.

        Args:
            input (torch.Tensor): Input tensor.
            values (torch.Tensor): Values tensor for input == 0.

        Returns:
            torch.Tensor: Tensor after applying heaviside.
        """
        return torch.heaviside(input, values)

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)

def get_inputs():
    # heaviside takes two tensors (input and values)
    return [
        torch.rand(batch_size, *input_shape), 
        torch.rand(batch_size, *input_shape)
    ]

def get_init_inputs():
    return []