import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that constructs a tensor filled with a specific value.
    """
    def __init__(self, fill_value: float = 1.0):
        """
        Initializes the full_like model.

        Args:
            fill_value (float, optional): The value to fill the tensor with. Defaults to 1.0.
        """
        super(Model, self).__init__()
        self.fill_value = fill_value
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Constructs a tensor filled with fill_value having the same size as input.

        Args:
            input (torch.Tensor): Input tensor defining the shape and dtype.

        Returns:
            torch.Tensor: Tensor filled with fill_value, same shape as input.
        """
        return torch.full_like(input, self.fill_value)

# Configuration for full_like
dim = 1024
batch_size = 64

def get_inputs():
    # Input tensor to mimic shape
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return [5.0]  # Provide fill_value for initialization