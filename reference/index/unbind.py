import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs torch.unbind operation.
    """

    def __init__(self, dim):
        """
        Initialize the Model.

        Args:
            dim (int): The dimension to unbind.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass, unbinding the tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple of Tensors: Unbound tensors.
        """
        return torch.unbind(x, dim=self.dim)

def get_inputs():
    # x: (2, 3, 4)
    return [torch.rand(2, 3, 4)]

def get_init_inputs():
    return [0]