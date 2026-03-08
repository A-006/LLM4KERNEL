import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs torch.swapdims operation.
    """

    def __init__(self, dim1, dim2):
        """
        Initialize the Model.

        Args:
            dim1 (int): First dimension.
            dim2 (int): Second dimension.
        """
        super(Model, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        """
        Forward pass, swapping dimensions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with swapped dimensions.
        """
        return torch.swapdims(x, dim1=self.dim1, dim2=self.dim2)

def get_inputs():
    return [torch.rand(2, 3, 4)]

def get_init_inputs():
    # Swap dim 0 and 2
    return [0, 2]