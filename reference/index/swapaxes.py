import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs torch.swapaxes operation.
    """

    def __init__(self, axis1, axis2):
        """
        Initialize the Model.

        Args:
            axis1 (int): First axis.
            axis2 (int): Second axis.
        """
        super(Model, self).__init__()
        self.axis1 = axis1
        self.axis2 = axis2

    def forward(self, x):
        """
        Forward pass, swapping axes.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with swapped axes.
        """
        return torch.swapaxes(x, axis1=self.axis1, axis2=self.axis2)

def get_inputs():
    return [torch.rand(2, 3, 4)]

def get_init_inputs():
    # Swap axis 0 and 2
    return [0, 2]