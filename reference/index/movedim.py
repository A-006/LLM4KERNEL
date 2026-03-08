import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs torch.movedim operation.
    """

    def __init__(self, source, destination):
        """
        Initialize the Model.

        Args:
            source (int or sequence of ints): The dimension(s) to move.
            destination (int or sequence of ints): The destination position(s).
        """
        super(Model, self).__init__()
        self.source = source
        self.destination = destination

    def forward(self, x):
        """
        Forward pass, moving dimensions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with moved dimensions.
        """
        return torch.movedim(x, source=self.source, destination=self.destination)

def get_inputs():
    return [torch.rand(2, 3, 4)]

def get_init_inputs():
    # Move dim 0 to position 2
    return [0, 2]