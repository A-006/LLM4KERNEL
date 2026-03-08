import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs torch.slice_scatter operation.
    """

    def __init__(self, dim, start, end, step=1):
        """
        Initialize the Model.

        Args:
            dim (int): The dimension along which to slice.
            start (int): The start index.
            end (int): The end index.
            step (int): The step size.
        """
        super(Model, self).__init__()
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

    def forward(self, input, src):
        """
        Forward pass, scattering src into input along a slice.

        Args:
            input (torch.Tensor): The base tensor.
            src (torch.Tensor): The tensor to scatter.

        Returns:
            torch.Tensor: Output tensor.
        """
        return torch.slice_scatter(input, src, dim=self.dim, start=self.start, end=self.end, step=self.step)

def get_inputs():
    # input: (2, 4, 4), src must match the slice shape. 
    # If dim=1, start=1, end=3, slice is size 2. So src: (2, 2, 4)
    input_tensor = torch.rand(2, 4, 4)
    src_tensor = torch.rand(2, 2, 4)
    return [input_tensor, src_tensor]

def get_init_inputs():
    return [1, 1, 3, 1]