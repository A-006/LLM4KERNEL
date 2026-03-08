import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs torch.scatter_reduce operation.
    """

    def __init__(self, dim, reduce):
        """
        Initialize the Model.

        Args:
            dim (int): The dimension along which to index.
            reduce (str): The reduce operation ('sum', 'prod', 'mean', 'amin', 'amax').
        """
        super(Model, self).__init__()
        self.dim = dim
        self.reduce = reduce

    def forward(self, input, index, src):
        """
        Forward pass, reducing values from src to input at indices.

        Args:
            input (torch.Tensor): The base tensor.
            index (torch.Tensor): The indices tensor.
            src (torch.Tensor): The source tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return torch.scatter_reduce(input, dim=self.dim, index=index, src=src, reduce=self.reduce)

def get_inputs():
    # Similar to scatter_add
    input_tensor = torch.rand(2, 3)
    index_tensor = torch.randint(0, 3, (2, 3))
    src_tensor = torch.rand(2, 3)
    return [input_tensor, index_tensor, src_tensor]

def get_init_inputs():
    # reduce options: 'sum', 'prod', 'mean', 'amin', 'amax'
    return [1, 'sum']