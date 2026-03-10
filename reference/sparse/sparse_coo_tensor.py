import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that constructs a COO sparse tensor.
    """
    def __init__(self, shape: tuple):
        """
        Initializes the COO sparse tensor model.

        Args:
            shape (tuple): The shape of the sparse tensor.
        """
        super(Model, self).__init__()
        self.shape = shape
    
    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Constructs a sparse COO tensor from indices and values.

        Args:
            indices (torch.Tensor): Index tensor of shape (2, nnz).
            values (torch.Tensor): Value tensor of shape (nnz,).

        Returns:
            torch.Tensor: Sparse COO tensor.
        """
        return torch.sparse_coo_tensor(indices, values, self.shape)

# Adjusted dimensions for sparse tensor safety
rows = 100
cols = 100
nnz = 200

def get_inputs():
    # Generate random unique indices
    indices = torch.randint(0, rows, (2, nnz))
    values = torch.rand(nnz)
    return [indices, values]

def get_init_inputs():
    return [(rows, cols)]  # Provide shape for initialization