import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that constructs a CSC sparse tensor.
    """
    def __init__(self, shape: tuple):
        """
        Initializes the CSC sparse tensor model.

        Args:
            shape (tuple): The shape of the sparse tensor.
        """
        super(Model, self).__init__()
        self.shape = shape
    
    def forward(self, ccol_indices: torch.Tensor, row_indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Constructs a sparse CSC tensor from components.

        Args:
            ccol_indices (torch.Tensor): Compressed column indices.
            row_indices (torch.Tensor): Row indices.
            values (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Sparse CSC tensor.
        """
        return torch.sparse_csc_tensor(ccol_indices, row_indices, values, self.shape)

# Adjusted dimensions for sparse tensor safety
rows = 100
cols = 100

def get_inputs():
    # Generate valid CSC components
    col_counts = torch.randint(0, 5, (cols,))
    ccol_indices = torch.cat([torch.tensor([0]), torch.cumsum(col_counts, dim=0)], dim=0)
    actual_nnz = ccol_indices[-1].item()
    if actual_nnz == 0:
        actual_nnz = 1
        ccol_indices[-1] = 1

    row_indices = torch.randint(0, rows, (actual_nnz,))
    values = torch.rand(actual_nnz)
    return [ccol_indices, row_indices, values]

def get_init_inputs():
    return [(rows, cols)]  # Provide shape for initialization