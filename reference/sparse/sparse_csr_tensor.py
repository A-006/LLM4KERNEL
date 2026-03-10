import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that constructs a CSR sparse tensor.
    """
    def __init__(self, shape: tuple):
        """
        Initializes the CSR sparse tensor model.

        Args:
            shape (tuple): The shape of the sparse tensor.
        """
        super(Model, self).__init__()
        self.shape = shape
    
    def forward(self, crow_indices: torch.Tensor, col_indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Constructs a sparse CSR tensor from components.

        Args:
            crow_indices (torch.Tensor): Compressed row indices.
            col_indices (torch.Tensor): Column indices.
            values (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Sparse CSR tensor.
        """
        return torch.sparse_csr_tensor(crow_indices, col_indices, values, self.shape)

# Adjusted dimensions for sparse tensor safety
rows = 100
cols = 100
nnz = 200

def get_inputs():
    # Generate valid CSR components
    # crow_indices must be sorted and start with 0
    row_counts = torch.randint(0, 5, (rows,))
    crow_indices = torch.cat([torch.tensor([0]), torch.cumsum(row_counts, dim=0)], dim=0)
    # Ensure nnz matches
    actual_nnz = crow_indices[-1].item()
    if actual_nnz == 0: 
        actual_nnz = 1 # Ensure at least one element for valid tensor
        crow_indices[-1] = 1
        
    col_indices = torch.randint(0, cols, (actual_nnz,))
    values = torch.rand(actual_nnz)
    return [crow_indices, col_indices, values]

def get_init_inputs():
    return [(rows, cols)]  # Provide shape for initialization