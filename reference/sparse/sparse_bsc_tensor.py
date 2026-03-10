import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that constructs a BSC sparse tensor.
    """
    def __init__(self, shape: tuple, blocksize: tuple):
        """
        Initializes the BSC sparse tensor model.

        Args:
            shape (tuple): The shape of the sparse tensor.
            blocksize (tuple): The size of the dense blocks (H, W).
        """
        super(Model, self).__init__()
        self.shape = shape
        self.blocksize = blocksize
    
    def forward(self, ccol_indices: torch.Tensor, row_indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Constructs a sparse BSC tensor from components.

        Args:
            ccol_indices (torch.Tensor): Compressed column indices (block level).
            row_indices (torch.Tensor): Row indices (block level).
            values (torch.Tensor): Value tensor (flattened blocks).

        Returns:
            torch.Tensor: Sparse BSC tensor.
        """
        return torch.sparse_bsc_tensor(ccol_indices, row_indices, values, self.blocksize, self.shape)

# Adjusted dimensions for sparse tensor safety
rows = 100
cols = 100
block_h = 2
block_w = 2
# Shape must be divisible by blocksize
shape_rows = (rows // block_h) * block_h
shape_cols = (cols // block_w) * block_w
n_blocks_rows = shape_rows // block_h
n_blocks_cols = shape_cols // block_w

def get_inputs():
    # Generate valid BSC components at block level
    block_counts = torch.randint(0, 3, (n_blocks_cols,))
    ccol_indices = torch.cat([torch.tensor([0]), torch.cumsum(block_counts, dim=0)], dim=0)
    actual_nnz_blocks = ccol_indices[-1].item()
    if actual_nnz_blocks == 0:
        actual_nnz_blocks = 1
        ccol_indices[-1] = 1

    row_indices = torch.randint(0, n_blocks_rows, (actual_nnz_blocks,))
    # Values shape: (nnz_blocks, block_h, block_w)
    values = torch.rand(actual_nnz_blocks, block_h, block_w)
    return [ccol_indices, row_indices, values]

def get_init_inputs():
    return [(shape_rows, shape_cols), (block_h, block_w)]  # Provide shape and blocksize