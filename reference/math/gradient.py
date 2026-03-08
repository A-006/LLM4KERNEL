import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes the gradient of a tensor.
    """

    def __init__(self):
        """
        Initialize the GradientModel.
        """
        super(Model, self).__init__()

    def forward(self, x):
        """
        Forward pass, computing gradient.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Gradient tensor (returning first dim gradient to match Tensor type hint).
        """
        # torch.gradient returns a tuple of tensors. 
        # We return the first one to comply with forward -> torch.Tensor hint.
        grads = torch.gradient(x)
        return grads[0]

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []