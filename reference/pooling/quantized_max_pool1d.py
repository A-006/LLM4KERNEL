import torch
import torch.nn as nn
import torch.nn.quantized.functional as Fq

class Model(nn.Module):
    """
    A model that performs torch.nn.quantized.functional.max_pool1d operation.
    """

    def __init__(self, kernel_size, stride, padding, dilation):
        """
        Initialize the Model.

        Args:
            kernel_size (int or tuple): Size of the sliding window.
            stride (int or tuple): Stride of the sliding window.
            padding (int or tuple): Zero padding added to both sides of the input.
            dilation (int or tuple): Spacing between kernel elements.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        """
        Forward pass, applying quantized max pooling 1D.

        Args:
            x (torch.Tensor): Quantized input tensor of shape (N, C, L).

        Returns:
            torch.Tensor: Quantized output tensor.
        """
        return Fq.max_pool1d(
            x, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, dilation=self.dilation
        )

def get_inputs():
    # Input shape: (N, C, L) = (1, 1, 8)
    input_float = torch.rand(1, 1, 8)
    # Quantize the input tensor
    input_quant = torch.quantize_per_tensor(input_float, 0.1, 0, torch.quint8)
    return [input_quant]

def get_init_inputs():
    # Kernel 2, Stride 2, Padding 0, Dilation 1
    return [2, 2, 0, 1]