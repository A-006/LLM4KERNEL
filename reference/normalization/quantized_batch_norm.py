import torch
import torch.nn as nn
import torch.nn.quantized.functional as Fq

class Model(nn.Module):
    """
    A model that performs torch.nn.quantized.functional.batch_norm operation.
    """

    def __init__(self, eps, momentum, scale, zero_point):
        """
        Initialize the Model.

        Args:
            eps (float): A value added to the denominator for numerical stability.
            momentum (float): The value used for the running_mean and running_var computation.
            scale (float): Output quantization scale.
            zero_point (int): Output quantization zero_point.
        """
        super(Model, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input, running_mean, running_var, weight, bias):
        """
        Forward pass, applying quantized batch normalization.

        Args:
            input (torch.Tensor): Quantized input tensor.
            running_mean (torch.Tensor): Running mean (float).
            running_var (torch.Tensor): Running variance (float).
            weight (torch.Tensor): Quantized weight.
            bias (torch.Tensor): Quantized bias.

        Returns:
            torch.Tensor: Quantized output tensor.
        """
        return Fq.batch_norm(
            input, running_mean, running_var, weight, bias,
            eps=self.eps, momentum=self.momentum,
            scale=self.scale, zero_point=self.zero_point
        )

def get_inputs():
    # Prepare quantized tensors for input, weight, and bias
    # Input shape: (N, C, H, W) = (1, 2, 4, 4)
    input_float = torch.rand(1, 2, 4, 4)
    input_quant = torch.quantize_per_tensor(input_float, 0.1, 0, torch.quint8)
    
    # Weight and Bias shape: (C,) = (2,)
    weight_float = torch.rand(2)
    weight_quant = torch.quantize_per_tensor(weight_float, 0.1, 0, torch.quint8)
    
    bias_float = torch.rand(2)
    bias_quant = torch.quantize_per_tensor(bias_float, 0.1, 0, torch.quint8)
    
    # Running mean and var are float tensors
    running_mean = torch.zeros(2)
    running_var = torch.ones(2)
    
    return [input_quant, running_mean, running_var, weight_quant, bias_quant]

def get_init_inputs():
    return [1e-5, 0.1, 0.1, 0]