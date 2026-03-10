import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that constructs a logarithmically spaced tensor.
    """
    def __init__(self, steps: int = 100, base: float = 10.0):
        """
        Initializes the logspace model.

        Args:
            steps (int, optional): The number of steps to generate. Defaults to 100.
            base (float, optional): The base of the logarithm. Defaults to 10.0.
        """
        super(Model, self).__init__()
        self.steps = steps
        self.base = base
    
    def forward(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """
        Constructs a tensor of values spaced logarithmically on a given base.

        Args:
            start (torch.Tensor): The starting value of the sequence (0-dim tensor).
            end (torch.Tensor): The ending value of the sequence (0-dim tensor).

        Returns:
            torch.Tensor: 1D tensor of shape (steps,).
        """
        return torch.logspace(start, end, steps=self.steps, base=self.base)

# Configuration for logspace
steps_default = 100
base_default = 10.0

def get_inputs():
    # 0-dim tensors for start and end exponents (base^start to base^end)
    start = torch.tensor(0.0)
    end = torch.tensor(2.0)
    return [start, end]

def get_init_inputs():
    return [steps_default, base_default]  # Provide steps and base for initialization