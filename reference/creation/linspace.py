import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that constructs a linearly spaced tensor.
    """
    def __init__(self, steps: int = 100):
        """
        Initializes the linspace model.

        Args:
            steps (int, optional): The number of steps to generate. Defaults to 100.
        """
        super(Model, self).__init__()
        self.steps = steps
    
    def forward(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """
        Constructs a tensor of evenly spaced values over a specified interval.

        Args:
            start (torch.Tensor): The starting value of the sequence (0-dim tensor).
            end (torch.Tensor): The ending value of the sequence (0-dim tensor).

        Returns:
            torch.Tensor: 1D tensor of shape (steps,).
        """
        return torch.linspace(start, end, steps=self.steps)

# Configuration for linspace
steps_default = 100

def get_inputs():
    # 0-dim tensors for start and end values
    start = torch.tensor(0.0)
    end = torch.tensor(10.0)
    return [start, end]

def get_init_inputs():
    return [steps_default]  # Provide steps value for initialization