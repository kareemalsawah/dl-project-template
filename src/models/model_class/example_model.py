"""
Example Model Class
"""
import torch
from torch import nn

from .. import register_model


@register_model("example_model")
class ExampleModel(nn.Module):
    """
    Example Model Class

    Parameters
    ----------
    model_size : int
        Size of the model
    """

    def __init__(self, model_size: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(2, model_size),
            nn.ReLU(),
            nn.Linear(model_size, model_size),
            nn.ReLU(),
            nn.Linear(model_size, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor, shape=(batch_size, 2)
        """
        return self.fc(x)
