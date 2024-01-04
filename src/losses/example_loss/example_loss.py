"""
Example Loss Class
"""
from typing import Tuple, Dict, Any
import torch
from torch import nn

from .. import register_loss


@register_loss("example_loss")
class ExampleLoss(nn.Module):
    """
    Example Loss Class
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.NLLLoss()

    def forward(
        self, model: nn.Module, data: Dict[str, torch.Tensor], epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass of the loss

        Parameters
        ----------
        model: nn.Module
            Model to train, should return logits in forward
        data: Dict[str, torch.Tensor]
            Dictionary of data, should contain observations and labels

        Returns
        -------
        torch.Tensor
            Loss value, shape=(1,)
        """
        y_pred = model.forward(data["observations"])
        loss = self.criterion(y_pred, data["labels"].reshape(-1).long())
        to_log = {"loss": loss.item(), "epoch": epoch}
        return loss, to_log
