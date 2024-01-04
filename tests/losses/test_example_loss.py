"""
Template for losses unittests
"""
import unittest
import torch
from torch import nn
from src.losses import init_loss


class TestExampleLoss(unittest.TestCase):
    """
    Test example loss
    """

    def test_example_loss_forward(self) -> None:
        """
        Test the NLLLoss
        """
        configs = {}
        loss_obj = init_loss("example_loss", **configs)
        model = nn.Linear(2, 2)
        model.weight.data = torch.zeros(model.weight.data.shape)
        model.bias.data = torch.ones(model.bias.data.shape)
        data = {
            "observations": torch.randn(10, 2),
            "labels": torch.ones((10,)).long(),
        }

        loss, to_log = loss_obj(model, data, epoch=6)

        self.assertEqual(
            loss.item(),
            nn.NLLLoss()(torch.ones(10, 2), torch.ones((10,)).long()).item(),
        )
        self.assertEqual(to_log["loss"], loss.item())
        self.assertEqual(to_log["epoch"], 6)


if __name__ == "__main__":
    unittest.main()
