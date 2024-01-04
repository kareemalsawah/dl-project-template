"""
Template for models unittests
"""
import unittest
import torch
from src.models import init_model


class TestExampleModel(unittest.TestCase):
    """
    Test example model
    """

    def test_example_model(self) -> None:
        """
        Test the example model
        """
        configs = {"model_size": 20}
        model = init_model("example_model", **configs)
        out = model(torch.ones(10, 2))

        self.assertEqual(out.shape, (10, 2))


if __name__ == "__main__":
    unittest.main()
