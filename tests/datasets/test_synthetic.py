"""
Template for dataset unittests
"""
import unittest
from src.datasets import init_dataset


class TestSyntheticDataset(unittest.TestCase):
    """
    Test gaussian_synthetic dataset
    """

    def test_gaussian_synthetic(self) -> None:
        """
        Simple shape test
        """
        configs = {
            "n_train_samples": 128,
            "n_val_samples": 32,
            "n_test_samples": 32,
            "batch_size": 16,
        }
        dataloaders = init_dataset("gaussian_synthetic", **configs)

        self.assertEqual(len(dataloaders["train"]), 8)
        self.assertEqual(len(dataloaders["val"]), 2)
        self.assertEqual(len(dataloaders["test"]), 2)

        data = next(iter(dataloaders["train"]))
        self.assertEqual(data["observations"].shape, (16, 2))
        self.assertEqual(data["labels"].shape, (16, 1))


if __name__ == "__main__":
    unittest.main()
