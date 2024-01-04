"""
Base Synthetic Dataset
"""
from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset


class SyntheticBase(Dataset):
    """
    Base Synthetic Dataset

    Parameters
    ----------
    n_samples: int
        Number of samples to generate
    """

    def __init__(
        self,
        n_samples: int = 5000,
    ):
        self.n_samples = n_samples

        self.data = self.generate_data(n_samples)

    def generate_data(self, n_samples: int) -> Dict[str, torch.Tensor]:
        """
        Generate data samples

        Parameters
        ----------
        n_samples: int
            Number of samples to generate

        Returns
        -------
        Generated data
        """
        raise NotImplementedError

    def plot(self, n_samples: int = 5, figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        Plot a few samples from the dataset

        Parameters
        ----------
        n_samples : int, by default 5
            Number of samples to plot
        figsize : Tuple[int, int], by default (12, 4)
            Figure size
        """
        raise NotImplementedError

    def __len__(self):
        keys = list(self.data.keys())
        return len(self.data[keys[0]])

    def __getitem__(self, idx):
        return {
            "observations": self.data["observations"][idx],
            "labels": self.data["labels"][idx],
        }
