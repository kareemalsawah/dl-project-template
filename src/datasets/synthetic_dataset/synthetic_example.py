"""
Example Synthetic Dataset with two classes
"""
from typing import Tuple, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from .synthetic_base import SyntheticBase


class SyntheticExample(SyntheticBase):
    """
    Synthetic Example Dataset
    """

    def generate_data(self, n_samples: int) -> Dict[str, torch.Tensor]:
        """
        Generate data samples.
        Currently, just a 2 class gaussian

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        Returns
        -------
        Dictionary of generated data
        Contains observations and labels
        """
        class_0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
        class_1 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
        observations = np.vstack([class_0, class_1])
        labels = np.concatenate(
            [np.zeros(n_samples // 2), np.ones(n_samples // 2)]
        ).reshape(-1, 1)

        observations = torch.from_numpy(observations).float()
        labels = torch.from_numpy(labels).float()

        # Shuffle
        perm = torch.randperm(n_samples)
        observations = observations[perm]
        labels = labels[perm]

        return {"observations": observations, "labels": labels}

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

        sample_idx = np.random.choice(len(self.data), n_samples, replace=False)

        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.scatter(self.data[sample_idx, 0], self.data[sample_idx, 1])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Synthetic Example")
        plt.show()
