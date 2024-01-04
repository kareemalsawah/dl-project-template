"""
Dataloader for the synthetic datasets in dmm_datasets.py
"""
from typing import Dict
from torch.utils.data import DataLoader
from .synthetic_example import SyntheticExample
from .. import register_dataset


@register_dataset("gaussian_synthetic")
def get_synthetic_dataloaders(
    n_train_samples: int,
    n_val_samples: int,
    n_test_samples: int,
    batch_size: int,
) -> Dict[str, DataLoader]:
    """
    General function to get dataloaders for GSSM datasets

    Parameters
    ----------
    n_train_samples : int
        Number of training samples
    n_val_samples : int
        Number of validation samples
    n_test_samples : int
        Number of test samples
    batch_size : int
        Batch size

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary of dataloaders for train, val, and test sets
    """
    train_dataset = SyntheticExample(n_train_samples)
    val_dataset = SyntheticExample(n_val_samples)
    test_dataset = SyntheticExample(n_test_samples)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}
