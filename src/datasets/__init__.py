"""
Registering and initializing datasets
"""
import os
import importlib
from typing import Any, Callable, Dict, TypeVar

from torch.utils.data import DataLoader

DATASETS = {}

T = TypeVar("T", bound=Callable[..., Dict[str, DataLoader]])


def register_dataset(
    dataset_name: str,
) -> Callable[[T], T]:
    """
    Decorator to add a dataset to the DATASETS dictionary

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to add
    """

    def decorator(func: T) -> T:
        """
        Save the function in the DATASETS dictionary

        Parameters
        ----------
        func : Callable[[Any], Dict[str, Dataloader]]
            Function to create a dictionary of dataloaders

        Returns
        -------
        Same function as given in the parameter
        """
        DATASETS[dataset_name] = func
        return func

    return decorator


def init_dataset(dataset_name: str, *args, **kwargs) -> Dict[str, DataLoader]:
    """
    Initialize a dataset from the DATASETS dictionary

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to initialize

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary of dataloaders for train, val, and test sets
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found in registered datasets")
    return DATASETS[dataset_name](*args, **kwargs)


def import_datasets() -> None:
    """
    Import all datasets in the datasets directory
    """
    for file in os.listdir(os.path.dirname(__file__)):
        # check if file is directory
        if os.path.isdir(os.path.join(os.path.dirname(__file__), file)):
            # import all files in directory
            importlib.import_module(f".{file}", package="src.datasets")


import_datasets()
