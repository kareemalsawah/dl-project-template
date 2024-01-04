"""
Registering and initializing losses
"""
import os
import importlib
from typing import Any, Dict, TypeVar

LOSSES = {}

T = TypeVar("T")


def register_loss(
    loss_name: str,
):
    """
    Decorator to add a loss to the LOSSES dictionary

    Parameters
    ----------
    loss_name : str
        Name of the loss to add
    """

    def decorator(func: T) -> T:
        """
        Save the class in the LOSSES dictionary

        Parameters
        ----------
        func : Any
            Class to create a loss

        Returns
        -------
        Same class as given in the parameter
        """
        LOSSES[loss_name] = func
        return func

    return decorator


def init_loss(loss_name: str, *args, **kwargs) -> Any:
    """
    Initialize a loss from the LOSSES dictionary

    Parameters
    ----------
    loss_name : str
        Name of the loss to initialize

    Returns
    -------
    Initialized loss
    """
    if loss_name not in LOSSES:
        raise ValueError(f"loss {loss_name} not found in registered LOSSES")
    return LOSSES[loss_name](*args, **kwargs)


def import_losses() -> None:
    """
    Import all losses in the losses directory
    """
    for file in os.listdir(os.path.dirname(__file__)):
        # check if file is directory
        if os.path.isdir(os.path.join(os.path.dirname(__file__), file)):
            # import all files in directory
            importlib.import_module(f".{file}", package="src.losses")


import_losses()
