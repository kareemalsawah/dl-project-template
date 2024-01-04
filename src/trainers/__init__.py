"""
Registering and initializing trainers
"""
import os
import importlib
from typing import Any, Dict, TypeVar

TRAINERS = {}

T = TypeVar("T")


def register_trainer(
    trainer_name: str,
):
    """
    Decorator to add a trainer to the TRAINERS dictionary

    Parameters
    ----------
    trainer_name : str
        Name of the trainer to add
    """

    def decorator(func: T) -> T:
        """
        Save the class in the TRAINERS dictionary

        Parameters
        ----------
        func : Any
            Class to create a trainer

        Returns
        -------
        Same class as given in the parameter
        """
        TRAINERS[trainer_name] = func
        return func

    return decorator


def init_trainer(trainer_name: str, *args, **kwargs) -> Any:
    """
    Initialize a trainer from the TRAINERS dictionary

    Parameters
    ----------
    trainer_name : str
        Name of the trainer to initialize

    Returns
    -------
    Initialized trainer
    """
    if trainer_name not in TRAINERS:
        raise ValueError(f"Trainer {trainer_name} not found in registered TRAINERS")
    return TRAINERS[trainer_name](*args, **kwargs)


def import_trainers() -> None:
    """
    Import all trainers in the trainers directory
    """
    for file in os.listdir(os.path.dirname(__file__)):
        # check if file is directory
        if os.path.isdir(os.path.join(os.path.dirname(__file__), file)):
            # import all files in directory
            importlib.import_module(f".{file}", package="src.trainers")


import_trainers()
