"""
Registering and initializing models
"""
import os
import importlib
from typing import Any, Dict, TypeVar

MODELS = {}

T = TypeVar("T")


def register_model(
    model_name: str,
):
    """
    Decorator to add a model to the MODELS dictionary

    Parameters
    ----------
    model_name : str
        Name of the model to add
    """

    def decorator(func: T) -> T:
        """
        Save the class in the MODELS dictionary

        Parameters
        ----------
        func : Any
            Class to create a model

        Returns
        -------
        Same class as given in the parameter
        """
        MODELS[model_name] = func
        return func

    return decorator


def init_model(model_name: str, *args, **kwargs) -> Any:
    """
    Initialize a model from the MODELS dictionary

    Parameters
    ----------
    model_name : str
        Name of the model to initialize

    Returns
    -------
    Initialized model
    """
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in registered MODELS")
    return MODELS[model_name](*args, **kwargs)


def import_models() -> None:
    """
    Import all models in the models directory
    """
    for file in os.listdir(os.path.dirname(__file__)):
        # check if file is directory
        if os.path.isdir(os.path.join(os.path.dirname(__file__), file)):
            # import all files in directory
            importlib.import_module(f".{file}", package="src.models")


import_models()
