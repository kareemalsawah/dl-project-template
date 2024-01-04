"""
Registering and initializing loggers
"""
import os
import importlib
from typing import Any, Dict, TypeVar

LOGGERS = {}

T = TypeVar("T")


def register_logger(
    logger_name: str,
):
    """
    Decorator to add a logger to the LOGGERS dictionary

    Parameters
    ----------
    logger_name : str
        Name of the logger to add
    """

    def decorator(func: T) -> T:
        """
        Save the class in the LOGGERS dictionary

        Parameters
        ----------
        func : Any
            Class to create a logger

        Returns
        -------
        Same class as given in the parameter
        """
        LOGGERS[logger_name] = func
        return func

    return decorator


def init_logger(logger_name: str, *args, **kwargs) -> Any:
    """
    Initialize a logger from the LOGGERS dictionary

    Parameters
    ----------
    logger_name : str
        Name of the logger to initialize

    Returns
    -------
    Logger
    """
    if logger_name not in LOGGERS:
        raise ValueError(f"Logger {logger_name} not found in registered LOGGERS")
    return LOGGERS[logger_name](*args, **kwargs)


def import_loggers() -> None:
    """
    Import all loggers in the loggers directory
    """
    for file in os.listdir(os.path.dirname(__file__)):
        # check if file is directory
        if os.path.isdir(os.path.join(os.path.dirname(__file__), file)):
            # import all files in directory
            importlib.import_module(f".{file}", package="src.loggers")


import_loggers()
