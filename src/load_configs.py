"""
Initialize all objects from config
"""
from typing import Dict, List

import torch
from .datasets import init_dataset, DATASETS
from .models import init_model, MODELS
from .losses import init_loss, LOSSES
from .loggers import init_logger, LOGGERS
from .trainers import init_trainer, TRAINERS


def init_from_config(config: dict) -> dict:
    """
    Initialize all objects from config

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        Dictionary of initialized objects
        Contains: dataloaders, model, loss, logger, trainer
    """
    dataloaders = init_dataset(config["dataset_name"], **config["dataset_params"])

    model = init_model(config["model_name"], **config["model_params"])
    # Load model checkpoint if path is provided
    if config["load_model_path"] is not None:
        try:
            model.load_state_dict(torch.load(config["load_model_path"]))
        except FileNotFoundError as file_not_found:  # if path not found
            raise FileNotFoundError(
                f"Model path {config['load_model_path']} not found."
            ) from file_not_found
        except RuntimeError as runtime_error:  # if model architecture doesn't match
            raise RuntimeError(
                f"Model architecture does not match checkpoint at {config['load_model_path']}."
            ) from runtime_error

    loss = init_loss(config["loss_name"], **config["loss_params"])

    logger = init_logger(config["logger_name"], config)

    trainer_name = config["trainer_name"]
    trainer_params = config["trainer_params"]
    trainer = init_trainer(
        trainer_name, model, dataloaders, loss, logger, trainer_params
    )

    to_return = {
        "dataloaders": dataloaders,
        "model": model,
        "loss": loss,
        "logger": logger,
        "trainer": trainer,
    }
    return to_return


def list_registered_objects() -> Dict[str, List[str]]:
    """
    List all registered objects

    Returns
    -------
    Dict[str, List[str]]
        Dictionary of registered objects
        Contains: datasets, models, losses, loggers, trainers
    """
    to_return = {
        "datasets": list(DATASETS.keys()),
        "models": list(MODELS.keys()),
        "losses": list(LOSSES.keys()),
        "loggers": list(LOGGERS.keys()),
        "trainers": list(TRAINERS.keys()),
    }
    return to_return
