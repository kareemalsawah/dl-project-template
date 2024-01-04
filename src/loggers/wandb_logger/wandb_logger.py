"""
Logger classes for training and evaluation
"""
from typing import Any, Dict

import wandb
from .. import register_logger


@register_logger("wandb_logger")
class WandbLogger:
    """
    Weights and Biases logger

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        project_name = config["logger_params"]["project_name"]
        entity = config["logger_params"]["entity"]
        self.run = wandb.init(project=project_name, entity=entity, config=config)

    def log_train(self, to_log: Dict[str, Any], epoch: int):
        """
        Log training metrics

        Parameters
        ----------
        to_log : Dict[str, Any]
            Dictionary of metrics to log
        epoch : int
            Current epoch
        """
        self.run.log(
            {"train": to_log},
        )

    def log_eval(self, to_log: Dict[str, Any], epoch: int):
        """
        Log validation metrics during training

        Parameters
        ----------
        to_log : Dict[str, Any]
            Dictionary of metrics to log
        epoch : int
            Current epoch
        """
        self.run.log(
            {"val": to_log},
        )

    def log_test(self, to_log: Dict[str, Any]) -> None:
        """
        Log test metrics after training is complete

        Parameters
        ----------
        to_log : Dict[str, Any]
            Dictionary of metrics to log
        """
        self.run.log(
            {"test": to_log},
        )
        self.run.finish()
