"""
Print Logger for training and evaluation
"""
from typing import Any, Dict
import yaml

from .. import register_logger


@register_logger("print_logger")
class PrintLogger:
    """
    Simple logger that prints to console

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        print("Training Configuration:")
        print(yaml.dump(config, allow_unicode=True, default_flow_style=False))

    def log_train(self, *args, **kwargs) -> None:
        """
        Print nothing during training
        """

    def log_eval(self, to_log: Dict[str, Any], epoch: int) -> None:
        """
        Print evaluation metrics during training

        Parameters
        ----------
        to_log : Dict[str, Any]
            Dictionary of metrics to log
        epoch : int
            Current epoch
        """
        print(f"Validation Metrics, Epoch: {epoch}")
        print(yaml.dump(to_log, allow_unicode=True, default_flow_style=False))

    def log_test(self, to_log: Dict[str, Any]) -> None:
        """
        Print test metrics after training is complete

        Parameters
        ----------
        to_log : Dict[str, Any]
            Dictionary of metrics to log
        """
        print("-" * 20)
        print("Yay, Training Completed")
        print("-" * 20)
        print("Test Metrics:")
        print(yaml.dump(to_log, allow_unicode=True, default_flow_style=False))
