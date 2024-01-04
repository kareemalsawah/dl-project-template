"""
Main trainer class for training all deep learning models
"""
import os
from typing import Dict, Any
from tqdm import tqdm

import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

from .. import register_trainer


@register_trainer("deep_trainer")
class DeepTrainer:
    """
    Trainer class for training models

    Parameters
    ----------
    model: nn.Module
        Model to train
    dataloaders: Dict[str, DataLoader]
        Dictionary of dataloaders for train, val, and test sets
    loss: nn.Module
        Loss function to use
        forward should take as input: model output, data, and epoch
    logger: Logger
        Used to log metrics during training
    trainer_params: Dict[str, Any]
        Dictionary of parameters for the trainer
        Should contain lr, n_epochs, val_freq, device, save_path, verbose
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        loss: nn.Module,
        logger: Any,
        trainer_params: Dict[str, Any],
    ):
        self.model = model
        self.dataloaders = dataloaders
        self.loss = loss
        self.logger = logger
        self.params = trainer_params

    def run(self):
        """
        Run training and validation loop
        Save the model at the end
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])
        device = self.params["device"]

        self.model.to(device)

        self.model.train()
        for epoch in tqdm(
            range(self.params["n_epochs"]),
            disable=not self.params["verbose"],
            leave=False,
        ):
            for data in self.dataloaders["train"]:
                for key in data.keys():
                    data[key] = data[key].to(device)

                loss, to_log = self.loss(self.model, data, epoch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.logger.log_train(to_log, epoch)

            if epoch % self.params["val_freq"] == 0:
                self.model.eval()
                to_log = self.run_eval(self.dataloaders["val"], epoch=epoch)
                self.logger.log_eval(to_log, epoch)
                self.model.train()

        # Eval on test set
        self.model.eval()
        to_log = self.run_eval(self.dataloaders["test"], epoch=self.params["n_epochs"])
        self.logger.log_test(to_log)

        # Save model
        if self.params["save_path"] is not None:
            self.save_run_results(to_log)

    def save_run_results(self, test_metrics: Dict[str, float]) -> None:
        """
        Save run results to disk
        Saves model.pth, config.yaml, and metrics.txt

        Parameters
        ----------
        test_metrics : Dict[str, float]
            Dictionary of test metrics
        """
        # Create folder if not exists
        os.makedirs(os.path.dirname(self.params["save_path"]), exist_ok=True)

        # Save config yaml
        with open(
            os.path.join(self.params["save_path"], "config.yaml"), "w", encoding="utf-8"
        ) as config_file:
            config_file.write(
                yaml.dump(
                    self.logger.config, allow_unicode=True, default_flow_style=False
                )
            )

        # Save model
        torch.save(
            self.model.state_dict(), os.path.join(self.params["save_path"], "model.pt")
        )

        # Save metrics as txt
        with open(
            os.path.join(self.params["save_path"], "test_metrics.txt"),
            "w",
            encoding="utf-8",
        ) as metric_file:
            metric_file.write(
                yaml.dump(test_metrics, allow_unicode=True, default_flow_style=False)
            )

    def run_eval(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        Run evaluation loop

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader to use for evaluation
        epoch : int
            Current epoch, used in loss function

        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        device = self.params["device"]
        with torch.no_grad():
            eval_metrics = {}
            for data in dataloader:
                for key in data.keys():
                    data[key] = data[key].to(device)

                _, to_log = self.loss(self.model, data, epoch)
                for key, value in to_log.items():
                    if key not in eval_metrics:
                        eval_metrics[key] = []

                    eval_metrics[key].append(value)

            for key in eval_metrics:
                eval_metrics[key] = sum(eval_metrics[key]) / len(eval_metrics[key])

        return eval_metrics
