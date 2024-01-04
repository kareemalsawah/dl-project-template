"""
Template for models unittests
"""
import unittest
import torch
import numpy as np

from src.loggers import init_logger
from src.models import init_model
from src.losses import init_loss
from src.datasets import init_dataset
from src.trainers import init_trainer


class TestDeepTrainer(unittest.TestCase):
    """
    Test DeepTrainer
    """

    def test_deep_trainer_full(self) -> None:
        """
        Test the deep trainer on a full example
        """
        np.random.seed(42)
        torch.manual_seed(42)

        def eval_model(model, dataloaders):
            data = dataloaders["test"].dataset.data
            preds = model.forward(data["observations"])
            accuracy = torch.mean(
                (preds.argmax(dim=1) == data["labels"].reshape(-1).long()).float()
            )
            return accuracy.item() * 100

        model = init_model("example_model", **{"model_size": 10})
        loss = init_loss("example_loss", **{})
        dataset_configs = {
            "n_train_samples": 128,
            "n_val_samples": 32,
            "n_test_samples": 256,
            "batch_size": 32,
        }
        dataloaders = init_dataset("gaussian_synthetic", **dataset_configs)
        logger_configs = {"config": {"model_params": "anything"}}
        logger = init_logger("print_logger", **logger_configs)
        trainer_params = {
            "lr": 0.001,
            "device": "cpu",
            "n_epochs": 100,
            "verbose": False,
            "val_freq": 50,
            "save_path": None,
        }

        configs = {
            "model": model,
            "dataloaders": dataloaders,
            "loss": loss,
            "logger": logger,
            "trainer_params": trainer_params,
        }
        trainer = init_trainer("deep_trainer", **configs)

        print(
            "Accuracy before training: {:.2f}%".format(eval_model(model, dataloaders))
        )
        trainer.run()
        trained_acc = eval_model(model, dataloaders)
        print("Accuracy after training: {:.2f}%".format(trained_acc))
        self.assertGreater(trained_acc, 95)


if __name__ == "__main__":
    unittest.main()
