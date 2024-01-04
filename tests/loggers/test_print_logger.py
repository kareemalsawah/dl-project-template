"""
Template for logger unittests
"""
import sys
import unittest

from io import StringIO
from src.loggers import init_logger


class TestPrintLogger(unittest.TestCase):
    """
    Test print logger
    """

    def test_print_logger(self) -> None:
        """
        Print logger test
        """
        configs = {"config": {"model_params": "anything"}}

        captured_output = StringIO()
        sys.stdout = captured_output
        print_logger = init_logger("print_logger", **configs)
        print_logger.log_train()
        print_logger.log_eval({"metric": 0.5}, 1)
        print_logger.log_test({"metric": 0.5})

        init_expected_output = "Training Configuration:\nmodel_params: anything\n\n"
        train_expected_output = ""
        eval_expected_output = "Validation Metrics, Epoch: 1\nmetric: 0.5\n\n"
        test_expected_output = (
            "-" * 20
            + "\nYay, Training Completed\n"
            + "-" * 20
            + "\nTest Metrics:\nmetric: 0.5\n\n"
        )
        self.assertEqual(
            captured_output.getvalue(),
            init_expected_output
            + train_expected_output
            + eval_expected_output
            + test_expected_output,
        )

        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    unittest.main()
