# Deep Learning Template Project

## Project Structure
```
├── src  (for adding to this, see the Extending section below)
│   ├── datasets
│   ├── loggers
│   ├── losses
│   ├── models
│   ├── trainers
├── configs (yaml configs for any experiements ran)
│   ├── model_class
│   │   ├── dataset_name
│   │   │   ├── config.yaml
├── data (any dataset files)
├── models (any saved model weights)
├── notebooks (any notebooks for experiements for visualization)
├── tests
│   ├── datasets
│   ├── loggers
│   ├── losses
│   ├── models
│   ├── trainers
└── requirements.txt
```
#### How to run
```
pip install -r requirements.txt
python main.py "configs/model_class/dataset_name/config_name.yaml"
```

## Extending
### Datasets
To create a new dataset:
- Create a new src/datasets/your_dataset_name directory
- Create a file.py in the directory (see synthetic_dataloaders.py for an example)
- In the directory, also add a __init__.py with one line from .file import *
- In that file, add from .. import register_dataset
- Create a function with a decorator @register_dataset("dataset_name")
- This function can have any parameters (you have to set them in the config.yaml file)
- The function should return a dictionary with 3 dataloaders at the keys "train", "val", and "test"

### Loggers
Two basic loggers are provided:
- print_logger: used for printing metrics during training using stdout
- wandb_logger: used for logging to wandb. To use it you need to login as follows:
```
wandb login
```
This will prompt you to enter you API key; you can find it on your account on wandb. Note: this only needs to be done once for each machine. To use the wandb logger, make sure the config has "wand_logger" as the logger name.

To create a new logger:
- Create a new src/loggers/your_logger_name directory
- Create a file.py in the directory
- In the directory, also add a __init__.py with one line from .file import *
- In the file add from .. import register_logger
- Create a class with a decorator @register_logger("logger_name")
- The class init should take as input the config.yaml file parsed as a dictionary
- If you need any parameters, read them directory from the dictionary
- It should implement an interface similar to the print logger

### Losses
To create a new loss:
- Create a new src/losses/your_loss_name directory
- Create a file.py in the directory (see example_loss.py for an example)
- In the directory, also add a __init__.py with one line from .file import *
- In that file, add from .. import register_loss
- Create a class with a decorator @register_loss("loss_name")
- This class can have any parameters in the init (you have to set them in the config.yaml file)
- The class can implement any functions as long as it works with the model and trainer used with it
Note: some utils have been provided in utils.py

### Models
To create a new model:
- Create a new src/models/model_class directory
- Create a file.py in the directory (see example_model.py for an example)
- In the directory, also add a __init__.py with one line from .file import *
- In that file, add from .. import register_model
- Create a class with a decorator @register_model("model_name")
- This class can have any parameters in the init (you have to set them in the config.yaml file)
- The class can implement any functions as long as it works with the loss and trainer used with it

### Trainers
To create a new trainer:
- Create a new src/trainer/trainer_name directory
- Create a file.py in the directory (see deep_trainer.py for an example)
- In the directory, also add a __init__.py with one line from .file import *
- In that file, add from .. import register_trainer
- Create a class with a decorator @register_trainer("trainer_name")
- This class needs to have the same init interface as deep_trainer (takes model, loss, dataloader, logger, and trainer_params which can be set in the config.yaml)
- The class also need to implement a .run() function