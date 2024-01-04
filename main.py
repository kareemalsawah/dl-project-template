"""
Main Training Script
"""
import argparse
import yaml

from src import init_from_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("config", type=str, help="Path to the config yaml file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    initialized_objects = init_from_config(config)
    initialized_objects["trainer"].run()
