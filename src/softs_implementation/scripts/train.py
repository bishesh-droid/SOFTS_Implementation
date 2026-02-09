import argparse
import sys
import os

# Temporarily add the project root to sys.path to allow importing from src
# This is a fallback for direct script execution and should ideally be replaced by
# proper package installation or PYTHONPATH setup for more complex environments.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.train import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOFTS model")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml", help="Path to experiment config")
    args = parser.parse_args()
    
    train_model(args.config)