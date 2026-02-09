import argparse
import sys
import os

# Temporarily add the project root to sys.path to allow importing from src
# This is a fallback for direct script execution and should ideally be replaced by
# proper package installation or PYTHONPATH setup for more complex environments.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation.evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SOFTS model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to training config")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    
    evaluate_model(args.config, args.model_config, args.checkpoint)