import argparse
import yaml
import os
import torch

from softs_implementation.scripts.train import train_model
from softs_implementation.scripts.evaluate import evaluate_model
from softs_implementation.data.scripts.download_data import download_all_datasets

def main():
    parser = argparse.ArgumentParser(description="Run SOFTS model training or evaluation.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'download_data'],
                        help='Mode to run: train, evaluate, or download_data.')
    parser.add_argument('--train_config', type=str, default='configs/train_config.yaml',
                        help='Path to the training configuration file.')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml',
                        help='Path to the model configuration file.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for evaluation.')

    args = parser.parse_args()

    if args.mode == 'download_data':
        print("Downloading/preparing data...")
        download_all_datasets()
        print("Data download/preparation complete.")
    elif args.mode == 'train':
        print("Starting model training...")
        train_model(args.train_config, args.model_config)
        print("Model training finished.")
    elif args.mode == 'evaluate':
        print("Starting model evaluation...")
        evaluate_model(args.train_config, args.model_config, args.checkpoint)
        print("Model evaluation finished.")

if __name__ == '__main__':
    main()