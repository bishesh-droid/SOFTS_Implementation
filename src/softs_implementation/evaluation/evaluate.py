import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
import numpy as np

from softs_implementation.models.softs_model import SOFTS
from softs_implementation.data_processing.dataset import TimeSeriesDataset
from softs_implementation.data_processing.transforms import ReversibleInstanceNormalization
from softs_implementation.utils.metrics import MSE, MAE
from softs_implementation.utils.helpers import set_seed

def evaluate_model(config_path="configs/train_config.yaml", model_config_path="configs/model_config.yaml", checkpoint_path=None, data_config_path=None):
    with open(config_path, 'r') as f:
        train_config = yaml.safe_load(f)['train']
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)['model']

    if data_config_path:
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)['data']
    else:
        data_config = train_config

    set_seed(train_config['seed'])

    device = torch.device(train_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    dataset_name = data_config['dataset_name']
    print(f"Loading dataset: {dataset_name}")

    target_cols = [f'feature_{i}' for i in range(model_config['input_dim'])]

    # For evaluation, we typically use a dedicated test set.
    # Here, we'll just use the full dataset and assume a split for simplicity.
    full_dataset = TimeSeriesDataset(
        dataset_name=dataset_name,
        seq_len=model_config['seq_len'],
        pred_len=model_config['pred_len'],
        target_cols=target_cols
    )
    # Simple split: 80% train, 20% test
    train_size = int(0.8 * len(full_dataset))
    test_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))


    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)

    # --- Model Loading ---
    model = SOFTS(
        input_dim=model_config['input_dim'],
        seq_len=model_config['seq_len'],
        pred_len=model_config['pred_len'],
        hidden_dim=model_config['hidden_dim'],
        core_dim=model_config['core_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No checkpoint path provided. Evaluating untrained model or randomly initialized model.")

    norm_layer = ReversibleInstanceNormalization(model_config['input_dim']).to(device)

    # --- Evaluation Loop ---
    model.eval()
    total_mae = 0
    total_mse = 0
    num_batches = 0

    predictions_list = []
    targets_list = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Evaluating"):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            normalized_x, mean, std = norm_layer(batch_x)
            output = model(normalized_x)
            output_denorm = norm_layer.inverse(output, mean, std)

            total_mae += MAE(output_denorm, batch_y).item()
            total_mse += MSE(output_denorm, batch_y).item()
            num_batches += 1

            predictions_list.append(output_denorm.cpu().numpy())
            targets_list.append(batch_y.cpu().numpy())

    avg_mae = total_mae / num_batches
    avg_mse = total_mse / num_batches

    print(f"--- Evaluation Results ---")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"--------------------------")

    # Optionally save predictions and targets for further analysis
    all_predictions = np.concatenate(predictions_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)
    # np.save("predictions.npy", all_predictions)
    # np.save("targets.npy", all_targets)
    # print("Saved predictions and targets to predictions.npy and targets.npy")