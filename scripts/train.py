import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
import numpy as np

# Assuming src is in the python path or relative import works
from src.models.softs_model import SOFTS
from src.data_processing.dataset import TimeSeriesDataset
from src.data_processing.transforms import ReversibleInstanceNormalization
from src.utils.metrics import MSE, MAE
from src.utils.helpers import set_seed

def train_model(config_path="configs/train_config.yaml", model_config_path="configs/model_config.yaml"):
    with open(config_path, 'r') as f:
        train_config = yaml.safe_load(f)['train']
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)['model']

    set_seed(train_config['seed'])

    device = torch.device(train_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    dataset_name = train_config['dataset_name']
    print(f"Loading dataset: {dataset_name}")

    # Assuming target_cols are all features for multivariate forecasting
    target_cols = [f'feature_{i}' for i in range(model_config['input_dim'])]

    # Split data (simple split for demonstration)
    full_dataset = TimeSeriesDataset(
        dataset_name=dataset_name,
        seq_len=model_config['seq_len'],
        pred_len=model_config['pred_len'],
        target_cols=target_cols
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)

    # --- Model, Optimizer, Loss ---
    model = SOFTS(
        input_dim=model_config['input_dim'],
        seq_len=model_config['seq_len'],
        pred_len=model_config['pred_len'],
        hidden_dim=model_config['hidden_dim'],
        core_dim=model_config['core_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)

    optimizer = getattr(optim, train_config['optimizer'])(model.parameters(), lr=train_config['learning_rate'])
    criterion = nn.MSELoss() # Using MSE as per paper

    # Normalization layer
    norm_layer = ReversibleInstanceNormalization(model_config['input_dim']).to(device)

    # --- Training Loop ---
    os.makedirs(train_config['save_path'], exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(train_config['epochs']):
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']} [Train]"):
            batch_x = batch_x.float().to(device) # (batch_size, seq_len, input_dim)
            batch_y = batch_y.float().to(device) # (batch_size, pred_len, input_dim)

            optimizer.zero_grad()

            # Apply normalization
            normalized_x, mean, std = norm_layer(batch_x)

            # Forward pass
            output = model(normalized_x) # (batch_size, pred_len, input_dim)

            # Inverse normalization for output to calculate loss on original scale
            output_denorm = norm_layer.inverse(output, mean, std)

            loss = criterion(output_denorm, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']} [Val]"):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                normalized_x, mean, std = norm_layer(batch_x)
                output = model(normalized_x)
                output_denorm = norm_layer.inverse(output, mean, std)

                loss = criterion(output_denorm, batch_y)
                val_loss += loss.item()

                val_mae += MAE(output_denorm, batch_y).item()
                val_mse += MSE(output_denorm, batch_y).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} | Val MSE: {avg_val_mse:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(train_config['save_path'], f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

    print("Training complete.")

if __name__ == "__main__":
    train_model()