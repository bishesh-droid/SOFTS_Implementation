import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import argparse
import sys
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.softs_model import SOFTS
from src.data_processing.dataset import TimeSeriesDataset
from src.data_processing.transforms import ReversibleInstanceNormalization
from src.utils.metrics import MSE, MAE
from src.utils.helpers import set_seed

def train_model(config_path="configs/experiment_config.yaml"):
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Unpack config
    experiment_config = config['experiment']
    data_config = config['data']
    model_config = config['model']
    train_config = config['train']

    print(f"Starting Experiment: {experiment_config['name']}")
    
    # Reproducibility
    set_seed(experiment_config['seed'])

    device = torch.device(experiment_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    print(f"Loading dataset: {data_config['dataset_name']}")

    # Assuming target_cols are all features for multivariate forecasting
    # If target_cols is None, dataset infers it, but we need input_dim for model
    # We will instantiate dataset first to get dimensions
    
    full_dataset = TimeSeriesDataset(
        dataset_name=data_config['dataset_name'],
        seq_len=data_config['seq_len'],
        pred_len=data_config['pred_len'],
        target_cols=data_config['target_cols'],
        noise_level=data_config.get('noise_level', 0.0)
    )

    # Infer input_dim if possible from dataset sample
    sample_x, _ = full_dataset[0]
    input_dim = sample_x.shape[-1]
    print(f"Inferred input_dim: {input_dim}")
    
    # Update model config with inferred input_dim
    model_config['input_dim'] = input_dim

    # Split data
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)

    # --- Model, Optimizer, Loss ---
    print(f"Initializing Model: {model_config['name']} (SimpleCore: {model_config.get('use_simple_core', False)})")
    
    model = SOFTS(
        input_dim=model_config['input_dim'],
        seq_len=data_config['seq_len'], # model config usually matched but using data config ensures alignment
        pred_len=data_config['pred_len'],
        hidden_dim=model_config['hidden_dim'],
        core_dim=model_config['core_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        use_simple_core=model_config.get('use_simple_core', False)
    ).to(device)

    optimizer_cls = getattr(optim, train_config['optimizer'])
    optimizer = optimizer_cls(model.parameters(), lr=train_config['learning_rate'])
    criterion = nn.MSELoss()

    # Normalization layer
    norm_layer = ReversibleInstanceNormalization(model_config['input_dim']).to(device)

    # --- Training Loop ---
    save_dir = os.path.join(train_config['save_path'], experiment_config['name'], datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_epoch = -1

    print("Starting Training Loop...")
    
    for epoch in range(train_config['epochs']):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']} [Train]")
        
        for batch_x, batch_y in loop:
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
            
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
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

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} | Val RMSE: {np.sqrt(avg_val_mse):.4f}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            model_save_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': config
            }, model_save_path)
            print(f"--> Saved best model to {model_save_path}")

    print(f"Training complete. Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOFTS model")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml", help="Path to experiment config")
    args = parser.parse_args()
    
    train_model(args.config)