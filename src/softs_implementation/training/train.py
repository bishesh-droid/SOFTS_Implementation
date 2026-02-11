import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime
import warnings

from softs_implementation.models.softs_model import SOFTS
from softs_implementation.data_processing.dataset import TimeSeriesDataset
from softs_implementation.data_processing.transforms import ReversibleInstanceNormalization
from softs_implementation.utils.metrics import MSE, MAE, RMSE
from softs_implementation.utils.helpers import set_seed

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
        use_simple_core=model_config.get('use_simple_core', False),
        use_layer_norm=model_config.get('use_layer_norm', False)
    ).to(device)

    optimizer_cls = getattr(optim, train_config['optimizer'])
    optimizer = optimizer_cls(model.parameters(), lr=train_config['learning_rate'])
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = None
    if 'scheduler' in train_config and train_config['scheduler']:
        scheduler_name = train_config['scheduler']
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_config['epochs'],
                eta_min=train_config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=train_config.get('step_size', 10),
                gamma=train_config.get('gamma', 0.1)
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=train_config.get('factor', 0.5),
                patience=train_config.get('lr_patience', 5),
                min_lr=train_config.get('min_lr', 1e-6)
            )
        print(f"Using {scheduler_name} scheduler")
    
    # Gradient clipping threshold
    gradient_clip = train_config.get('gradient_clip', None)
    if gradient_clip:
        print(f"Gradient clipping enabled with threshold: {gradient_clip}")
    
    # Early stopping parameters
    early_stopping_patience = train_config.get('early_stopping_patience', None)
    if early_stopping_patience:
        print(f"Early stopping enabled with patience: {early_stopping_patience}")

    # Normalization layer
    norm_layer = ReversibleInstanceNormalization(model_config['input_dim']).to(device)

    # --- Training Loop ---
    save_dir = os.path.join(train_config['save_path'], experiment_config['name'], datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    training_history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [], 'learning_rate': []}

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
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf detected in loss at epoch {epoch+1}. Skipping batch.")
                continue
            
            loss.backward()
            
            # Gradient clipping
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
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
        avg_val_rmse = np.sqrt(avg_val_mse)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics in history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_mae'].append(avg_val_mae)
        training_history['val_rmse'].append(avg_val_rmse)
        training_history['learning_rate'].append(current_lr)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} | Val RMSE: {avg_val_rmse:.4f} | LR: {current_lr:.2e}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            model_save_path = os.path.join(save_dir, "best_model.pth")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': config,
                'training_history': training_history
            }
            
            # Save scheduler state if available
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, model_save_path)
            print(f"--> Saved best model to {model_save_path}")
        else:
            epochs_without_improvement += 1
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {epochs_without_improvement} epochs)")
            break

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    
    # Save final training history
    history_path = os.path.join(save_dir, "training_history.npy")
    np.save(history_path, training_history)
    print(f"Saved training history to {history_path}")