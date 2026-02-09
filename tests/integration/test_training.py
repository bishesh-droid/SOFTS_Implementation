import pytest
import os
import torch
import yaml
from softs_implementation.training.train import train_model
from softs_implementation.evaluation.evaluate import evaluate_model
from softs_implementation.data.scripts.download_data import download_ett_data

# Define paths relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints_test') # Use a separate dir for test checkpoints

@pytest.fixture(scope="module", autouse=True)
def setup_data_and_configs():
    """
    Ensures dummy data and test configs are present before running tests.
    """
    # Create dummy data
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    dummy_data_path = os.path.join(DATA_RAW_DIR, "ETTh1.csv")
    if not os.path.exists(dummy_data_path):
        download_ett_data()

    # Create test config file
    test_experiment_config_path = os.path.join(CONFIGS_DIR, "test_experiment_config.yaml")

    # Unified experiment config for testing
    test_experiment_config_content = {
        'experiment': {
            'name': 'test_experiment',
            'seed': 42,
            'device': "cpu"
        },
        'data': {
            'dataset_name': 'ETTh1',
            'seq_len': 24,
            'pred_len': 12,
            'target_cols': None, # Let dataset infer
            'noise_level': 0.0
        },
        'model': {
            'name': 'SOFTS',
            'input_dim': 7, # Will be inferred by train_model, but good to have a default
            'output_dim': 7,
            'seq_len': 24,
            'pred_len': 12,
            'hidden_dim': 16,
            'core_dim': 8,
            'num_layers': 1,
            'dropout': 0.0,
            'use_simple_core': False
        },
        'train': {
            'epochs': 1, # Only 1 epoch for quick test
            'batch_size': 4,
            'learning_rate': 0.001,
            'optimizer': "Adam",
            'loss_function': "MSE",
            'scheduler': "CosineAnnealingLR",
            'device': "cpu", # Force CPU for CI/CD
            'seed': 42,
            'save_path': CHECKPOINTS_DIR,
            'log_interval': 1
        }
    }
    with open(test_experiment_config_path, 'w') as f:
        yaml.safe_dump(test_experiment_config_content, f)

    yield

    # Teardown: clean up generated files
    if os.path.exists(dummy_data_path):
        os.remove(dummy_data_path)
    if os.path.exists(test_experiment_config_path):
        os.remove(test_experiment_config_path)
    if os.path.exists(CHECKPOINTS_DIR):
        for f in os.listdir(CHECKPOINTS_DIR):
            os.remove(os.path.join(CHECKPOINTS_DIR, f))
        os.rmdir(CHECKPOINTS_DIR)

def test_training_run():
    """
    Tests if the training script runs without errors for a minimal configuration.
    Checks if a model checkpoint is saved.
    """
    test_experiment_config_path = os.path.join(CONFIGS_DIR, "test_experiment_config.yaml")

    # Run training
    train_model(test_experiment_config_path)

    # Check if a checkpoint was saved
    checkpoints = os.listdir(CHECKPOINTS_DIR)
    assert len(checkpoints) > 0, "No model checkpoint was saved during training."
    assert any(f.endswith('.pth') for f in checkpoints), "Saved file is not a .pth checkpoint."

def test_evaluation_run():
    """
    Tests if the evaluation script runs without errors using a saved checkpoint.
    """
    test_experiment_config_path = os.path.join(CONFIGS_DIR, "test_experiment_config.yaml")

    # Ensure a model is trained and saved first
    train_model(test_experiment_config_path)
    checkpoints = os.listdir(CHECKPOINTS_DIR)
    latest_checkpoint = sorted([os.path.join(CHECKPOINTS_DIR, f) for f in checkpoints if f.endswith('.pth')], key=os.path.getmtime, reverse=True)[0]

    # Load the experiment config to extract train and model config parts for evaluate_model
    with open(test_experiment_config_path, 'r') as f:
        experiment_config = yaml.safe_load(f)
    
    # Create temporary config files for evaluate_model
    temp_train_config_path = os.path.join(CONFIGS_DIR, "temp_eval_train_config.yaml")
    temp_model_config_path = os.path.join(CONFIGS_DIR, "temp_eval_model_config.yaml")

    with open(temp_train_config_path, 'w') as f:
        yaml.safe_dump({'train': experiment_config['train']}, f)
    with open(temp_model_config_path, 'w') as f:
        yaml.safe_dump({'model': experiment_config['model']}, f)

    # Run evaluation
    evaluate_model(temp_train_config_path, temp_model_config_path, latest_checkpoint)

    # Clean up temporary config files
    os.remove(temp_train_config_path)
    os.remove(temp_model_config_path)

    # No direct assertion on metrics, just checking for no errors during run.
    # Further assertions could involve checking if metrics are within a reasonable range.
