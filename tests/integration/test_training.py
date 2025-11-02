import pytest
import os
import torch
import yaml
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from data.scripts.download_data import download_ett_data

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
        download_ett_data(raw_data_path=DATA_RAW_DIR)

    # Create test config files
    test_train_config_path = os.path.join(CONFIGS_DIR, "test_train_config.yaml")
    test_model_config_path = os.path.join(CONFIGS_DIR, "test_model_config.yaml")

    # Minimal train config for testing
    test_train_config_content = {
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
    with open(test_train_config_path, 'w') as f:
        yaml.safe_dump(test_train_config_content, f)

    # Minimal model config for testing
    test_model_config_content = {
        'model': {
            'input_dim': 7,
            'output_dim': 7,
            'seq_len': 24, # Smaller seq_len for quick test
            'pred_len': 12, # Smaller pred_len for quick test
            'hidden_dim': 16,
            'core_dim': 8,
            'num_layers': 1,
            'dropout': 0.0
        }
    }
    with open(test_model_config_path, 'w') as f:
        yaml.safe_dump(test_model_config_content, f)

    # Yield control to tests
    yield

    # Teardown: clean up generated files
    if os.path.exists(dummy_data_path):
        os.remove(dummy_data_path)
    if os.path.exists(test_train_config_path):
        os.remove(test_train_config_path)
    if os.path.exists(test_model_config_path):
        os.remove(test_model_config_path)
    if os.path.exists(CHECKPOINTS_DIR):
        for f in os.listdir(CHECKPOINTS_DIR):
            os.remove(os.path.join(CHECKPOINTS_DIR, f))
        os.rmdir(CHECKPOINTS_DIR)

def test_training_run():
    """
    Tests if the training script runs without errors for a minimal configuration.
    Checks if a model checkpoint is saved.
    """
    test_train_config_path = os.path.join(CONFIGS_DIR, "test_train_config.yaml")
    test_model_config_path = os.path.join(CONFIGS_DIR, "test_model_config.yaml")

    # Run training
    train_model(test_train_config_path, test_model_config_path)

    # Check if a checkpoint was saved
    checkpoints = os.listdir(CHECKPOINTS_DIR)
    assert len(checkpoints) > 0, "No model checkpoint was saved during training."
    assert any(f.endswith('.pth') for f in checkpoints), "Saved file is not a .pth checkpoint."

def test_evaluation_run():
    """
    Tests if the evaluation script runs without errors using a saved checkpoint.
    """
    test_train_config_path = os.path.join(CONFIGS_DIR, "test_train_config.yaml")
    test_model_config_path = os.path.join(CONFIGS_DIR, "test_model_config.yaml")

    # Ensure a model is trained and saved first
    train_model(test_train_config_path, test_model_config_path)
    checkpoints = os.listdir(CHECKPOINTS_DIR)
    latest_checkpoint = sorted([os.path.join(CHECKPOINTS_DIR, f) for f in checkpoints if f.endswith('.pth')], key=os.path.getmtime, reverse=True)[0]

    # Run evaluation
    evaluate_model(test_train_config_path, test_model_config_path, latest_checkpoint)

    # No direct assertion on metrics, just checking for no errors during run.
    # Further assertions could involve checking if metrics are within a reasonable range.
