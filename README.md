# SOFTS_Implementation

This repository contains an enhanced implementation of the "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion" paper (NeurIPS 2024).

## Project Overview

This project reproduces and extends the SOFTS model for multivariate time series forecasting. The SOFTS model introduces a novel STar Aggregate-Redistribute (STAR) module to efficiently capture channel correlations with linear complexity.

### Key Features

- ✅ **True Stochastic Pooling**: Proper stochastic sampling during training, deterministic averaging during evaluation
- ✅ **Layer Normalization**: Optional layer normalization in STAR modules for improved training stability
- ✅ **Comprehensive Metrics**: MSE, MAE, RMSE, MAPE, sMAPE, R² score
- ✅ **Advanced Training**: Learning rate schedulers, gradient clipping, early stopping
- ✅ **Robust Error Handling**: NaN/Inf detection, shape validation, informative error messages
- ✅ **Extensive Testing**: Unit and integration tests for all components

## Getting Started

### Prerequisites

*   Python 3.9+
*   PyTorch
*   Pandas
*   NumPy
*   Scikit-learn
*   PyYAML

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your_username/SOFTS_Implementation.git
    cd SOFTS_Implementation
    ```
2.  Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Development

### Building and Running

*   **Run Training:**
    ```bash
    python src/softs_implementation/scripts/train.py --config configs/experiment_config.yaml
    ```
*   **Run Tests:**
    ```bash
    pytest tests/ -v
    ```
*   **Run Specific Test Suite:**
    ```bash
    pytest tests/unit/ -v           # Unit tests only
    pytest tests/integration/ -v    # Integration tests only
    ```

### Configuration

The model behavior can be customized via `configs/experiment_config.yaml`:

**Model Configuration:**
- `use_layer_norm`: Enable layer normalization in STAR modules (default: false)
- `use_simple_core`: Use SimpleCoreModule instead of STarModule (default: false)
- `hidden_dim`, `core_dim`, `num_layers`: Model architecture parameters

**Training Configuration:**
- `scheduler`: Learning rate scheduler (CosineAnnealingLR, StepLR, ReduceLROROnPlateau, or null)
- `gradient_clip`: Gradient clipping threshold (set to null to disable)
- `early_stopping_patience`: Number of epochs without improvement before stopping (set to null to disable)
- `learning_rate`, `batch_size`, `epochs`: Standard training hyperparameters

See `configs/experiment_config.yaml` for all available options.

### Coding Conventions

*   Follow PEP 8 guidelines.
*   Use type hints.
*   Add docstrings to functions and classes.

## Key Files

*   `src/softs_implementation/models/softs_model.py`: Core SOFTS model and STAR module implementation
*   `src/softs_implementation/data_processing/dataset.py`: Data loading and preprocessing
*   `src/softs_implementation/training/train.py`: Training script with advanced features
*   `src/softs_implementation/evaluation/evaluate.py`: Model evaluation script
*   `src/softs_implementation/utils/metrics.py`: Comprehensive metrics suite
*   `configs/`: Configuration files
*   `tests/`: Unit and integration tests

## What's New

This implementation includes several enhancements beyond the base paper:

1. **True Stochastic Pooling** - Matches paper specification with proper train/eval mode behavior
2. **Additional Metrics** - RMSE, MAPE, sMAPE, and R² for comprehensive evaluation
3. **Training Enhancements** - Learning rate scheduling, gradient clipping, early stopping
4. **Robust Implementation** - Extensive error handling and numerical stability checks
5. **Comprehensive Testing** - Full test coverage for all components

See `CHANGELOG.md` for detailed documentation of all changes.

## Troubleshooting

**KeyError about column names:**
- Ensure `target_cols` in your config matches the actual column names in your dataset
- Set `target_cols: null` to automatically use all available columns

**NaN/Inf during training:**
- Enable gradient clipping: `gradient_clip: 1.0`
- Reduce learning rate
- Check input data for extreme values

**Model not improving:**
- Try enabling layer normalization: `use_layer_norm: true`
- Adjust learning rate scheduler parameters
- Increase model capacity (hidden_dim, num_layers)
