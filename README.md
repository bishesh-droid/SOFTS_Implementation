# SOFTS_Implementation

An enhanced implementation of **"SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion"** (NeurIPS 2024).

> This project faithfully reproduces the SOFTS model while adding practical training enhancements, comprehensive metrics, and full PyTorch 2.6+ compatibility.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Download Dataset](#download-dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Jupyter Notebook](#jupyter-notebook)
- [Configuration](#configuration)
- [Testing](#testing)
- [CI/CD](#cicd)
- [What's New](#whats-new)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Project Overview

The **SOFTS** model (Series-cOre Fused Time Series forecaster) addresses a core challenge in multivariate time series forecasting: efficiently capturing inter-channel correlations without the quadratic complexity of attention-based approaches.

The key innovation is the **STar Aggregate-Redistribute (STAR) module**, which aggregates all channel representations into a single global "core" vector and redistributes it back to each channel — achieving **linear complexity O(C)** instead of O(C²).

This implementation is based on the original NeurIPS 2024 paper and extends it with improved training utilities, broader metrics, and production-ready robustness.

---

## Key Features

- **True Stochastic Pooling** — Proper multinomial sampling during training; deterministic weighted averaging during evaluation (matches paper specification)
- **Optional Layer Normalization** — Toggle layer norm in STAR modules for improved training stability
- **Comprehensive Metrics** — MSE, MAE, RMSE, MAPE, sMAPE, R² score
- **Advanced Training** — Cosine annealing, StepLR, ReduceLROnPlateau schedulers; gradient clipping; early stopping
- **Robust Error Handling** — NaN/Inf detection at runtime, shape validation, informative error messages
- **Full Test Coverage** — Unit and integration tests for all components
- **PyTorch 2.6+ Compatible** — Correct checkpoint loading with `weights_only=False`
- **Flexible Configuration** — All hyperparameters and training options via YAML

---

## Project Structure

```
SOFTS_Implementation/
├── configs/                          # YAML configuration files
│   ├── experiment_config.yaml        # Main unified config (model + training + data)
│   ├── model_config.yaml             # Model hyperparameters only
│   ├── train_config.yaml             # Training and data config
│   └── temp_eval_*.yaml              # Lightweight configs used by CI/CD tests
│
├── docs/
│   └── architecture.md               # Detailed architecture documentation
│
├── notebooks/
│   └── exploration.ipynb             # Exploratory analysis notebook
│
├── src/softs_implementation/
│   ├── models/
│   │   └── softs_model.py            # Core SOFTS + STAR module implementation
│   ├── data/
│   │   └── scripts/
│   │       └── download_data.py      # Dataset download utilities
│   ├── data_processing/
│   │   ├── dataset.py                # TimeSeriesDataset (sliding window)
│   │   └── transforms.py             # ReversibleInstanceNormalization
│   ├── training/
│   │   └── train.py                  # Full training loop
│   ├── evaluation/
│   │   └── evaluate.py               # Evaluation on held-out test split
│   ├── scripts/
│   │   ├── train.py                  # Training entry point (CLI)
│   │   └── evaluate.py               # Evaluation entry point (CLI)
│   └── utils/
│       ├── metrics.py                # Metric functions (MSE, MAE, RMSE, ...)
│       └── helpers.py                # Seed setting and misc utilities
│
├── tests/
│   ├── unit/
│   │   ├── test_softs_model.py       # Model component tests
│   │   ├── test_metrics.py           # Metrics validation tests
│   │   └── test_transforms.py        # Normalization transform tests
│   └── integration/
│       └── test_training.py          # End-to-end training + evaluation tests
│
├── .github/workflows/main.yml        # GitHub Actions CI/CD pipeline
├── CHANGELOG.md                      # Detailed change log
├── LICENSE                           # MIT License
├── pyproject.toml                    # Poetry project configuration
├── requirements.txt                  # pip-compatible dependency list
└── NeurIPS-2024-softs-...Paper.pdf   # Original NeurIPS 2024 paper
```

---

## Model Architecture

```
Input: (batch, seq_len, C)   ← C = number of channels/variables
         │
         ▼
  ┌─────────────────────────────────┐
  │  Temporal Embedding (per channel)│   seq_len → hidden_dim
  └─────────────────────────────────┘
         │
         ▼  (repeated N times)
  ┌───────────────────────────────────────────────────────┐
  │               STar Module (STAR)                      │
  │                                                       │
  │  Aggregation:  [MLP1] → [StochasticPooling] → core   │
  │                   ↓ (each channel)     ↓ (global)     │
  │  Redistribution: concat(channel, core) → [MLP2]       │
  │                          + residual connection         │
  │                          + optional LayerNorm          │
  └───────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────┐
  │  Linear Predictor (per channel) │   hidden_dim → pred_len
  └─────────────────────────────────┘
         │
         ▼
Output: (batch, pred_len, C)
```

**StochasticPooling** (key mechanism):
- **Training**: Samples from multinomial distribution over softmax-normalized channel weights → stochastic global core
- **Evaluation**: Deterministic weighted average of all channel representations → stable global core

**Complexity**: O(C) per layer (linear in number of channels), versus O(C²) for full attention.

---

## Getting Started

### Prerequisites

- Python 3.9–3.13
- PyTorch 1.10+ (tested up to PyTorch 2.6+)
- [Poetry](https://python-poetry.org/) (recommended) or pip

### Installation

**Option 1 — Poetry (recommended):**
```bash
git clone https://github.com/your_username/SOFTS_Implementation.git
cd SOFTS_Implementation
poetry install --with dev
```

**Option 2 — pip:**
```bash
git clone https://github.com/your_username/SOFTS_Implementation.git
cd SOFTS_Implementation
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

> **PyTorch 2.6+ note**: Checkpoint loading uses `weights_only=False` for trusted model files. This is already handled internally — no action needed.

### Download Dataset

The ETTh1 dataset (used by default) can be downloaded automatically:

```bash
mkdir -p data/raw
wget -O data/raw/ETTh1.csv \
  https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
```

Other supported datasets (ETTh2, ETTm1, ETTm2, Weather, Solar, PEMS) require manual download. See `src/softs_implementation/data/scripts/download_data.py` for helpers.

---

## Usage

### Training

```bash
# Using Poetry
poetry run python src/softs_implementation/scripts/train.py \
    --config configs/experiment_config.yaml

# Using pip / activated venv
python src/softs_implementation/scripts/train.py \
    --config configs/experiment_config.yaml
```

Checkpoints, training history (loss, metrics, learning rate), and config snapshots are saved under `checkpoints/<timestamp>/`.

### Evaluation

```bash
poetry run python src/softs_implementation/scripts/evaluate.py \
    --train_config configs/train_config.yaml \
    --model_config configs/model_config.yaml
```

Reports MAE and MSE on the held-out test split (last 20% of data).

### Jupyter Notebook

```bash
poetry run jupyter notebook notebooks/exploration.ipynb
```

---

## Configuration

All behavior is controlled via `configs/experiment_config.yaml`. Key options:

```yaml
experiment:
  name: 'softs_experiment'
  seed: 42
  device: 'cuda'           # or 'cpu'

data:
  dataset_name: 'ETTh1'
  seq_len: 96              # Lookback window length
  pred_len: 96             # Forecasting horizon
  target_cols: null        # null = use all columns; or list column names

model:
  hidden_dim: 128          # Per-channel embedding dimension
  core_dim: 64             # Global core vector dimension
  num_layers: 2            # Number of STAR layers
  dropout: 0.1
  use_layer_norm: false    # Enable LayerNorm in STAR modules
  use_simple_core: false   # Use deterministic mean pooling instead of stochastic

train:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: 'Adam'
  loss_function: 'MSE'
  scheduler: 'CosineAnnealingLR'   # CosineAnnealingLR | StepLR | ReduceLROnPlateau | null
  min_lr: 1.0e-6
  gradient_clip: 1.0               # Set to null to disable
  early_stopping_patience: 10      # Set to null to disable
  save_path: 'checkpoints'
```

See `configs/experiment_config.yaml` for the full reference with all defaults.

---

## Testing

```bash
# Run all tests
poetry run pytest tests/ -v

# Unit tests only
poetry run pytest tests/unit/ -v

# Integration tests only
poetry run pytest tests/integration/ -v
```

**Coverage includes:**
- STAR module forward pass and residual connections
- StochasticPooling train vs eval mode behavior
- Gradient flow verification
- Metrics correctness (MSE, MAE, RMSE, MAPE, sMAPE, R²)
- ReversibleInstanceNormalization forward and inverse
- End-to-end training run + checkpoint saving
- Evaluation from saved checkpoint

---

## CI/CD

GitHub Actions runs on every push and pull request to `main`:

1. **Install** — Poetry install with dev dependencies (Python 3.9)
2. **Data** — Downloads ETTh1 dataset automatically
3. **Test** — Runs full test suite via `pytest tests/`
4. **Lint** — Runs `flake8` (error checks + complexity ≤ 10, line length ≤ 120)

See `.github/workflows/main.yml` for the pipeline definition.

---

## What's New

Enhancements beyond the base NeurIPS 2024 paper:

| Enhancement | Detail |
|---|---|
| True Stochastic Pooling | Multinomial sampling during training (not softmax averaging) |
| Additional Metrics | RMSE, MAPE, sMAPE, R² alongside MSE and MAE |
| LR Schedulers | CosineAnnealingLR, StepLR, ReduceLROnPlateau |
| Gradient Clipping | Configurable clipping threshold to prevent divergence |
| Early Stopping | Patience-based termination on validation loss |
| Training History | Metrics and LR curves saved with each checkpoint |
| Layer Normalization | Optional LayerNorm in STAR and SimpleCoreModule |
| PyTorch 2.6 Support | Fixed `torch.load` for latest PyTorch versions |
| Comprehensive Tests | Full unit + integration test coverage |

See `CHANGELOG.md` for full details.

---

## Troubleshooting

**KeyError about column names:**
- Set `target_cols: null` in config to auto-use all columns
- Or ensure names match the CSV header exactly
- The error message lists all available column names

**NaN/Inf during training:**
- Enable gradient clipping: `gradient_clip: 1.0`
- Reduce `learning_rate`
- Enable layer normalization: `use_layer_norm: true`
- Check input data for extreme or missing values

**Model not improving:**
- Try `use_layer_norm: true`
- Increase `hidden_dim` or `num_layers`
- Tune the learning rate scheduler
- Check if early stopping is triggering prematurely (increase `early_stopping_patience`)

**PyTorch 2.6 checkpoint errors:**
- Already resolved: the implementation uses `weights_only=False`
- If loading external checkpoints, ensure they are from trusted sources

**Integration test failures:**
- Install the package: `pip install -e .` or `poetry install`
- Ensure `data/raw/ETTh1.csv` exists (see [Download Dataset](#download-dataset))
- Confirm Python version is 3.9–3.13

---

## Contributing

Contributions are welcome! Please ensure:

- All tests pass: `poetry run pytest tests/ -v`
- Code follows PEP 8 (checked by flake8)
- New features include unit tests
- Docstrings and type hints are added for new functions/classes
- `CHANGELOG.md` is updated

---

## License

MIT License — see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{han2024softs,
  title     = {SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion},
  author    = {Han, Lu and Chen, Xu-Yang and Ye, Han-Jia and Zhan, De-Chuan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```
