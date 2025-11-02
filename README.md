# SOFTS_Implementation

This repository contains an implementation of the "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion" paper (NeurIPS 2024).

## Project Overview

This project aims to reproduce and extend the SOFTS model for multivariate time series forecasting. The SOFTS model introduces a novel STar Aggregate-Redistribute (STAR) module to efficiently capture channel correlations with linear complexity.

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

*   **Build:** (No specific build step for Python, just ensure dependencies are installed)
*   **Run:**
    ```bash
    python scripts/train.py --config configs/train_config.yaml
    ```
*   **Test:**
    ```bash
    pytest tests/
    ```

### Coding Conventions

*   Follow PEP 8 guidelines.
*   Use type hints.
*   Add docstrings to functions and classes.

## Key Files

*   `src/models/softs_model.py`: Contains the core implementation of the SOFTS model and the STAR module.
*   `src/data_processing/dataset.py`: Handles data loading and preprocessing.
*   `scripts/train.py`: Script for training the model.
*   `configs/`: Directory for configuration files.
*   `tests/`: Unit and integration tests.
