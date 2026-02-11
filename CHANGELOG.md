# Changelog

All notable changes to the SOFTS implementation will be documented in this file.

## [Unreleased] - 2026-02-11

### Added
- **True Stochastic Pooling**: Implemented proper stochastic sampling during training (multinomial distribution) and deterministic averaging during evaluation, matching the NeurIPS-2024 paper specification
- **Layer Normalization**: Added optional layer normalization in STAR modules and SimpleCoreModule for improved training stability
- **Comprehensive Metrics Suite**: 
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error) with zero-value handling
  - sMAPE (Symmetric Mean Absolute Percentage Error)
  - R² Score (Coefficient of Determination)
- **Learning Rate Schedulers**: 
  - CosineAnnealingLR
  - StepLR
  - ReduceLROnPlateau
- **Gradient Clipping**: Configurable gradient clipping to prevent exploding gradients
- **Early Stopping**: Automatic training termination based on validation loss patience
- **Training History Tracking**: Saves training/validation metrics and learning rate history
- **Enhanced Checkpointing**: Checkpoints now include scheduler state and training history
- **Comprehensive Unit Tests**:
  - Tests for stochastic pooling train vs eval mode behavior
  - Gradient flow verification tests
  - Layer normalization tests
  - Metrics validation tests
  - Transform reversibility tests
  - Shape validation tests

### Fixed
- **Dataset Column Handling**: Improved error messages when target columns don't match CSV columns, preventing cryptic KeyErrors
- **Numerical Stability**: Added clamping in stochastic pooling to prevent overflow
- **NaN/Inf Detection**: Added runtime checks in STAR modules and training loop for numerical stability
- **Type Hints**: Added comprehensive type hints throughout the codebase
- **Shape Assertions**: Added validation to catch dimension mismatches early
- **PyTorch 2.6 Compatibility**: Fixed checkpoint loading by adding `weights_only=False` parameter to `torch.load` for trusted checkpoint files

### Changed
- **StochasticPooling**: Changed from deterministic softmax-based averaging to true stochastic sampling during training
- **Model Initialization**: Added `use_layer_norm` parameter to SOFTS, STarModule, and SimpleCoreModule
- **Configuration Format**: Updated experiment config with new training parameters (gradient_clip, early_stopping_patience, scheduler options)
- **Training Loop**: Enhanced with NaN/Inf checking, gradient clipping, scheduler updates, and early stopping logic
- **Evaluation Metrics**: Now reports RMSE in addition to MAE and MSE

### Documentation
- Updated README with new features and configuration options
- Added comprehensive docstrings to all new methods
- Created CHANGELOG to track all modifications
- Enhanced inline comments for complex operations

## Project Information

This implementation is based on:
**"SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion"**  
NeurIPS 2024  
Authors: Lu Han, Xu-Yang Chen, Han-Jia Ye, De-Chuan Zhan
