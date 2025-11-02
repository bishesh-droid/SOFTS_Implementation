# SOFTS Model Architecture Documentation

This document details the architecture of the SOFTS (Series-cOre Fused Time Series forecaster) model as implemented in this repository, based on the NeurIPS 2024 paper.

## 1. Overview

The SOFTS model is an efficient MLP-based model designed for multivariate time series forecasting. Its core innovation is the STar Aggregate-Redistribute (STAR) module, which effectively captures channel correlations with linear complexity.

## 2. Core Components

### 2.1. Reversible Instance Normalization

*   **Purpose:** To calibrate the distribution of input data and stabilize predictions.
*   **Mechanism:** Centers series to zero mean, scales to unit variance, and reverses normalization on forecasted series.

### 2.2. Series Embedding

*   **Purpose:** To embed the input time series data into a higher-dimensional space.
*   **Mechanism:** A linear projection is used to embed each channel's series into a hidden dimension `d`. This is analogous to patch embedding but without introducing an extra dimension, making it less complex.

### 2.3. STar Aggregate-Redistribute (STAR) Module

*   **Purpose:** To efficiently capture dependencies between channels. This is the central component of SOFTS.
*   **Mechanism:**
    1.  **Aggregation:** Instead of distributed interactions (like attention), STAR employs a centralized strategy. It first aggregates information from all channels to form a global "core representation." This is achieved by applying an MLP to the series embeddings and then using a stochastic pooling mechanism (combining mean and max pooling).
    2.  **Redistribution/Fusion:** The global core representation is then dispatched and fused with individual series representations. This indirect interaction reduces comparison complexity and improves robustness against anomalies.
    3.  **Linear Complexity:** The design ensures linear complexity with respect to the number of channels, making it scalable for large channel counts.

### 2.4. Linear Predictor

*   **Purpose:** To produce the final forecasting results from the processed series representations.
*   **Mechanism:** After `N` layers of the STAR module, a linear layer projects the final series representation to the desired prediction horizon `H`.

## 3. Model Flow

1.  **Input:** Multivariate time series `X` with `C` channels and lookback window `L`.
2.  **Normalization:** Apply reversible instance normalization.
3.  **Series Embedding:** Each channel's series is linearly projected to `S_0` (C x d).
4.  **STAR Layers:** `N` sequential STAR modules process `S_0`.
    *   Inside each STAR module:
        *   **Core Representation:** `o = Stoch_Pool(MLP_1(S_{i-1}))`
        *   **Fusion:** `F_i = Repeat_Concat(S_{i-1}, o_i)` (concatenate core with each series)
        *   **Projection:** `S_i = MLP_2(F_i) + S_{i-1}` (project back to hidden dimension with residual connection)
5.  **Linear Predictor:** The final series representation `S_N` is passed through a linear layer to produce the forecast `Y` (C x H).
6.  **Denormalization:** Reverse the instance normalization to get the final predictions.

## 4. Complexity Analysis

The paper states that SOFTS achieves linear complexity with respect to the number of channels (`C`), window length (`L`), and forecasting horizon (`H`). This is a significant advantage over quadratic complexity methods like traditional attention mechanisms.

## 5. Future Work / Extensions

*   Experiment with different pooling methods within the STAR module.
*   Explore alternative aggregate-redistribute strategies.
*   Investigate the impact of various embedding techniques.
*   Apply SOFTS to a wider range of real-world datasets and domains.
