import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class TimeSeriesDataset(Dataset):
    """
    A custom PyTorch Dataset for time series forecasting.
    Loads data from a CSV, creates input-output pairs based on sequence length and prediction length.
    """
    def __init__(self, dataset_name, seq_len, pred_len, target_cols=None, freq='H', noise_level=0.0):
        """
        Args:
            dataset_name (str): Name of the dataset file relative to data/raw/ (e.g., "ETT/ETTh1.csv").
            seq_len (int): Length of the input sequence (lookback window).
            pred_len (int): Length of the prediction horizon.
            target_cols (list, optional): List of column names to be used as features and targets.
                                         If None, all columns (except 'date' if present) are used.
            freq (str): Frequency of the time series data (e.g., 'H' for hourly, 'D' for daily).
                        Used for date indexing if needed, though not strictly used in this basic version.
            noise_level (float): Standard deviation of Gaussian noise to add to input data. Default: 0.0
        """
        self.dataset_name = dataset_name
        self.data_path = os.path.join("data/raw", dataset_name)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_cols = target_cols
        self.freq = freq
        self.noise_level = noise_level
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.data_path}")

        df = pd.read_csv(self.data_path, index_col='date', parse_dates=True)

        if self.target_cols:
            self.data = df[self.target_cols].values
        else:
            self.data = df.values

        # Convert to float32
        self.data = self.data.astype(np.float32)

        # Ensure enough data for at least one sample
        if len(self.data) < self.seq_len + self.pred_len:
            raise ValueError(f"Not enough data for seq_len={self.seq_len} and pred_len={self.pred_len}. "
                             f"Available data points: {len(self.data)}")

    def __len__(self):
        # The number of samples is the total length minus the sequence and prediction lengths
        # plus one (for the first sample starting at index 0).
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # Get input sequence (x)
        s_begin = idx
        s_end = s_begin + self.seq_len
        seq_x = self.data[s_begin:s_end]

        # Get target sequence (y)
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_y = self.data[r_begin:r_end]

        # Convert to tensor
        seq_x = torch.from_numpy(seq_x)
        seq_y = torch.from_numpy(seq_y)

        # Add noise if specified and in training mode (simple heuristic: if noise_level > 0)
        # Note: Ideally this should be controlled by a mode flag in __init__, but for now
        # we assume this dataset is used for training if noise is enabled.
        # To be cleaner, we could add a 'train' mode flag.
        if hasattr(self, 'noise_level') and self.noise_level > 0:
             noise = torch.randn_like(seq_x) * self.noise_level
             seq_x = seq_x + noise

        return seq_x, seq_y

if __name__ == '__main__':
    # Example Usage:
    # Create a dummy CSV file for testing
    dummy_raw_dir = "data/raw/TestDataset"
    os.makedirs(dummy_raw_dir, exist_ok=True)
    dummy_data_path = os.path.join(dummy_raw_dir, 'dummy_time_series.csv')

    dummy_data = pd.DataFrame(np.random.rand(200, 5), columns=[f'feature_{i}' for i in range(5)])
    dates = pd.date_range(start='2023-01-01', periods=200, freq='H')
    dummy_data['date'] = dates
    dummy_data.set_index('date', inplace=True)
    dummy_data.to_csv(dummy_data_path)

    # Instantiate the dataset
    seq_len = 96
    pred_len = 24
    target_cols = ['feature_0', 'feature_1', 'feature_2'] # Example: use 3 features
    dataset = TimeSeriesDataset(
        dataset_name='TestDataset/dummy_time_series.csv',
        seq_len=seq_len,
        pred_len=pred_len,
        target_cols=target_cols
    )

    print(f"Dataset length: {len(dataset)}")

    # Get a sample
    x, y = dataset[0]
    print(f"Input sequence (x) shape: {x.shape}") # Should be (seq_len, num_target_cols)
    print(f"Target sequence (y) shape: {y.shape}") # Should be (pred_len, num_target_cols)

    # Clean up dummy file and directory
    os.remove(dummy_data_path)
    os.rmdir(dummy_raw_dir)