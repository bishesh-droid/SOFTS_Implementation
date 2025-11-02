import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron.
    Used for Embedding, MLP1, and MLP2 in the SOFTS paper.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim # Default to identity if not specified
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() # GELU activation as per paper

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class StochasticPooling(nn.Module):
    """
    Stochastic Pooling as described in the paper (Appendix B.2).
    Combines advantages of mean and max pooling.
    The paper describes sampling from a multinomial distribution during training
    and probabilistic averaging during testing. For simplicity and stability,
    this implementation uses a weighted average based on softmax for both training and evaluation,
    which aligns with the test-time description and is a common approximation for stochastic pooling.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (batch_size, C, hidden_dim)
        # Apply softmax over channels for each feature dimension to get weights
        weights = F.softmax(x, dim=1) # (batch_size, C, hidden_dim)
        # Compute weighted sum across channels
        pooled_output = (weights * x).sum(dim=1) # (batch_size, hidden_dim)
        return pooled_output

class STarModule(nn.Module):
    """
    STar Aggregate-Redistribute Module as described in the paper.
    """
    def __init__(self, hidden_dim, core_dim, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.core_dim = core_dim

        # MLP1: Projects series representation from hidden_dim to core_dim
        self.mlp1 = MLP(hidden_dim, core_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.stoch_pool = StochasticPooling(core_dim)

        # MLP2: Projects concatenated (series + core) back to hidden_dim
        self.mlp2 = MLP(hidden_dim + core_dim, hidden_dim, hidden_dim=hidden_dim + core_dim, dropout=dropout)

    def forward(self, x):
        # x shape: (batch_size, C, hidden_dim)
        batch_size, C, _ = x.shape

        # 1. Get core representation
        # Apply MLP1 to each series independently
        x_mlp1 = self.mlp1(x) # (batch_size, C, core_dim)

        # Stochastic Pooling to get global core representation
        core_representation = self.stoch_pool(x_mlp1) # (batch_size, core_dim)

        # 2. Repeat and Concatenate (Fuse)
        # Repeat core_representation for each channel
        core_repeated = core_representation.unsqueeze(1).expand(-1, C, -1) # (batch_size, C, core_dim)

        # Concatenate series representation with core_repeated
        fused_representation = torch.cat([x, core_repeated], dim=-1) # (batch_size, C, hidden_dim + core_dim)

        # 3. Apply MLP2 and add residual connection
        output = self.mlp2(fused_representation) # (batch_size, C, hidden_dim)
        output = output + x # Residual connection

        return output

class SOFTS(nn.Module):
    """
    Series-cOre Fused Time Series forecaster (SOFTS) model.
    """
    def __init__(self, input_dim, seq_len, pred_len, hidden_dim, core_dim, num_layers, dropout=0.0): # Removed output_dim as it's implicitly input_dim
        super().__init__()
        self.input_dim = input_dim
        # self.output_dim = output_dim # Removed as it's implicitly input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.core_dim = core_dim
        self.num_layers = num_layers



        # Stack of STar Modules
        self.star_layers = nn.ModuleList([
            STarModule(hidden_dim, core_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Linear Predictor: maps hidden_dim to pred_len for each channel.
        # The paper's Algorithm 1: Y = Linear(SN) where SN is (C, d) and Y is (C, H).
        # This means the final linear layer maps d -> H for each channel.

        self.temporal_embedding = MLP(seq_len, hidden_dim, dropout=dropout) # Maps L -> d

        # Final Linear Predictor: maps hidden_dim to pred_len * output_dim, then reshape
        # Or, maps hidden_dim to output_dim, and we need to ensure the temporal dimension is correct.
        # The paper's Algorithm 1: Y = Linear(SN) where SN is (C, d) and Y is (C, H)
        # This means the final linear layer maps d -> H for each channel.
        self.final_predictor = MLP(hidden_dim, pred_len, dropout=dropout)


    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape

        # Transpose to (batch_size, input_dim, seq_len) for channel-wise processing
        # The paper's Algorithm 1 uses X.transpose to get (C, L)
        # Let's assume input_dim is C and seq_len is L.
        # So input x is (batch_size, L, C)
        # We need to process each channel's time series (L,) independently first.

        # The paper's description is a bit ambiguous on the exact input/output shapes
        # for the embedding and STAR module.
        # "series embedding on the lookback window"
        # "linear projection to embed the series of each channel to S0 = RCxd"
        # This implies that for each channel, the time series of length L is mapped to a vector of length d.
        # So, if input is (batch_size, L, C), we need to get (batch_size, C, L) first.
        # Then apply embedding to map L -> d for each channel.

        # Let's assume input x is (batch_size, seq_len, input_dim) where input_dim is C.
        # Transpose to (batch_size, input_dim, seq_len) to process each channel's time series
        x_transposed = x.permute(0, 2, 1) # (batch_size, C, L)

        # Apply temporal embedding to each channel's time series
        # Each (L,) vector for a channel becomes a (d,) vector
        s0 = self.temporal_embedding(x_transposed) # (batch_size, C, hidden_dim)

        # Pass through STar layers
        s_n = s0
        for star_layer in self.star_layers:
            s_n = star_layer(s_n) # (batch_size, C, hidden_dim)

        # Final Linear Predictor
        # The output of `final_predictor` is (batch_size, C, pred_len).

        # Permute to (batch_size, pred_len, C, output_dim) if output_dim is 1 (single feature per channel)
        # Or if output_dim is the number of channels, then (batch_size, pred_len, output_dim)
        # Assuming output_dim is the number of features to predict per channel, and C is input_dim.
        # The output of `final_predictor` is (batch_size, C, pred_len).
        # We need to permute it to (batch_size, pred_len, C) to match typical time series output format.
        # Here, C is `self.input_dim`.
        output = output.permute(0, 2, 1) # (batch_size, pred_len, self.input_dim)

        return output