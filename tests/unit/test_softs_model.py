import pytest
import torch
from softs_implementation.models.softs_model import SOFTS, STarModule, MLP, StochasticPooling
from softs_implementation.data_processing.transforms import ReversibleInstanceNormalization

# --- Test MLP ---
def test_mlp_forward():
    input_dim = 10
    output_dim = 5
    hidden_dim = 20
    mlp = MLP(input_dim, output_dim, hidden_dim)
    x = torch.randn(32, input_dim)
    output = mlp(x)
    assert output.shape == (32, output_dim)

def test_mlp_dropout():
    input_dim = 10
    output_dim = 5
    mlp = MLP(input_dim, output_dim, dropout=0.5)
    mlp.train()
    x = torch.randn(32, input_dim)
    output = mlp(x)
    assert output.shape == (32, output_dim)

# --- Test StochasticPooling ---
def test_stochastic_pooling_forward():
    input_dim = 10 # hidden_dim for STAR
    C = 7 # Number of channels
    batch_size = 4
    pooling = StochasticPooling(input_dim)
    x = torch.randn(batch_size, C, input_dim)
    output = pooling(x)
    assert output.shape == (batch_size, input_dim)

# --- Test STarModule ---
def test_star_module_forward():
    hidden_dim = 64
    core_dim = 32
    C = 7 # Number of channels
    batch_size = 4
    star_module = STarModule(hidden_dim, core_dim)
    x = torch.randn(batch_size, C, hidden_dim)
    output = star_module(x)
    assert output.shape == (batch_size, C, hidden_dim)

def test_softs_model_forward():
    input_dim = 7
    # output_dim = 7 # Predicting all input channels
    seq_len = 96
    pred_len = 24
    hidden_dim = 64
    core_dim = 32
    num_layers = 2
    dropout = 0.1

    model = SOFTS(input_dim, seq_len, pred_len, hidden_dim, core_dim, num_layers, dropout)
    x = torch.randn(32, seq_len, input_dim) # (batch_size, seq_len, input_dim)
    output = model(x)
    assert output.shape == (32, pred_len, input_dim)

def test_softs_model_with_normalization():
    input_dim = 7
    # output_dim = 7
    seq_len = 96
    pred_len = 24
    hidden_dim = 64
    core_dim = 32
    num_layers = 2
    dropout = 0.1

    model = SOFTS(input_dim, seq_len, pred_len, hidden_dim, core_dim, num_layers, dropout)
    norm_layer = ReversibleInstanceNormalization(input_dim)

    x = torch.randn(32, seq_len, input_dim)
    normalized_x, mean, std = norm_layer(x)

    output = model(normalized_x)
    # For inverse, we need mean/std corresponding to the output shape.
    # Assuming output_dim == input_dim, and pred_len is the temporal dimension.
    # We need to slice mean/std to match the output's temporal and feature dimensions.
    # In SOFTS, the output is (batch_size, pred_len, output_dim)
    # mean/std are (batch_size, 1, input_dim)
    # So, we can use mean/std directly as they will broadcast correctly.
    denormalized_output = norm_layer.inverse(output, mean, std)

    assert denormalized_output.shape == (32, pred_len, input_dim)
    # Further checks could involve comparing denormalized_output with some expected range
    # or ensuring no NaNs/Infs.
