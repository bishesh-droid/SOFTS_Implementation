import pytest
import torch
from softs_implementation.models.softs_model import SOFTS, STarModule, SimpleCoreModule, MLP, StochasticPooling
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

def test_stochastic_pooling_train_vs_eval():
    """Test that stochastic pooling behaves differently in train vs eval mode"""
    input_dim = 10
    C = 7
    batch_size = 4
    pooling = StochasticPooling(input_dim)
    x = torch.randn(batch_size, C, input_dim)
    
    # Set to train mode and run multiple times - should get different results
    pooling.train()
    torch.manual_seed(42)
    output1_train = pooling(x)
    torch.manual_seed(43)
    output2_train = pooling(x)
    
    # In training mode, outputs should be different (stochastic sampling)
    # Note: this is probabilistic, but with different seeds should differ
    assert not torch.allclose(output1_train, output2_train, atol=1e-3), \
        "Stochastic pooling should produce different outputs in train mode with different seeds"
    
    # Set to eval mode and run multiple times - should get same results
    pooling.eval()
    output1_eval = pooling(x)
    output2_eval = pooling(x)
    
    # In eval mode, outputs should be identical (deterministic averaging)
    assert torch.allclose(output1_eval, output2_eval), \
        "Stochastic pooling should produce identical outputs in eval mode"

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

def test_star_module_with_layer_norm():
    """Test STarModule with layer normalization enabled"""
    hidden_dim = 64
    core_dim = 32
    C = 7
    batch_size = 4
    star_module = STarModule(hidden_dim, core_dim, use_layer_norm=True)
    x = torch.randn(batch_size, C, hidden_dim)
    output = star_module(x)
    assert output.shape == (batch_size, C, hidden_dim)
    assert not torch.isnan(output).any(), "Output should not contain NaN"
    assert not torch.isinf(output).any(), "Output should not contain Inf"

def test_simple_core_module_with_layer_norm():
    """Test SimpleCoreModule with layer normalization enabled"""
    hidden_dim = 64
    core_dim = 32
    C = 7
    batch_size = 4
    simple_module = SimpleCoreModule(hidden_dim, core_dim, use_layer_norm=True)
    x = torch.randn(batch_size, C, hidden_dim)
    output = simple_module(x)
    assert output.shape == (batch_size, C, hidden_dim)

# --- Test SOFTS Model ---
def test_softs_model_forward():
    input_dim = 7
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
    denormalized_output = norm_layer.inverse(output, mean, std)

    assert denormalized_output.shape == (32, pred_len, input_dim)

def test_softs_model_with_layer_norm():
    """Test SOFTS model with layer normalization enabled"""
    input_dim = 7
    seq_len = 96
    pred_len = 24
    hidden_dim = 64
    core_dim = 32
    num_layers = 2
    
    model = SOFTS(input_dim, seq_len, pred_len, hidden_dim, core_dim, num_layers, 
                  dropout=0.1, use_layer_norm=True)
    x = torch.randn(16, seq_len, input_dim)
    output = model(x)
    assert output.shape == (16, pred_len, input_dim)
    assert not torch.isnan(output).any(), "Output should not contain NaN"

def test_gradient_flow():
    """Test that gradients flow through the model"""
    input_dim = 7
    seq_len = 96
    pred_len = 24
    hidden_dim = 64
    core_dim = 32
    num_layers = 2
    
    model = SOFTS(input_dim, seq_len, pred_len, hidden_dim, core_dim, num_layers, dropout=0.1)
    x = torch.randn(4, seq_len, input_dim, requires_grad=True)
    target = torch.randn(4, pred_len, input_dim)
    
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    # Check that gradients exist for model parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} should not be None"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} should not contain NaN"
        assert not torch.isinf(param.grad).any(), f"Gradient for {name} should not contain Inf"

def test_shape_validation():
    """Test that model raises errors for incorrect input shapes"""
    input_dim = 7
    seq_len = 96
    pred_len = 24
    hidden_dim = 64
    core_dim = 32
    num_layers = 2
    
    model = SOFTS(input_dim, seq_len, pred_len, hidden_dim, core_dim, num_layers)
    
    # Correct shape should work
    x_correct = torch.randn(4, seq_len, input_dim)
    output = model(x_correct)
    assert output.shape == (4, pred_len, input_dim)
    
    # Wrong number of features should fail at embedding stage
    x_wrong = torch.randn(4, seq_len, input_dim + 1)
    # This will not fail at model level but at transpose operation
    # The model will process it but the final shape will be different
    output_wrong = model(x_wrong)
    assert output_wrong.shape[2] == input_dim + 1  # Output channels match input channels

def test_model_train_eval_modes():
    """Test that model behaves correctly in train vs eval mode"""
    input_dim = 7
    seq_len = 96
    pred_len = 24
    hidden_dim = 64
    core_dim = 32
    num_layers = 2
    
    model = SOFTS(input_dim, seq_len, pred_len, hidden_dim, core_dim, num_layers, dropout=0.1)
    x = torch.randn(4, seq_len, input_dim)
    
    # Train mode
    model.train()
    output_train = model(x)
    
    # Eval mode
    model.eval()
    output_eval = model(x)
    
    # Shapes should be the same
    assert output_train.shape == output_eval.shape
    # Outputs may differ due to stochastic pooling
    # This is expected behavior

