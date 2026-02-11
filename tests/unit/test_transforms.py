import pytest
import torch
from softs_implementation.data_processing.transforms import ReversibleInstanceNormalization

def test_reversible_normalization():
    """Test that normalization is reversible"""
    batch_size = 4
    seq_len = 96
    input_dim = 7
    
    x = torch.randn(batch_size, seq_len, input_dim) * 10 + 5
    norm_layer = ReversibleInstanceNormalization(input_dim)
    
    normalized_x, mean, std = norm_layer(x)
    denormalized_x = norm_layer.inverse(normalized_x, mean, std)
    
    assert torch.allclose(x, denormalized_x, atol=1e-5), "Normalization should be reversible"

def test_normalized_statistics():
    """Test that normalized data has zero mean and unit variance"""
    batch_size = 4
    seq_len = 96
    input_dim = 7
    
    x = torch.randn(batch_size, seq_len, input_dim) * 10 + 5
    norm_layer = ReversibleInstanceNormalization(input_dim)
    
    normalized_x, mean, std = norm_layer(x)
    
    # Check mean is close to zero
    normalized_mean = normalized_x.mean(dim=1)
    assert torch.allclose(normalized_mean, torch.zeros_like(normalized_mean), atol=1e-5), \
        "Normalized data should have zero mean"
    
    # Check std is close to one
    normalized_std = normalized_x.std(dim=1)
    assert torch.allclose(normalized_std, torch.ones_like(normalized_std), atol=1e-4), \
        "Normalized data should have unit variance"

def test_normalization_shapes():
    """Test that normalization preserves shapes correctly"""
    batch_size = 8
    seq_len = 48
    input_dim = 5
    
    x = torch.randn(batch_size, seq_len, input_dim)
    norm_layer = ReversibleInstanceNormalization(input_dim)
    
    normalized_x, mean, std = norm_layer(x)
    
    assert normalized_x.shape == x.shape, "Normalized data should have same shape as input"
    assert mean.shape == (batch_size, 1, input_dim), "Mean should have correct broadcast shape"
    assert std.shape == (batch_size, 1, input_dim), "Std should have correct broadcast shape"

def test_inverse_with_different_length():
    """Test inverse normalization works with different sequence lengths (for predictions)"""
    batch_size = 4
    seq_len = 96
    pred_len = 24
    input_dim = 7
    
    x = torch.randn(batch_size, seq_len, input_dim)
    norm_layer = ReversibleInstanceNormalization(input_dim)
    
    normalized_x, mean, std = norm_layer(x)
    
    # Simulate prediction with different length
    pred = torch.randn(batch_size, pred_len, input_dim)
    denormalized_pred = norm_layer.inverse(pred, mean, std)
    
    assert denormalized_pred.shape == (batch_size, pred_len, input_dim), \
        "Denormalized prediction should have correct shape"

def test_numerical_stability():
    """Test normalization handles extreme values"""
    batch_size = 4
    seq_len = 96
    input_dim = 7
    
    # Test with very large values
    x_large = torch.randn(batch_size, seq_len, input_dim) * 1e6
    norm_layer = ReversibleInstanceNormalization(input_dim, eps=1e-5)
    
    normalized_large, mean_large, std_large = norm_layer(x_large)
    denormalized_large = norm_layer.inverse(normalized_large, mean_large, std_large)
    
    assert torch.allclose(x_large, denormalized_large, rtol=1e-4), \
        "Normalization should handle large values"
    assert not torch.isnan(normalized_large).any(), "No NaN in normalized large values"
    assert not torch.isinf(normalized_large).any(), "No Inf in normalized large values"
