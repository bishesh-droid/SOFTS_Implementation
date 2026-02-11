import pytest
import torch
from softs_implementation.utils.metrics import MSE, MAE, RMSE, MAPE, sMAPE, R2_Score

def test_mse():
    """Test Mean Squared Error metric"""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mse = MSE(pred, true)
    assert mse.item() < 1e-6, "MSE should be near zero for identical tensors"

def test_mae():
    """Test Mean Absolute Error metric"""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mae = MAE(pred, true)
    assert mae.item() < 1e-6, "MAE should be near zero for identical tensors"

def test_rmse():
    """Test Root Mean Squared Error metric"""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    true = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    rmse = RMSE(pred, true)
    expected_rmse = 0.5
    assert abs(rmse.item() - expected_rmse) < 1e-5, f"RMSE should be close to {expected_rmse}"

def test_mape():
    """Test Mean Absolute Percentage Error"""
    pred = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    true = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    mape = MAPE(pred, true)
    assert mape.item() < 1e-3, "MAPE should be near zero for identical tensors"
    
    # Test with different values
    pred2 = torch.tensor([[9.0, 18.0]])
    true2 = torch.tensor([[10.0, 20.0]])
    mape2 = MAPE(pred2, true2)
    assert 9.0 < mape2.item() < 11.0, "MAPE should be around 10% for 10% error"

def test_smape():
    """Test Symmetric Mean Absolute Percentage Error"""
    pred = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    true = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    smape = sMAPE(pred, true)
    assert smape.item() < 1e-3, "sMAPE should be near zero for identical tensors"

def test_r2_score():
    """Test R-squared metric"""
    # Perfect prediction
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    true = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    r2 = R2_Score(pred, true)
    assert abs(r2.item() - 1.0) < 1e-5, "R² should be 1.0 for perfect prediction"

def test_metrics_with_batch():
    """Test metrics with batched data"""
    batch_size = 4
    seq_len = 10
    features = 5
    
    pred = torch.randn(batch_size, seq_len, features)
    true = pred + torch.randn(batch_size, seq_len, features) * 0.1
    
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    
    assert not torch.isnan(mse), "MSE should not be NaN"
    assert not torch.isnan(mae), "MAE should not be NaN"
    assert not torch.isnan(rmse), "RMSE should not be NaN"
    assert mse.item() > 0, "MSE should be positive for different tensors"
    assert mae.item() > 0, "MAE should be positive for different tensors"

def test_mape_with_zeros():
    """Test MAPE handles near-zero values gracefully"""
    pred = torch.tensor([[0.1, 0.2]])
    true = torch.tensor([[0.0, 0.0]])
    mape = MAPE(pred, true)
    # Should not raise error due to epsilon handling
    assert not torch.isnan(mape), "MAPE should handle zeros gracefully"

def test_smape_symmetry():
    """Test sMAPE is symmetric"""
    pred = torch.tensor([[10.0, 20.0]])
    true = torch.tensor([[20.0, 10.0]])
    
    smape1 = sMAPE(pred, true)
    smape2 = sMAPE(true, pred)
    
    assert abs(smape1.item() - smape2.item()) < 1e-5, "sMAPE should be symmetric"
