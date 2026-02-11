import torch
import torch.nn.functional as F
from typing import Union

def MSE(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error
    
    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.
        
    Returns:
        torch.Tensor: MSE value.
    """
    return F.mse_loss(pred, true)

def MAE(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error
    
    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.
        
    Returns:
        torch.Tensor: MAE value.
    """
    return F.l1_loss(pred, true)

def RMSE(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Squared Error
    
    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.
        
    Returns:
        torch.Tensor: RMSE value.
    """
    return torch.sqrt(F.mse_loss(pred, true))

def MAPE(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean Absolute Percentage Error
    
    Note: MAPE is undefined when true values are zero. This implementation
    adds a small epsilon to avoid division by zero.
    
    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.
        eps (float): Small constant to avoid division by zero.
        
    Returns:
        torch.Tensor: MAPE value (as a percentage, e.g., 10.5 for 10.5%).
    """
    # Avoid division by zero
    denominator = torch.abs(true) + eps
    mape = torch.mean(torch.abs((true - pred) / denominator)) * 100.0
    return mape

def sMAPE(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric Mean Absolute Percentage Error
    
    sMAPE is more robust than MAPE when dealing with values close to zero,
    as it uses the average of absolute values in the denominator.
    
    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.
        eps (float): Small constant to avoid division by zero.
        
    Returns:
        torch.Tensor: sMAPE value (as a percentage, 0-200%).
    """
    numerator = torch.abs(true - pred)
    denominator = (torch.abs(true) + torch.abs(pred)) / 2.0 + eps
    smape = torch.mean(numerator / denominator) * 100.0
    return smape

def R2_Score(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    R-squared (Coefficient of Determination)
    
    R² score measures the proportion of variance in the dependent variable
    that is predictable from the independent variable(s).
    
    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.
        
    Returns:
        torch.Tensor: R² score (ranges from -inf to 1, where 1 is perfect).
    """
    ss_res = torch.sum((true - pred) ** 2)
    ss_tot = torch.sum((true - torch.mean(true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))  # Add epsilon to avoid division by zero
    return r2

if __name__ == '__main__':
    # Example Usage
    pred_tensor = torch.randn(10, 5)
    true_tensor = torch.randn(10, 5)

    mse_val = MSE(pred_tensor, true_tensor)
    mae_val = MAE(pred_tensor, true_tensor)
    rmse_val = RMSE(pred_tensor, true_tensor)
    mape_val = MAPE(pred_tensor, true_tensor)
    smape_val = sMAPE(pred_tensor, true_tensor)
    r2_val = R2_Score(pred_tensor, true_tensor)

    print(f"MSE: {mse_val.item():.4f}")
    print(f"MAE: {mae_val.item():.4f}")
    print(f"RMSE: {rmse_val.item():.4f}")
    print(f"MAPE: {mape_val.item():.2f}%")
    print(f"sMAPE: {smape_val.item():.2f}%")
    print(f"R²: {r2_val.item():.4f}")

    # Test with identical tensors
    mse_zero = MSE(pred_tensor, pred_tensor)
    mae_zero = MAE(pred_tensor, pred_tensor)
    rmse_zero = RMSE(pred_tensor, pred_tensor)
    print(f"\nMSE (identical): {mse_zero.item():.4f}")
    print(f"MAE (identical): {mae_zero.item():.4f}")
    print(f"RMSE (identical): {rmse_zero.item():.4f}")
    assert mse_zero < 1e-6 and mae_zero < 1e-6, "Metrics should be near zero for identical tensors"
    print("Metrics tests passed!")

