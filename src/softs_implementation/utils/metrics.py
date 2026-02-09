import torch
import torch.nn.functional as F

def MSE(pred, true):
    """
    Mean Squared Error
    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.
    Returns:
        torch.Tensor: MSE value.
    """
    return F.mse_loss(pred, true)

def MAE(pred, true):
    """
    Mean Absolute Error
    Args:
        pred (torch.Tensor): Predicted values.
        true (torch.Tensor): True values.
    Returns:
        torch.Tensor: MAE value.
    """
    return F.l1_loss(pred, true)

if __name__ == '__main__':
    # Example Usage
    pred_tensor = torch.randn(10, 5)
    true_tensor = torch.randn(10, 5)

    mse_val = MSE(pred_tensor, true_tensor)
    mae_val = MAE(pred_tensor, true_tensor)

    print(f"MSE: {mse_val.item():.4f}")
    print(f"MAE: {mae_val.item():.4f}")

    # Test with identical tensors
    mse_zero = MSE(pred_tensor, pred_tensor)
    mae_zero = MAE(pred_tensor, pred_tensor)
    print(f"MSE (identical): {mse_zero.item():.4f}")
    print(f"MAE (identical): {mae_zero.item():.4f}")
    assert mse_zero < 1e-6 and mae_zero < 1e-6, "Metrics should be near zero for identical tensors"
    print("Metrics tests passed!")
