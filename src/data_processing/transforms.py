import torch
import torch.nn as nn

class ReversibleInstanceNormalization(nn.Module):
    """
    Reversible Instance Normalization layer as described in the SOFTS paper.
    Centers the series to zero mean, scales to unit variance, and reverses normalization.
    """
    def __init__(self, input_dim, eps=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Normalize across the sequence length for each feature independently.
        # Calculate mean and std for each feature across the sequence length and batch.
        # The paper mentions "instance normalization which centers the series to zero means,
        # scales them to unit variance". This typically means normalizing each instance (batch item)
        # independently across its temporal dimension.

        # Let's assume normalization is applied per batch item, per feature, across the sequence length.
        # x: (batch_size, seq_len, input_dim)
        # Calculate mean and std along the seq_len dimension for each feature and batch item.
        mean = x.mean(dim=1, keepdim=True) # (batch_size, 1, input_dim)
        std = x.std(dim=1, keepdim=True) + self.eps # (batch_size, 1, input_dim)

        normalized_x = (x - mean) / std

        return normalized_x, mean, std

    def inverse(self, normalized_x, mean, std):
        # Inverse normalization
        # normalized_x: (batch_size, pred_len, output_dim)
        # mean, std: (batch_size, 1, input_dim) or (batch_size, pred_len, output_dim) if adapted
        # For inverse, ensure mean and std match the dimensions of normalized_x.
        # If mean/std were computed over the input sequence, they need to be broadcastable
        # or specifically sliced/expanded for the prediction length and output dimensions.

        # Assuming mean and std provided here are already correctly shaped for the output.
        # E.g., if mean/std are (batch_size, 1, input_dim), they will broadcast.
        # If they are (batch_size, pred_len, output_dim), they are directly used.
        denormalized_x = normalized_x * std + mean
        return denormalized_x

if __name__ == '__main__':
    # Example Usage
    batch_size = 4
    seq_len = 96
    input_dim = 7
    x = torch.randn(batch_size, seq_len, input_dim) * 10 + 5 # Dummy data with some mean and std

    norm_layer = ReversibleInstanceNormalization(input_dim)

    normalized_x, mean, std = norm_layer(x)

    print(f"Original x shape: {x.shape}")
    print(f"Normalized x shape: {normalized_x.shape}")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")

    print(f"Original x mean (first batch, first feature): {x[0, :, 0].mean():.4f}")
    print(f"Original x std (first batch, first feature): {x[0, :, 0].std():.4f}")
    print(f"Normalized x mean (first batch, first feature): {normalized_x[0, :, 0].mean():.4f}")
    print(f"Normalized x std (first batch, first feature): {normalized_x[0, :, 0].std():.4f}")

    # Test inverse
    denormalized_x = norm_layer.inverse(normalized_x, mean, std)
    print(f"Denormalized x shape: {denormalized_x.shape}")

    # Check if original and denormalized are close
    assert torch.allclose(x, denormalized_x, atol=1e-5)
    print("Reversible Instance Normalization test passed!")
