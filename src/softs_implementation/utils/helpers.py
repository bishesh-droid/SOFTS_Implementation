import torch
import numpy as np
import random

def set_seed(seed):
    """
    Sets the seed for reproducibility across torch, numpy, and random.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility.")

if __name__ == '__main__':
    # Example Usage
    seed_value = 42
    set_seed(seed_value)

    # Test reproducibility
    rand_torch = torch.randn(2, 2)
    rand_numpy = np.random.rand(2, 2)
    rand_random = random.random()

    set_seed(seed_value) # Reset seed

    rand_torch_2 = torch.randn(2, 2)
    rand_numpy_2 = np.random.rand(2, 2)
    rand_random_2 = random.random()

    assert torch.equal(rand_torch, rand_torch_2)
    assert np.array_equal(rand_numpy, rand_numpy_2)
    assert rand_random == rand_random_2

    print("Reproducibility test passed!")
