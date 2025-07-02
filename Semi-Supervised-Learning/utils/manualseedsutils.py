# utils.py
import torch
import numpy as np
import random

def set_seed(seed):
    """
    Sets the random seed for reproducibility across various modules.

    This function ensures consistent results by setting the random seed for 
    Python's `random` module, NumPy, and PyTorch (both CPU and CUDA, if available). 
    Additionally, it configures PyTorch's CUDA backend to be deterministic, ensuring
    reproducibility at the cost of potential performance trade-offs.

    Args:
        seed (int): The seed value to set for random number generators.

    Notes:
        - This function ensures reproducibility by fixing the seed across different libraries.
        - For CUDA-based operations, it sets deterministic behavior at the cost of performance.
    """
    
    # Set the seed for Python's random number generator
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch's random number generator (CPU)
    torch.manual_seed(seed)
    
    # If CUDA is available, set the seed for GPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set the deterministic behavior for CUDA (reproducibility)
    torch.backends.cudnn.deterministic = True

    # Disable auto-tuning to ensure deterministic results (but may impact performance)
    torch.backends.cudnn.benchmark = False
