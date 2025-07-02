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
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If you are using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
