import torch
import numpy as np

def sharpen(p, T=0.6):
    """
    Applies a sharpening function to a probability distribution `p` by 
    adjusting its temperature `T`. Sharpening makes the distribution 
    more confident by reducing entropy.
    
    Args:
        p (torch.Tensor): Probability distribution tensor (e.g., class probabilities).
        T (float, optional): Temperature parameter for sharpening. Lower values 
            make the distribution sharper. Default is 0.6.
    
    Returns:
        torch.Tensor: Sharpened probability distribution.
    """
    p = p ** (1 / T)
    return p / p.sum(dim=1, keepdim=True)


def mixup_data(x, y, alpha=0.2):
    """
    Generates mixed inputs and targets using the mixup data augmentation 
    strategy. Mixup blends two examples, resulting in a new sample that is 
    a linear combination of the original inputs and their respective labels.
    
    Args:
        x (torch.Tensor): Input data tensor (e.g., images or feature vectors).
        y (torch.Tensor): Target labels tensor.
        alpha (float, optional): Parameter for the beta distribution used to 
            sample the mixing coefficient `lam`. Default is 0.2. Higher values 
            increase mixing between examples.
    
    Returns:
        tuple: Mixed inputs (`mixed_x`), original targets (`y_a`, `y_b`), and 
        the mixing coefficient (`lam`).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the mixup loss for predictions by applying a weighted sum 
    of losses between the predicted values and the targets `y_a` and `y_b`.
    
    Args:
        criterion (function): Loss function to apply.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): First set of target labels.
        y_b (torch.Tensor): Second set of target labels.
        lam (float): Mixing coefficient used to blend targets.
    
    Returns:
        torch.Tensor: Computed mixup loss value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
