import torch
import torch.nn as nn
from torchvision import models

def define_model(layer_freeze_upto, fc_units, dropout_rate, num_classes):    
    """
    Defines and modifies a pre-trained EfficientNet model with custom layers and optional layer freezing.

    This function loads a pre-trained EfficientNet model from `torchvision.models`, freezes layers 
    up to a specified layer, and then replaces the original classifier with a custom fully connected 
    classification head. The new head consists of a linear layer, a ReLU activation, dropout, and 
    a final linear layer for classification.

    Args:
        layer_freeze_upto (str): The name of the layer up to which parameters should be frozen. 
            All layers up to this name (inclusive) will have `requires_grad = False`.
        fc_units (int): Number of units in the first fully connected layer of the custom classifier.
        dropout_rate (float): Dropout rate to apply after the ReLU activation in the custom classifier.
        num_classes (int): Number of output classes for the final linear layer of the classifier.

    Returns:
        torch.nn.Module: Modified EfficientNet model with custom classifier and specified frozen layers.

    Notes:
        - This function uses the `efficientnet_v2_s` variant with `EfficientNet_V2_S_Weights.IMAGENET1K_V1` 
          pre-trained weights.
        - The `layer_freeze_upto` parameter allows fine-grained control over which layers to freeze 
          during training, useful for transfer learning.
    """
   
    # Load the pre-trained EfficientNet model
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # Freeze layers up to the specified point
    cutoff_reached = False
    for name, param in model.named_parameters():
        if not cutoff_reached:
            if name == layer_freeze_upto:
                cutoff_reached = True
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Modify the classifier to add custom fully connected layers
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, fc_units),  # Fully connected layer
        nn.ReLU(),                     # ReLU activation
        nn.Dropout(dropout_rate),       # Dropout for regularization
        nn.Linear(fc_units, num_classes),  # Final output layer
    )
    
    return model
