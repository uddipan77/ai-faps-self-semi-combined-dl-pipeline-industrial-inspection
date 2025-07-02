import torch.nn as nn

class CustomDINONormModel(nn.Module):
    """
    Custom model built on top of a DINO model backbone with an additional classifier for output prediction.

    The `CustomDINONormModel` class allows for fine-tuning and extending a pre-trained DINO model by 
    adding a custom classification head. The classification head consists of fully connected layers, 
    layer normalization, and a ReLU activation for non-linearity.

    Args:
        dino_model (nn.Module): Pre-trained DINO model or backbone used as a feature extractor.
        num_classes (int, optional): Number of output classes for the classifier. Default is 3.

    Attributes:
        dino_model (nn.Module): The base DINO model used for feature extraction.
        classifier (nn.Sequential): Custom classifier head with fully connected layers, layer normalization, 
            and a ReLU activation.

    Methods:
        forward(x):
            Defines the forward pass through the DINO model and custom classifier layers.

        Args:
            x (torch.Tensor): Input tensor (e.g., images).
        
        Returns:
            torch.Tensor: Output tensor with predictions for each class.
    """
    
    def __init__(self, dino_model, num_classes=3):
        """
        Initialize the custom DINO model with a classifier head.

        Args:
            dino_model (nn.Module): Pre-trained DINO model or backbone used for feature extraction.
            num_classes (int): The number of output classes for the classification head. Default is 3.
        """
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),  # Fully connected layer from 1024 to 256
            nn.LayerNorm(256),     # Layer normalization
            nn.Linear(256, 128),   # Fully connected layer from 256 to 128
            nn.ReLU(),             # ReLU activation
            nn.Linear(128, num_classes),  # Final fully connected layer to output classes
        )

    def forward(self, x):
        """
        Forward pass through the DINO model and the custom classifier.

        Args:
            x (torch.Tensor): Input tensor (e.g., images).

        Returns:
            torch.Tensor: The output tensor with predictions for each class.
        """
        x = self.dino_model(x)  # Pass through the pre-trained DINO model
        x = self.classifier(x)  # Pass through the custom classifier head
        return x
