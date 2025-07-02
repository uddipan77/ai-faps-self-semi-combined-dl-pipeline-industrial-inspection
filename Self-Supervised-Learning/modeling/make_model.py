import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

"""
This script contains functions for loading models and classifying using the DINOv2 pretrained model.
It includes functionality to add a classifier head to the pretrained model.
"""

def load_model(
    model_name,
    num_classes=1000,
    ssl=False,
    checkpoint_path=None,
    pretrain=True,
    backbone="False",
):
    """
    Load a specified model and replace the final layer with a new layer with num_classes output features.

    Args:
        model_name (str): Name of the model to be loaded. Options: 'resnet50', 'resnet101', 'efficientnet-b6', 'EfficientNet_v2_m', 'efficientnet_v2_l', 'efficientnet_v2_s', 'vit'.
        num_classes (int): Number of output classes.
        ssl (bool): If True, the model is loaded without the final layer.
        checkpoint_path (str): Path to the checkpoint file to load the model from.
        pretrain (bool): If True, the model is loaded with pretrained weights.
        backbone (str): If not False, the model is loaded from the checkpoint file with the specified backbone name (used for SSL).

    Returns:
        torch.nn.Module: The loaded model.
    """
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name == "efficientnet-b6":
        model = EfficientNet.from_pretrained("efficientnet-b6", num_classes=num_classes)
    elif model_name == "EfficientNet_v2_m":
        weights = "IMAGENET1K_V1" if pretrain else None
        model = models.efficientnet_v2_m(weights=weights)
        if num_classes != 1000:
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(num_ftrs, num_classes, bias=True),
            )
    elif model_name == "efficientnet_v2_l":
        weights = "IMAGENET1K_V1" if pretrain else None
        model = models.efficientnet_v2_l(weights=weights)
        if num_classes != 1000:
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(num_ftrs, num_classes, bias=True),
            )
    elif model_name == "efficientnet_v2_s":
        weights = "IMAGENET1K_V1" if pretrain else None
        model = models.efficientnet_v2_s(weights=weights)
        if num_classes != 1000:
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(num_ftrs, num_classes, bias=True),
            )
    # elif model_name == "vit":
    #     model = MaskedCausalVisionTransformer(
    #         img_size=224,
    #         patch_size=32,
    #         embed_dim=768,
    #         depth=12,
    #         num_heads=12,
    #         qk_norm=False,
    #         class_token=False,
    #         no_embed_class=True,
    #     )
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    if ssl:
        # Remove the final layer if SSL is True in case of model trained through SSL technique
        model = nn.Sequential(*list(model.children())[:-1])

    if checkpoint_path:
        
        # Load the model from the checkpoint file
        if backbone == "False":
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            model.load_state_dict(torch.load(checkpoint_path)[backbone])

    return model


class DinoVisionTransformerClassifier(nn.Module):
    """
    A classifier head for the DINOv2 Vision Transformer model.

    Args:
        dino_model (nn.Module): The DINOv2 model.
        config (dict): Configuration dictionary containing the number of output classes.

    Attributes:
        dino_model (nn.Module): The DINOv2 model.
        num_classes (int): Number of output classes.
        classifier (nn.Sequential): The classifier head.
    """
    def __init__(self, dino_model, config):
        """
        Initialize the classifier head for the DINOv2 Vision Transformer model.

        Args:
            dino_model (nn.Module): The DINOv2 model.
            config (dict): Configuration dictionary containing the number of output classes.
        """
        super(DinoVisionTransformerClassifier, self).__init__()
        self.dino_model = dino_model
        self.num_classes = config["num_classes"]
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.dino_model(x)
        x = self.dino_model.norm(x)
        x = self.classifier(x)
        return x
