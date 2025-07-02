"""
This script implements the SimCLR model using EfficientNet-v2 as the backbone for self-supervised learning (SSL).
It includes the training loop, early stopping, and model saving features.

Functions:
- SimCLR: A class implementing the SimCLR model with a backbone and projection head.
"""
import torch
import tqdm
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.Utils import EarlyStopping, create_json, setup_wandb
from data.Dataset import FilteredImageDataset, get_data
from modeling.make_model import load_model

print("debug - 1")

# Configuration dictionary containing model settings, learning rate, etc.
config_dict = {
    "model_name": "efficientnet_v2_s",
    "batch_size": 50,
    "input_dim": 1280,
    "lr": 0.0006,
    "weight_decay": 0.04,
    "epochs": 50,
    "expriment_save_dir": "/home/vault/iwfa/iwfa111h/SSL_REPRODUCTION/simCLR_EfficientNet_v2_s",
    "num_workers": 8,
}

# Create the directory for saving experiments if it does not exist
if not os.path.exists(config_dict["expriment_save_dir"]):
    os.makedirs(config_dict["expriment_save_dir"])

print("debug - 2")

class SimCLR(nn.Module):
    """
    A SimCLR model with a backbone and projection head for self-supervised learning.

    Args:
        backbone (torch.nn.Module): The backbone model (e.g., EfficientNet).

    Attributes:
        backbone (torch.nn.Module): The backbone model.
        projection_head (torch.nn.Module): The projection head for mapping features to lower-dimensional space.
    """
    def __init__(self, backbone):
        """
        Initialize the SimCLR model with the specified backbone.

        Args:
            backbone (torch.nn.Module): The backbone model (e.g., EfficientNet).
        """
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(1280, 512, 128)

    def forward(self, x):
        """
        Forward pass through the SimCLR model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the backbone and projection head.
        """
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


# Load the backbone model and create the SimCLR model
backbone = load_model(model_name=config_dict["model_name"], ssl=True)
model = SimCLR(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define the transformation pipeline for SimCLR
transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)

# Define patterns to ignore during dataset filtering
patterns_to_ignore = [
    "Spule001",
    "Spule014",
    "Spule018",
    "Suple023",
    "Spule033",
    "Spule034",
    "Spule002",
    "Spule017",
    "Spule024",
    "Spule029",
    "Spule031",
    "Spule039",
]  # Add your specific patterns here

# Initialize the custom dataset with the defined transformations
dataset = FilteredImageDataset(
    root_dir="/home/woody/iwfa/iwfa111h/unalablled_data",
    patterns_to_ignore=patterns_to_ignore,
    transform=transform,
)

# Create the data loader for training
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config_dict["batch_size"],
    shuffle=True,
    drop_last=True,
    num_workers=config_dict["num_workers"],
)

# Initialize early stopping with a patience of 10 epochs
early_stopping = EarlyStopping(
    patience=10,
    verbose=True,
    path=os.path.join(config_dict["expriment_save_dir"], f"Earlystopping.pth"),
)

# Define the loss function and optimizer
criterion = NTXentLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config_dict["lr"])

# Start the training loop
print("Starting Training")
for epoch in range(config_dict["epochs"]):
    total_loss = 0
    for batch in tqdm.tqdm(dataloader):
        x0, x1 = batch[0]
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    early_stopping(avg_loss, model.backbone)

    if early_stopping.early_stop:
        print("Early stopping")
        torch.save(
            {
                "projection_head": model.projection_head.state_dict(),
                "backbone": model.backbone.state_dict(),
                "epoch": epoch,
            },
            os.path.join(config_dict["expriment_save_dir"], f"Earlystopping.pth"),
        )
        break

    # Write results to a JSON file
    create_json(avg_loss, epoch, config_dict["expriment_save_dir"])

    # Save a checkpoint every 50 epochs
    if epoch % 50 == 0:
        torch.save(
            {
                "projection_head": model.projection_head.state_dict(),
                "backbone": model.backbone.state_dict(),
                "epoch": epoch,
            },
            os.path.join(config_dict["expriment_save_dir"], f"checkpoint_{epoch}.pth"),
        )

    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
