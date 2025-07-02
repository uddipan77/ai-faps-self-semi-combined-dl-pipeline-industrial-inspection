import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from PIL import Image
import torch
import numpy as np
import random
import copy
import os
import sys
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import optuna
from optuna.exceptions import TrialPruned
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from timm.data.auto_augment import rand_augment_transform
from contextlib import nullcontext  # Import nullcontext

# Add your project-specific paths and imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.manualseedsutils import set_seed
# from models.customefficientnet import SSLClassifier
from dataset.datasets import CustomImageDataset, FixmatchUnlabeledImageDataset
from utils.checkpoint import CheckpointManager
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    if model_name == "efficientnet_v2_s":
        weights = "IMAGENET1K_V1" if pretrain else None
        model = models.efficientnet_v2_s(weights=weights)
        if num_classes != 1000:
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(num_ftrs, num_classes, bias=True),
            )
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    if ssl:
        # Remove the final layer if SSL is True in case of model trained through SSL technique
        model = nn.Sequential(*list(model.children())[:-1])

    if checkpoint_path:
        # Load the model from the checkpoint file
        if backbone == "False":
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')[backbone])

    return model

# Define the image directory paths
image_dir = r'/home/woody/iwfa/iwfa111h/Supervised_Data/linear_winding_images_with_labels'

# Load the CSV files
test_df = pd.read_csv('/home/woody/iwfa/iwfa111h/Supervised_Data/Splits/test_v2024-03-18.csv')

# Specify the column containing class labels
y_columns = test_df.drop(columns=["image", "binary_NOK"]).columns  
num_classes = len(y_columns)

# Define the image transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the datasets
test_dataset = CustomImageDataset(test_df, image_dir, y_columns, transform=test_transform)

# Create the data loaders
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4)

# Initialize the list to store scores
all_scores = []

# Load the best model
best_model_path = '/home/vault/iwfa/iwfa111h/COMBINATION_LOGIC_TRAINING/v100-train_100%_1/finalmodel.pth'
backbone = load_model(
    model_name="efficientnet_v2_s",
    num_classes=num_classes,  # Dynamically set based on your dataset
    ssl=True,
)

model = nn.Sequential(
    backbone, nn.Flatten(), nn.Linear(1280, num_classes)
)
model.to(device)  # Move the model to the device

# Load the saved model state
model.load_state_dict(torch.load(best_model_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for i in range(1):
    model.eval()

    all_preds_test = []
    all_labels_test = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs))
            all_preds_test.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())

    test_f1 = f1_score(all_labels_test, all_preds_test, average='macro')
    print(f'Test F1-score for trial {i+1}: {test_f1:.4f}')
    all_scores.append(test_f1)

# Print the summary of all trials
print(f'All Test F1-scores: {all_scores}')
