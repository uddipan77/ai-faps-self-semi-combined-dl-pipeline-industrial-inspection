"""
Script for Evaluating a Custom EfficientNet Model on Test Data

This script evaluates a pre-trained Custom EfficientNet model on a test dataset to compute the macro F1-score. 
It loads image data from a specified directory, applies transformations, and uses the model for inference on 
test images. The predictions are evaluated against ground truth labels using scikit-learn's F1-score computation.

Main Components:
- Data loading and preprocessing using a custom dataset class.
- Initialization of a Custom EfficientNet model using user-defined parameters.
- Loading of pre-trained model weights from a specified checkpoint.
- Evaluation of the model in inference mode to compute the macro F1-score.

Workflow:
1. Load test data from a specified CSV file and perform image transformations.
2. Initialize the Custom EfficientNet model with the specified parameters.
3. Load pre-trained model weights from a checkpoint file.
4. Move the model to the specified device (CPU or GPU).
5. Evaluate the model on the test set:

Usage:
1. Update `best_model_path` to point to the pre-trained model weights.
2. Update `dropout_rate', 'fc_units', 'layer_freeze_upto'
3. Run the script to evaluate the model on the test data and print the macro F1-score.
"""

import sys
import os

# Adjust the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import torch
import numpy as np
from dataset.datasets import CustomImageDataset
from models.customefficientnet import define_model
from torchvision import transforms, models
import torchvision.transforms as transforms  
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score

# Define the image directory paths
image_dir = r'/home/woody/iwfa/iwfa111h/Supervised_Data/linear_winding_images_with_labels'

# Load the CSV files
test_df = pd.read_csv('/home/woody/iwfa/iwfa111h/Supervised_Data/Splits/test_v2024-03-18.csv')


# Specify the column containing class labels
y_columns = test_df.drop(columns=["image", "binary_NOK"]).columns  

# Define the image transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the datasets
test_dataset = CustomImageDataset(test_df, image_dir, y_columns, transform=test_transform)

# Create the data loaders
test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

# Initialize the list to store scores
all_scores = []
num_classes = 3
dropout_rate = 0.5
fc_units = 512
layer_freeze_upto = 'features.0.1.bias'

# Load the best model
best_model_path = '/home/vault/iwfa/iwfa111h/models/fixmtachefficientnet_50%_run1_best_valF1_0.8990.pth'
model = define_model(layer_freeze_upto, fc_units, dropout_rate, num_classes)

# Load the saved model state
model.load_state_dict(torch.load(best_model_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


for i in range(1):
    model.eval()  # Set the model to evaluation mode

    all_preds_test = []
    all_labels_test = []

    # Inference loop
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs))  # Convert logits to binary predictions
            all_preds_test.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())

    # Calculate macro F1-score
    test_f1 = f1_score(all_labels_test, all_preds_test, average='macro')
    print(f'Test F1-score for trial {i+1}: {test_f1:.4f}')
    all_scores.append(test_f1)

# Print the summary of all trials
print(f'All Test F1-scores: {all_scores}')
