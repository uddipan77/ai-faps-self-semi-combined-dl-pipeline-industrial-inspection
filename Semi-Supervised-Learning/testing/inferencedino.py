"""
Script for Evaluating a Dino Model on Test Data

This script evaluates a pre-trained CustomDINONormModel based on DINO using test data. 
The evaluation computes and prints the macro F1-score on the test set. The script uses 
PyTorch for model loading, data transformations, and computation of predictions, and it 
relies on scikit-learn to calculate the F1-score.

Main Components:
- Custom model loading using DINO as a backbone.
- Data loading and transformation pipeline for test data.
- Model evaluation in inference mode.
- Calculation of macro F1-score to assess model performance.

Workflow:
1. Load the test data and perform necessary image transformations.
2. Initialize the DINO-based custom model and load the pre-trained model weights.
3. Move the model to the specified device (CPU or GPU).
4. Evaluate the model on the test set:

Usage:
1. Update `best_model_path` to point to the pre-trained model weights
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
from models.customdinomodel import CustomDINONormModel
import torchvision.transforms as transforms  
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader
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

# Load the best model
local_model_path = '/home/hpc/iwfa/iwfa111h/.cache/torch/hub/facebookresearch_dinov2_main'
dino_model = torch.hub.load(local_model_path, 'dinov2_vitl14', source='local')
best_model_path = '/home/vault/iwfa/iwfa111h/models/fixmtachefficientnet_10%_run1_best_valF1_0.8743.pth'
model = CustomDINONormModel(dino_model, num_classes=len(y_columns))

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
    print(f'Test F1-score for run {i+1}: {test_f1:.4f}')
    all_scores.append(test_f1)

# Print the summary of all trials
print(f'All Test F1-scores: {all_scores}')
