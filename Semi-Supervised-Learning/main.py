"""
Script for Training a Custom Model with Configurable Parameters

This script is designed to train a custom deep learning model (either an EfficientNet variant or a DINO model) 
using a specified algorithm (FixMatch or MixMatch) for Semi-supervised learning. The training configurations 
are loaded from a provided YAML file, and the script supports multiple trials with hyperparameter tuning using 
Optuna.

The script handles data loading, transformation, model initialization, and training, including checkpoints 
and logging of results.

Main Components:
- Configurable model architecture: EfficientNet or DINO with customizable parameters.
- Configurable data augmentation and transformations.
- Support for FixMatch and MixMatch semi-supervised learning algorithms.
- Loading of training, validation, and test datasets from CSV files.
- Saving and loading of model checkpoints.
- Logging and monitoring of training progress.

Usage:
    python train_model_script.py --config path/to/config.yaml

Args:
    --config (str): Path to the YAML configuration file containing training parameters.
"""

import pandas as pd
from dataset.datasets import CustomImageDataset, FixmatchUnlabeledImageDataset, MixmatchUnlabeledImageDataset
from models.customdinomodel import CustomDINONormModel
from models.customefficientnet import define_model
from utils.manualseedsutils import set_seed
import torchvision.transforms as transforms
import torch
import yaml
from train.train import train_model
from torch.utils.data import Dataset, DataLoader
import timm
from timm.data.auto_augment import rand_augment_transform  
import argparse
import sys

# Parsing arguments
parser = argparse.ArgumentParser(description='Training Model with Configurations')
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

# Load configuration from config.yaml
with open(args.config, 'r') as file:  
    config = yaml.safe_load(file)

# Load paths and other configuration parameters
paths = config['paths']
train_csv = paths['train_csv']
val_csv = paths['val_csv']
test_csv = paths['test_csv']
image_dir = paths['image_dir']
unlabeled_image_dir = paths['unlabeled_image_dir']
checkpoint_dir = paths['checkpoint_dir']
checkpoint_path = paths['checkpoint_path']
best_model_dir = paths['best_model_dir']
log_dir = paths['log_dir']
model_type = config['model_type']
algorithm = config['algorithm']
num_epochs = config['num_epochs']
batch_size = config['batch_size']
unlabeled_batch_size = config['unlabeled_batch_size']
patience = config['patience']
hparams = config['hparams']
#num_trials = config['num_trials']
num_classes = config['num_classes']
transform_weights = config['transform_weights']
magnitude = hparams['magnitude']
mstd = hparams['mstd']
num_layers = hparams['num_layers']
probability = hparams['probability']
#added - shouvik
trial_num = config['run_num']
experiment_name = config['experiment_name']    

# Load CSV files and set up datasets and transformations
train_df = pd.read_excel(train_csv)
val_df = pd.read_excel(val_csv)
test_df = pd.read_csv(test_csv)
y_columns = train_df.drop(columns=["image", "binary_NOK"]).columns

# Model initialization based on model type
if model_type == 'Efficientnet':
    model = define_model(config['efficientnet_params']['layer_freeze_upto'], config['efficientnet_params']['fc_units'],
                         config['efficientnet_params']['dropout_rate'], num_classes=num_classes)
elif model_type == 'dinov2':
    local_model_path = '/home/hpc/iwfa/iwfa111h/.cache/torch/hub/facebookresearch_dinov2_main'
    dino_model = torch.hub.load(local_model_path, 'dinov2_vitl14', source='local')
    model = CustomDINONormModel(dino_model, num_classes=num_classes)
else:
    raise ValueError(f"Unknown model type: {config['model_type']}")

# Define image transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

normalize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

weak_transform = transforms.Compose([transforms.RandomHorizontalFlip()])

# Configure transformations and datasets for FixMatch or MixMatch
if algorithm == 'fixmatch':
    strong_transform = rand_augment_transform(
        config_str=f'rand-m{magnitude}-mstd{mstd}-n{num_layers}-p{probability}',
        hparams=hparams, transforms=transform_weights
    )
    unlabeled_dataset = FixmatchUnlabeledImageDataset(
        unlabeled_image_dir, normalize, weak_transform=weak_transform, strong_transform=strong_transform
    )

elif algorithm == 'mixmatch':
    k_augmentations = hparams['K']
    strong_transform = rand_augment_transform(
        config_str='rand-m9-mstd0.5', 
        hparams={}
    )
    unlabeled_dataset = MixmatchUnlabeledImageDataset(
        unlabeled_image_dir, transform=normalize, strong_transform=strong_transform, k_augmentations=k_augmentations
    )
else:
    raise ValueError(f"Unknown algorithm: {algorithm}")

# Initialize datasets
train_dataset = CustomImageDataset(train_df, image_dir, y_columns, transform=train_transform)
val_dataset = CustomImageDataset(val_df, image_dir, y_columns, transform=test_transform)

# Set up data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=False, num_workers=4)

# Start training loop
print(f"Starting Run {trial_num}...")
sys.stdout.flush()
    
trial_seed = 42 + trial_num
set_seed(trial_seed)
    
for epoch_info in train_model(model, train_loader, val_loader, unlabeled_loader, train_dataset, unlabeled_dataset, val_dataset, config['params'], config['hparams'], num_epochs, patience, checkpoint_dir, best_model_dir, checkpoint_path, log_dir, trial_num, algorithm, experiment_name):
    print(f'Run {trial_num}, Epoch {epoch_info["epoch"]}/{num_epochs}, '
            f'Training Loss: {epoch_info["epoch_loss"]:.4f}, Unsupervised Loss: {epoch_info["epoch_unsupervised_loss"]:.4f}, '
            f'Training F1: {epoch_info["train_f1"]:.4f}, Validation Loss: {epoch_info["val_epoch_loss"]:.4f}, Validation F1: {epoch_info["val_f1"]:.4f}')
    sys.stdout.flush()
