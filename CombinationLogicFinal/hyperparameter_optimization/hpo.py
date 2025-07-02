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
#from utils.manualseedsutils import set_seed
# from models.customefficientnet import SSLClassifier
from dataset.datasets import CustomImageDataset, FixmatchUnlabeledImageDataset
#from utils.checkpoint import CheckpointManager
import argparse

# ===============================
# Model Loading and Definition
# ===============================

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
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')[backbone])

    return model

# ===============================
# Objective Function for Optuna
# ===============================

def objective(trial, args):
    """
    Defines the objective function for the Optuna optimization.

    Args:
        trial (optuna.Trial): The current Optuna trial object.
        args (Namespace): The command line arguments containing paths and configuration settings.

    Returns:
        float: The best validation F1 score achieved during the trial.
    """
    # ===============================
    # Hyperparameter Suggestions
    # ===============================
    # Threshold for pseudo-labeling
    threshold = trial.suggest_categorical('threshold', [0.4, 0.5, 0.6, 0.7, 0.8])
    
    # Augmentation parameters
    magnitude = trial.suggest_int("magnitude", 0, 10)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    probability = trial.suggest_float("probability", 0.1, 1.0)
    mstd = trial.suggest_float("mstd", 0.0, 1.0)
    increasing = trial.suggest_categorical("increasing", [0, 1])
    
    # Transformation Weights for RandAugment
    transform_weights = {
        'Rotate': trial.suggest_float('Rotate_weight', 0.0, 3.0),
        'ShearX': trial.suggest_float('ShearX_weight', 0.0, 2.0),
        'ShearY': trial.suggest_float('ShearY_weight', 0.0, 2.0),
        'TranslateXRel': trial.suggest_float('TranslateXRel_weight', 0.0, 1.0),
        'TranslateYRel': trial.suggest_float('TranslateYRel_weight', 0.0, 1.0),
        'ColorIncreasing': trial.suggest_float('ColorIncreasing_weight', 0.0, 1.0),
        'SharpnessIncreasing': trial.suggest_float('SharpnessIncreasing_weight', 0.0, 1.0),
        'AutoContrast': trial.suggest_float('AutoContrast_weight', 0.0, 1.0),
        'SolarizeIncreasing': trial.suggest_float('SolarizeIncreasing_weight', 0.0, 1.0),
        'SolarizeAdd': trial.suggest_float('SolarizeAdd_weight', 0.0, 1.0),
        'ContrastIncreasing': trial.suggest_float('ContrastIncreasing_weight', 0.0, 1.0),
        'BrightnessIncreasing': trial.suggest_float('BrightnessIncreasing_weight', 0.0, 1.0),
        'Equalize': trial.suggest_float('Equalize_weight', 0.0, 1.0),
        'PosterizeIncreasing': trial.suggest_float('PosterizeIncreasing_weight', 0.0, 1.0),
        'Invert': trial.suggest_float('Invert_weight', 0.0, 1.0),
    }
    
    # Hyperparameters for RandAugment
    hparams = {
        'magnitude': magnitude,
        'magnitude_std': mstd,
        'magnitude_max': 10,
        'increasing': increasing
    }
    
    # Define Strong Transform using RandAugment
    strong_transform = rand_augment_transform(
        config_str=f'rand-m{magnitude}-mstd{mstd}-n{num_layers}-p{probability}',
        hparams=hparams,
        transforms=transform_weights
    )
    
    # ===============================
    # Batch Size as Hyperparameter
    # ===============================
    batch_size = 8
    
    # Unlabeled batch size can be a multiple of the labeled batch size, e.g., mu = 7
    # Fixed as per your setup
    unlabeled_batch_size = 16
    
    # ===============================
    # Freeze Parameters
    # ===============================
    freez_or_not = trial.suggest_categorical("freeze", [True, False])
    freeze_percentage = trial.suggest_float("freeze_percentage", 0.1, 0.9)
    
    # ===============================
    # Optimizer and Scheduler Parameters
    # ===============================
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    scheduler_name = trial.suggest_categorical("scheduler", ["ReduceLROnPlateau", "CosineAnnealingLR"])
    
    # Scheduler-specific hyperparameters
    factor = trial.suggest_float("factor", 0.1, 0.9)
    patience = trial.suggest_int("patience", 0, 50)
    T_max = trial.suggest_int("T_max", 50, 1000)
    eta_min = trial.suggest_float("eta_min", 0, 0.1)
    
    # Learning rate
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    
    # ===============================
    # Device Configuration
    # ===============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ===============================
    # Load Data
    # ===============================
    # Define paths from args
    data_dir = args.data_dir
    unlabeled_image_dir = args.unlabeled_data_dir
    
    train_csv = args.train_csv
    val_csv = args.val_csv
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_excel(val_csv)
    
    y_columns = train_df.drop(columns=["image", "binary_NOK"]).columns
    
    # Define Data Transforms
    train_transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(), 
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    
    test_transform = transforms.Compose([ 
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    
    normalize = transforms.Compose([ 
        transforms.Resize((224, 224)),                       
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    
    weak_transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip() 
    ])
    
    # Initialize Datasets
    train_dataset = CustomImageDataset(train_df, data_dir, y_columns, transform=train_transform)
    val_dataset = CustomImageDataset(val_df, data_dir, y_columns, transform=test_transform)
    unlabeled_dataset = FixmatchUnlabeledImageDataset(
        unlabeled_image_dir, 
        normalize, 
        weak_transform=weak_transform,  
        strong_transform=strong_transform
    )
    
    # Initialize DataLoaders with Variable Batch Sizes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=True, num_workers=4)
    
    # ===============================
    # Initialize Model
    # ===============================
    num_classes = len(y_columns)  # Adjust based on your dataset
    backbone = load_model(
        model_name="efficientnet_v2_s",
        num_classes=num_classes,  # Dynamically set based on your dataset
        checkpoint_path=args.selfsup_model_path,
        backbone="False",
        pretrain=True,
        ssl=True,
    )

    print(f'Loaded self-supervised pretrained backbone from {args.selfsup_model_path}')
    model = nn.Sequential(
        backbone, nn.Flatten(), nn.Linear(1280, num_classes)
    )
    
    model.to(device)  # Move the model to the device
    
    # Apply Freezing Logic
    if freez_or_not:
        frozen_layers = []
        total_layers = len(list(model.named_children()))
        print(f"freeze_percentage: {freeze_percentage}")
        num_layers_to_freeze = int(total_layers * freeze_percentage)
        named_layers = list(model.named_children())
        for name, layer in named_layers[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
            frozen_layers.append(name)
        print(f"Frozen layers: {frozen_layers}")
    else:
        print("No layers are frozen.")
    
    # Define the optimizer and scheduler
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=patience, factor=factor
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    
    # ===============================
    # Define Loss Function
    # ===============================
    loss_function = nn.BCEWithLogitsLoss()
    
    # ===============================
    # Initialize Mixed Precision Components
    # ===============================
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # ===============================
    # Training Loop with Mixed Precision
    # ===============================
    max_epochs = 20  # Adjust the number of epochs as needed
    best_val_f1 = -np.inf
    best_model_path = os.path.join(args.output_dir, 'finalmodel.pth')
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        all_labels = []
        all_predictions = []
        running_supervised_loss = 0.0
        running_unsupervised_loss = 0.0

        labeled_iter = iter(train_loader)
        unlabeled_iter = iter(unlabeled_loader)
        num_batches = len(train_loader)

        for _ in range(num_batches):
            try:
                labeled_inputs, labels = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(train_loader)
                labeled_inputs, labels = next(labeled_iter)

            try:
                weak_inputs, strong_inputs = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                weak_inputs, strong_inputs = next(unlabeled_iter)

            # Move data to device
            labeled_inputs, labels = labeled_inputs.to(device), labels.to(device).float()  # Convert labels to float
            weak_inputs, strong_inputs = weak_inputs.to(device), strong_inputs.to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast() if scaler else nullcontext():
                all_outputs = model(torch.cat([labeled_inputs, weak_inputs, strong_inputs], dim=0))
                # Split outputs
                labeled_outputs = all_outputs[:labeled_inputs.size(0)]
                unlabeled_outputs = all_outputs[labeled_inputs.size(0):]
                weak_outputs, strong_outputs = torch.chunk(unlabeled_outputs, 2, dim=0)
                # Compute losses
                supervised_loss = loss_function(labeled_outputs, labels)
                # Generate Pseudo Labels for Unsupervised Loss
                with torch.no_grad():
                    weak_probs = torch.sigmoid(weak_outputs)
                    pseudo_labels = torch.where(weak_probs > threshold, 1., 0.)
                unsupervised_loss = loss_function(strong_outputs, pseudo_labels)
                # Total Loss
                loss = supervised_loss + unsupervised_loss

            # Backward pass with mixed precision
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            # Accumulate Losses
            running_supervised_loss += supervised_loss.item() * labeled_inputs.size(0)
            running_unsupervised_loss += unsupervised_loss.item() * labeled_inputs.size(0)

            # Collect Predictions for F1-score
            preds = torch.round(torch.sigmoid(labeled_outputs)).detach().int()  # Convert to int
            all_labels.append(labels.cpu().numpy().astype(int))               # Use append instead of extend
            all_predictions.append(preds.cpu().numpy().astype(int))          # Use append instead of extend

        # Stack lists into 2D NumPy arrays
        all_labels = np.vstack(all_labels)
        all_predictions = np.vstack(all_predictions)

        # Debugging: Print shapes and data types
        print(f"Epoch {epoch} Training Labels shape: {all_labels.shape}, dtype: {all_labels.dtype}")
        print(f"Epoch {epoch} Training Predictions shape: {all_predictions.shape}, dtype: {all_predictions.dtype}")

        # Compute Training Metrics
        epoch_supervised_loss = running_supervised_loss / len(train_dataset)
        epoch_unsupervised_loss = running_unsupervised_loss / len(unlabeled_dataset)
        train_f1_score = f1_score(all_labels, all_predictions, average='macro')

        # ===============================
        # Validation Phase
        # ===============================
        model.eval()
        all_preds_val = []
        all_labels_val = []
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()  # Convert labels to float
                with torch.cuda.amp.autocast() if scaler else nullcontext():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.round(torch.sigmoid(outputs)).int()                # Convert to int
                all_preds_val.append(preds.cpu().numpy().astype(int))             # Use append instead of extend
                all_labels_val.append(labels.cpu().numpy().astype(int))           # Use append instead of extend

        # Stack lists into 2D NumPy arrays
        all_preds_val = np.vstack(all_preds_val)
        all_labels_val = np.vstack(all_labels_val)

        # Debugging: Print shapes and data types
        print(f"Epoch {epoch} Validation Labels shape: {all_labels_val.shape}, dtype: {all_labels_val.dtype}")
        print(f"Epoch {epoch} Validation Predictions shape: {all_preds_val.shape}, dtype: {all_preds_val.dtype}")

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_f1_score = f1_score(all_labels_val, all_preds_val, average='macro')

        # ===============================
        # Logging
        # ===============================
        print(f'Epoch {epoch}, '
              f'Training Supervised Loss: {epoch_supervised_loss:.4f}, '
              f'Unsupervised Loss: {epoch_unsupervised_loss:.4f}, '
              f'Training F1-score: {train_f1_score:.4f}, '
              f'Validation Loss: {val_epoch_loss:.4f}, '
              f'Validation F1-score: {val_f1_score:.4f}')

        # ===============================
        # Scheduler Step
        # ===============================
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(val_f1_score)  # Step based on validation F1-score
        else:
            scheduler.step()

        # ===============================
        # Optuna Trial Reporting and Pruning
        # ===============================
        # Log the validation F1 score for each epoch
        trial.report(val_f1_score, epoch)

        # Check for improvement
        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            best_model_state = copy.deepcopy(model.state_dict())
            # Save the best model as 'finalmodel.pth' in the study directory
            torch.save(best_model_state, best_model_path)
            print(f"Saved new best model at {best_model_path}")

        # Prune the trial if the validation F1 score is less than the best accuracy so far
        if trial.should_prune():
            print("Pruning trial due to insufficient performance.")
            raise TrialPruned()

    return best_val_f1

    # ===============================
    # Main Function to Run Optuna Study
    # ===============================

def main():
    """
    Main function to run the Optuna hyperparameter optimization study.

    Parses command-line arguments, sets up the study, and runs the optimization process.
    """
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization for FixMatch Model')
    parser.add_argument('--data_dir', type=str, default='/home/woody/iwfa/iwfa111h/Supervised_Data/linear_winding_images_with_labels',
                        help='Path to the labeled dataset directory')
    parser.add_argument('--unlabeled_data_dir', type=str, default='/home/woody/iwfa/iwfa111h/unalablled_data',
                        help='Path to the unlabeled dataset directory')
    parser.add_argument('--train_csv', type=str, default='/home/woody/iwfa/iwfa111h/Supervised_Data/Percentage_Splits/train_v2024-03-18_25%.csv',
                        help='Path to the training CSV file')
    parser.add_argument('--val_csv', type=str, default='/home/woody/iwfa/iwfa111h/Supervised_Data/Splits/validation_v2024-03-18.xlsx',
                        help='Path to the validation CSV file')
    parser.add_argument('--selfsup_model_path', type=str, default='/home/vault/iwfa/iwfa111h/badar_data/Code/Pretrained_models_and_logs/SSL_Pretrained_Backbones/SimCLR_EfficicentNetV2_S.pth',
                        help='Path to the self-supervised pretrained model checkpoint')
    # Updated default output directory to the COMBINATION_LOGIC path
    parser.add_argument('--output_dir', type=str, default='/home/vault/iwfa/iwfa111h/COMBINATION_LOGIC',
                        help='Base directory to save the best model and logs')
    parser.add_argument('--n_trials', type=int, default=2, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Time limit for Optuna study in seconds')
    parser.add_argument('--study_name', type=str, required=True, help='Name of the Optuna study (e.g., study_10_percent)')
    parser.add_argument('--storage', type=str, default='sqlite:///fixmatch_hpo.db', help='Storage URL for Optuna study (e.g., sqlite:///fixmatch_hpo.db)')
    parser.add_argument('--direction', type=str, default='maximize', choices=['maximize', 'minimize'], help='Optimization direction for the study')
    args = parser.parse_args()

    # ===============================
    # Update Output Directory to Include Study Name
    # ===============================
    # Create a subdirectory for the study within the base output directory
    # For example: /home/vault/iwfa/iwfa111h/COMBINATION_LOGIC/study_10_percent/
    args.output_dir = os.path.join(args.output_dir, args.study_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # ===============================
    # Define the Objective Function with Fixed Arguments
    # ===============================
    def wrapped_objective(trial):
        return objective(trial, args)

    # ===============================
    # Start Optuna Study
    # ===============================
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1),
        load_if_exists=True  # Load existing study if it exists
    )

    # ===============================
    # Optimize the Objective Function
    # ===============================
    try:
        study.optimize(wrapped_objective, n_trials=args.n_trials, timeout=args.timeout)
    except Exception as e:
        print(f"An error occurred during the Optuna study: {e}")

    # ===============================
    # Post-Optimization: Display Best Trial and Confirm Final Model Saving
    # ===============================
    if study.best_trial:
        best_trial = study.best_trial
        print("\n=== Best Trial ===")
        print(f"Trial Number: {best_trial.number}")
        print(f"Best Validation F1-score: {best_trial.value}")
        print("Best Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo trials completed successfully.")

if __name__ == '__main__':
    main()
