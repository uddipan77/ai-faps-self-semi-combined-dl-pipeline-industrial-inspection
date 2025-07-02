import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
import torch
import numpy as np
import random
import copy
import os
import sys
from sklearn.metrics import f1_score
import argparse
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
        model_name (str): Name of the model to be loaded.
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

def train_model(args, best_params, experiment_output_dir):
    """
    Train the model with the best hyperparameters from Optuna study.

    Args:
        args (argparse.Namespace): Command line arguments passed to the script.
        best_params (dict): Best hyperparameters obtained from Optuna study.
        experiment_output_dir (str): Directory to save the trained model.
    """
    # ===============================
    # Set Seed for Reproducibility
    # ===============================
    set_seed(42)  # You can make this an argument if needed

    # ===============================
    # Device Configuration
    # ===============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===============================
    # Load Data
    # ===============================
    data_dir = args.data_dir
    unlabeled_image_dir = args.unlabeled_data_dir

    train_csv = args.train_csv
    val_csv = args.val_csv

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_excel(val_csv)

    y_columns = train_df.drop(columns=["image", "binary_NOK"]).columns

    # Define Strong Transform using RandAugment with best hyperparameters
    transform_weights = {
        'Rotate': best_params.get('Rotate_weight', 0.0),
        'ShearX': best_params.get('ShearX_weight', 0.0),
        'ShearY': best_params.get('ShearY_weight', 0.0),
        'TranslateXRel': best_params.get('TranslateXRel_weight', 0.0),
        'TranslateYRel': best_params.get('TranslateYRel_weight', 0.0),
        'ColorIncreasing': best_params.get('ColorIncreasing_weight', 0.0),
        'SharpnessIncreasing': best_params.get('SharpnessIncreasing_weight', 0.0),
        'AutoContrast': best_params.get('AutoContrast_weight', 0.0),
        'SolarizeIncreasing': best_params.get('SolarizeIncreasing_weight', 0.0),
        'SolarizeAdd': best_params.get('SolarizeAdd_weight', 0.0),
        'ContrastIncreasing': best_params.get('ContrastIncreasing_weight', 0.0),
        'BrightnessIncreasing': best_params.get('BrightnessIncreasing_weight', 0.0),
        'Equalize': best_params.get('Equalize_weight', 0.0),
        'PosterizeIncreasing': best_params.get('PosterizeIncreasing_weight', 0.0),
        'Invert': best_params.get('Invert_weight', 0.0),
    }

    hparams = {
        'magnitude': best_params.get('magnitude', 5),
        'magnitude_std': best_params.get('mstd', 0.0),
        'magnitude_max': 10,
        'increasing': best_params.get('increasing', 0)
    }

    strong_transform = rand_augment_transform(
        config_str=f'rand-m{hparams["magnitude"]}-mstd{hparams["magnitude_std"]}-n{best_params.get("num_layers",1)}-p{best_params.get("probability",0.5)}',
        hparams=hparams,
        transforms=transform_weights
    )

    # Define Weak Transform
    weak_transform = transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])

    # Define Normalize Transform
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),                       
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define Other Transforms
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

    # Initialize Datasets
    train_dataset = CustomImageDataset(train_df, data_dir, y_columns, transform=train_transform)
    val_dataset = CustomImageDataset(val_df, data_dir, y_columns, transform=test_transform)
    unlabeled_dataset = FixmatchUnlabeledImageDataset(
        unlabeled_image_dir, 
        normalize, 
        weak_transform=weak_transform,  
        strong_transform=strong_transform
    )

    # Initialize DataLoaders with Best Hyperparameters
    batch_size = best_params.get('batch_size', 8)  # Default to 8 if not specified
    unlabeled_batch_size = best_params.get('unlabeled_batch_size', 16)  # Default to 16 if not specified

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

    # Apply Freezing Logic based on Best Hyperparameters
    if best_params.get("freeze", False):
        freeze_percentage = best_params.get("freeze_percentage", 0.1)
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

    # ===============================
    # Define the optimizer and scheduler
    # ===============================
    optimizer_name = best_params.get("optimizer", "Adam")
    scheduler_name = best_params.get("scheduler", "ReduceLROnPlateau")

    lr = best_params.get("lr", 1e-3)

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    if scheduler_name == "ReduceLROnPlateau":
        factor = best_params.get("factor", 0.1)
        patience = best_params.get("patience", 10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=patience, factor=factor
        )
    elif scheduler_name == "CosineAnnealingLR":
        T_max = best_params.get("T_max", 50)
        eta_min = best_params.get("eta_min", 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

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
    max_epochs = 50
    threshold = best_params.get('threshold', 0.5)  # Default threshold
    best_val_f1 = -np.inf
    best_model_path = os.path.join(experiment_output_dir, 'finalmodel.pth')

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
        # Save the Best Model
        # ===============================
        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            best_model_state = copy.deepcopy(model.state_dict())
            # Save the best model as 'finalmodel.pth' in the output directory
            torch.save(best_model_state, best_model_path)
            print(f"Saved new best model at {best_model_path}")

    print(f"Training completed. Best Validation F1-score: {best_val_f1:.4f}")
    print(f"Best model saved at {best_model_path}")

def main():
    """
    Main function to run the training process with the best hyperparameters from Optuna study.

    Parses command-line arguments, loads the best trial from the Optuna study, and trains the model.
    """
    parser = argparse.ArgumentParser(description='Train FixMatch Model with Best Hyperparameters from Optuna Study')
    parser.add_argument('--study_name', type=str, required=True, help='Name of the Optuna study (e.g., study_10_percent)')
    parser.add_argument('--storage', type=str, default='sqlite:///fixmatch_hpo.db', help='Storage URL for Optuna study (e.g., sqlite:///fixmatch_hpo.db)')
    # Add Experiment Name Argument
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment (e.g., experiment_1)')
    parser.add_argument('--output_dir', type=str, required=True, help='Base directory to save the trained model')
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
    args = parser.parse_args()

    # ===============================
    # Load Optuna Study
    # ===============================
    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage
    )

    if study.best_trial is None:
        print("No completed trials found in the study.")
        sys.exit(1)

    best_trial = study.best_trial
    best_params = best_trial.params

    print("\n=== Best Trial ===")
    print(f"Trial Number: {best_trial.number}")
    print(f"Best Validation F1-score: {best_trial.value}")
    print("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # ===============================
    # Update Output Directory to Include Experiment Name
    # ===============================
    # Create a subdirectory for the experiment within the base output directory
    # For example: /home/vault/iwfa/iwfa111h/COMBINATION_LOGIC/experiment_1/
    experiment_output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    # ===============================
    # Start Training with Best Hyperparameters
    # ===============================
    train_model(args, best_params, experiment_output_dir)

if __name__ == '__main__':
    main()
