import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score
import copy
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.checkpoint import CheckpointManager
from utils.mixmatchutils import sharpen, mixup_data, mixup_criterion

def train_model(model, train_loader, val_loader, unlabeled_loader, train_dataset, unlabeled_dataset, val_dataset, params, hparams, num_epochs, patience, checkpoint_dir, best_model_dir, checkpoint_path, log_dir, trial_num, algorithm, experiment_name):
    """
    Training function for semi-supervised learning using FixMatch and MixMatch algorithms.

    This function selects the appropriate training algorithm (FixMatch or MixMatch) 
    based on the given configuration and starts the training process.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for labeled training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        unlabeled_loader (torch.utils.data.DataLoader): DataLoader for unlabeled data.
        train_dataset (torch.utils.data.Dataset): Dataset for labeled data.
        unlabeled_dataset (torch.utils.data.Dataset): Dataset for unlabeled data.
        val_dataset (torch.utils.data.Dataset): Dataset for validation data.
        params (dict): Parameters such as learning rate and thresholds.
        hparams (dict): Hyperparameters such as temperature and alpha.
        num_epochs (int): The number of epochs to train the model.
        patience (int): Early stopping patience based on validation performance.
        checkpoint_dir (str): Directory to save checkpoints.
        best_model_dir (str): Directory to save the best model.
        checkpoint_path (str): Path to load a specific checkpoint, if available.
        log_dir (str): Directory to save TensorBoard logs.
        trial_num (int): Trial number for hyperparameter tuning.
        algorithm (str): Algorithm to use ('fixmatch' or 'mixmatch').
        experiment_name (str): Name of the experiment.

    Returns:
        Generator: Yields a dictionary containing training metrics for each epoch.
    """
    
    if algorithm == 'fixmatch':
        return train_fixmatch(model, train_loader, val_loader, unlabeled_loader, train_dataset, unlabeled_dataset, val_dataset, params, hparams, num_epochs, patience, checkpoint_dir, best_model_dir, checkpoint_path, log_dir, trial_num, experiment_name) 
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm}")
    

def train_fixmatch(model, train_loader, val_loader, unlabeled_loader, train_dataset, unlabeled_dataset, val_dataset, params, hparams, num_epochs, patience, checkpoint_dir=None, best_model_dir=None, checkpoint_path=None, log_dir=None, trial_num=1, experiment_name=None):
    """
    Trains the model using the FixMatch algorithm with both labeled and unlabeled data. 
    The training includes both supervised and unsupervised loss calculations, and 
    early stopping based on validation F1 score.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for labeled training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        unlabeled_loader (torch.utils.data.DataLoader): DataLoader for unlabeled data.
        train_dataset (torch.utils.data.Dataset): Dataset for labeled training data.
        unlabeled_dataset (torch.utils.data.Dataset): Dataset for unlabeled data.
        val_dataset (torch.utils.data.Dataset): Dataset for validation data.
        params (dict): Parameters such as learning rate and threshold.
        hparams (dict): Hyperparameters such as temperature and alpha.
        num_epochs (int): The number of epochs to train the model.
        patience (int): Early stopping patience based on validation performance.
        checkpoint_dir (str, optional): Directory to save checkpoints.
        best_model_dir (str, optional): Directory to save the best model.
        checkpoint_path (str, optional): Path to load a specific checkpoint, if available.
        log_dir (str, optional): Directory to save TensorBoard logs.
        trial_num (int, optional): Trial number for hyperparameter tuning.
        experiment_name (str, optional): Name of the experiment.

    Returns:
        Generator: Yields training and validation metrics for each epoch.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    loss_function = nn.BCEWithLogitsLoss()
    start_epoch = 0
    best_val_f1 = 0  
    early_stop_counter = 0  
    best_model_state = None

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            start_epoch, best_val_f1, early_stop_counter = checkpoint_manager.load_checkpoint(checkpoint_path, model, optimizer)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print("Starting from scratch.")

    early_stop_counter = 0  # Initialize early stop counter

    logs = []
    writer = SummaryWriter(f'{log_dir}/log_{experiment_name}')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        all_labels = []
        all_predictions = []
        running_loss = 0.0
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

            labeled_inputs, labels = labeled_inputs.to(device), labels.to(device)
            weak_inputs, strong_inputs = weak_inputs.to(device), strong_inputs.to(device)

            all_inputs = torch.cat([labeled_inputs, weak_inputs, strong_inputs], dim=0)
            optimizer.zero_grad()
            all_outputs = model(all_inputs)

            labeled_outputs = all_outputs[:labeled_inputs.size(0)]
            weak_outputs, strong_outputs = torch.chunk(all_outputs[labeled_inputs.size(0):], 2, dim=0)

            # Supervised loss
            supervised_loss = loss_function(labeled_outputs, labels)

            # Unsupervised loss with pseudo-labels
            with torch.no_grad():
                weak_outputs = torch.sigmoid(weak_outputs)
                pseudo_labels = torch.where(weak_outputs > params['threshold'], 1., 0.)
            unsupervised_loss = loss_function(strong_outputs, pseudo_labels)

            loss = supervised_loss + unsupervised_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labeled_inputs.size(0)
            running_unsupervised_loss += unsupervised_loss.item() * labeled_inputs.size(0)

            preds = torch.round(torch.sigmoid(labeled_outputs)).detach()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_dataset)
        epoch_unsupervised_loss = running_unsupervised_loss / len(unlabeled_dataset)
        train_f1 = f1_score(all_labels, all_predictions, average='macro')

        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.round(torch.sigmoid(outputs))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        writer.add_scalar('Loss/Training', epoch_loss, epoch)
        writer.add_scalar('Loss/Validation', val_epoch_loss, epoch)
        writer.add_scalar('F1/Training', train_f1, epoch)
        writer.add_scalar('F1/Validation', val_f1, epoch)
        
        yield {
            'epoch': epoch + 1,
            'epoch_loss': epoch_loss,
            'epoch_unsupervised_loss': epoch_unsupervised_loss,
            'train_f1': train_f1,
            'val_epoch_loss': val_epoch_loss,
            'val_f1': val_f1,
        }

        logs.append([epoch + 1, epoch_loss, epoch_unsupervised_loss, train_f1, val_epoch_loss, val_f1])

        # Save the best model based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, os.path.join(best_model_dir, f'{experiment_name}_best_valF1_{best_val_f1:.4f}.pth'))
            early_stop_counter = 0 
        else:
            early_stop_counter += 1

        # Save checkpoint after each epoch
        checkpoint_manager.save_checkpoint(epoch + 1, best_val_f1, train_f1, epoch_loss, epoch_unsupervised_loss, early_stop_counter, model, optimizer, experiment_name)    

        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
