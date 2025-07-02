"""
This script contains functions for training, validating, and testing a model using a mix of precision training.
It also includes a function for finding the best threshold for each class that maximizes the F1 score.

Functions:
- train_one_epoch_mix: Train the model for one epoch and calculate training metrics.
- validate_one_epoch_mix: Validate the model on the validation dataset for one epoch.
- test_loop_mix: Perform the testing loop for a given model on the test dataset.
- train_validation: Train and validate the model for one epoch.
- find_best_threshold_base: Find the best threshold for each class that maximizes the F1 score.
"""

import tqdm
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from utils.Utils import get_metrics
import torchmetrics

import numpy as np
from sklearn.metrics import f1_score, classification_report


def train_one_epoch_mix(
    model, train_loader, optimizer, loss_fn, scaler, device, metrics
):
    """
    Train the model for one epoch and calculate training metrics.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        loss_fn (torch.nn.Module): Loss function.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        device (torch.device): Device to run the training on (CPU or GPU).
        metrics (tuple): List of metric objects to calculate during training.

    Returns:
        tuple: Average training loss and calculated metrics (accuracy, precision, recall, F1 score).
    """
    model.train()
    total_loss = 0
    accuracy, precision, recall, f1_score = metrics

    for data, target in tqdm.tqdm(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True
        )
        optimizer.zero_grad()

        with autocast():
            output = model(data)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * data.size(0)
        preds = torch.round(torch.sigmoid(output)).detach()

        accuracy.update(preds, target)
        precision.update(preds, target)
        recall.update(preds, target)
        f1_score.update(preds, target)

    avg_train_loss = total_loss / len(train_loader.dataset)
    train_acc = accuracy.compute()
    train_prec = precision.compute()
    train_rec = recall.compute()
    train_f1 = f1_score.compute()

    # Reset metrics for next epoch
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()

    return avg_train_loss, train_acc, train_prec, train_rec, train_f1


def validate_one_epoch_mix(model, valid_loader, loss_fn, device, metrics):
    """
    Validate the model on the validation dataset for one epoch.

    Args:
        model (torch.nn.Module): The model to be validated.
        valid_loader (torch.utils.data.DataLoader): The validation data loader.
        loss_fn (torch.nn.Module): The loss function used for validation.
        device (torch.device): The device on which the validation will be performed.
        metrics (tuple): A tuple containing the metrics to be computed during validation.

    Returns:
        tuple: A tuple containing the average validation loss, validation accuracy,
            validation precision, validation recall, and validation F1 score.
    """
    model.eval()
    total_val_loss = 0
    accuracy, precision, recall, f1_score = metrics

    with torch.no_grad():
        for data, target in tqdm.tqdm(valid_loader):
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )

            with autocast():
                output = model(data)
                loss = loss_fn(output, target)

            total_val_loss += loss.item() * data.size(0)
            preds = torch.round(torch.sigmoid(output))

            accuracy.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            f1_score.update(preds, target)

    avg_val_loss = total_val_loss / len(valid_loader.dataset)
    val_acc = accuracy.compute()
    val_prec = precision.compute()
    val_rec = recall.compute()
    val_f1 = f1_score.compute()

    # Reset metrics for next epoch
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()

    return avg_val_loss, val_acc, val_prec, val_rec, val_f1


def test_loop_mix(model, test_loader, loss_fn, device, metrics, threshold):
    """
    Perform the testing loop for a given model on the test dataset.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        loss_fn (torch.nn.Module): The loss function used for evaluation.
        device (torch.device): The device on which the computation will be performed.
        metrics (tuple): A tuple containing the metrics to be computed during testing.
        threshold (float): The threshold value for binary classification.

    Returns:
        tuple: A tuple containing the average test loss, test accuracy, test precision,
                test recall, and test F1 score.
    """
    model.eval()
    total_val_loss = 0
    accuracy, precision, recall, f1_score = metrics

    all_preds = []
    all_labels = []
    threshold = torch.tensor(threshold).to(device)
    threshold = threshold[None, :]

    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader):
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )

            with autocast():
                output = model(data)
                loss = loss_fn(output, target)

            total_val_loss += loss.item() * data.size(0)
            model_output = torch.sigmoid(output)
            preds = (model_output > threshold).float()

            accuracy.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            f1_score.update(preds, target)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    avg_test_loss = total_val_loss / len(test_loader.dataset)
    test_acc = accuracy.compute()
    test_prec = precision.compute()
    test_rec = recall.compute()
    test_f1 = f1_score.compute()
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_preds = all_preds.astype(int)
    all_labels = all_labels.astype(int)
    class_names = ["multi-label_double_winding", "multi-label_gap", "multi-label_crossing"]

    sklearn_report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(sklearn_report)

    accuracy.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()

    return avg_test_loss, test_acc, test_prec, test_rec, test_f1


def train_validation(
    model, train_loader, valid_loader, optimizer, loss_fn, epoch, device
):
    """
    Train and validate the model.

    Args:
        model (torch.nn.Module): The model to be trained and validated.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        loss_fn (torch.nn.Module): Loss function.
        epoch (int): The current epoch number.
        device (torch.device): Device to run the training on (CPU or GPU).

    Returns:
        tuple: A tuple containing the average validation loss, validation accuracy,
            validation precision, validation recall, validation F1 score, training accuracy,
            average training loss, training precision, training recall, and training F1 score.
    """
    scaler = GradScaler()
    model.to(device)
    metrics = get_metrics(task="multilabel", num_classes=3, device=device)

    avg_train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch_mix(
        model, train_loader, optimizer, loss_fn, scaler, device, metrics
    )

    avg_val_loss, val_acc, val_prec, val_rec, val_f1 = validate_one_epoch_mix(
        model, valid_loader, loss_fn, device, metrics
    )

    print(
        f"Epoch {epoch}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_acc:.4f}, Training Precision: {train_prec:.4f}, Training Recall: {train_rec:.4f}, Training F1-score: {train_f1:.4f}"
    )
    print(
        f"Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation Precision: {val_prec:.4f}, Validation Recall: {val_rec:.4f}, Validation F1-score: {val_f1:.4f}"
    )

    return (
        avg_val_loss,
        val_acc,
        val_prec,
        val_rec,
        val_f1,
        train_acc,
        avg_train_loss,
        train_prec,
        train_rec,
        train_f1,
    )


def find_best_threshold_base(model, data_loader, DEVICE):
    """
    Finds the best threshold for each class that maximizes the F1 score.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        val_loader (torch.utils.data.DataLoader): The DataLoader for the validation data.
        DEVICE (torch.device): The device type (CPU or GPU).

    Returns:
        list: A list of the best thresholds for each class.
    """
    num_classes = 3
    best_thresholds = []
    model.eval()
    with torch.no_grad():
        for class_idx in range(num_classes):
            all_preds = []
            all_labels = []
            for batch_idx, (data, target) in tqdm.tqdm(enumerate(data_loader)):
                data = data.to(DEVICE)
                target = target.to(DEVICE)

                with autocast():
                    model_output = model(data)
                sigmoid = nn.Sigmoid()
                model_output = sigmoid(model_output)
                preds = model_output[:, class_idx]
                all_preds.append(preds.cpu().numpy())
                all_labels.append(target[:, class_idx].cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            thresholds = np.linspace(0, 1, num=100)
            f1_scores = []
            for threshold in thresholds:
                preds_binary = (all_preds > threshold).astype(int)
                f1 = f1_score(all_labels, preds_binary)
                f1_scores.append(f1)
            print(np.max(f1_scores))
            best_threshold = thresholds[np.argmax(f1_scores)]
            best_thresholds.append(best_threshold)

    return best_thresholds
