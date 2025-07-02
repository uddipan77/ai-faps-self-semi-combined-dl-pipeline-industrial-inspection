"""
This script contains utility functions for various tasks related to model training and evaluation.

The functions in this script include:
- `freeze_model`: Freezes the layers of a model up to a specified ratio.
- `write_results`: Manually writes the results to a JSON file.
- `EarlyStopping`: Class for early stopping during training based on validation loss.
- `setup_wandb`: Sets up Weights & Biases (wandb) for logging.
- `get_metrics`: Retrieves metrics for model evaluation.
- `create_json`: Creates a JSON file to store loss values.
- `save_checkpoint`: Saves the model checkpoint.
- `set_seed`: Sets the random seed for reproducibility.
"""

import json
import os
import numpy as np
import torch
import wandb
import os
import torchmetrics
import random


def freeze_model(model, freeze_ratio):
    """
    Freezes the layers of the model up to the specified ratio.

    Args:
        model (torch.nn.Module): The model whose layers are to be frozen.
        freeze_ratio (float): The ratio of layers to freeze (between 0 and 1).
    """
    total_layers = len(list(model.named_children()))
    num_layers_to_freeze = int(total_layers * freeze_ratio)
    named_layers = list(model.named_children())

    for name, layer in named_layers[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    print(f"Froze {num_layers_to_freeze} out of {total_layers} layers.")


def write_results(file_path, epoch=0, lr=0.0, error=0.0, f1_score=0.0, precision=0.0):
    """
    Manually write the results to a JSON file.

    Args:
        file_path (str): The path to the file where results will be written.
        epoch (int): The epoch number.
        lr (float): The learning rate.
        error (float): The error value.
        f1_score (float): The F1 score.
        precision (float): The precision value.
    """
    result = {
        "epoch": epoch.item() if torch.is_tensor(epoch) else epoch,
        "error": error.item() if torch.is_tensor(error) else error,
        "f1_score": f1_score.item() if torch.is_tensor(f1_score) else f1_score,
        "precision": precision.item() if torch.is_tensor(precision) else precision,
        "lr": lr.item() if torch.is_tensor(lr) else lr,
    }
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

        data.append(result)

        with open(file_path, "w") as file:
            json.dump(data, file)
    else:
        with open(file_path, "w") as file:
            json.dump([result], file)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
            trace_func (function): Trace print function. Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        Called to check whether early stopping should occur based on validation loss.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model to save if the validation loss decreases.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreases.

        Args:
            val_loss (float): The validation loss.
            model (torch.nn.Module): The model to be saved.
        """
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def setup_wandb(api_key, project, name, notes, config_dict):
    """
    Setup Weights & Biases (wandb) for experiment logging.

    Args:
        api_key (str): The API key for the wandb.
        project (str): The project name in wandb.
        name (str): The name of the current experiment run.
        notes (str): Additional notes for the experiment.
        config_dict (dict): The configuration dictionary for the experiment.
    """
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "dryrun"
    os.environ["WANDB_API_KEY"] = api_key

    # Ensure project name is less than or equal to 128 characters
    if len(project) > 128:
        project = project[:128]

    wandb.init(project=project, name=name, notes=notes)
    config = wandb.config
    for key, value in config_dict.items():
        setattr(config, key, value)


def get_metrics(task, num_classes, device):
    """
    Get the metrics for model evaluation.

    Args:
        task (str): The task to perform, either "multiclass" or "multilabel".
        num_classes (int): The number of classes in the dataset for multiclass tasks.
        device (torch.device): The device to use for computation (CPU or GPU).

    Returns:
        tuple: Accuracy, precision, recall, and F1 score metrics.
    """
    accuracy = torchmetrics.Accuracy(
        task=task, average="macro", num_labels=num_classes
    ).to(device)
    precision = torchmetrics.Precision(
        task=task, average="macro", num_labels=num_classes
    ).to(device)
    recall = torchmetrics.Recall(task=task, average="macro", num_labels=num_classes).to(
        device
    )
    f1_score = torchmetrics.F1Score(
        task=task, average="macro", num_labels=num_classes
    ).to(device)

    return accuracy, precision, recall, f1_score


def create_json(loss, epoch, result_dir):
    """
    Create a JSON file to store the loss values.

    Args:
        loss (float): The loss value to be stored.
        epoch (int): The current epoch number.
        result_dir (str): The directory where the JSON file will be saved.
    """
    data = {}
    data["loss"] = (
        loss.item() if isinstance(loss, torch.Tensor) else loss
    )  # convert Tensor to number
    data["epoch"] = epoch
    file_path = os.path.join(result_dir, "loss.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            existing_data = json.load(file)
        existing_data.append(data)

        with open(file_path, "w") as file:
            json.dump(existing_data, file)
    else:
        with open(file_path, "w") as file:
            json.dump([data], file)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The epoch number.
        loss (float): The loss value.
        path (str): The path to save the checkpoint.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def set_seed(seed):
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
