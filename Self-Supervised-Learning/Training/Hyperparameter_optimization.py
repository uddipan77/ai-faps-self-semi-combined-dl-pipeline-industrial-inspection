"""
Script for Hyperparameter optimization for SSL_downstream or supervised learning process.

This script imports necessary libraries and modules for optimizing a downstream task using Optuna. 
It defines an objective function that takes Optuna trial as input and performs the optimization process.
The objective function creates a model, sets up the optimizer and scheduler, and trains the model using the provided dataset. 

The script also includes the main function that creates an Optuna study, sets up the storage and pruner, and starts the optimization process. 
The study is loaded if it already exists, and the remaining trials are executed. The best trial and study statistics are printed at the end.

Usage:
    - python Hyperparameter_optimization.py --model_name efficientnet_v2_s --experiment_name Quater_dataset --training_dataset /home/vault/iwfa/iwfa100h/ai-faps-badar-alam/data-csv/Quater_dataset.csv
"""
import sys
import os
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)) )
import argparse
from modeling.make_model import load_model
from modeling.train_validation_test import train_validation
from data.Dataset import get_data
import numpy as np
from utils.Utils import setup_wandb, set_seed
from optuna.pruners import MedianPruner, PatientPruner
import torch.utils.data
import torch.optim as optim
from torch import nn
import torch
from optuna.trial import TrialState
import optuna
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

# Set up the argument parser
parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
parser.add_argument("--model_name", type=str, default="efficientnet_v2_s")
parser.add_argument("--experiment_name", type=str, default="Full_dataset")
parser.add_argument(
    "--training_dataset",
    type=str,
    default="/home/vault/iwfa/iwfa111h/Ten_percentage_datatset.csv",
)

args = parser.parse_args()

# Set the random seed for reproducibility
set_seed(42)

# Define the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration dictionary for the experiment parameters
config_dict = {
    "epochs": 10,  # Number of epochs for each trial model is trained for
    "backbone": "False",  # Used to load the backbone from dictionary of saved checkpoints  like backbone in case of SIMCLR. Hint: Have look where we are saving the checkpoint during SSL training
    "model": args.model_name,  # Name of the model # Type of dataset, e.g., half, full, quarter_dataset, etc.
    "dataset": "clean_lbl_v2",
    "SSL_downstream": True,  # If true, add the classifier layer which is removed during SSL training to load the SSL trained weights and then append the classifier layer
    "loss_fn": "BCEWithLogitsLoss",  # Loss function to calculate the loss
    "num_classes": 3,  # Number of classes
    # Name of the experiment either is SSL algorithm (Simclr,Barlow Twins etc) or SL (supervised learning)
    "expriment_name": f"optuna_exirments{args.model_name}_simclr_700coil_{args.experiment_name}",
    # Path to save the Optuna trials
    "expriment_results_saved": f"optuna_{args.model_name}__simclr_700coil_{args.experiment_name}",
    # Path of .csv file to train the model
    "training_dataset": args.training_dataset,
    # Contains the checkpoint path which is provided in case of SSL downstreaming and in case of supervised training it is "False"
    "checkpoint_dir": "/home/vault/iwfa/iwfa111h/SSL_REPRODUCTION/simCLR_EfficientNet_v2_s/Earlystopping.pth",
    # If True, load the pretrained weights of the model like ImageNet weights
    "pretrained": True,
    "num_workers": 8,  # Number of workers to load the data
}

# setup_wandb(
#     api_key="",
#     project="optuna expriemnets",
#     name=config_dict["expriment_name"],
#     notes=f"optuna expriments for {config_dict['expriment_name']}",
#     config_dict=config_dict,
# )


def objective(trial):
    """
    Objective function for the hyperparameter optimization process. The objective function trains the model using the given trial parameters and evaluates it.
    
    Args:
        trial (optuna.Trial): The current trial object containing hyperparameter suggestions.
    
    Returns:
        float: The average F1 score for the validation dataset after training.
    """
    print("trial number: ", trial.number)

    # Define the classification criterion (BCEWithLogitsLoss)
    classification_criterion = nn.BCEWithLogitsLoss()

    # Suggest a batch size for this trial
    BATCHSIZE = trial.suggest_int("batch_size", 50, 128)
    best_F1 = 0

    # Load the model based on the configuration dictionary
    backbone = load_model(
        model_name=config_dict["model"],
        num_classes=config_dict["num_classes"],
        checkpoint_path=config_dict["checkpoint_dir"],
        backbone=config_dict["backbone"],
        pretrain=config_dict["pretrained"],
        ssl=config_dict["SSL_downstream"],
    ).to(DEVICE)

    # Append the classifier layer if SSL_downstream is True
    if config_dict["SSL_downstream"]:
        model = nn.Sequential(
            backbone, nn.Flatten(), nn.Linear(1280, config_dict["num_classes"])
        )
    else:
        model = backbone
    model = model.to(DEVICE)

    # Freeze the layers based on trial suggestion
    freez_or_not = trial.suggest_categorical("freeze", [True, False])
    if freez_or_not:
        frozen_layers = []
        total_layers = len(list(model.named_children()))

        # Freeze the layers based on the freeze percentage suggested by the trial
        freeze_percentage = trial.suggest_float("freeze_percentage", 0.1, 0.9)
        print(f"freeze_percentage: {freeze_percentage}")
        num_layers_to_freeze = int(total_layers * freeze_percentage)
        named_layers = list(model.named_children())
        for name, layer in named_layers[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
                frozen_layers.append(name)

        trial.set_user_attr("frozen_layers", frozen_layers)
    else:
        pass

    # Define optimizer and scheduler based on trial suggestions
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "SGD"]
    )
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["ReduceLROnPlateau", "CosineAnnealingLR"]
    )
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    image_input = trial.suggest_int("image_input", 100, 224)

    if scheduler_name == "ReduceLROnPlateau":
        factor = trial.suggest_float("factor", 0.1, 0.9)
        patience = trial.suggest_int("patience", 0, 50)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=patience, factor=factor
        )
    elif scheduler_name == "CosineAnnealingLR":
        T_max = trial.suggest_int("T_max", 50, 1000)
        eta_min = trial.suggest_float("eta_min", 0, 0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

    f1_score_lst = []

    # Load training and validation datasets
    train_loader, valid_loader = get_data(
        train_csv=config_dict["training_dataset"],
        validation_csv="/home/vault/iwfa/iwfa111h/Validation_datatset.csv",
        input_size=image_input,
        BATCHSIZE=BATCHSIZE,
        base_dir="/home/woody/iwfa/iwfa111h/Supervised_Data/linear_winding_images_with_labels",
        NW=config_dict["num_workers"],
    )

    # Train the model and calculate validation metrics
    for epoch in range(config_dict["epochs"]):
        (
            val_loss,
            val_acc,
            val_prec,
            val_rec,
            val_f1,
            train_acc,
            train_loss,
            train_prec,
            train_rec,
            train_f1,
        ) = train_validation(
            model,
            train_loader,
            valid_loader,
            optimizer,
            classification_criterion,
            epoch,
            DEVICE,
        )

        # Update the scheduler based on validation loss
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Log the validation F1 score
        trial.report(val_f1.cpu().numpy(), epoch)
        f1_score_lst.append(val_f1.cpu().numpy())

        # Prune the trial if needed
        if trial.should_prune():
            print("pruning")
            raise optuna.TrialPruned()

        # Update the best F1 score
        if best_F1 < val_f1:
            best_F1 = val_f1

    return np.average(f1_score_lst)


if __name__ == "__main__":
    """
    Main function to set up the Optuna study and start the optimization process.
    """

    # Set up the study name and storage
    study_name = f"optuna-exp__{args.model_name}_BYOL{args.experiment_name}"
    storage_name = f"sqlite:///{args.model_name}_BYOL{args.experiment_name}.db"
    
    # Set up the pruner
    pruner = PatientPruner(MedianPruner(), patience=5)

    # Create the study and start optimizing the objective function
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    # Get the number of completed and remaining trials
    completed_trials = len(
        study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    )
    remaining_trials = max(0, 200 - completed_trials)
    print(f"completed_trials: {completed_trials}")
    print(f"remaining_trials: {remaining_trials}")

    # Optimize the objective function
    study.optimize(objective, n_trials=remaining_trials)

    # Get pruned and complete trials
    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    # Print the study statistics and the best trial
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
