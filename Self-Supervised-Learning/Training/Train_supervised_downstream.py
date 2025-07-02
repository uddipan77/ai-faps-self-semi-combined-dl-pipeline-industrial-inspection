"""
This script is used for supervised training or for downstreaming on SSL models.

To train a model on supervised learning, adjust the configuration dictionary values:
- SSL_downstream=False
- checkpoint_path=False

To train a model on downstreaming, adjust the configuration dictionary values:
- SSL_downstream=True
- checkpoint_path="path to SSL trained model"

Usage:
    python Train_supervised_downstream.py --experiment_number 1 --model efficientnet_v2_s
"""

import sys
import os
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)) )
import argparse
from utils.Utils import EarlyStopping, setup_wandb, freeze_model, set_seed
from data.Dataset import get_data
from modeling.train_validation_test import train_validation
from modeling.make_model import load_model
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--expriment_number", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument('--train_csv', type=str, default='/home/woody/iwfa/iwfa111h/Supervised_Data/Percentage_Splits/train_v2024-03-18_25%.csv',
                        help='Path to the training CSV file')
args = parser.parse_args()

# Configuration dictionary for experiment parameters
config_dict = {
    "batch_size": 124,  # Batch size for training
    "epochs": 300,  # Number of epochs for training
    "learning_rate": 0.00012606252121446308,  # Learning rate for the optimizer
    "model": args.model,  # Model type (e.g., efficientnet_v2_s)
    "dataset": "clean_lbl_v1",  # Dataset version being used
    "input_size": 203,  # Input size for the model
    "optimizer": "Adam",  # Optimizer type
    "loss_fn": "BCEWithLogitsLoss",  # Loss function used
    "scheduler": "CosineAnnealingLR",  # Learning rate scheduler
    "freeze": False,  # Whether to freeze model layers
    "freeze_ratio": 0.0,  # Ratio of layers to freeze
    "num_classes": 3,  # Number of output classes
    "checkpoint_point": "/home/vault/iwfa/iwfa111h/SSL_REPRODUCTION/simCLR_EfficientNet_v2_s/Earlystopping.pth",  # Pretrained model checkpoint for downstream tasks
    "SSL_downstream": True,  # Flag to indicate if SSL downstream task is being performed
    "expriment_name": f"{args.model}-{args.expriment_number}Full_dataset_data_clean_lbl_v1",  # Experiment name for tracking
    "expriment_results_saved": f"{args.expriment_number}th_{args.model}-{args.experiment_name}",  # Path to save the experiment results
    "training_dataset": args.train_csv,  # Path to the training dataset
}


def main_run():
    """
    Main function to run the training process.
    - Initializes the model, optimizer, and other training configurations.
    - Runs the training loop for the specified number of epochs.
    """
    # Set up device and create results directory to save the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    if not os.path.exists(config_dict["expriment_results_saved"]):
        os.makedirs(config_dict["expriment_results_saved"])

    # Set up wandb for experiment tracking (currently commented out)
    # setup_wandb(
    #     api_key="b2705a044aea7761c3c51b7c28d51c0994f4c766",
    #     project="efficient_net",
    #     name=config_dict["expriment_name"],
    #     notes="effcientb6 is trained with cleanlab labels version 1",
    #     config_dict=config_dict,
    # )

    # Loading the train and validation data
    train_loader, valid_data = get_data(
        train_csv=config_dict["training_dataset"],
        validation_csv="/home/vault/iwfa/iwfa111h/Validation_datatset.csv",
        input_size=config_dict["input_size"],
        BATCHSIZE=config_dict["batch_size"],
        base_dir="/home/woody/iwfa/iwfa111h/Supervised_Data/linear_winding_images_with_labels",
        NW=8,
    )

    # Initialize model
    model = load_model(config_dict["model"], num_classes=config_dict["num_classes"], ssl=config_dict["SSL_downstream"], checkpoint_path=config_dict["checkpoint_point"]).to(
        DEVICE
    )
    
    # Add the classifier layer if SSL_downstream is True
    if config_dict["SSL_downstream"]:
        model = nn.Sequential(
            model, nn.Flatten(), nn.Linear(1280, config_dict["num_classes"])
        )

    # Freeze the model if freeze is True
    if config_dict["freeze"]:
        freeze_model(model, config_dict["freeze_ratio"])

    # Defining the model, loss function, optimizer, and scheduler
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config_dict["learning_rate"])

    # Scheduler setup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=625, eta_min=0.00920122303024355
    )

    # Early stopping initialization and saving the best model on loss decrease
    earlystopping = EarlyStopping(
        patience=15,
        verbose=True,
        path=f'{config_dict["expriment_results_saved"]}/{args.experiment_name}-number-{args.expriment_number}.pt',
    )

    # Training loop
    for epoch in range(0, config_dict["epochs"]):
        print(f"Epoch {epoch}")

        # Train the model and evaluate on validation data
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
            valid_data,
            optimizer,
            loss_fn,
            epoch,
            DEVICE,
        )

        # Log the model checkpoints every 20 epochs
        if epoch % 20 == 0:
            torch.save(
                model.state_dict(),
                f"{config_dict['expriment_results_saved']}/model-{epoch}.pt",
            )

        # Step the scheduler with validation loss
        scheduler.step(val_loss)

        # Apply early stopping based on validation loss
        earlystopping(val_loss, model)
        if earlystopping.early_stop:
            print("Early stopping")
            break

    # Final wandb logging (currently commented out)
    # wandb.finish()


if __name__ == "__main__":
    main_run()
# start decay lr if val_loss not decrease
