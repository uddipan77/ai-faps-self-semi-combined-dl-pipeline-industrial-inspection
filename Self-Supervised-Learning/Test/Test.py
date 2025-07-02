"""
This script is used to test multiple models trained through either self-supervised learning or downstream tasks. It loads the test data, loads the models, 
and evaluates their performance using various metrics such as accuracy, precision, recall, and F1 score.
"""

import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import torch.nn as nn
from data.Dataset import load_test_data
import torch.nn.functional as F
import torch

# from make_model import load_model
from utils.Utils import get_metrics
from modeling.train_validation_test import find_best_threshold_base, test_loop_mix
from modeling.make_model import load_model
import os

"""
Configuration dictionary is used for testing the model on the test dataset.
downstream: True if the model is trained on downstream task, False if the model is trained on pure supervised learning.
"""
config_dict = {
    "batch_size": 52,
    "num_workers": 8,
    "input_size": 221,
    "num_classes": 3,
    "base_dir": "/home/woody/iwfa/iwfa111h/Supervised_Data/linear_winding_images_with_labels",
    "test_csv": "/home/vault/iwfa/iwfa111h/Test_dataset.csv",
    "model_name": "efficientnet_v2_s",
    "checkpoint_path": "/home/hpc/iwfa/iwfa111h/shouvik/ai-faps-badar-alam/1th_efficientnet_v2_s-SIMCLR_EFFICIENT_NET_V2_S_25%_DOWNSTREAM_01/SIMCLR_EFFICIENT_NET_V2_S_25%_DOWNSTREAM_01-number-1.pt",
    "downstream": True, 
}


if __name__ == "__main__":
    """
    Main function to load the test data, model, and evaluate its performance on the test dataset.
    """
    # Load the test data using the provided paths
    test_loader = load_test_data(
        base_dir=config_dict["base_dir"],
        test_csv=config_dict["test_csv"],
        BATCHSIZE=config_dict["batch_size"],
        NW=config_dict["num_workers"],
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the evaluation metrics (accuracy, precision, recall, F1 score)
    metrics = get_metrics(
        task="multilabel", num_classes=config_dict["num_classes"], device=DEVICE
    )

    # Load the model based on whether it's trained on a downstream task or not
    if config_dict["downstream"] == True:
        backbone = load_model(
            config_dict["model_name"],
            config_dict["num_classes"],
            ssl=config_dict["downstream"],
        )

        model = nn.Sequential(
            backbone, nn.Flatten(), nn.Linear(1280, config_dict["num_classes"])
        )
    else:
        model = load_model(
            config_dict["model_name"], config_dict["num_classes"], ssl=False
        )

    # Load the model weights from the checkpoint file
    model.load_state_dict(
        torch.load(config_dict["checkpoint_path"], map_location=DEVICE)
    )
    model = model.to(DEVICE)

    # Define the loss function for evaluation
    loss = nn.BCEWithLogitsLoss()

    # Find the best threshold for each class to maximize F1 score
    threshold = find_best_threshold_base(model, test_loader, DEVICE)
    
    # Test the model on the test dataset with the best threshold for classification
    avg_test_loss, test_acc, test_prec, test_rec, test_f1 = test_loop_mix(
        model, test_loader, loss, DEVICE, metrics, threshold
    )

    # Print the final test results
    print(
        f"Test loss: {avg_test_loss}, Test accuracy: {test_acc}, Test precision: {test_prec}, Test recall: {test_rec}, Test f1: {test_f1}"
    )
