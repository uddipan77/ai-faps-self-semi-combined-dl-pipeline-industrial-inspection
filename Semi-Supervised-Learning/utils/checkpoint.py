import os
import torch

class CheckpointManager:
    """
    A utility class for managing model checkpoints during training.

    This class handles saving and loading of model and optimizer states at various 
    points during the training process. It also supports saving additional training 
    information, such as best validation scores and early stopping counters.

    Attributes:
        checkpoint_dir (str): Directory where checkpoints will be saved and loaded from.
    """

    def __init__(self, checkpoint_dir):
        """
        Initialize the CheckpointManager.

        Args:
            checkpoint_dir (str): Directory to save and load checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch, best_val_f1, train_f1, epoch_loss, epoch_unsupervised_loss, early_stop_counter, model, optimizer, experiment_name):
        """
        Save the current model and optimizer states, along with additional training information.

        Args:
            epoch (int): The current epoch number.
            best_val_f1 (float): The best validation F1 score achieved so far.
            train_f1 (float): The current training F1 score.
            epoch_loss (float): The current epoch loss.
            epoch_unsupervised_loss (float): The current unsupervised loss.
            early_stop_counter (int): The number of epochs since the last improvement in validation F1.
            model (torch.nn.Module): The model whose state will be saved.
            optimizer (torch.optim.Optimizer): The optimizer whose state will be saved.
            experiment_name (str): The name of the experiment, used to create the checkpoint filename.

        Saves the checkpoint as a `.pth` file in the specified `checkpoint_dir`.
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_f1": best_val_f1,
            "train_f1": train_f1,
            "loss": epoch_loss,
            "unsupervised_loss": epoch_unsupervised_loss,
            "early_stop_counter": early_stop_counter,
            'experiment_name': experiment_name
        }

        checkpoint_filename = f"{experiment_name}.pth"
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, checkpoint_filename))
        print(f"Checkpoint saved at epoch {epoch}. Best Validation F1: {best_val_f1:.4f}")

    def load_checkpoint(self, checkpoint_path, model, optimizer):
        """
        Load a checkpoint and resume training from the specified epoch.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            model (torch.nn.Module): The model to load the state dict into.
            optimizer (torch.optim.Optimizer): The optimizer to load the state dict into.

        Returns:
            tuple: 
                - int: The epoch number to resume training from.
                - float: The best validation F1 score loaded from the checkpoint.
                - int: The early stop counter, indicating the number of epochs since the last improvement.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_f1 = checkpoint['best_val_f1']
        early_stop_counter = checkpoint['early_stop_counter']
        
        print(f"Checkpoint loaded: resume training from epoch {start_epoch}, with best validation F1-score of {best_val_f1:.4f}")
        
        return start_epoch, best_val_f1, early_stop_counter
