"""
This script contains the data loading functions for supervised learning and self-supervised learning.

Supervised Learning Data Loading:
- Winding_Dataset: Custom dataset class for the linear winding dataset in supervised training.
- get_data: Load the data for supervised training.
- load_test_data: Load the test data for model testing.
- get_data_cifar10: Load the CIFAR-10 dataset for supervised training.
- get_cifar10_testdataset: Load the CIFAR-10 test dataset.

Self-Supervised Learning Data Loading:
- FilteredImageDataset: Dataloading function for self-supervised training with filtering the test and validation coils.

Helper Functions:
- has_file_allowed_extension: Checks if a file has an allowed extension.
"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import re
import PIL
from PIL import Image, ImageFile
import warnings
import os


class Winding_Dataset(Dataset):
    """
    Custom dataset class for the linear winding dataset in supervised training.

    Args:
        csv_file (str): Path to the CSV file containing the dataset information.
        root_dir (str): Path to the root directory containing the images.
        input_size (int, optional): The size of the input image (default is 224).
        mode (str, optional): The mode for image conversion (default is "L").
    """

    def __init__(self, csv_file, root_dir, input_size=224, mode="L"):
        """
        Initialize the dataset class.

        Args:
            csv_file (str): Path to the CSV file.
            root_dir (str): Path to the root directory.
            input_size (int): The size of the input image.
            mode (str): The mode for image conversion (default is "L").
        """
        self.data = pd.read_csv(csv_file, sep=",")
        self.input_size = input_size
        self.mode = mode
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.main_dir = root_dir
        self.total_imgs = len(self.data)

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            int: The total number of images.
        """
        return self.total_imgs

    def __getitem__(self, idx):
        """
        Get a single image and its corresponding labels.

        Args:
            idx (int): The index of the image to be retrieved.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The image tensor and its corresponding labels tensor.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img_loc1 = os.path.join(self.main_dir, self.data["image"][idx])
            x = (
                PIL.Image.open(img_loc1)
                .convert(self.mode)
                .resize((self.input_size, self.input_size))
            )
            column_names = self.data.columns[1:-1]

            indices = [self.data.columns.get_loc(c) for c in column_names]
            labels = self.data.iloc[idx, indices]  # Select data using column names
            labels = torch.tensor(labels, dtype=torch.float32)
            image = self.transform(x)
            return image, labels


def get_data(train_csv, validation_csv, base_dir, input_size, BATCHSIZE=20, NW=4):
    """
    Load the data for the supervised training.

    Args:
        train_csv (str): Path to the training CSV file.
        validation_csv (str): Path to the validation CSV file.
        base_dir (str): Path to the root directory containing the images.
        input_size (int): The size of the input image.
        BATCHSIZE (int, optional): The batch size (default is 20).
        NW (int, optional): The number of workers (default is 4).

    Returns:
        Tuple[DataLoader, DataLoader]: The training and validation data loaders.
    """
    mode = "RGB"
    train_dataset = Winding_Dataset(
        csv_file=train_csv,
        root_dir=base_dir,
        input_size=input_size,
        mode=mode,
    )
    valid_dataset = Winding_Dataset(
        csv_file=validation_csv,
        root_dir=base_dir,
        input_size=input_size,
        mode=mode,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCHSIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NW,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=BATCHSIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NW,
    )

    print("data loaded")
    return train_loader, valid_loader


def load_test_data(base_dir, test_csv, BATCHSIZE=20, NW=4):
    """
    Load the test data for the model testing.

    Args:
        base_dir (str): Path to the root directory containing the images.
        test_csv (str): Path to the test CSV file.
        BATCHSIZE (int, optional): The batch size (default is 20).
        NW (int, optional): The number of workers (default is 4).

    Returns:
        DataLoader: The test data loader.
    """
    test_dataset = Winding_Dataset(csv_file=test_csv, root_dir=base_dir, mode="RGB")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCHSIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=NW,
    )
    print("data loaded")
    return test_loader


def get_data_cifar10(input_size, BATCHSIZE=50, NW=4):
    """
    Load the CIFAR-10 dataset for supervised training.

    Args:
        input_size (int): The size of the input image.
        BATCHSIZE (int, optional): The batch size (default is 50).
        NW (int, optional): The number of workers (default is 4).

    Returns:
        Tuple[DataLoader, DataLoader]: The training and validation data loaders.
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="/home/vault/iwfa/iwfa100h/SSL_Just_Train/data",
        train=True,
        download=True,
        transform=transform,
    )
    valid_dataset = torchvision.datasets.CIFAR10(
        root="/home/vault/iwfa/iwfa100h/SSL_Just_Train/data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=NW
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=NW
    )

    return train_loader, valid_loader


def get_cifar10_testdataset(BATCHSIZE):
    """
    Load the CIFAR-10 test dataset.

    Args:
        BATCHSIZE (int): The batch size.

    Returns:
        DataLoader: The test data loader.
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=8
    )
    return test_loader


class FilteredImageDataset(Dataset):
    """
    Dataloading function for self-supervised training with filtering the test and validation coils.

    Args:
        root_dir (str): Path to the root directory containing the images.
        patterns_to_ignore (list): List of patterns to ignore containing the coil numbers of the test and validation coils.
        transform (callable, optional): The transform function (default is None).
    """

    def __init__(self, root_dir, patterns_to_ignore, transform=None):
        """
        Initialize the dataset class.

        Args:
            root_dir (str): Path to the root directory.
            patterns_to_ignore (list): List of patterns to ignore.
            transform (callable): The transform function.
        """
        self.root_dir = root_dir
        self.patterns_to_ignore = patterns_to_ignore
        self.transform = transform
        self.image_paths = self._load_image_paths()
        self.suple_identifiers = self._extract_suple_identifiers()

    def _load_image_paths(self):
        """
        Load the image paths excluding those with the patterns to ignore.

        Returns:
            list: A list of valid image paths.
        """
        image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                    if not any(pattern in file for pattern in self.patterns_to_ignore):
                        image_paths.append(os.path.join(root, file))
        return image_paths

    def _extract_suple_identifiers(self):
        """
        Extract the unique identifiers from the image paths.

        Returns:
            set: A set of unique identifiers.
        """
        suple_identifiers = set()
        suple_pattern = re.compile(r"Spule\d+")
        for path in self.image_paths:
            match = suple_pattern.search(os.path.basename(path))
            if match:
                suple_identifiers.add(match.group())
        return suple_identifiers

    def get_unique_suple_count(self):
        """
        Get the count of unique suple identifiers.

        Returns:
            int: The number of unique suple identifiers.
        """
        return len(self.suple_identifiers)

    def get_unique_suple_identifiers(self):
        """
        Get the unique suple identifiers.

        Returns:
            set: A set of unique suple identifiers.
        """
        return self.suple_identifiers

    def __len__(self):
        """
        Return the number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a pair of images for self-supervised training.

        Args:
            idx (int): The index of the image pair to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The two image tensors.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image1 = self.transform(image)
                image2 = self.transform(image)
        return image1, image2


def has_file_allowed_extension(filename, extensions):
    """
    Checks if a file has an allowed extension.

    Args:
        filename (str): The name of the file.
        extensions (tuple): A tuple of allowed file extensions.

    Returns:
        bool: True if the file has an allowed extension, False otherwise.
    """
    return filename.lower().endswith(extensions)
