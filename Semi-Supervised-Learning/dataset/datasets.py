import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Custom dataset for labeled images
class CustomImageDataset(Dataset):
    """
    Dataset for labeled images with transformations.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image file names and labels.
        image_dir (str): Directory containing the image files.
        y_columns (list): List of columns in `dataframe` corresponding to the target labels.
        transform (callable, optional): Transformations to apply to images.
    """
    
    def __init__(self, dataframe, image_dir, y_columns, transform=None):
        """
        Initialize the dataset with the given parameters.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image file names and labels.
            image_dir (str): Directory containing the image files.
            y_columns (list): List of columns in `dataframe` corresponding to the target labels.
            transform (callable, optional): Transformations to apply to images.
        """
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.y_columns = y_columns
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Loads and returns an image and its corresponding label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Transformed image and its labels as a tensor.
        """
        img_name = f"{self.image_dir}/{self.dataframe.iloc[idx, 0]}"
        image = Image.open(img_name).convert('L').convert('RGB')
        labels = torch.tensor(self.dataframe.iloc[idx][self.y_columns].to_numpy().astype('float32'))
        if self.transform:
            image = self.transform(image)
        return image, labels

# Custom dataset for unlabeled images
class FixmatchUnlabeledImageDataset(Dataset):
    """
    Dataset for unlabeled images with weak and strong augmentations for the FixMatch algorithm.

    Args:
        image_dir (str): Directory containing the unlabeled image files.
        normalize (callable): Normalization transformation to apply to images.
        weak_transform (callable): Weak transformation for weakly augmented images.
        strong_transform (callable): Strong transformation for strongly augmented images.
    """
    
    def __init__(self, image_dir, normalize, weak_transform=None, strong_transform=None):
        """
        Initialize the FixMatch dataset with the given transformations.

        Args:
            image_dir (str): Directory containing the unlabeled image files.
            normalize (callable): Normalization transformation to apply to images.
            weak_transform (callable, optional): Weak transformation for weakly augmented images.
            strong_transform (callable, optional): Strong transformation for strongly augmented images.
        """
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.normalize = normalize
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: The number of unlabeled samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves and returns a weakly and strongly augmented version of an unlabeled image.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing weakly augmented and strongly augmented images.
        """
        img_name = f"{self.image_dir}/{self.image_files[idx]}"
        image = Image.open(img_name).convert('L').convert('RGB')
        weak_image = self.weak_transform(image) if self.weak_transform else image
        strong_image = self.strong_transform(image) if self.strong_transform else image
        weak_image = self.normalize(weak_image)
        strong_image = self.normalize(strong_image)
        return weak_image, strong_image

class MixmatchUnlabeledImageDataset(Dataset):
    """
    Dataset for unlabeled images with multiple augmentations for the MixMatch algorithm.

    Args:
        image_dir (str): Directory containing the unlabeled image files.
        new_transform (callable, optional): Additional transformation applied to images before `transform`.
        transform (callable): Transformation applied to images after `new_transform`.
        k_augmentations (int): Number of augmented versions of each image to produce.
    """
    
    def __init__(self, image_dir, new_transform=None, transform=None, k_augmentations=3):
        """
        Initialize the MixMatch dataset with the given parameters.

        Args:
            image_dir (str): Directory containing the unlabeled image files.
            new_transform (callable, optional): Additional transformation applied to images before `transform`.
            transform (callable): Transformation applied to images after `new_transform`.
            k_augmentations (int): Number of augmented versions of each image to produce.
        """
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform
        self.new_transform = new_transform
        self.k_augmentations = k_augmentations

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: The number of unlabeled samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves and returns multiple augmented versions of an unlabeled image.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: A tensor containing multiple augmented versions of the image.
        """
        img_name = f"{self.image_dir}/{self.image_files[idx]}"
        image = Image.open(img_name).convert('L').convert('RGB')
        
        augmented_images = []
        for _ in range(self.k_augmentations):
            if self.new_transform:
                augmented_image = self.new_transform(image)
            else:
                augmented_image = image
            
            if self.transform:
                augmented_image = self.transform(augmented_image)
            
            augmented_images.append(augmented_image)
        
        # Stack the augmented images into a single tensor
        augmented_images_tensor = torch.stack(augmented_images)
        
        return augmented_images_tensor
