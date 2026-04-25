"""
dataset.py
----------
Dataset class and dataloader factory for the Chest X-Ray Pneumonia detection project.

Classes
-------
ChestXrayDataset
    Custom PyTorch Dataset for loading chest X-ray images from a directory
    structured as:
        root/
            NORMAL/
            PNEUMONIA/

Functions
---------
get_dataloaders(processed_dir, batch_size)
    Returns train, validation, and test DataLoader objects.

Label Convention
----------------
    0 = NORMAL
    1 = PNEUMONIA
"""

import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

class ChestXrayDataset(Dataset):
    """
    Custom Dataset for chest X-ray binary classification.

    Reads images from a directory containing two subfolders:
    NORMAL (label 0) and PNEUMONIA (label 1).

    Parameters
    ----------
    data_dir : str
        Path to the directory containing NORMAL and PNEUMONIA subfolders.
    transform : callable, optional
        A torchvision transform pipeline to apply to each image.

    Attributes
    ----------
    img_paths : list of str
        File paths to all valid images in the dataset.
    labels : list of int
        Corresponding class labels (0 = NORMAL, 1 = PNEUMONIA).
    """

    SUPPORTED_EXTENSIONS = ('.jpeg', '.jpg', '.png')

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self._load_image_paths()

    def _load_image_paths(self):
        """Scan subdirectories and populate img_paths and labels lists."""
        for label, subfolder in enumerate(['NORMAL', 'PNEUMONIA']):
            subfolder_path = os.path.join(self.data_dir, subfolder)
            if not os.path.exists(subfolder_path):
                raise FileNotFoundError(
                    f"Expected subfolder not found: {subfolder_path}"
                )
            for img_name in os.listdir(subfolder_path):
                if img_name.lower().endswith(self.SUPPORTED_EXTENSIONS):
                    self.img_paths.append(os.path.join(subfolder_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Load and return a single image and its label.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        img : torch.Tensor
            Transformed image tensor of shape (3, H, W).
        label : int
            Class label (0 = NORMAL, 1 = PNEUMONIA).
        """
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# ImageNet normalisation statistics used for all pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Dataloader Factory
# ---------------------------------------------------------------------------

def get_dataloaders(processed_dir, batch_size=32):
    """
    Build and return DataLoader objects for train, validation, and test splits.

    Augmentation is applied only to the training set. Validation and test
    sets use deterministic preprocessing only.

    Parameters
    ----------
    processed_dir : str
        Root directory containing 'train', 'val', and 'test' subdirectories.
    batch_size : int, optional
        Number of samples per batch. Default is 32.

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    test_loader  : DataLoader
    """
    train_dataset = ChestXrayDataset(
        os.path.join(processed_dir, 'train'),
        transform=train_transforms
    )
    val_dataset = ChestXrayDataset(
        os.path.join(processed_dir, 'val'),
        transform=val_test_transforms
    )
    test_dataset = ChestXrayDataset(
        os.path.join(processed_dir, 'test'),
        transform=val_test_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader