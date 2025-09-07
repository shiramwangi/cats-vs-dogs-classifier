from typing import Tuple
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_transforms(image_size: int = 128):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, val_tfms


def create_dataloaders(
    data_dir: str,
    image_size: int = 128,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, datasets.ImageFolder]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_tfms, val_tfms = build_transforms(image_size)

    # Check if data is already split into train/val folders
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    
    if train_dir.exists() and val_dir.exists():
        # Data is already split
        train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
        val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_tfms)
        print(f"Using pre-split data: train={len(train_dataset)}, val={len(val_dataset)}")
    else:
        # Data needs to be split
        full_dataset = datasets.ImageFolder(root=data_dir, transform=train_tfms)
        val_dataset = datasets.ImageFolder(root=data_dir, transform=val_tfms)
        
        total_size = len(full_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = torch.utils.data.random_split(
            range(total_size), [train_size, val_size], generator=generator
        )

        train_subset = torch.utils.data.Subset(full_dataset, train_indices.indices)
        val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
        
        train_dataset = train_subset
        val_dataset = val_subset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset
