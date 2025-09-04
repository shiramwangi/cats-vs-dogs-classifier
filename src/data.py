from typing import Tuple
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_transforms(image_size: int = 128):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
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

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_tfms)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_tfms)

    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Ensure validation uses deterministic transforms
    val_subset.dataset = val_dataset

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, full_dataset
